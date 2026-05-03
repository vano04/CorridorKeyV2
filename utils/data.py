from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F

from matting_common import (
    composite_fg as _composite_fg,
    fg_target_from_premul as _fg_target_from_premul,
    validate_fg_representation as _validate_fg_representation,
)

EPS = 1e-6
RAW_COLOR_MAX = 256.0

# Tri-state switch for the B=1 zero-copy collate fast path:
#   "auto" (default): enable when the per-sample tensor dtype is smaller
#       than fp32 (e.g. bf16/fp16), legacy alloc+copy when fp32. The fast
#       path keeps the worker's torch.stack output alive through the full
#       IPC -> pin -> H2D pipeline; at fp32 with many workers x deep
#       prefetch queues this was measured to evict OS page cache of the
#       EXR shards (-14% throughput at 6 workers, prefetch_factor=6,
#       2048^2 native res). Halving the dtype halves the held memory and
#       the regression vanishes -- so we only auto-enable when the user
#       has already opted into a smaller dtype via ``data.host_dtype``.
#   "1"/"true"/"on": force the fast path regardless of dtype. Useful for
#       1-2 worker dev runs where the page-cache pressure is irrelevant
#       and the worker-CPU saving (~17% throughput at 1 worker) wins.
#   "0"/"false"/"off": force the legacy alloc+copy path. Always safe.
_COLLATE_FAST_MODE: str = os.environ.get("CORRIDORKEY_COLLATE_FAST", "auto").strip().lower()
if _COLLATE_FAST_MODE in {"", "auto"}:
    _COLLATE_FAST_FORCE: Optional[bool] = None
elif _COLLATE_FAST_MODE in {"1", "true", "on", "yes"}:
    _COLLATE_FAST_FORCE = True
elif _COLLATE_FAST_MODE in {"0", "false", "off", "no"}:
    _COLLATE_FAST_FORCE = False
else:
    raise ValueError(
        f"CORRIDORKEY_COLLATE_FAST={_COLLATE_FAST_MODE!r} is not recognised. "
        "Use one of: auto (default), 1/true/on, 0/false/off."
    )


def _ensure_color_three_channels(x: Tensor) -> Tensor:
    # Color tensors are expected as [T, C, H, W] and normalized to RGB channels.
    if x.ndim != 4:
        raise ValueError(f"Expected color tensor with shape [T, C, H, W], got {tuple(x.shape)}")

    c = int(x.shape[1])
    if c == 3:
        return x
    if c == 1:
        return x.repeat(1, 3, 1, 1)
    if c == 2:
        return torch.cat([x, x[:, :1]], dim=1)
    if c > 3:
        return x[:, :3]

    raise ValueError(f"Invalid channel count for color tensor: {c}")


def _ensure_alpha_single_channel(alpha: Tensor) -> Tensor:
    # alpha is expected [T, C, H, W]
    if alpha.shape[1] == 1:
        return alpha
    return alpha[:, :1]


def _sanitize_color_tensor(x: Tensor) -> Tensor:
    """Keep source HDR finite without crushing real highlights.

    CorridorKey EXRs are HDR, so values above 1.0 are expected.  We only
    replace non-finite values and cap implausible infinities/overflows high
    enough to preserve observed HDR clips before the final LDR training clamp.
    """
    return torch.nan_to_num(x, nan=0.0, posinf=RAW_COLOR_MAX, neginf=0.0).clamp_min(0.0).clamp_max(RAW_COLOR_MAX)


def _sanitize_alpha_tensor(alpha: Tensor) -> Tensor:
    return torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


def _green_foreground_challenge_premul(
    fg_premul: Tensor,
    alpha: Tensor,
    strength: float,
    green_rgb: Tuple[float, float, float],
    eps: float = 1e-3,
) -> Tensor:
    """Tint visible FG toward key-green while preserving alpha/shading.

    The source FG is premultiplied here. We temporarily un-premultiply in
    visible pixels, recolor by luminance, then premultiply again so the target
    alpha remains "foreground" even for strongly green subject regions.
    """
    alpha_safe = alpha.clamp_min(eps)
    visible = (alpha > 0.05).to(fg_premul.dtype)
    straight = fg_premul / alpha_safe
    luma = 0.299 * straight[:, 0:1] + 0.587 * straight[:, 1:2] + 0.114 * straight[:, 2:3]

    color = torch.tensor(green_rgb, dtype=fg_premul.dtype, device=fg_premul.device).view(1, 3, 1, 1)
    color_luma = max(EPS, 0.299 * green_rgb[0] + 0.587 * green_rgb[1] + 0.114 * green_rgb[2])
    target = luma * (color / color_luma)
    blend = visible * float(max(0.0, min(1.0, strength)))
    straight = straight * (1.0 - blend) + target * blend
    return straight * alpha


def _gaussian_blur(x: Tensor, kernel_size: int = 5, sigma: float = 1.0) -> Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1

    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-(coords**2) / (2 * sigma * sigma + EPS))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    c = x.shape[1]
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(c, 1, -1, -1)
    return F.conv2d(x, kernel, padding=radius, groups=c)


def _morphological(x: Tensor, op: str, kernel: int) -> Tensor:
    pad = kernel // 2
    if op == "dilate":
        return F.max_pool2d(x, kernel_size=kernel, stride=1, padding=pad)
    if op == "erode":
        return -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=pad)
    raise ValueError(f"Unknown op: {op}")


def generate_coarse_alpha_init(alpha0: Tensor) -> Tensor:
    # alpha0: [1, H, W]
    alpha = alpha0.unsqueeze(0)

    # Low-resolution degradation.
    factor = random.choice([2, 2, 3, 4, 6])
    h, w = alpha.shape[-2:]
    h2 = max(1, h // factor)
    w2 = max(1, w // factor)
    alpha = F.interpolate(alpha, size=(h2, w2), mode="bilinear", align_corners=False)
    alpha = F.interpolate(alpha, size=(h, w), mode="bilinear", align_corners=False)

    # Biased morphology: more often erode than dilate.
    if random.random() < 0.8:
        alpha = _morphological(alpha, op="erode", kernel=random.choice([3, 5, 7]))
    else:
        alpha = _morphological(alpha, op="dilate", kernel=random.choice([3, 5]))

    # Boundary jitter via subpixel shifts.
    shift_x = random.uniform(-2.0, 2.0)
    shift_y = random.uniform(-2.0, 2.0)
    theta = torch.tensor(
        [[[1.0, 0.0, 2.0 * shift_x / max(1.0, w)], [0.0, 1.0, 2.0 * shift_y / max(1.0, h)]]],
        device=alpha.device,
        dtype=alpha.dtype,
    )
    grid = F.affine_grid(theta, alpha.shape, align_corners=False)
    alpha = F.grid_sample(alpha, grid, mode="bilinear", padding_mode="border", align_corners=False)

    # Blur to produce soft but imperfect guidance.
    k = random.choice([3, 5, 7, 9])
    sigma = random.uniform(0.7, 2.5)
    alpha = _gaussian_blur(alpha, kernel_size=k, sigma=sigma)

    # Add slight structural noise.
    noise = torch.randn_like(alpha) * random.uniform(0.0, 0.03)
    alpha = (alpha + noise).clamp(0.0, 1.0)

    return alpha.squeeze(0)


def _boundary_band(alpha: Tensor, kernel_size: int = 3) -> Tensor:
    # alpha: [T, 1, H, W]
    dil = F.max_pool2d(alpha, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    ero = -F.max_pool2d(-alpha, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (dil - ero).clamp(0.0, 1.0)


def _random_resize(video_tensors: List[Tensor], max_side_target: int) -> List[Tensor]:
    # Tensors are [T, C, H, W]
    h, w = video_tensors[0].shape[-2:]
    scale = max_side_target / float(max(h, w))
    h2 = max(1, int(round(h * scale)))
    w2 = max(1, int(round(w * scale)))

    out = []
    for x in video_tensors:
        mode = "bilinear" if x.shape[1] > 1 else "nearest"
        out.append(F.interpolate(x, size=(h2, w2), mode=mode, align_corners=False if mode == "bilinear" else None))
    return out


def _resize_short_side(video_tensors: List[Tensor], short_side_target: int) -> List[Tensor]:
    """Resize so the *shorter* axis equals ``short_side_target``.

    Used as the pre-step for a fixed-size crop: guarantees that the crop
    can always be placed (i.e. ``min(h, w) >= crop_size``) without
    distorting the source aspect ratio.
    """
    h, w = video_tensors[0].shape[-2:]
    scale = short_side_target / float(min(h, w))
    h2 = max(1, int(round(h * scale)))
    w2 = max(1, int(round(w * scale)))

    out = []
    for x in video_tensors:
        mode = "bilinear" if x.shape[1] > 1 else "nearest"
        out.append(F.interpolate(x, size=(h2, w2), mode=mode, align_corners=False if mode == "bilinear" else None))
    return out


def _fixed_crop_with_unknown_bias(
    video_tensors: List[Tensor], alpha: Tensor, crop_size: int
) -> List[Tensor]:
    """Fixed-size, position-randomized crop, biased toward the unknown band.

    NOT to be confused with the removed size-randomized crop: ``crop_size``
    is fixed across the whole run and matches the inference tile size, so
    every training sample is the exact shape the model will see at tile
    inference. Only the *position* varies, which is the augment a tiling
    model actually benefits from (subjects can land anywhere inside a tile).
    """
    _, _, h, w = alpha.shape
    crop_h = min(crop_size, h)
    crop_w = min(crop_size, w)

    band = _boundary_band(alpha).mean(dim=0, keepdim=False)[0]
    ys, xs = torch.where(band > 0.03)

    if ys.numel() > 0 and random.random() < 0.8:
        idx = random.randrange(ys.numel())
        cy = int(ys[idx].item())
        cx = int(xs[idx].item())
    else:
        cy = random.randrange(h)
        cx = random.randrange(w)

    y0 = max(0, min(h - crop_h, cy - crop_h // 2))
    x0 = max(0, min(w - crop_w, cx - crop_w // 2))
    return [x[..., y0 : y0 + crop_h, x0 : x0 + crop_w] for x in video_tensors]


# --- Photometric / sensor / codec augmentation primitives -----------------
#
# All operate on tensors shaped ``[T, C, H, W]`` in float32, range roughly
# [0, 1]. Each call site decides whether to clamp; many of the augments
# (especially gain + noise) intentionally let values briefly exceed the unit
# range so the loss-side composite still sees the original radiometric ratios.
# Final clamp is performed once at the end of the transform.


def _saturation_jitter(rgb: Tensor, factor: float) -> Tensor:
    """Scale chroma about luma. ``factor=1`` is a no-op."""
    luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    return luma + (rgb - luma) * factor


def _hue_jitter(rgb: Tensor, angle_rad: float) -> Tensor:
    """Rotate the YIQ chrominance plane by ``angle_rad``.

    Cheap, branch-free, and bounded — much faster than a full RGB->HSV->RGB
    round-trip and visually indistinguishable for the small angles
    (|angle| < ~0.4 rad) used in matting augmentation.
    """
    if angle_rad == 0.0:
        return rgb
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.274 * g - 0.322 * b
    q = 0.211 * r - 0.523 * g + 0.312 * b
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    i2 = i * cos_a - q * sin_a
    q2 = i * sin_a + q * cos_a
    r2 = y + 0.956 * i2 + 0.621 * q2
    g2 = y - 0.272 * i2 - 0.647 * q2
    b2 = y - 1.106 * i2 + 1.703 * q2
    return torch.cat([r2, g2, b2], dim=1)


def _motion_blur(x: Tensor, kernel_size: int, direction: str) -> Tensor:
    """Apply a discrete-direction motion blur.

    ``direction`` ∈ ``{"h", "v", "d1", "d2"}`` selects horizontal, vertical,
    main-diagonal (\\) or anti-diagonal (/). Per-clip uniform direction +
    length is what camera shake actually looks like — frame-to-frame jitter
    of the kernel produces flicker artifacts the model would learn to chase.
    """
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    c = x.shape[1]

    if direction == "h":
        kernel = torch.full((1, 1, 1, kernel_size), 1.0 / kernel_size, dtype=x.dtype, device=x.device)
        kernel = kernel.expand(c, 1, 1, kernel_size).contiguous()
        return F.conv2d(x, kernel, padding=(0, kernel_size // 2), groups=c)
    if direction == "v":
        kernel = torch.full((1, 1, kernel_size, 1), 1.0 / kernel_size, dtype=x.dtype, device=x.device)
        kernel = kernel.expand(c, 1, kernel_size, 1).contiguous()
        return F.conv2d(x, kernel, padding=(kernel_size // 2, 0), groups=c)
    # Diagonal kernels. ``Tensor.flip`` returns a copy (not a view), so we
    # have to write the anti-diagonal via explicit index-assignment instead
    # of the seductive ``base.flip(0).diagonal().fill_(1.0)`` -- the latter
    # leaves ``base`` all zeros and turns the kernel into 0/0 = NaN.
    base = torch.zeros((kernel_size, kernel_size), dtype=x.dtype, device=x.device)
    idx = torch.arange(kernel_size, device=x.device)
    if direction == "d1":
        base[idx, idx] = 1.0
    elif direction == "d2":
        base[idx, kernel_size - 1 - idx] = 1.0
    else:
        raise ValueError(f"Unknown motion-blur direction: {direction!r}")
    base = base / base.sum()
    kernel = base.view(1, 1, kernel_size, kernel_size).expand(c, 1, kernel_size, kernel_size).contiguous()
    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=c)


def _compression_proxy(x: Tensor, downscale: float) -> Tensor:
    """Cheap JPEG/H.264-like degradation via lossy resample.

    Halving and re-upsampling at random factors in roughly [1.05, 1.4]
    discards the same high-frequency band a real codec quantizes away,
    without the per-block DCT cost. The output retains the input shape.
    """
    if downscale <= 1.0:
        return x
    h, w = x.shape[-2:]
    h2 = max(8, int(round(h / downscale)))
    w2 = max(8, int(round(w / downscale)))
    down = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False, antialias=True)
    up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
    return up


@dataclass
class CorridorMattingTransform:
    clip_len_min: int = 4
    clip_len_max: int = 12

    # Spatial: long-side cap only. No random *size* crops — each clip
    # retains its source aspect ratio scaled to ``target_side`` on the long
    # axis. Tile inference always sees a fixed tile_size, and the lowres
    # global pass always sees a fixed long-side cap, so size-randomized
    # crops were producing scale invariance the inference path never
    # exercises.
    resolution_buckets: Sequence[int] = (512,)
    resize_scale_min: float = 1.0
    resize_scale_max: float = 1.0

    # Optional fixed-size, position-randomized square crop. Set to the
    # model's inference tile_size when training a tiling model; leave at 0
    # to keep the long-side-resize path (correct for the lowres global
    # pass). When >0, the resize step targets short-side instead of
    # long-side so the crop can always be placed without aspect-ratio
    # distortion.
    fixed_crop_size: int = 0

    # Geometric / clip-level.
    horizontal_flip_p: float = 0.5
    background_replace_p: float = 0.3
    spill_augment_p: float = 0.4
    green_foreground_augment_p: float = 0.0
    green_foreground_strength_min: float = 0.6
    green_foreground_strength_max: float = 1.0
    temporal_jitter_p: float = 0.1
    skip_temporal_sample: bool = False
    disable_temporal_augment: bool = False
    fg_representation: str = "premul"

    # When ``True``, ``__call__`` exits after the channel/dtype normalisation
    # and temporal subsample/jitter steps, returning ``fg_gt`` / ``bg_gt`` /
    # ``alpha_gt`` at *native* source resolution and intentionally omitting the
    # ``video_rgb`` / ``input_gt`` / ``coarse_alpha_init`` keys. The training
    # loop is then expected to apply ``utils.device_transform.DeviceMattingTransform``
    # after the H2D copy to fill in those keys (boundary-band crop, all
    # photometric augments, representation-specific composite,
    # coarse_alpha_init).
    #
    # Use this to take the per-clip CPU work off the dataloader workers when
    # they are the bottleneck — see ``configs/memory_vmatte_tile1024_perf.yaml``
    # and the docstring of ``utils.device_transform`` for the full rationale.
    # Leave at ``False`` (default) for any host-only / debug / non-CUDA run; the
    # device transform requires a CUDA batch.
    device_offload: bool = False

    # Photometric: scalar gain on subject/background.
    subject_gain_min: float = 0.7
    subject_gain_max: float = 1.3
    bg_gain_min: float = 0.65
    bg_gain_max: float = 1.35

    # White-balance jitter: per-channel multiplier with center on green.
    # ``wb_jitter_strength=0.15`` => R,B in [0.85, 1.15], G in [0.95, 1.05].
    wb_jitter_p: float = 0.7
    wb_jitter_strength: float = 0.15

    # HSV-style colour jitter applied to the composited input only.
    color_jitter_p: float = 0.4
    saturation_min: float = 0.7
    saturation_max: float = 1.3
    hue_jitter_max: float = 0.15  # radians (~8.6 degrees)

    # Sensor noise. Combined gaussian (read noise) + brightness-scaled shot
    # noise approximation. Both are applied independently per frame so the
    # network sees temporally-uncorrelated grain — matching real footage.
    noise_p: float = 0.8
    noise_sigma_min: float = 0.005
    noise_sigma_max: float = 0.04
    shot_noise_p: float = 0.3
    shot_noise_strength: float = 0.025

    # Lens softness / out-of-focus blur (per-clip uniform).
    blur_p: float = 0.2
    blur_sigma_min: float = 0.3
    blur_sigma_max: float = 0.8

    # Motion blur (per-clip uniform direction + length, mimicking real
    # camera shake. Per-frame randomization would create flicker the model
    # would learn to chase.)
    motion_blur_p: float = 0.25
    motion_blur_kernel_min: int = 3
    motion_blur_kernel_max: int = 7

    # Codec / compression artefacts.
    compression_p: float = 0.2
    compression_downscale_min: float = 1.05
    compression_downscale_max: float = 1.35

    # Worker-side dtype for the device-offload short-circuit. Setting this
    # to ``torch.bfloat16`` (or ``torch.float16``) halves the bytes shipped
    # through shared-memory IPC, the pin_memory daemon, and the H2D DMA,
    # roughly halves the pinned-buffer pool footprint (which lets the
    # prefetch queue grow without hammering host RAM), and keeps the GPU
    # transform numerically identical because ``DeviceMattingTransform``
    # already inherits its working dtype from its inputs (``dtype=x.dtype``
    # everywhere). The cost is one extra fp32->bf16 cast per modality on
    # the worker -- ~20-30 ms per native-2048^2 clip, which is negligible
    # next to the EXR decode floor of ~600+ ms when workers are the
    # bottleneck. Leave at ``None`` for the safe fp32 path; explicit
    # ``torch.float32`` also forces fp32 (same as None).
    #
    # NOTE: only honoured in ``device_offload=True`` mode. The host-side
    # photometric/blur ops in the non-offload branch require fp32 for
    # correctness on CPU.
    host_dtype: Optional[torch.dtype] = None

    def _temporal_sample(self, tensors: List[Tensor]) -> List[Tensor]:
        t = tensors[0].shape[0]
        clip_len = min(t, random.randint(self.clip_len_min, min(self.clip_len_max, t)))
        start = random.randint(0, max(0, t - clip_len))
        out = [x[start : start + clip_len] for x in tensors]

        if random.random() < 0.5:
            out = [torch.flip(x, dims=[0]) for x in out]

        if clip_len >= 4 and random.random() < self.temporal_jitter_p:
            # Duplicate or drop one frame to improve temporal robustness.
            frame_id = random.randint(1, clip_len - 2)
            if random.random() < 0.5:
                out = [torch.cat([x[:frame_id], x[frame_id : frame_id + 1], x[frame_id:]], dim=0) for x in out]
            else:
                out = [torch.cat([x[:frame_id], x[frame_id + 1 :]], dim=0) for x in out]

        return out

    def _temporal_augment(self, tensors: List[Tensor]) -> List[Tensor]:
        """Lightweight temporal augments (reverse, frame dup/drop) without re-sampling the window."""
        t = tensors[0].shape[0]
        out = list(tensors)

        if random.random() < 0.5:
            out = [torch.flip(x, dims=[0]) for x in out]

        if t >= 4 and random.random() < self.temporal_jitter_p:
            frame_id = random.randint(1, t - 2)
            if random.random() < 0.5:
                out = [torch.cat([x[:frame_id], x[frame_id : frame_id + 1], x[frame_id:]], dim=0) for x in out]
            else:
                out = [torch.cat([x[:frame_id], x[frame_id + 1 :]], dim=0) for x in out]

        return out

    def __call__(self, sample: Dict[str, object]) -> Dict[str, object]:
        fg_representation = _validate_fg_representation(self.fg_representation)
        fg_gt = _sanitize_color_tensor(_ensure_color_three_channels(sample["FG"].to(torch.float32)))
        bg_gt = _sanitize_color_tensor(_ensure_color_three_channels(sample["BG"].to(torch.float32)))
        alpha_gt = _sanitize_alpha_tensor(_ensure_alpha_single_channel(sample["Alpha"].to(torch.float32)))

        input_sample = sample.get("Input")
        input_provided = input_sample is not None
        if input_provided:
            input_gt: Optional[Tensor] = _sanitize_color_tensor(
                _ensure_color_three_channels(input_sample.to(torch.float32))
            )
        else:
            # Input is synthesized from FG/BG/Alpha below. We intentionally
            # defer that compose until AFTER the resize: on 2048x2048 fp32 the
            # representation-specific composite fuses ~600 MB tensors at
            # 2048x2048 fp32 and wastes ~200-400 ms per sample that the
            # downstream re-compose (same math at 1280x1280) makes completely
            # redundant.
            input_gt = None

        global_input_sample = sample.get("global_Input")
        if global_input_sample is None:
            global_input_sample = sample.get("global_input_gt")
        global_fg_sample = sample.get("global_FG")
        if global_fg_sample is None:
            global_fg_sample = sample.get("global_fg_gt")
        global_alpha_sample = sample.get("global_Alpha")
        if global_alpha_sample is None:
            global_alpha_sample = sample.get("global_alpha_gt")

        global_input_gt: Optional[Tensor] = None
        global_fg_gt: Optional[Tensor] = None
        global_alpha_gt: Optional[Tensor] = None
        if global_input_sample is not None:
            global_input_gt = _sanitize_color_tensor(
                _ensure_color_three_channels(global_input_sample.to(torch.float32))
            )
        if global_fg_sample is not None:
            global_fg_gt = _sanitize_color_tensor(_ensure_color_three_channels(global_fg_sample.to(torch.float32)))
        if global_alpha_sample is not None:
            global_alpha_gt = _sanitize_alpha_tensor(
                _ensure_alpha_single_channel(global_alpha_sample.to(torch.float32))
            )

        # Apply the temporal reverse/dup/drop to every temporally-aligned tensor,
        # including optional precomputed global sidecars. This keeps local crops,
        # frame indices, and global context in the same temporal order.
        temporal_names: List[str] = []
        temporal_tensors: List[Tensor] = []
        if input_provided and input_gt is not None:
            temporal_names.append("input")
            temporal_tensors.append(input_gt)
        temporal_names.extend(["fg", "bg", "alpha"])
        temporal_tensors.extend([fg_gt, bg_gt, alpha_gt])
        if global_input_gt is not None:
            temporal_names.append("global_input")
            temporal_tensors.append(global_input_gt)
        if global_fg_gt is not None:
            temporal_names.append("global_fg")
            temporal_tensors.append(global_fg_gt)
        if global_alpha_gt is not None:
            temporal_names.append("global_alpha")
            temporal_tensors.append(global_alpha_gt)

        if self.disable_temporal_augment:
            temporal_tensors = list(temporal_tensors)
        elif self.skip_temporal_sample:
            temporal_tensors = self._temporal_augment(temporal_tensors)
        else:
            temporal_tensors = self._temporal_sample(temporal_tensors)
        temporal = dict(zip(temporal_names, temporal_tensors))

        if input_provided:
            input_gt = temporal["input"]
        fg_gt = temporal["fg"]
        bg_gt = temporal["bg"]
        alpha_gt = temporal["alpha"]
        global_input_gt = temporal.get("global_input")
        global_fg_gt = temporal.get("global_fg")
        global_alpha_gt = temporal.get("global_alpha")

        if self.device_offload:
            # Hand the worker's job back to the GPU. Everything below this
            # point — boundary-band crop, photometric/sensor/codec augments,
            # the representation-specific composite, and coarse_alpha_init — runs in
            # ``utils.device_transform.DeviceMattingTransform`` after the H2D
            # copy. The collate (``pad_collate_video``) is updated to tolerate
            # the missing ``video_rgb`` / ``input_gt`` / ``coarse_alpha_init``
            # keys by sizing buffers from ``fg_gt`` instead.
            #
            # Optional dtype downcast (bf16/fp16) to halve the bytes that
            # cross the IPC -> pin -> PCIe path. Only meaningful when set
            # to a smaller-than-fp32 dtype; ``None`` and ``float32`` are
            # both no-ops.
            if self.host_dtype is not None and self.host_dtype != torch.float32:
                fg_gt = fg_gt.to(self.host_dtype)
                bg_gt = bg_gt.to(self.host_dtype)
                alpha_gt = alpha_gt.to(self.host_dtype)
                if global_input_gt is not None:
                    global_input_gt = global_input_gt.to(self.host_dtype)
                if global_fg_gt is not None:
                    global_fg_gt = global_fg_gt.to(self.host_dtype)
                if global_alpha_gt is not None:
                    global_alpha_gt = global_alpha_gt.to(self.host_dtype)
            out: Dict[str, object] = {
                "fg_gt": fg_gt,
                "bg_gt": bg_gt,
                "alpha_gt": alpha_gt,
                "clip_id": sample.get("clip_name", "unknown"),
                "frame_indices": sample.get("frame_numbers"),
            }
            if global_input_gt is not None:
                out["global_input_gt"] = global_input_gt
            if global_fg_gt is not None:
                out["global_fg_gt"] = global_fg_gt
            if global_alpha_gt is not None:
                out["global_alpha_gt"] = global_alpha_gt
            for meta_key in ("tile_coords", "source_hw", "tile_grid"):
                if meta_key in sample:
                    out[meta_key] = sample[meta_key]
            return out


        if random.random() < self.green_foreground_augment_p:
            strength_min = float(self.green_foreground_strength_min)
            strength_max = max(strength_min, float(self.green_foreground_strength_max))
            fg_gt = _green_foreground_challenge_premul(
                fg_gt,
                alpha_gt,
                strength=random.uniform(strength_min, strength_max),
                green_rgb=(
                    random.uniform(0.03, 0.18),
                    random.uniform(0.85, 1.25),
                    random.uniform(0.03, 0.22),
                ),
            )

        target_side = random.choice(tuple(self.resolution_buckets))
        scale = random.uniform(self.resize_scale_min, self.resize_scale_max)
        side_target = max(128, int(round(target_side * scale)))

        # Spatial pipeline. Two mutually-exclusive paths:
        #   * fixed_crop_size > 0  (tiling-model training):
        #       Take a fixed-size, position-randomized square crop biased
        #       toward the unknown band, AT NATIVE PIXEL RESOLUTION. The
        #       crop *is* one of the actual inference tiles the model will
        #       see. We only invoke the resize fallback if source short-
        #       side < crop_size (otherwise we'd have nowhere to crop
        #       from). For 2048^2 source + crop_size=1024 this is a pure
        #       1024x1024 native crop -- no downsample, no upsample.
        #   * fixed_crop_size == 0 (lowres / global-pass training):
        #       resize so the LONG side == side_target, keep source
        #       aspect ratio. No crop.
        if self.fixed_crop_size > 0:
            crop_size = int(self.fixed_crop_size)
            src_h, src_w = alpha_gt.shape[-2:]
            if min(src_h, src_w) < crop_size:
                # Source is smaller than one tile -- the only case where
                # we have to resize. Bring short-side up to crop_size so
                # the subsequent crop can be placed without distortion.
                # In production with 2048^2 shards this branch is dead.
                if input_provided:
                    input_gt, fg_gt, bg_gt, alpha_gt = _resize_short_side(
                        [input_gt, fg_gt, bg_gt, alpha_gt], crop_size
                    )
                else:
                    fg_gt, bg_gt, alpha_gt = _resize_short_side(
                        [fg_gt, bg_gt, alpha_gt], crop_size
                    )
            if input_provided:
                input_gt, fg_gt, bg_gt, alpha_gt = _fixed_crop_with_unknown_bias(
                    [input_gt, fg_gt, bg_gt, alpha_gt], alpha_gt, crop_size=crop_size
                )
            else:
                fg_gt, bg_gt, alpha_gt = _fixed_crop_with_unknown_bias(
                    [fg_gt, bg_gt, alpha_gt], alpha_gt, crop_size=crop_size
                )
        else:
            if input_provided:
                input_gt, fg_gt, bg_gt, alpha_gt = _random_resize(
                    [input_gt, fg_gt, bg_gt, alpha_gt], side_target
                )
            else:
                fg_gt, bg_gt, alpha_gt = _random_resize([fg_gt, bg_gt, alpha_gt], side_target)

        # Horizontal flip.
        if random.random() < self.horizontal_flip_p:
            fg_gt = torch.flip(fg_gt, dims=[-1])
            bg_gt = torch.flip(bg_gt, dims=[-1])
            alpha_gt = torch.flip(alpha_gt, dims=[-1])
            if input_provided:
                input_gt = torch.flip(input_gt, dims=[-1])

        # Background replacement robustness.
        if random.random() < self.background_replace_p and bg_gt.shape[0] > 1:
            order = torch.randperm(bg_gt.shape[0])
            bg_gt = bg_gt[order]

        # Independent exposure gains for subject/background.
        # NOTE: ``fg_gt`` is the *premultiplied* foreground (FG = alpha * color)
        # as it ships in the CorridorKey EXR dataset. Multiplying premul FG by a
        # scalar gain preserves the premul invariant (g * alpha * color = alpha *
        # (g * color)), so the gain is safe to apply before recompositing.
        subject_gain = random.uniform(self.subject_gain_min, self.subject_gain_max)
        bg_gain = random.uniform(self.bg_gain_min, self.bg_gain_max)
        fg_gt = fg_gt * subject_gain
        bg_gt = bg_gt * bg_gain

        # Per-channel white-balance jitter. Applied to BOTH branches with the
        # same multiplier so the loss-side composite stays radiometrically
        # consistent with the input branch (this is a *capture-side* WB shift,
        # not a post-process). Centered on green because Bayer interpolation
        # makes G the most stable channel in real sensors.
        if self.wb_jitter_strength > 0 and random.random() < self.wb_jitter_p:
            s = float(self.wb_jitter_strength)
            wb = torch.tensor(
                [
                    random.uniform(1.0 - s, 1.0 + s),
                    random.uniform(1.0 - s / 3.0, 1.0 + s / 3.0),
                    random.uniform(1.0 - s, 1.0 + s),
                ],
                dtype=fg_gt.dtype,
                device=fg_gt.device,
            ).view(1, 3, 1, 1)
            fg_gt = fg_gt * wb
            bg_gt = bg_gt * wb

        # Source FG is premultiplied on disk.  Convert the supervised target
        # after capture-side gain/WB augments, then synthesize the input with the
        # representation-specific over formula.  In straight mode this teaches
        # the network to predict unmixed colour directly; the composite remains
        # physically equivalent to the original premul plate.
        fg_gt = _fg_target_from_premul(fg_gt, alpha_gt, fg_representation)
        input_gt = _composite_fg(fg_gt, bg_gt, alpha_gt, fg_representation)

        # Synthetic boundary spill on input branch only.
        if random.random() < self.spill_augment_p:
            edge = _boundary_band(alpha_gt)
            spill_strength = random.uniform(0.05, 0.25)
            spill = torch.zeros_like(input_gt)
            spill[:, 1:2] = spill_strength * edge
            input_gt = (input_gt + spill).clamp_min(0.0)

        # ---- Input-branch-only photometric / sensor / codec degradations ----
        # These are deliberately applied AFTER the composite so the loss-side
        # ``fg_gt`` / ``bg_gt`` retain clean radiometry; only what the model
        # actually consumes (``input_gt`` / ``video_rgb``) gets degraded.

        # HSV-style chroma jitter (saturation + hue rotation in YIQ).
        if random.random() < self.color_jitter_p:
            sat = random.uniform(self.saturation_min, self.saturation_max)
            if sat != 1.0:
                input_gt = _saturation_jitter(input_gt, sat)
            if self.hue_jitter_max > 0:
                hue = random.uniform(-self.hue_jitter_max, self.hue_jitter_max)
                input_gt = _hue_jitter(input_gt, hue)

        # Per-clip directional motion blur. Done before sensor noise so the
        # noise field doesn't get smeared (real cameras add read noise after
        # the optical blur path).
        if random.random() < self.motion_blur_p:
            kmin = max(1, int(self.motion_blur_kernel_min))
            kmax = max(kmin, int(self.motion_blur_kernel_max))
            k = random.randint(kmin, kmax)
            direction = random.choice(("h", "v", "d1", "d2"))
            input_gt = _motion_blur(input_gt, kernel_size=k, direction=direction)

        # Lens softness / out-of-focus gaussian blur.
        if random.random() < self.blur_p:
            sigma = random.uniform(self.blur_sigma_min, self.blur_sigma_max)
            radius = max(1, int(round(2.0 * sigma)))
            kernel_size = 2 * radius + 1
            input_gt = _gaussian_blur(input_gt, kernel_size=kernel_size, sigma=sigma)

        # Compression / codec proxy (downsample-upsample at random factor).
        if random.random() < self.compression_p:
            ds = random.uniform(self.compression_downscale_min, self.compression_downscale_max)
            input_gt = _compression_proxy(input_gt, downscale=ds)

        # Sensor noise: gaussian read noise + brightness-scaled shot noise.
        if random.random() < self.noise_p:
            sigma = random.uniform(self.noise_sigma_min, self.noise_sigma_max)
            input_gt = input_gt + torch.randn_like(input_gt) * sigma
        if random.random() < self.shot_noise_p:
            input_gt = input_gt + torch.randn_like(input_gt) * (
                input_gt.clamp_min(0.0).sqrt() * float(self.shot_noise_strength)
            )

        # Final clamp to keep downstream losses well-behaved. We also clamp
        # ``fg_gt`` (and ``bg_gt``) into ``[0, 1]`` so that ``l_fg`` and
        # ``l_comp`` push the model towards a *consistent* target. CorridorKey's
        # premultiplied FG can carry HDR radiometry (>1) but the model output
        # is bounded by sigmoid in ``[0, 1]``; mixing an HDR ``fg_gt`` target
        # with a clamped ``input_gt`` target makes ``l_fg`` and ``l_comp``
        # disagree on the same pixel and their gradients cancel out.
        input_gt = input_gt.clamp(0.0, 1.0)
        fg_gt = fg_gt.clamp(0.0, 1.0)
        bg_gt = bg_gt.clamp(0.0, 1.0)

        coarse_alpha_init = generate_coarse_alpha_init(alpha_gt[0])

        # Runtime input branch equals composited frame branch.
        video_rgb = input_gt

        out: Dict[str, object] = {
            "video_rgb": video_rgb,
            "alpha_gt": alpha_gt,
            "fg_gt": fg_gt,
            "bg_gt": bg_gt,
            "input_gt": input_gt,
            "coarse_alpha_init": coarse_alpha_init,
            "clip_id": sample.get("clip_name", "unknown"),
            "frame_indices": sample.get("frame_numbers"),
        }
        return out

def pad_collate_video(batch: List[Dict[str, object]], pad_multiple: int = 8) -> Dict[str, object]:
    """Stack variable-length video samples into padded batch tensors.

    Performance notes:
    * **B=1 zero-copy fast path.** When the batch contains a single sample,
      no temporal or spatial padding is needed (the common case for the
      tile1024 perf configs and any fixed-shape training run), we ``unsqueeze
      (0)`` the worker tensors directly instead of allocating a fresh
      destination and copying. This eliminates a ~448 MB alloc + 448 MB copy
      per batch on the worker (measured ~90 ms / batch at native 2048^2
      fp32 with device_offload), which was the largest avoidable cost in the
      device-offload pipeline. The pinned-memory daemon then copies the
      worker's stacked tensors directly to pinned RAM, saving the entire
      double-copy on the worker side.
    * **No-padding shortcut for B>1.** When every sample already matches
      ``max_t`` / ``max_h`` / ``max_w`` we use ``torch.empty(...)`` instead
      of ``torch.zeros(...)`` because the assignment loop is guaranteed to
      overwrite every element -- the ``zeros()`` write was pure waste in
      that case (~45 ms / batch at native 2048^2 fp32).
    * Zero-initialised destination buffers double as the padding when
      padding is actually needed, so we assign each sample into its
      ``[:, :t, :, :h, :w]`` slice directly. This avoids the prior
      per-sample padded allocation, which produced a padded copy that was
      then copied again into the destination.
    * ``_ensure_color_three_channels`` is already applied in the transform,
      so we don't re-run it here.
    * ``valid_mask`` and ``spatial_valid_mask`` are only emitted when the
      batch actually needs them (i.e. at least one item is smaller than the
      batch maximum). When every sample already matches ``max_t`` / ``max_h``
      / ``max_w`` we skip the masks entirely. The loss and training loop
      treat a missing mask as "all valid".
    * Device-offload mode: when the per-sample dict omits ``video_rgb`` /
      ``input_gt`` / ``coarse_alpha_init`` (because the
      ``CorridorMattingTransform(device_offload=True)`` short-circuit left
      them for the GPU side to materialise), we size buffers from ``fg_gt``
      and skip allocating the absent keys. The training loop then runs
      ``DeviceMattingTransform`` after H2D, which fills them in.
    """
    flattened_batch: List[Dict[str, object]] = []
    saw_prebatched = False
    for item in batch:
        prebatched = item.get("_prebatched_samples") if isinstance(item, dict) else None
        if isinstance(prebatched, list):
            saw_prebatched = True
            flattened_batch.extend([sample for sample in prebatched if isinstance(sample, dict)])
        else:
            flattened_batch.append(item)
    if saw_prebatched:
        batch = flattened_batch
        if not batch:
            raise ValueError("Received an empty prebatched sample list")

    b = len(batch)
    # Pick whichever of the standard tensors is present to size the batch
    # buffers. ``video_rgb`` is the host-mode anchor; ``fg_gt`` is the
    # device-offload anchor (always present in either mode).
    size_anchor_key = "video_rgb" if "video_rgb" in batch[0] else "fg_gt"
    has_video_rgb = size_anchor_key == "video_rgb"
    has_coarse = "coarse_alpha_init" in batch[0]
    ts = [int(item[size_anchor_key].shape[0]) for item in batch]
    hs = [int(item[size_anchor_key].shape[-2]) for item in batch]
    ws = [int(item[size_anchor_key].shape[-1]) for item in batch]

    max_t = max(ts)
    raw_max_h = max(hs)
    raw_max_w = max(ws)
    if pad_multiple > 1:
        max_h = int(math.ceil(raw_max_h / pad_multiple) * pad_multiple)
        max_w = int(math.ceil(raw_max_w / pad_multiple) * pad_multiple)
    else:
        max_h = raw_max_h
        max_w = raw_max_w

    needs_temporal_mask = any(t < max_t for t in ts)
    needs_spatial_mask = any(h < max_h or w < max_w for h, w in zip(hs, ws))

    # ---- B=1 zero-copy fast path -------------------------------------------
    # By far the hottest collate config in this codebase. The worker has
    # already produced contiguous [T,C,H,W] tensors via torch.stack inside
    # the dataset; an ``unsqueeze(0)`` is a zero-copy view that the pinned-
    # memory daemon can copy directly. We additionally require that no
    # padding is needed (the slicing fast-path can't express "pad on the
    # right" without an alloc) -- which is true any time the source clip is
    # already a multiple of pad_multiple on H,W and matches max_t, i.e. the
    # production fixed_crop_size configs and the device_offload native-res
    # path (2048 % 8 == 0).
    # See the ``_COLLATE_FAST_MODE`` comment for the auto vs forced policy.
    sample_dtype = batch[0][size_anchor_key].dtype
    if _COLLATE_FAST_FORCE is None:
        # ``auto``: enable when the worker has already downcast to a
        # smaller dtype than fp32 (e.g. via ``data.host_dtype: bfloat16``),
        # which bounds the held shared-mem footprint enough to avoid the
        # page-cache pressure regression measured at fp32.
        fast_path_enabled = sample_dtype != torch.float32
    else:
        fast_path_enabled = _COLLATE_FAST_FORCE

    if (
        fast_path_enabled
        and b == 1
        and not needs_temporal_mask
        and not needs_spatial_mask
        and raw_max_h == max_h
        and raw_max_w == max_w
    ):
        item = batch[0]
        fg = item["fg_gt"]
        bg = item["bg_gt"]
        a = item["alpha_gt"]

        clip_ids = [str(item.get("clip_id", "unknown"))]
        fi = item.get("frame_indices")
        if isinstance(fi, Tensor):
            frame_indices = fi.to(dtype=torch.long)[:max_t].unsqueeze(0)
        else:
            frame_indices = torch.full((1, max_t), -1, dtype=torch.long)

        out: Dict[str, object] = {
            "alpha_gt": a.unsqueeze(0),
            "fg_gt": fg.unsqueeze(0),
            "bg_gt": bg.unsqueeze(0),
            "clip_id": clip_ids,
            "frame_indices": frame_indices,
        }
        if has_video_rgb:
            video_rgb = item["video_rgb"].unsqueeze(0)
            out["video_rgb"] = video_rgb
            # Runtime input branch and model video branch share storage in
            # the host-mode pipeline (see the slow-path branch below); keep
            # the alias so downstream code can rely on it.
            out["input_gt"] = video_rgb
        if has_coarse:
            out["coarse_alpha_init"] = item["coarse_alpha_init"].unsqueeze(0)
        for extra_key in ("global_input_gt", "global_alpha_gt", "global_fg_gt", "global_bg_gt"):
            extra = item.get(extra_key)
            if isinstance(extra, Tensor):
                out[extra_key] = extra.unsqueeze(0)
        for meta_key in ("tile_coords", "source_hw", "tile_grid"):
            meta = item.get(meta_key)
            if isinstance(meta, Tensor):
                out[meta_key] = meta.unsqueeze(0)
        for meta_key in ("temporal_stream_chunk_index", "temporal_stream_num_chunks", "temporal_stream_full_frames", "temporal_stream_chunk_start", "temporal_stream_chunk_end", "temporal_stream_logical_batch"):
            meta = item.get(meta_key)
            if meta is not None:
                out[meta_key] = torch.tensor([int(meta)], dtype=torch.long)
        return out

    # ---- General slow path: stack into a padded destination ---------------
    device = batch[0][size_anchor_key].device
    dtype = batch[0][size_anchor_key].dtype

    # ``empty`` is safe whenever there is no padding (we will overwrite
    # every element below). With padding we MUST use ``zeros`` because the
    # padded margins are read by downstream losses as ground-truth zeros.
    needs_zero_fill = needs_temporal_mask or needs_spatial_mask
    alloc = torch.zeros if needs_zero_fill else torch.empty

    def alloc_video(ch: int) -> Tensor:
        return alloc((b, max_t, ch, max_h, max_w), dtype=dtype, device=device)

    video_rgb = alloc_video(3) if has_video_rgb else None
    alpha_gt = alloc_video(1)
    fg_gt = alloc_video(3)
    bg_gt = alloc_video(3)
    coarse_alpha_init = (
        alloc((b, 1, max_h, max_w), dtype=dtype, device=device) if has_coarse else None
    )

    valid_mask: Optional[Tensor] = (
        torch.zeros((b, max_t), dtype=dtype, device=device) if needs_temporal_mask else None
    )
    spatial_valid_mask: Optional[Tensor] = (
        torch.zeros((b, 1, max_h, max_w), dtype=dtype, device=device) if needs_spatial_mask else None
    )

    clip_ids: List[str] = []
    frame_indices = torch.full((b, max_t), -1, dtype=torch.long, device=device)

    for i, item in enumerate(batch):
        a = item["alpha_gt"]
        fg = item["fg_gt"]
        bg = item["bg_gt"]

        t, _, h, w = fg.shape
        alpha_gt[i, :t, :, :h, :w] = a
        fg_gt[i, :t, :, :h, :w] = fg
        bg_gt[i, :t, :, :h, :w] = bg

        if video_rgb is not None:
            video_rgb[i, :t, :, :h, :w] = item["video_rgb"]
        if coarse_alpha_init is not None:
            coarse_alpha_init[i, :, :h, :w] = item["coarse_alpha_init"]

        if valid_mask is not None:
            valid_mask[i, :t] = 1.0
        if spatial_valid_mask is not None:
            spatial_valid_mask[i, :, :h, :w] = 1.0

        clip_ids.append(str(item.get("clip_id", "unknown")))

        fi = item.get("frame_indices")
        if isinstance(fi, Tensor):
            fi_t = fi.to(dtype=torch.long)[:t]
            frame_indices[i, : fi_t.shape[0]] = fi_t

    out: Dict[str, object] = {
        "alpha_gt": alpha_gt,
        "fg_gt": fg_gt,
        "bg_gt": bg_gt,
        "clip_id": clip_ids,
        "frame_indices": frame_indices,
    }
    if video_rgb is not None:
        out["video_rgb"] = video_rgb
        # Runtime input and model video branch are equivalent in this pipeline.
        # Reuse storage to avoid a duplicate H2D copy.
        out["input_gt"] = video_rgb
    if coarse_alpha_init is not None:
        out["coarse_alpha_init"] = coarse_alpha_init
    if valid_mask is not None:
        out["valid_mask"] = valid_mask
    if spatial_valid_mask is not None:
        out["spatial_valid_mask"] = spatial_valid_mask

    for extra_key in ("global_input_gt", "global_alpha_gt", "global_fg_gt", "global_bg_gt"):
        if extra_key not in batch[0]:
            continue
        tensors = [item[extra_key] for item in batch if isinstance(item.get(extra_key), Tensor)]
        if len(tensors) != b:
            continue
        if all(tuple(t.shape) == tuple(tensors[0].shape) for t in tensors):
            out[extra_key] = torch.stack(tensors, dim=0)
            continue
        et = max(int(t.shape[0]) for t in tensors)
        ec = int(tensors[0].shape[1])
        eh = max(int(t.shape[-2]) for t in tensors)
        ew = max(int(t.shape[-1]) for t in tensors)
        if pad_multiple > 1:
            eh = int(math.ceil(eh / pad_multiple) * pad_multiple)
            ew = int(math.ceil(ew / pad_multiple) * pad_multiple)
        dest = torch.zeros((b, et, ec, eh, ew), dtype=tensors[0].dtype, device=tensors[0].device)
        for i, tensor in enumerate(tensors):
            tt, _, hh, ww = tensor.shape
            dest[i, :tt, :, :hh, :ww] = tensor
        out[extra_key] = dest

    for meta_key in ("tile_coords", "source_hw", "tile_grid"):
        metas = [item.get(meta_key) for item in batch]
        if all(isinstance(m, Tensor) for m in metas):
            out[meta_key] = torch.stack([m for m in metas if isinstance(m, Tensor)], dim=0)

    for meta_key in ("temporal_stream_chunk_index", "temporal_stream_num_chunks", "temporal_stream_full_frames", "temporal_stream_chunk_start", "temporal_stream_chunk_end", "temporal_stream_logical_batch"):
        metas = [item.get(meta_key) for item in batch]
        if all(m is not None for m in metas):
            out[meta_key] = torch.tensor([int(m) for m in metas], dtype=torch.long, device=device)
    return out
