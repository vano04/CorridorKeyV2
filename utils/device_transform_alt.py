"""GPU-side implementation of CorridorMattingTransform's spatial + photometric pipeline.

Why this module exists
======================

The CPU transform was the documented bottleneck for the tile1024 training
configurations: a 6-worker dataloader at T=4 / 1024^2 produced ~2 clips/s of
decoded+augmented samples while the (compile + no-checkpoint) GPU step time
settled around 0.5 s, leaving the GPU idle ~70% of every step. The expensive
pieces -- the boundary-band ``max_pool2d`` on full-resolution alpha, the
`F.interpolate` + per-frame conv augmentations, the per-pixel ``randn`` noise
fields -- are all already pure ``torch.nn.functional`` ops. Running them on
the GPU costs single-digit ms vs tens to hundreds of ms on the CPU, while
also turning unused VRAM and idle SMs into useful work.

The split
=========

* **CPU side** (in DataLoader workers, via
  ``CorridorMattingTransform(device_offload=True)``): EXR decode, channel /
  dtype normalisation, temporal subsample / jitter. Output is FG / BG / Alpha
  at *native* source resolution (typically 2048x2048 for the CorridorKey
  shards).

* **GPU side** (this module, applied after H2D in the training loop):
  boundary-band fixed-size crop → horizontal flip → BG temporal shuffle →
  subject / BG gain → WB jitter → representation-specific composite →
  green spill → saturation / hue jitter → motion blur → gaussian blur →
  compression proxy → gaussian read + shot noise → final clamp →
  coarse_alpha_init.

Semantics match ``utils.data.CorridorMattingTransform.__call__`` step for
step; the function order, the per-clip stochastic decisions, and the
augmentation primitives are deliberately verbatim ports of the CPU helpers
in ``utils.data``.

RNG strategy
============

* **Per-clip scalar decisions** (flip yes/no, gain values, kernel sizes,
  motion blur direction, ...) are sampled with Python ``random`` -- same
  generator the CPU transform uses. ``train.set_seed`` already seeds it,
  so device-mode and host-mode runs are bitwise reproducible
  modulo floating-point reassociation differences between CPU and CUDA.

* **Per-pixel stochastic fields** (``torch.randn_like`` for sensor noise)
  use the default CUDA generator, also seeded by ``train.set_seed`` via
  ``torch.cuda.manual_seed``.

Caveats
=======

* **Batch-vs-sample augmentation diversity**: per-clip params are sampled
  once per *batch* call and broadcast across all ``B`` samples in the batch.
  The CPU transform samples per *sample* (called once per
  ``Dataset.__getitem__``). For ``batch_size=1`` this is identical;
  for ``B>1`` the device path applies the same augmentation params (flip
  decision, gain values, blur kernel, ...) to every sample in the batch,
  which is slightly less diverse. The relevant tile1024 perf configs all
  use B=1, but if you bump it, this is the parity caveat to remember.

* **Crop position**: per-sample on GPU. The boundary-band sampling produces
  a ``[B]``-shaped pair of crop offsets, so spatial diversity within a batch
  is preserved even when the photometric params aren't.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from matting_common import (
    composite_fg as _composite_fg,
    fg_target_from_premul as _fg_target_from_premul,
    validate_fg_representation as _validate_fg_representation,
)


_EPS = 1e-6


def apply_green_foreground_augmentation(
    fg: Tensor,
    alpha: Tensor,
    *,
    prob: float = 0.25,
    strength_min: float = 0.25,
    strength_max: float = 0.75,
    alpha_thresh: float = 0.70,
) -> Tensor:
    """Tint confident true foreground regions green while preserving shading.

    fg:    [B,T,3,H,W]
    alpha: [B,T,1,H,W]

    Source FG is premultiplied at this point. Convert visible pixels to
    straight colour, recolour by luminance, then premultiply again so the
    later compose creates a self-consistent faux greenscreen input.
    """
    if prob <= 0.0:
        return fg

    if torch.rand((), device=fg.device) >= float(prob):
        return fg

    visible = (alpha > float(alpha_thresh)).to(dtype=fg.dtype)
    s_min = float(strength_min)
    s_max = max(s_min, float(strength_max))
    strength = torch.empty(
        (fg.shape[0], 1, 1, 1, 1),
        device=fg.device,
        dtype=fg.dtype,
    ).uniform_(s_min, s_max)

    green = torch.empty(
        (fg.shape[0], 1, 3, 1, 1),
        device=fg.device,
        dtype=fg.dtype,
    )
    green[:, :, 0:1].uniform_(0.03, 0.18)
    green[:, :, 1:2].uniform_(0.85, 1.25)
    green[:, :, 2:3].uniform_(0.03, 0.22)

    alpha_safe = alpha.clamp_min(_EPS)
    straight = fg / alpha_safe
    luma = 0.299 * straight[:, :, 0:1] + 0.587 * straight[:, :, 1:2] + 0.114 * straight[:, :, 2:3]
    green_luma = (0.299 * green[:, :, 0:1] + 0.587 * green[:, :, 1:2] + 0.114 * green[:, :, 2:3]).clamp_min(_EPS)
    target = luma * (green / green_luma)

    blend = visible * strength
    straight_aug = straight * (1.0 - blend) + target * blend
    return (straight_aug * alpha).clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Verbatim GPU ports of the per-frame primitives from utils.data.
#
# These accept tensors of shape [N, C, H, W] (i.e. with the temporal axis
# folded into the batch axis) so the per-frame conv kernels work unchanged.
# Callers in DeviceMattingTransform.forward fold/unfold T as needed.
# ---------------------------------------------------------------------------


def _gaussian_blur(x: Tensor, kernel_size: int, sigma: float) -> Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma + _EPS))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    c = x.shape[1]
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(c, 1, kernel_size, kernel_size)
    return F.conv2d(x, kernel, padding=radius, groups=c)


def _morphological(x: Tensor, op: str, kernel: int) -> Tensor:
    pad = kernel // 2
    if op == "dilate":
        return F.max_pool2d(x, kernel_size=kernel, stride=1, padding=pad)
    if op == "erode":
        return -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=pad)
    raise ValueError(f"Unknown morphological op: {op}")


def _boundary_band(alpha: Tensor, kernel_size: int = 3) -> Tensor:
    dil = F.max_pool2d(alpha, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    ero = -F.max_pool2d(-alpha, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    return (dil - ero).clamp(0.0, 1.0)


def _saturation_jitter(rgb: Tensor, factor: float) -> Tensor:
    luma = 0.299 * rgb[:, 0:1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    return luma + (rgb - luma) * factor


def _hue_jitter(rgb: Tensor, angle_rad: float) -> Tensor:
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
    if kernel_size <= 1:
        return x
    if kernel_size % 2 == 0:
        kernel_size += 1
    c = x.shape[1]

    if direction == "h":
        kernel = torch.full(
            (1, 1, 1, kernel_size), 1.0 / kernel_size, dtype=x.dtype, device=x.device
        )
        kernel = kernel.expand(c, 1, 1, kernel_size).contiguous()
        return F.conv2d(x, kernel, padding=(0, kernel_size // 2), groups=c)
    if direction == "v":
        kernel = torch.full(
            (1, 1, kernel_size, 1), 1.0 / kernel_size, dtype=x.dtype, device=x.device
        )
        kernel = kernel.expand(c, 1, kernel_size, 1).contiguous()
        return F.conv2d(x, kernel, padding=(kernel_size // 2, 0), groups=c)

    base = torch.zeros((kernel_size, kernel_size), dtype=x.dtype, device=x.device)
    idx = torch.arange(kernel_size, device=x.device)
    if direction == "d1":
        base[idx, idx] = 1.0
    elif direction == "d2":
        base[idx, kernel_size - 1 - idx] = 1.0
    else:
        raise ValueError(f"Unknown motion-blur direction: {direction!r}")
    base = base / base.sum()
    kernel = (
        base.view(1, 1, kernel_size, kernel_size)
        .expand(c, 1, kernel_size, kernel_size)
        .contiguous()
    )
    return F.conv2d(x, kernel, padding=kernel_size // 2, groups=c)


def _compression_proxy(x: Tensor, downscale: float) -> Tensor:
    if downscale <= 1.0:
        return x
    h, w = x.shape[-2:]
    h2 = max(8, int(round(h / downscale)))
    w2 = max(8, int(round(w / downscale)))
    down = F.interpolate(x, size=(h2, w2), mode="bilinear", align_corners=False, antialias=True)
    up = F.interpolate(down, size=(h, w), mode="bilinear", align_corners=False)
    return up


# ---------------------------------------------------------------------------
# Coarse-alpha init port. ``utils.data.generate_coarse_alpha_init`` operates
# on a single [1, H, W] tensor (alpha_gt[0]); the device version here works on
# a batched [B, 1, H, W] so all per-clip param choices are still scalar but
# applied to the whole batch in one go.
# ---------------------------------------------------------------------------


def _generate_coarse_alpha_init_device(alpha0: Tensor) -> Tensor:
    """Match ``utils.data.generate_coarse_alpha_init`` semantics on GPU.

    ``alpha0`` is the first-frame alpha for every sample in the batch with
    shape ``[B, 1, H, W]``. Same scalar param distribution; same op order.
    """
    b, _, h, w = alpha0.shape

    factor = random.choice([2, 2, 3, 4, 6])
    h2 = max(1, h // factor)
    w2 = max(1, w // factor)
    alpha = F.interpolate(alpha0, size=(h2, w2), mode="bilinear", align_corners=False)
    alpha = F.interpolate(alpha, size=(h, w), mode="bilinear", align_corners=False)

    if random.random() < 0.8:
        alpha = _morphological(alpha, op="erode", kernel=random.choice([3, 5, 7]))
    else:
        alpha = _morphological(alpha, op="dilate", kernel=random.choice([3, 5]))

    shift_x = random.uniform(-2.0, 2.0)
    shift_y = random.uniform(-2.0, 2.0)
    theta = torch.tensor(
        [[1.0, 0.0, 2.0 * shift_x / max(1.0, w)], [0.0, 1.0, 2.0 * shift_y / max(1.0, h)]],
        device=alpha.device,
        dtype=alpha.dtype,
    ).unsqueeze(0).expand(b, -1, -1)
    grid = F.affine_grid(theta, alpha.shape, align_corners=False)
    alpha = F.grid_sample(alpha, grid, mode="bilinear", padding_mode="border", align_corners=False)

    k = random.choice([3, 5, 7, 9])
    sigma = random.uniform(0.7, 2.5)
    alpha = _gaussian_blur(alpha, kernel_size=k, sigma=sigma)

    noise = torch.randn_like(alpha) * random.uniform(0.0, 0.03)
    alpha = (alpha + noise).clamp(0.0, 1.0)
    return alpha


def _resize_short_side(
    tensors: Sequence[Tensor], short_side_target: int
) -> Sequence[Tensor]:
    """Match ``utils.data._resize_short_side`` for the rare case that a
    native shard happens to be smaller than ``crop_size``."""
    h, w = tensors[0].shape[-2:]
    scale = short_side_target / float(min(h, w))
    h2 = max(1, int(round(h * scale)))
    w2 = max(1, int(round(w * scale)))

    out = []
    for x in tensors:
        b, t, c, _, _ = x.shape
        flat = x.view(b * t, c, h, w)
        mode = "bilinear" if c > 1 else "nearest"
        if mode == "bilinear":
            resized = F.interpolate(flat, size=(h2, w2), mode=mode, align_corners=False)
        else:
            resized = F.interpolate(flat, size=(h2, w2), mode=mode)
        out.append(resized.reshape(b, t, c, h2, w2))
    return out


def _resize_long_side(tensors: Sequence[Tensor], long_side_cap: int) -> Sequence[Tensor]:
    h, w = tensors[0].shape[-2:]
    cap = int(long_side_cap)
    if cap <= 0 or max(h, w) <= cap:
        return list(tensors)

    scale = cap / float(max(h, w))
    h2 = max(1, int(round(h * scale)))
    w2 = max(1, int(round(w * scale)))

    out = []
    for x in tensors:
        b, t, c, _, _ = x.shape
        flat = x.reshape(b * t, c, h, w)
        resized = F.interpolate(flat, size=(h2, w2), mode="bilinear", align_corners=False)
        out.append(resized.view(b, t, c, h2, w2))
    return out


def _resize_alpha_long_side(alpha: Tensor, long_side_cap: int) -> Tensor:
    h, w = alpha.shape[-2:]
    cap = int(long_side_cap)
    if cap <= 0 or max(h, w) <= cap:
        return alpha
    scale = cap / float(max(h, w))
    h2 = max(1, int(round(h * scale)))
    w2 = max(1, int(round(w * scale)))
    return F.interpolate(alpha, size=(h2, w2), mode="bilinear", align_corners=False)


def _batched_crop_5d(x: Tensor, y0: Tensor, x0: Tensor, crop_h: int, crop_w: int) -> Tensor:
    """Crop one [T,C,H,W] window per batch item without GPU->CPU sync."""
    b, t, c, h, w = x.shape
    if int(y0.shape[0]) != b or int(x0.shape[0]) != b:
        raise ValueError(f"Crop offsets must have shape [{b}], got {tuple(y0.shape)} and {tuple(x0.shape)}")

    if b == 1:
        # The throughput configs use B=1. A direct slice avoids materializing
        # hundreds of MB of gather index tensors for each 2048 -> 1024 crop.
        yy = int(y0[0].item())
        xx = int(x0[0].item())
        yy = max(0, min(yy, max(0, h - crop_h)))
        xx = max(0, min(xx, max(0, w - crop_w)))
        return x[..., yy : yy + crop_h, xx : xx + crop_w].contiguous()

    yy = y0.to(device=x.device, dtype=torch.long)[:, None] + torch.arange(crop_h, device=x.device)[None, :]
    xx = x0.to(device=x.device, dtype=torch.long)[:, None] + torch.arange(crop_w, device=x.device)[None, :]
    yy = yy.clamp_(0, max(0, h - 1))
    xx = xx.clamp_(0, max(0, w - 1))

    y_index = yy.view(b, 1, 1, crop_h, 1).expand(b, t, c, crop_h, w)
    cropped_h = x.gather(dim=-2, index=y_index)
    x_index = xx.view(b, 1, 1, 1, crop_w).expand(b, t, c, crop_h, crop_w)
    return cropped_h.gather(dim=-1, index=x_index)


# ---------------------------------------------------------------------------
# Hyperparameter container. Mirrors the dataclass fields on
# ``CorridorMattingTransform`` that the device pipeline actually consumes
# (the temporal-sample fields stay on the CPU side and are not duplicated
# here).
# ---------------------------------------------------------------------------


@dataclass
class DeviceMattingTransformConfig:
    fixed_crop_size: int = 1024
    global_context_long_side: int = 0
    global_context_pre_resize: bool = False
    fg_representation: str = "premul"

    horizontal_flip_p: float = 0.5
    background_replace_p: float = 0.3
    spill_augment_p: float = 0.4
    green_foreground_prob: float = 0.0
    green_foreground_strength_min: float = 0.25
    green_foreground_strength_max: float = 0.75
    green_foreground_alpha_thresh: float = 0.70
    coarse_alpha_drop_prob: float = 0.0

    subject_gain_min: float = 0.7
    subject_gain_max: float = 1.3
    bg_gain_min: float = 0.65
    bg_gain_max: float = 1.35

    wb_jitter_p: float = 0.7
    wb_jitter_strength: float = 0.15

    color_jitter_p: float = 0.4
    saturation_min: float = 0.7
    saturation_max: float = 1.3
    hue_jitter_max: float = 0.15

    noise_p: float = 0.8
    noise_sigma_min: float = 0.005
    noise_sigma_max: float = 0.04
    shot_noise_p: float = 0.3
    shot_noise_strength: float = 0.025

    blur_p: float = 0.2
    blur_sigma_min: float = 0.3
    blur_sigma_max: float = 0.8

    motion_blur_p: float = 0.25
    motion_blur_kernel_min: int = 3
    motion_blur_kernel_max: int = 7

    compression_p: float = 0.2
    compression_downscale_min: float = 1.05
    compression_downscale_max: float = 1.35

    band_use_prob: float = 0.8
    band_threshold: float = 0.03


# ---------------------------------------------------------------------------
# The transform itself. nn.Module so it slots neatly into a torch graph and
# so .to(device) / .to(dtype) propagate any (currently zero) buffers it might
# acquire later.
# ---------------------------------------------------------------------------


class DeviceMattingTransform(nn.Module):
    """GPU port of the spatial + photometric stages of CorridorMattingTransform.

    Input batch (post-collate, post-H2D, in device-offload mode) must contain:

        fg_gt      : [B, T, 3, H_native, W_native]  float
        bg_gt      : [B, T, 3, H_native, W_native]  float
        alpha_gt   : [B, T, 1, H_native, W_native]  float in [0, 1]
        clip_id    : list[str]    (passthrough)
        frame_indices : [B, T] long  (passthrough)

    Output batch matches the schema produced by the host-side
    ``CorridorMattingTransform`` in non-offload mode:

        video_rgb         : [B, T, 3, H_crop, W_crop]
        input_gt          : alias of video_rgb (same storage)
        alpha_gt          : [B, T, 1, H_crop, W_crop]
        fg_gt, bg_gt      : [B, T, 3, H_crop, W_crop]
        coarse_alpha_init : [B, 1, H_crop, W_crop]
        clip_id, frame_indices : passthrough
        valid_mask, spatial_valid_mask : passthrough if present
    """

    def __init__(self, cfg: DeviceMattingTransformConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        cfg = self.cfg
        fg_representation = _validate_fg_representation(cfg.fg_representation)

        fg = batch["fg_gt"]
        bg = batch["bg_gt"]
        alpha = batch["alpha_gt"]
        emit_global_context = int(cfg.global_context_long_side) > 0
        has_precomputed_global_input = isinstance(batch.get("global_input_gt"), Tensor)
        has_precomputed_global_fg = isinstance(batch.get("global_fg_gt"), Tensor)
        has_precomputed_global_alpha = isinstance(batch.get("global_alpha_gt"), Tensor)
        precomputed_global = (
            emit_global_context
            and has_precomputed_global_alpha
            and (has_precomputed_global_input or has_precomputed_global_fg)
        )

        if fg.ndim != 5:
            raise ValueError(
                f"DeviceMattingTransform expected fg_gt with shape [B,T,C,H,W], got {tuple(fg.shape)}"
            )
        b, t, _, h, w = fg.shape

        # ---- Spatial: optional resize-if-too-small + boundary-band crop ----
        crop_size = int(cfg.fixed_crop_size)
        if crop_size > 0 and min(h, w) < crop_size:
            fg, bg, alpha = _resize_short_side([fg, bg, alpha], crop_size)
            h, w = fg.shape[-2:]

        if isinstance(batch.get("source_hw"), Tensor):
            source_hw = batch["source_hw"].to(device=fg.device, dtype=torch.float32)
            if source_hw.ndim == 1:
                source_hw = source_hw.unsqueeze(0).expand(b, -1)
            source_h = int(float(source_hw[0, 0].item()))
            source_w = int(float(source_hw[0, 1].item()))
        else:
            source_h, source_w = h, w
            source_hw = torch.tensor(
                [[float(source_h), float(source_w)] for _ in range(b)],
                device=fg.device,
                dtype=torch.float32,
            )
        fg = apply_green_foreground_augmentation(
            fg,
            alpha,
            prob=float(cfg.green_foreground_prob),
            strength_min=float(cfg.green_foreground_strength_min),
            strength_max=float(cfg.green_foreground_strength_max),
            alpha_thresh=float(cfg.green_foreground_alpha_thresh),
        )

        global_input_gt: Optional[Tensor]
        if emit_global_context and precomputed_global:
            global_input_gt = (
                batch["global_input_gt"].to(device=fg.device, dtype=fg.dtype)
                if has_precomputed_global_input
                else None
            )
            global_fg = (
                batch["global_fg_gt"].to(device=fg.device, dtype=fg.dtype)
                if has_precomputed_global_fg
                else None
            )
            global_alpha = batch["global_alpha_gt"].to(device=fg.device, dtype=alpha.dtype).clamp(0.0, 1.0)
            global_bg = None
        elif emit_global_context:
            global_input_gt = None
            global_fg = fg
            global_bg = bg
            global_alpha = alpha
            if bool(cfg.global_context_pre_resize):
                global_fg, global_bg, global_alpha = _resize_long_side(
                    [global_fg, global_bg, global_alpha],
                    int(cfg.global_context_long_side),
                )
        else:
            global_input_gt = None
            global_fg = global_bg = global_alpha = None

        if isinstance(batch.get("tile_coords"), Tensor):
            tile_coords = batch["tile_coords"].to(device=fg.device, dtype=torch.float32)
            if tile_coords.ndim == 1:
                tile_coords = tile_coords.unsqueeze(0).expand(b, -1)
        else:
            tile_coords = torch.tensor(
                [[0.0, float(h), 0.0, float(w)] for _ in range(b)],
                device=fg.device,
                dtype=torch.float32,
            )

        if crop_size > 0:
            crop_h = min(crop_size, h)
            crop_w = min(crop_size, w)
            if crop_h < h or crop_w < w:
                # When the loader already decoded exactly one tile via
                # OpenEXR.readTiles(), this stage is a shape no-op and we skip
                # the boundary-band sampling work entirely.
                alpha_for_band = alpha.view(b * t, 1, h, w)
                band = _boundary_band(alpha_for_band).view(b, t, 1, h, w)
                band_per_batch = band.mean(dim=(1, 2))  # [B, H, W]

                use_band = random.random() < cfg.band_use_prob
                mask = (band_per_batch > cfg.band_threshold).to(alpha.dtype) if use_band else torch.zeros_like(band_per_batch)
                weight = mask + 1e-6
                score = weight * torch.rand_like(weight)
                flat = score.view(b, -1)
                linear_idx = flat.argmax(dim=1)
                cy = linear_idx // w
                cx = linear_idx % w
                y0 = (cy - crop_h // 2).clamp(min=0, max=max(0, h - crop_h))
                x0 = (cx - crop_w // 2).clamp(min=0, max=max(0, w - crop_w))

                base_y0 = tile_coords[:, 0].clone()
                base_x0 = tile_coords[:, 2].clone()
                tile_coords = torch.stack(
                    [
                        base_y0 + y0.to(dtype=torch.float32),
                        base_y0 + y0.to(dtype=torch.float32) + float(crop_h),
                        base_x0 + x0.to(dtype=torch.float32),
                        base_x0 + x0.to(dtype=torch.float32) + float(crop_w),
                    ],
                    dim=1,
                )
                fg = _batched_crop_5d(fg, y0=y0, x0=x0, crop_h=crop_h, crop_w=crop_w)
                bg = _batched_crop_5d(bg, y0=y0, x0=x0, crop_h=crop_h, crop_w=crop_w)
                alpha = _batched_crop_5d(alpha, y0=y0, x0=x0, crop_h=crop_h, crop_w=crop_w)

        # ---- Geometric: horizontal flip ----
        if random.random() < cfg.horizontal_flip_p:
            fg = torch.flip(fg, dims=[-1])
            bg = torch.flip(bg, dims=[-1])
            alpha = torch.flip(alpha, dims=[-1])
            if emit_global_context:
                assert global_alpha is not None
                if global_fg is not None and global_bg is not None:
                    global_fg = torch.flip(global_fg, dims=[-1])
                    global_bg = torch.flip(global_bg, dims=[-1])
                if global_input_gt is not None:
                    global_input_gt = torch.flip(global_input_gt, dims=[-1])
                global_alpha = torch.flip(global_alpha, dims=[-1])
                old_x0 = tile_coords[:, 2].clone()
                old_x1 = tile_coords[:, 3].clone()
                tile_coords[:, 2] = float(source_w) - old_x1
                tile_coords[:, 3] = float(source_w) - old_x0

        # ---- Background temporal shuffle ----
        if random.random() < cfg.background_replace_p and bg.shape[1] > 1:
            order = torch.randperm(bg.shape[1], device=bg.device)
            bg = bg.index_select(dim=1, index=order)
            if emit_global_context and global_bg is not None:
                global_bg = global_bg.index_select(dim=1, index=order)

        # ---- Photometric: subject / BG gain ----
        subject_gain = random.uniform(cfg.subject_gain_min, cfg.subject_gain_max)
        bg_gain = random.uniform(cfg.bg_gain_min, cfg.bg_gain_max)
        fg = fg * subject_gain
        bg = bg * bg_gain
        if emit_global_context and global_fg is not None:
            global_fg = global_fg * subject_gain
        if emit_global_context and global_bg is not None:
            global_bg = global_bg * bg_gain

        # ---- WB jitter ----
        if cfg.wb_jitter_strength > 0 and random.random() < cfg.wb_jitter_p:
            s = float(cfg.wb_jitter_strength)
            wb = torch.tensor(
                [
                    random.uniform(1.0 - s, 1.0 + s),
                    random.uniform(1.0 - s / 3.0, 1.0 + s / 3.0),
                    random.uniform(1.0 - s, 1.0 + s),
                ],
                dtype=fg.dtype,
                device=fg.device,
            ).view(1, 1, 3, 1, 1)
            fg = fg * wb
            bg = bg * wb
            if emit_global_context:
                if global_fg is not None:
                    global_fg = global_fg * wb
                if global_bg is not None:
                    global_bg = global_bg * wb
                elif global_input_gt is not None:
                    global_input_gt = global_input_gt * wb

        # ---- Representation-specific target + composite ----
        # Source FG is premultiplied on disk.  Straight mode supervises unmixed
        # colour directly, while the synthesized plate remains physically
        # equivalent via FG_straight * alpha + BG * (1-alpha).
        fg = _fg_target_from_premul(fg, alpha, fg_representation)
        input_gt = _composite_fg(fg, bg, alpha, fg_representation)
        if emit_global_context:
            assert global_alpha is not None
            if global_input_gt is None:
                if global_fg is not None and global_bg is not None:
                    global_fg = _fg_target_from_premul(global_fg, global_alpha, fg_representation)
                    global_input_gt = _composite_fg(global_fg, global_bg, global_alpha, fg_representation)
                elif global_fg is not None:
                    global_fg = _fg_target_from_premul(global_fg, global_alpha, fg_representation)
            elif global_fg is not None:
                global_fg = _fg_target_from_premul(global_fg, global_alpha, fg_representation)
            global_bg = None
        else:
            global_input_gt = None

        # ---- Synthetic green spill (input branch only) ----
        if random.random() < cfg.spill_augment_p:
            # _boundary_band wants [N, 1, H, W]; fold (B, T) → N for the conv.
            bb, tt, _, hh, ww = alpha.shape
            edge = _boundary_band(alpha.view(bb * tt, 1, hh, ww)).view(bb, tt, 1, hh, ww)
            spill_strength = random.uniform(0.05, 0.25)
            input_gt = input_gt.clone()
            input_gt[:, :, 1:2].add_(edge, alpha=spill_strength)
            input_gt.clamp_min_(0.0)
            if emit_global_context:
                assert global_alpha is not None and global_input_gt is not None
                gbb, gtt, _, ghh, gww = global_alpha.shape
                global_edge = _boundary_band(global_alpha.view(gbb * gtt, 1, ghh, gww)).view(
                    gbb, gtt, 1, ghh, gww
                )
                global_input_gt = global_input_gt.clone()
                global_input_gt[:, :, 1:2].add_(global_edge, alpha=spill_strength)
                global_input_gt.clamp_min_(0.0)

        # ---- Saturation + hue jitter (input branch only) ----
        if random.random() < cfg.color_jitter_p:
            sat = random.uniform(cfg.saturation_min, cfg.saturation_max)
            if sat != 1.0:
                bb, tt, c, hh, ww = input_gt.shape
                flat = input_gt.view(bb * tt, c, hh, ww)
                input_gt = _saturation_jitter(flat, sat).view(bb, tt, c, hh, ww)
                if emit_global_context:
                    assert global_input_gt is not None
                    gbb, gtt, gc, ghh, gww = global_input_gt.shape
                    gflat = global_input_gt.view(gbb * gtt, gc, ghh, gww)
                    global_input_gt = _saturation_jitter(gflat, sat).view(gbb, gtt, gc, ghh, gww)
            if cfg.hue_jitter_max > 0:
                hue = random.uniform(-cfg.hue_jitter_max, cfg.hue_jitter_max)
                bb, tt, c, hh, ww = input_gt.shape
                flat = input_gt.view(bb * tt, c, hh, ww)
                input_gt = _hue_jitter(flat, hue).view(bb, tt, c, hh, ww)
                if emit_global_context:
                    assert global_input_gt is not None
                    gbb, gtt, gc, ghh, gww = global_input_gt.shape
                    gflat = global_input_gt.view(gbb * gtt, gc, ghh, gww)
                    global_input_gt = _hue_jitter(gflat, hue).view(gbb, gtt, gc, ghh, gww)

        # ---- Motion blur (per-clip uniform direction + length) ----
        if random.random() < cfg.motion_blur_p:
            kmin = max(1, int(cfg.motion_blur_kernel_min))
            kmax = max(kmin, int(cfg.motion_blur_kernel_max))
            k = random.randint(kmin, kmax)
            direction = random.choice(("h", "v", "d1", "d2"))
            bb, tt, c, hh, ww = input_gt.shape
            flat = input_gt.view(bb * tt, c, hh, ww)
            input_gt = _motion_blur(flat, kernel_size=k, direction=direction).view(bb, tt, c, hh, ww)
            if emit_global_context:
                assert global_input_gt is not None
                gbb, gtt, gc, ghh, gww = global_input_gt.shape
                gflat = global_input_gt.view(gbb * gtt, gc, ghh, gww)
                global_input_gt = _motion_blur(gflat, kernel_size=k, direction=direction).view(
                    gbb, gtt, gc, ghh, gww
                )

        # ---- Lens softness gaussian blur ----
        if random.random() < cfg.blur_p:
            sigma = random.uniform(cfg.blur_sigma_min, cfg.blur_sigma_max)
            radius = max(1, int(round(2.0 * sigma)))
            kernel_size = 2 * radius + 1
            bb, tt, c, hh, ww = input_gt.shape
            flat = input_gt.view(bb * tt, c, hh, ww)
            input_gt = _gaussian_blur(flat, kernel_size=kernel_size, sigma=sigma).view(bb, tt, c, hh, ww)
            if emit_global_context:
                assert global_input_gt is not None
                gbb, gtt, gc, ghh, gww = global_input_gt.shape
                gflat = global_input_gt.view(gbb * gtt, gc, ghh, gww)
                global_input_gt = _gaussian_blur(gflat, kernel_size=kernel_size, sigma=sigma).view(
                    gbb, gtt, gc, ghh, gww
                )

        # ---- Compression / codec proxy ----
        if random.random() < cfg.compression_p:
            ds = random.uniform(cfg.compression_downscale_min, cfg.compression_downscale_max)
            bb, tt, c, hh, ww = input_gt.shape
            flat = input_gt.view(bb * tt, c, hh, ww)
            input_gt = _compression_proxy(flat, downscale=ds).view(bb, tt, c, hh, ww)
            if emit_global_context:
                assert global_input_gt is not None
                gbb, gtt, gc, ghh, gww = global_input_gt.shape
                gflat = global_input_gt.view(gbb * gtt, gc, ghh, gww)
                global_input_gt = _compression_proxy(gflat, downscale=ds).view(gbb, gtt, gc, ghh, gww)

        # ---- Sensor noise: gaussian read + brightness-scaled shot ----
        if random.random() < cfg.noise_p:
            sigma = random.uniform(cfg.noise_sigma_min, cfg.noise_sigma_max)
            input_gt = input_gt.clone()
            input_gt.add_(torch.randn_like(input_gt), alpha=sigma)
            if emit_global_context:
                assert global_input_gt is not None
                global_input_gt = global_input_gt.clone()
                global_input_gt.add_(torch.randn_like(global_input_gt), alpha=sigma)
        if random.random() < cfg.shot_noise_p:
            shot_scale = input_gt.clamp_min(0.0).sqrt_()
            input_gt.addcmul_(torch.randn_like(input_gt), shot_scale, value=float(cfg.shot_noise_strength))
            if emit_global_context:
                assert global_input_gt is not None
                global_shot_scale = global_input_gt.clamp_min(0.0).sqrt_()
                global_input_gt.addcmul_(
                    torch.randn_like(global_input_gt),
                    global_shot_scale,
                    value=float(cfg.shot_noise_strength),
                )

        # ---- Final clamp ----
        # Both ``input_gt`` (model input) and the loss-side ground-truth
        # ``fg_gt`` / ``bg_gt`` are squashed into ``[0, 1]``. The CorridorKey
        # premultiplied FG can carry HDR radiometry (>1) but the model output
        # is bounded by sigmoid in ``[0, 1]``; mixing an HDR ``fg_gt`` target
        # with a clamped ``input_gt`` target would let ``l_fg`` and ``l_comp``
        # disagree on the same pixel and their gradients would cancel out.
        input_gt.clamp_(0.0, 1.0)
        fg.clamp_(0.0, 1.0)
        bg.clamp_(0.0, 1.0)
        if emit_global_context:
            assert global_alpha is not None
            if global_fg is not None:
                global_fg = global_fg.clamp(0.0, 1.0)
                global_fg, _ = _resize_long_side(
                    [global_fg, global_alpha],
                    int(cfg.global_context_long_side),
                )
            if global_input_gt is not None:
                global_input_gt.clamp_(0.0, 1.0)
                global_input_gt, _ = _resize_long_side(
                    [global_input_gt, global_alpha],
                    int(cfg.global_context_long_side),
                )
            global_alpha = global_alpha.clamp(0.0, 1.0)

        # ---- coarse_alpha_init from first-frame alpha of every sample ----
        coarse = _generate_coarse_alpha_init_device(alpha[:, 0])
        drop_prob = float(cfg.coarse_alpha_drop_prob)
        if drop_prob > 0.0:
            drop = (torch.rand((b, 1, 1, 1), device=coarse.device) < drop_prob).to(coarse.dtype)
            coarse = coarse * (1.0 - drop)
        bb, tt, _, hh, ww = alpha.shape
        alpha_boundary_gt = (_boundary_band(alpha.view(bb * tt, 1, hh, ww)).view(bb, tt, 1, hh, ww) > 0.02).to(
            alpha.dtype
        )

        out = dict(batch)
        out["fg_gt"] = fg
        out["bg_gt"] = bg
        out["alpha_gt"] = alpha
        out["alpha_boundary_gt"] = alpha_boundary_gt
        out["video_rgb"] = input_gt
        out["input_gt"] = input_gt  # alias, same storage
        out["coarse_alpha_init"] = coarse
        if emit_global_context:
            assert global_alpha is not None
            global_coarse = _resize_alpha_long_side(
                _generate_coarse_alpha_init_device(global_alpha[:, 0]),
                int(cfg.global_context_long_side),
            )
            if drop_prob > 0.0:
                global_drop = (torch.rand((b, 1, 1, 1), device=global_coarse.device) < drop_prob).to(global_coarse.dtype)
                global_coarse = global_coarse * (1.0 - global_drop)
            if global_fg is not None:
                out["global_fg_gt"] = global_fg
            out["global_video_rgb"] = global_input_gt if global_input_gt is not None else global_fg
            out["global_coarse_alpha_init"] = global_coarse
            out["tile_coords"] = tile_coords
            out["source_hw"] = source_hw
        return out


def build_device_transform_from_data_cfg(
    data_cfg: Dict[str, Any], train_cfg: Optional[Dict[str, Any]] = None
) -> DeviceMattingTransform:
    """Construct a DeviceMattingTransform from the same ``data:`` block of a
    YAML config that drives ``CorridorMattingTransform``.

    Parameters that are stochastic on a per-clip basis come from ``data.*``
    (so the host- and device-mode runs are configured from a single source).
    The ``train.*`` block is accepted only for forward compatibility (none of
    its current keys influence the device transform directly).
    """
    cfg = DeviceMattingTransformConfig(
        fixed_crop_size=int(data_cfg.get("fixed_crop_size", 0)),
        global_context_long_side=int(
            data_cfg.get(
                "global_context_long_side",
                train_cfg.get("global_long_side_cap", 0) if train_cfg is not None else 0,
            )
            if (
                bool(data_cfg.get("emit_global_context", False))
                or (train_cfg is not None and str(train_cfg.get("inference_mode", "")).strip().lower() == "hybrid")
            )
            else 0
        ),
        fg_representation=str(data_cfg.get("fg_representation", "premul")),
        global_context_pre_resize=bool(data_cfg.get("global_context_pre_resize", False)),
        horizontal_flip_p=float(data_cfg.get("horizontal_flip_p", 0.5)),
        background_replace_p=float(data_cfg.get("background_replace_p", 0.3)),
        spill_augment_p=float(data_cfg.get("spill_augment_p", 0.4)),
        green_foreground_prob=float(data_cfg.get("green_foreground_prob", data_cfg.get("green_foreground_augment_p", 0.0))),
        green_foreground_strength_min=float(data_cfg.get("green_foreground_strength_min", 0.25)),
        green_foreground_strength_max=float(data_cfg.get("green_foreground_strength_max", 0.75)),
        green_foreground_alpha_thresh=float(data_cfg.get("green_foreground_alpha_thresh", 0.70)),
        coarse_alpha_drop_prob=float(data_cfg.get("coarse_alpha_drop_prob", 0.0)),
        subject_gain_min=float(data_cfg.get("subject_gain_min", 0.7)),
        subject_gain_max=float(data_cfg.get("subject_gain_max", 1.3)),
        bg_gain_min=float(data_cfg.get("bg_gain_min", 0.65)),
        bg_gain_max=float(data_cfg.get("bg_gain_max", 1.35)),
        wb_jitter_p=float(data_cfg.get("wb_jitter_p", 0.7)),
        wb_jitter_strength=float(data_cfg.get("wb_jitter_strength", 0.15)),
        color_jitter_p=float(data_cfg.get("color_jitter_p", 0.4)),
        saturation_min=float(data_cfg.get("saturation_min", 0.7)),
        saturation_max=float(data_cfg.get("saturation_max", 1.3)),
        hue_jitter_max=float(data_cfg.get("hue_jitter_max", 0.15)),
        noise_p=float(data_cfg.get("noise_p", 0.8)),
        noise_sigma_min=float(data_cfg.get("noise_sigma_min", 0.005)),
        noise_sigma_max=float(data_cfg.get("noise_sigma_max", 0.04)),
        shot_noise_p=float(data_cfg.get("shot_noise_p", 0.3)),
        shot_noise_strength=float(data_cfg.get("shot_noise_strength", 0.025)),
        blur_p=float(data_cfg.get("blur_p", 0.2)),
        blur_sigma_min=float(data_cfg.get("blur_sigma_min", 0.3)),
        blur_sigma_max=float(data_cfg.get("blur_sigma_max", 0.8)),
        motion_blur_p=float(data_cfg.get("motion_blur_p", 0.25)),
        motion_blur_kernel_min=int(data_cfg.get("motion_blur_kernel_min", 3)),
        motion_blur_kernel_max=int(data_cfg.get("motion_blur_kernel_max", 7)),
        compression_p=float(data_cfg.get("compression_p", 0.2)),
        compression_downscale_min=float(data_cfg.get("compression_downscale_min", 1.05)),
        compression_downscale_max=float(data_cfg.get("compression_downscale_max", 1.35)),
    )
    return DeviceMattingTransform(cfg)
