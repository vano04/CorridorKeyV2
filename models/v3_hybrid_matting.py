"""V3 hybrid video matting model.

Fresh nn.Module — no inheritance from V1/V2. Self-contained three-branch
architecture: global context + local tile encoder + native detail refiner.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple
from contextlib import nullcontext

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .global_context import GlobalContextBranch
from .local_tile_encoder import LocalTileEncoder
from .native_detail_refiner import NativeDetailRefiner
from .reference_memory import ReferenceMemoryBank
from .heads import AlphaHead, ForegroundHead, UncertaintyHead, SpillMaskHead, QualityEvalHead
from .positional import make_tile_coordinate_channels, make_default_tile_coords
from .transformer_engine_utils import resolve_fp8_config, transformer_engine_available


EPS = 1e-6
FG_REPRESENTATIONS = {"premul", "straight"}
FG_PREDICTION_MODES = {"decoder", "input_residual"}
GLOBAL_CONTEXT_ALPHA_MODES = {"seed", "zero"}


def _validate_fg_representation(value: str) -> str:
    rep = str(value).strip().lower()
    if rep not in FG_REPRESENTATIONS:
        raise ValueError(f"Unsupported fg_representation={value!r}")
    return rep


def _validate_fg_prediction_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in FG_PREDICTION_MODES:
        raise ValueError(f"Unsupported fg_prediction_mode={value!r}")
    return mode


def _validate_global_context_alpha_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("_", "-")
    if mode in {"none", "off", "false", "0"}:
        return "zero"
    if mode not in GLOBAL_CONTEXT_ALPHA_MODES:
        raise ValueError(f"Unsupported global_context_alpha_mode={value!r}")
    return mode


def _composite_fg(fg: Tensor, bg: Tensor, alpha: Tensor, representation: str) -> Tensor:
    if representation == "straight":
        return fg * alpha + (1.0 - alpha) * bg
    return fg + (1.0 - alpha) * bg


def _input_residual_fg_base(
    video: Tensor,
    alpha: Tensor,
    *,
    despill: bool,
    strength: float,
) -> Tensor:
    base = video * alpha
    if not despill or strength <= 0.0:
        return base.clamp(0.0, 1.0)

    r = video[:, :, 0:1]
    g = video[:, :, 1:2]
    b = video[:, :, 2:3]
    green_excess = (g - torch.maximum(r, b)).clamp_min(0.0)
    transition = (alpha * (1.0 - alpha) * 4.0).clamp(0.0, 1.0)
    correction = green_excess * alpha * (1.0 - alpha) * transition * float(strength)
    corrected_g = (base[:, :, 1:2] - correction).clamp_min(0.0)
    return torch.cat([base[:, :, 0:1], corrected_g, base[:, :, 2:3]], dim=2).clamp(0.0, 1.0)


def _alpha_only_edge_refine(
    alpha: Tensor,
    refine_mask: Optional[Tensor],
    *,
    strength: float,
    kernel_size: int,
) -> Tuple[Tensor, Tensor]:
    if strength <= 0.0:
        return alpha, torch.zeros_like(alpha)

    k = max(3, int(kernel_size))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    b, t, _, h, w = alpha.shape
    flat = alpha.reshape(b * t, 1, h, w)
    low = F.avg_pool2d(F.pad(flat, (pad, pad, pad, pad), mode="replicate"), kernel_size=k, stride=1)
    high = flat - low

    transition = (flat * (1.0 - flat) * 4.0).clamp(0.0, 1.0)
    gate = transition
    if refine_mask is not None:
        gate = gate * refine_mask.reshape(b * t, 1, h, w).clamp(0.0, 1.0)

    contrast = (flat - 0.5) * 0.5
    delta = (high + contrast) * gate * float(strength)
    refined = (flat + delta).clamp(0.0, 1.0)
    return refined.reshape_as(alpha), delta.reshape_as(alpha)


def _compute_padding(h: int, w: int, multiple: int) -> Tuple[int, int]:
    return (multiple - h % multiple) % multiple, (multiple - w % multiple) % multiple


def _safe_reflect_pad(x: Tensor, pad_w: int, pad_h: int) -> Tensor:
    h, w = x.shape[-2], x.shape[-1]
    if pad_h >= h or pad_w >= w:
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


def _pad_video(video: Tensor, multiple: int) -> Tuple[Tensor, Tuple[int, int]]:
    b, t, c, h, w = video.shape
    pad_h, pad_w = _compute_padding(h, w, multiple)
    if pad_h == 0 and pad_w == 0:
        return video, (0, 0)
    x = video.reshape(b * t, c, h, w)
    x = _safe_reflect_pad(x, pad_w, pad_h)
    return x.reshape(b, t, c, h + pad_h, w + pad_w), (pad_h, pad_w)


def _unpad_video(video: Tensor, pad_hw: Tuple[int, int]) -> Tensor:
    pad_h, pad_w = pad_hw
    if pad_h == 0 and pad_w == 0:
        return video
    return video[..., :video.shape[-2] - pad_h, :video.shape[-1] - pad_w]


def _resize_video(video: Tensor, out_hw: Tuple[int, int]) -> Tensor:
    b, t, c, h, w = video.shape
    x = video.reshape(b * t, c, h, w)
    x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
    return x.reshape(b, t, c, out_hw[0], out_hw[1])


@torch.jit.script
def _boundary_band(alpha: Tensor, kernel_size: int = 3) -> Tensor:
    """Morphological boundary band via dilation - erosion.

    Uses in-place .neg_() and .clamp_() to avoid extra kernel launches.
    """
    bt = alpha.shape[0] * alpha.shape[1]
    a = alpha.reshape(bt, 1, alpha.shape[-2], alpha.shape[-1])
    neg_a = a.neg()  # 1 launch
    dil = F.max_pool2d(a,     kernel_size, stride=1, padding=kernel_size // 2)
    ero = F.max_pool2d(neg_a, kernel_size, stride=1, padding=kernel_size // 2).neg_()  # in-place negate
    band = (dil - ero).clamp_(0.0, 1.0)  # in-place clamp
    return band.reshape(alpha.shape[0], alpha.shape[1], 1, alpha.shape[-2], alpha.shape[-1])


@torch.jit.script
def _green_excess_map(rgb: Tensor) -> Tensor:
    """Compute green excess: G - max(R, B), clamped to [0, 1]."""
    g = rgb[:, :, 1:2]
    r = rgb[:, :, 0:1]
    b = rgb[:, :, 2:3]
    return (g - torch.maximum(r, b)).clamp(0.0, 1.0)


@torch.jit.script
def _chroma_distance_map(
    rgb: Tensor,
    key_r: float = 0.0,
    key_g: float = 1.0,
    key_b: float = 0.0,
) -> Tensor:
    """Per-pixel chroma distance to key color.

    TorchScript fuses the luma, chroma-diff, and L2 chains into ~3 kernels.
    Key color passed as scalars (not a tuple) for TorchScript compatibility.
    """
    r = rgb[:, :, 0:1]
    g = rgb[:, :, 1:2]
    b = rgb[:, :, 2:3]
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    key_luma = 0.299 * key_r + 0.587 * key_g + 0.114 * key_b
    # Fused: chroma diff from key in one expression per channel
    dr = (r - luma) - (key_r - key_luma)
    dg = (g - luma) - (key_g - key_luma)
    db = (b - luma) - (key_b - key_luma)
    return (dr * dr + dg * dg + db * db).sqrt().clamp(0.0, 1.0)


def _unknown_band_from_hint(hint: Tensor, threshold: float = 0.02) -> Tensor:
    """Mark uncertain regions from coarse alpha hint."""
    return ((hint > threshold) & (hint < 1.0 - threshold)).to(hint.dtype)


@dataclass
class V3InferenceOptions:
    mode: str = "full"
    global_long_side_cap: int = 512
    tile_size: int = 1024
    tile_overlap: int = 64
    refine_on_uncertainty: bool = True
    refine_on_edges: bool = True
    refine_on_spill_regions: bool = True


class V3HybridVideoMattingModel(nn.Module):
    """V3 three-branch hybrid matting model.

    Branch 1: GlobalContextBranch — runs on downscaled full-frame, produces compact tokens
    Branch 2: LocalTileEncoder — processes 1024x1024 tiles with extended inputs
    Branch 3: NativeDetailRefiner — produces bounded residual deltas at native res
    """

    def __init__(
        self,
        # Patch / embedding
        patch_size: int = 8,
        embed_dims: Sequence[int] = (128, 256, 384, 512),
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (4, 8, 12, 16),
        window_sizes: Sequence[int] = (8, 8, 4, 4),
        memory_tokens: int = 64,
        temporal_window: int = 5,
        drop_path_rate: float = 0.1,
        # Input channels
        use_green_priors: bool = True,
        use_coordinate_channels: bool = True,
        use_unknown_band: bool = True,
        guidance_presence_flag: bool = True,
        green_prior_dropout: float = 0.0,
        coarse_alpha_drop_prob: float = 0.0,
        # Heads
        predict_fg: bool = True,
        predict_uncertainty: bool = True,
        predict_spill_mask: bool = True,
        predict_quality_eval: bool = True,
        fg_representation: str = "premul",
        fg_prediction_mode: str = "decoder",
        input_residual_despill_base: bool = False,
        input_residual_despill_strength: float = 1.0,
        alpha_only_refine: bool = False,
        alpha_only_refine_strength: float = 0.75,
        alpha_only_refine_kernel_size: int = 5,
        # Global context
        global_context_dim: int = 256,
        global_context_layers: int = 4,
        global_context_heads: int = 8,
        global_context_tokens: int = 64,
        use_global_fg_guidance: bool = False,
        global_context_alpha_mode: str = "seed",
        # Reference memory
        use_reference_memory: bool = True,
        reference_tokens_per_frame: int = 32,
        num_reference_frames: int = 2,
        reference_dropout: float = 0.25,
        # Native detail refiner
        use_native_refiner: bool = True,
        native_refiner_blocks: int = 8,
        native_refiner_hidden: int = 64,
        native_refiner_chunk_frames: int = 2,
        native_refiner_num_heads: int = 4,
        native_refiner_head_dim: int = 16,
        native_refiner_mlp_ratio: float = 2.0,
        native_refiner_window_sizes: Optional[Sequence[int]] = None,
        native_refiner_shift_every_other_block: bool = True,
        native_refiner_attention_backend: str = "linear",
        native_refiner_attention_types: Optional[Sequence[str]] = None,
        native_refiner_linear_feature_map: str = "elu_plus_one",
        native_refiner_linear_attention_eps: float = 1.0e-6,
        native_refiner_use_2d_rope: bool = True,
        native_refiner_use_checkpointing: Optional[bool] = None,
        max_alpha_delta: float = 0.03,
        max_fg_delta: float = 0.02,
        max_spill_delta: float = 0.15,
        # Performance
        gradient_checkpointing: bool = False,
        decoder_out_dim: int = 256,
        # FP8 / Transformer Engine
        fp8_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        self.fp8_cfg = resolve_fp8_config({"fp8": fp8_cfg}) if fp8_cfg is not None else resolve_fp8_config({})

        self.patch_size = patch_size
        self.predict_fg = predict_fg
        self.predict_uncertainty = predict_uncertainty
        self.predict_spill_mask = predict_spill_mask
        self.predict_quality_eval = predict_quality_eval
        self.fg_representation = _validate_fg_representation(fg_representation)
        self.fg_prediction_mode = _validate_fg_prediction_mode(fg_prediction_mode)
        self.input_residual_despill_base = bool(input_residual_despill_base)
        self.input_residual_despill_strength = float(input_residual_despill_strength)
        self.alpha_only_refine = bool(alpha_only_refine)
        self.alpha_only_refine_strength = float(alpha_only_refine_strength)
        self.alpha_only_refine_kernel_size = int(alpha_only_refine_kernel_size)
        self.use_green_priors = use_green_priors
        self.use_coordinate_channels = use_coordinate_channels
        self.use_unknown_band = use_unknown_band
        self.guidance_presence_flag = guidance_presence_flag
        self.use_global_fg_guidance = bool(use_global_fg_guidance)
        self.global_context_alpha_mode = _validate_global_context_alpha_mode(global_context_alpha_mode)
        self.green_prior_dropout = float(max(0.0, min(1.0, green_prior_dropout)))
        self.coarse_alpha_drop_prob = float(max(0.0, min(1.0, coarse_alpha_drop_prob)))
        self.use_native_refiner = use_native_refiner
        self.use_reference_memory = use_reference_memory
        self.native_refiner_chunk_frames = max(1, int(native_refiner_chunk_frames))

        # Compute input channels for local tile encoder
        in_ch = 4
        if guidance_presence_flag:
            in_ch += 1
        if use_green_priors:
            in_ch += 2
        if use_unknown_band:
            in_ch += 1
        if use_coordinate_channels:
            in_ch += 5
        self.local_in_channels = in_ch

        # Global context input
        global_in_ch = 4
        if use_green_priors:
            global_in_ch += 2
        if self.use_global_fg_guidance:
            global_in_ch += 3

        self.global_context = GlobalContextBranch(
            in_channels=global_in_ch,
            embed_dim=global_context_dim,
            num_layers=global_context_layers,
            num_heads=global_context_heads,
            patch_size=patch_size,
            memory_tokens=global_context_tokens,
            gradient_checkpointing=gradient_checkpointing,
            fp8_cfg=self.fp8_cfg,
        )

        self.local_encoder = LocalTileEncoder(
            in_channels=in_ch,
            patch_size=patch_size,
            embed_dims=embed_dims,
            depths=depths,
            num_heads=num_heads,
            window_sizes=window_sizes,
            memory_tokens=memory_tokens,
            temporal_window=temporal_window,
            drop_path_rate=drop_path_rate,
            gradient_checkpointing=gradient_checkpointing,
            global_context_dim=global_context_dim,
            decoder_out_dim=decoder_out_dim,
            fp8_cfg=self.fp8_cfg,
        )

        if use_reference_memory:
            self.reference_memory = ReferenceMemoryBank(
                dim=embed_dims[-1],
                num_reference_frames=num_reference_frames,
                tokens_per_reference=reference_tokens_per_frame,
                dropout=reference_dropout,
            )
        else:
            self.reference_memory = None

        self.alpha_head = AlphaHead(decoder_out_dim)
        self.fg_head = ForegroundHead(decoder_out_dim) if predict_fg else None
        self.uncertainty_head = UncertaintyHead(decoder_out_dim) if predict_uncertainty else None
        self.spill_head = SpillMaskHead(decoder_out_dim) if predict_spill_mask else None
        self.quality_eval_head = QualityEvalHead(decoder_out_dim) if predict_quality_eval else None

        if use_native_refiner:
            refiner_in = 3 + 1 + 3 + 1 + (1 if predict_spill_mask else 0)
            self.native_refiner = NativeDetailRefiner(
                in_channels=refiner_in,
                hidden_channels=native_refiner_hidden,
                num_blocks=native_refiner_blocks,
                max_alpha_delta=max_alpha_delta,
                max_fg_delta=max_fg_delta,
                max_spill_delta=max_spill_delta,
                predict_spill=predict_spill_mask,
                global_context_dim=global_context_dim,
                num_heads=native_refiner_num_heads,
                head_dim=native_refiner_head_dim,
                mlp_ratio=native_refiner_mlp_ratio,
                window_sizes=native_refiner_window_sizes,
                shift_every_other_block=native_refiner_shift_every_other_block,
                attention_backend=native_refiner_attention_backend,
                attention_types=native_refiner_attention_types,
                linear_feature_map=native_refiner_linear_feature_map,
                linear_attention_eps=native_refiner_linear_attention_eps,
                use_2d_rope=native_refiner_use_2d_rope,
                use_checkpointing=(
                    gradient_checkpointing
                    if native_refiner_use_checkpointing is None
                    else native_refiner_use_checkpointing
                ),
            )
        else:
            self.native_refiner = None

    def _build_local_input(
        self,
        video: Tensor,
        coarse_alpha_init: Tensor,
        tile_coords: Optional[Tensor],
        source_hw: Optional[Tensor],
        guidance_present: Optional[Tensor] = None,
        green_excess: Optional[Tensor] = None,
        chroma_dist: Optional[Tensor] = None,
    ) -> Tensor:
        """Build the local tile encoder input tensor.

        green_excess / chroma_dist can be passed in pre-computed to avoid
        redundant _green_excess_map / _chroma_distance_map calls when the same
        video tensor is used for both global and local inputs.
        """
        b, t, _, h, w = video.shape
        device = video.device
        dtype = video.dtype
        cond = torch.zeros(b, t, 1, h, w, device=device, dtype=dtype)
        cond[:, 0] = coarse_alpha_init
        parts = [video, cond]
        if self.guidance_presence_flag:
            if guidance_present is None:
                guidance_present = torch.ones(b, 1, 1, 1, device=device, dtype=dtype)
            else:
                guidance_present = guidance_present.to(device=device, dtype=dtype)
            flag = torch.zeros(b, t, 1, h, w, device=device, dtype=dtype)
            flag[:, 0] = guidance_present
            parts.append(flag)
        if self.use_green_priors:
            if green_excess is None:
                green_excess = _green_excess_map(video)
            if chroma_dist is None:
                chroma_dist = _chroma_distance_map(video)
            ge = green_excess
            cd = chroma_dist
            if self.training and self.green_prior_dropout > 0.0:
                keep = (
                    torch.rand((b, 1, 1, 1, 1), device=device)
                    >= self.green_prior_dropout
                ).to(dtype=dtype)
                ge = ge * keep
                cd = cd * keep
            parts.append(ge)
            parts.append(cd)
        if self.use_unknown_band:
            parts.append(_unknown_band_from_hint(cond))
        if self.use_coordinate_channels:
            if tile_coords is None or source_hw is None:
                tile_coords, source_hw = make_default_tile_coords(b, h, w, device, dtype)
            else:
                tile_coords = tile_coords.to(device=device, dtype=dtype)
                source_hw = source_hw.to(device=device, dtype=dtype)
            coord_maps = make_tile_coordinate_channels(tile_coords, source_hw, h, w)
            parts.append(coord_maps.unsqueeze(1).expand(b, t, -1, h, w))
        return torch.cat(parts, dim=2)

    def _build_global_input(
        self,
        global_video: Tensor,
        global_coarse_alpha_init: Tensor,
        global_fg_guidance: Optional[Tensor] = None,
        green_excess: Optional[Tensor] = None,
        chroma_dist: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
        """Build global context branch input tensors.

        green_excess / chroma_dist can be pre-computed and passed in to avoid
        a second call when global_video == video.
        """
        b, t, _, h, w = global_video.shape
        cond = torch.zeros(b, t, 1, h, w, device=global_video.device, dtype=global_video.dtype)
        if self.global_context_alpha_mode == "seed":
            cond[:, 0] = global_coarse_alpha_init
        fg_guidance = None
        if self.use_global_fg_guidance:
            if global_fg_guidance is not None:
                fg_guidance = global_fg_guidance.to(device=global_video.device, dtype=global_video.dtype)
            else:
                fg_guidance = torch.zeros(
                    b, t, 3, h, w,
                    device=global_video.device,
                    dtype=global_video.dtype,
                )
        green_priors = None
        if self.use_green_priors:
            if green_excess is None:
                green_excess = _green_excess_map(global_video)
            if chroma_dist is None:
                chroma_dist = _chroma_distance_map(global_video)
            ge = green_excess
            cd = chroma_dist
            if self.training and self.green_prior_dropout > 0.0:
                keep = (
                    torch.rand((b, 1, 1, 1, 1), device=global_video.device)
                    >= self.green_prior_dropout
                ).to(dtype=global_video.dtype)
                ge = ge * keep
                cd = cd * keep
            green_priors = torch.cat([ge, cd], dim=2)
        return cond, green_priors, fg_guidance

    def forward(
        self,
        video: Tensor,
        coarse_alpha_init: Tensor,
        valid_mask: Optional[Tensor] = None,
        bg_for_comp: Optional[Tensor] = None,
        inference_options: Optional[V3InferenceOptions] = None,
        global_video: Optional[Tensor] = None,
        global_coarse_alpha_init: Optional[Tensor] = None,
        global_fg_guidance: Optional[Tensor] = None,
        global_tokens: Optional[Tensor] = None,
        ref_tokens: Optional[Tensor] = None,
        tile_coords: Optional[Tensor] = None,
        source_hw: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        b, t, _, h, w = video.shape
        te = None
        if self.fp8_cfg["enabled"] and self.fp8_cfg["backend"] == "transformer_engine":
            try: import transformer_engine.pytorch as te
            except ImportError: pass
        fp8_context = te.fp8_autocast(enabled=True) if te is not None else nullcontext()

        with fp8_context:
            coarse_alpha_model = coarse_alpha_init
            global_coarse_alpha_model = (
                global_coarse_alpha_init if global_coarse_alpha_init is not None else coarse_alpha_init
            )
            guidance_present = torch.ones(b, 1, 1, 1, device=video.device, dtype=video.dtype)
            if self.training and self.coarse_alpha_drop_prob > 0.0:
                drop = (
                    torch.rand((b, 1, 1, 1), device=video.device)
                    < self.coarse_alpha_drop_prob
                )
                guidance_present = (~drop).to(dtype=video.dtype)
                coarse_alpha_model = torch.where(drop, torch.zeros_like(coarse_alpha_init), coarse_alpha_init)
                if global_coarse_alpha_model is not None:
                    global_coarse_alpha_model = torch.where(
                        drop,
                        torch.zeros_like(global_coarse_alpha_model),
                        global_coarse_alpha_model,
                    )

            shared_ge: Optional[Tensor] = None
            shared_cd: Optional[Tensor] = None
            if global_tokens is None:
                if global_video is not None and global_coarse_alpha_model is not None:
                    c_bc, gp, fg_guidance = self._build_global_input(
                        global_video,
                        global_coarse_alpha_model,
                        global_fg_guidance=global_fg_guidance,
                    )
                    global_tokens, _ = self.global_context(
                        video_rgb_global=global_video,
                        coarse_alpha_global=c_bc,
                        green_priors_global=gp,
                        fg_guidance_global=fg_guidance,
                    )
                else:
                    # global_video is None: both branches share the same tensor.
                    # Compute green priors once and pass to both builders.
                    if self.use_green_priors:
                        shared_ge = _green_excess_map(video)
                        shared_cd = _chroma_distance_map(video)
                    else:
                        shared_ge = None
                        shared_cd = None
                    c_bc, gp, fg_guidance = self._build_global_input(
                        video,
                        coarse_alpha_model,
                        green_excess=shared_ge,
                        chroma_dist=shared_cd,
                    )
                    global_tokens, _ = self.global_context(
                        video_rgb_global=video,
                        coarse_alpha_global=c_bc,
                        green_priors_global=gp,
                        fg_guidance_global=fg_guidance,
                    )

            local_input = self._build_local_input(
                video,
                coarse_alpha_model,
                tile_coords,
                source_hw,
                guidance_present=guidance_present,
                # Re-use pre-computed priors only in the shared-video path
                green_excess=shared_ge if global_video is None and self.use_green_priors else None,
                chroma_dist=shared_cd if global_video is None and self.use_green_priors else None,
            )
            local_input, pad_hw = _pad_video(local_input, self.patch_size)
            video_padded, _ = _pad_video(video, self.patch_size)
            coarse_padded = coarse_alpha_model
            if pad_hw[0] or pad_hw[1]:
                coarse_padded = _safe_reflect_pad(coarse_alpha_model, pad_hw[1], pad_hw[0])

            if self.reference_memory is not None:
                stage_feats = self.local_encoder.encode(x=local_input, coarse_alpha_init=coarse_padded, video_rgb=video_padded, valid_mask=valid_mask, global_tokens=global_tokens)
                deepest, ref_tokens_out = self.reference_memory(stage_feats[-1], ref_tokens)
                stage_feats[-1] = deepest
                decoded = self.local_encoder.decoder(stage_feats)
            else:
                decoded, stage_feats = self.local_encoder(x=local_input, coarse_alpha_init=coarse_padded, video_rgb=video_padded, valid_mask=valid_mask, global_tokens=global_tokens)
                ref_tokens_out = ref_tokens

            alpha_pred = self.alpha_head(decoded)
            decoder_fg_pred = self.fg_head(decoded) if self.fg_head else torch.zeros_like(alpha_pred).expand(-1, 3, -1, -1)
            fg_pred = decoder_fg_pred
            uncertainty_pred = self.uncertainty_head(decoded) if self.uncertainty_head else None
            spill_mask_pred = self.spill_head(decoded) if self.spill_head else None
            quality_eval_pred = self.quality_eval_head(decoded) if self.quality_eval_head and self.training else None

            target_h, target_w = h + pad_hw[0], w + pad_hw[1]
            if alpha_pred.shape[-2:] != (target_h, target_w):
                alpha_pred = F.interpolate(alpha_pred, size=(target_h, target_w), mode="bilinear", align_corners=False)
                fg_pred = F.interpolate(fg_pred, size=(target_h, target_w), mode="bilinear", align_corners=False)
                decoder_fg_pred = F.interpolate(decoder_fg_pred, size=(target_h, target_w), mode="bilinear", align_corners=False)
                if uncertainty_pred is not None: uncertainty_pred = F.interpolate(uncertainty_pred, size=(target_h, target_w), mode="bilinear", align_corners=False)
                if spill_mask_pred is not None: spill_mask_pred = F.interpolate(spill_mask_pred, size=(target_h, target_w), mode="bilinear", align_corners=False)
                if quality_eval_pred is not None: quality_eval_pred = F.interpolate(quality_eval_pred, size=(target_h, target_w), mode="bilinear", align_corners=False)

            alpha_pred = _unpad_video(alpha_pred.reshape(b, t, 1, target_h, target_w), pad_hw)
            fg_pred = _unpad_video(fg_pred.reshape(b, t, 3, target_h, target_w), pad_hw)
            decoder_fg_pred = _unpad_video(decoder_fg_pred.reshape(b, t, 3, target_h, target_w), pad_hw)
            if uncertainty_pred is not None: uncertainty_pred = _unpad_video(uncertainty_pred.reshape(b, t, 1, target_h, target_w), pad_hw)
            if spill_mask_pred is not None: spill_mask_pred = _unpad_video(spill_mask_pred.reshape(b, t, 1, target_h, target_w), pad_hw)
            if quality_eval_pred is not None: quality_eval_pred = _unpad_video(quality_eval_pred.reshape(b, t, 1, target_h, target_w), pad_hw)

            input_fg_base = _input_residual_fg_base(
                video,
                alpha_pred,
                despill=self.input_residual_despill_base,
                strength=self.input_residual_despill_strength,
            )
            if self.fg_prediction_mode == "input_residual":
                fg_pred = input_fg_base

            coarse_alpha_pred, coarse_fg_pred, coarse_spill_pred = alpha_pred, fg_pred, spill_mask_pred
            alpha_only_delta_pred: Optional[Tensor] = None
            alpha_only_refine_mask: Optional[Tensor] = None

            if self.native_refiner is not None:
                alpha_refine_mask_full = _boundary_band(alpha_pred)
                if uncertainty_pred is not None:
                    alpha_refine_mask_full = torch.maximum(alpha_refine_mask_full, uncertainty_pred)
                alpha_refine_mask_full = alpha_refine_mask_full.clamp(0, 1)

                # FG detail is not an edge-only problem. The decoder head predicts
                # from 1/patch_size features and is then upsampled, so the native
                # refiner must be allowed to adjust visible foreground interiors
                # using the full-resolution RGB tile.
                fg_refine_mask_full = alpha_pred.detach().clamp(0, 1)

                chunk_frames = max(1, min(self.native_refiner_chunk_frames, t))
                refiner_chunks: Dict[str, List[Tensor]] = {}

                for t0 in range(0, t, chunk_frames):
                    t1 = min(t, t0 + chunk_frames)
                    tc = t1 - t0
                    refiner_out_c = self.native_refiner(
                        rgb=video[:, t0:t1].reshape(b * tc, 3, h, w),
                        coarse_alpha=alpha_pred[:, t0:t1].reshape(b * tc, 1, h, w),
                        coarse_fg=fg_pred[:, t0:t1].reshape(b * tc, 3, h, w),
                        uncertainty=(uncertainty_pred[:, t0:t1].reshape(b * tc, 1, h, w) if uncertainty_pred is not None else None),
                        coarse_spill=(spill_mask_pred[:, t0:t1].reshape(b * tc, 1, h, w) if spill_mask_pred is not None else None),
                        alpha_refine_mask=alpha_refine_mask_full[:, t0:t1].reshape(b * tc, 1, h, w),
                        fg_refine_mask=fg_refine_mask_full[:, t0:t1].reshape(b * tc, 1, h, w),
                        spill_refine_mask=alpha_refine_mask_full[:, t0:t1].reshape(b * tc, 1, h, w),
                        global_tokens=global_tokens,
                    )
                    for key, value in refiner_out_c.items():
                        c = value.shape[1]
                        refiner_chunks.setdefault(key, []).append(value.reshape(b, tc, c, h, w))

                refiner_out = {key: torch.cat(values, dim=1) for key, values in refiner_chunks.items()}
                alpha_pred, fg_pred = refiner_out["alpha_refined"], refiner_out["fg_refined"]
                if "spill_refined" in refiner_out: spill_mask_pred = refiner_out["spill_refined"]
                if "uncertainty_refined" in refiner_out: uncertainty_pred = refiner_out["uncertainty_refined"]
                refine_mask = torch.maximum(alpha_refine_mask_full, fg_refine_mask_full).reshape(b*t, 1, h, w)

            if self.alpha_only_refine:
                alpha_only_refine_mask = _boundary_band(alpha_pred)
                if uncertainty_pred is not None:
                    alpha_only_refine_mask = torch.maximum(alpha_only_refine_mask, uncertainty_pred)
                alpha_only_refine_mask = alpha_only_refine_mask.clamp(0.0, 1.0)
                alpha_pred, alpha_only_delta_pred = _alpha_only_edge_refine(
                    alpha_pred,
                    alpha_only_refine_mask,
                    strength=self.alpha_only_refine_strength,
                    kernel_size=self.alpha_only_refine_kernel_size,
                )

            comp_pred = _composite_fg(fg_pred, bg_for_comp if bg_for_comp is not None else video, alpha_pred, self.fg_representation)

            out = {
                "alpha_pred": alpha_pred,
                "fg_pred": fg_pred,
                "comp_pred": comp_pred,
                "ref_tokens": ref_tokens_out,
                "coarse_alpha_pred": coarse_alpha_pred,
                "coarse_fg_pred": coarse_fg_pred,
                "input_fg_base_pred": input_fg_base,
                "decoder_fg_pred": decoder_fg_pred,
            }
            if uncertainty_pred is not None: out["uncertainty_pred"] = uncertainty_pred
            if spill_mask_pred is not None: out["spill_mask_pred"] = spill_mask_pred
            if quality_eval_pred is not None: out["quality_eval_pred"] = quality_eval_pred
            if alpha_only_delta_pred is not None:
                out["alpha_only_delta_pred"] = alpha_only_delta_pred
            if alpha_only_refine_mask is not None:
                out["alpha_only_refine_mask"] = alpha_only_refine_mask
            if self.native_refiner is not None:
                out.update({
                    "native_alpha_delta_pred": refiner_out["native_alpha_delta_pred"],
                    "native_fg_delta_pred": refiner_out["native_fg_delta_pred"],
                    "fg_residual_pred": (fg_pred - input_fg_base).clamp(-1.0, 1.0),
                    "despill_residual_pred": (fg_pred - input_fg_base).clamp(-1.0, 1.0),
                    "refine_mask": refine_mask.reshape(b, t, 1, h, w),
                })
                if "native_spill_delta_pred" in refiner_out:
                    out["native_spill_delta_pred"] = refiner_out["native_spill_delta_pred"]
                if "native_uncertainty_delta_pred" in refiner_out:
                    out["native_uncertainty_delta_pred"] = refiner_out["native_uncertainty_delta_pred"]
            return out


def build_v3_hybrid_video_matting_model(config: Dict[str, Any]) -> V3HybridVideoMattingModel:
    refine_head = config.get("refine_head") or {}
    if refine_head:
        head_type = str(refine_head.get("type", "native_window_linear_attention")).strip().lower()
        if head_type not in {"native_window_linear_attention", "native_window_linear", "linear"}:
            raise ValueError(f"Unsupported refine_head.type={head_type!r}")

    return V3HybridVideoMattingModel(
        patch_size=int(config.get("patch_size", 8)),
        embed_dims=tuple(config.get("embed_dims", [128, 256, 384, 512])),
        depths=tuple(config.get("depths", [2, 2, 6, 2])),
        num_heads=tuple(config.get("num_heads", [4, 8, 12, 16])),
        window_sizes=tuple(config.get("window_sizes", [8, 8, 4, 4])),
        memory_tokens=int(config.get("memory_tokens", 64)),
        temporal_window=int(config.get("temporal_window", 5)),
        drop_path_rate=float(config.get("drop_path_rate", 0.1)),
        use_green_priors=bool(config.get("use_green_priors", True)),
        use_coordinate_channels=bool(config.get("use_coordinate_channels", True)),
        use_unknown_band=bool(config.get("use_unknown_band", True)),
        guidance_presence_flag=bool(config.get("guidance_presence_flag", True)),
        green_prior_dropout=float(config.get("green_prior_dropout", 0.0)),
        coarse_alpha_drop_prob=float(config.get("coarse_alpha_drop_prob", 0.0)),
        predict_fg=bool(config.get("predict_fg", True)),
        predict_uncertainty=bool(config.get("predict_uncertainty", True)),
        predict_spill_mask=bool(config.get("predict_spill_mask", True)),
        predict_quality_eval=bool(config.get("predict_quality_eval", True)),
        fg_representation=str(config.get("fg_representation", "premul")),
        fg_prediction_mode=str(config.get("fg_prediction_mode", "decoder")),
        input_residual_despill_base=bool(config.get("input_residual_despill_base", False)),
        input_residual_despill_strength=float(config.get("input_residual_despill_strength", 1.0)),
        alpha_only_refine=bool(config.get("alpha_only_refine", False)),
        alpha_only_refine_strength=float(config.get("alpha_only_refine_strength", 0.75)),
        alpha_only_refine_kernel_size=int(config.get("alpha_only_refine_kernel_size", 5)),
        global_context_dim=int(config.get("global_context_dim", 256)),
        global_context_layers=int(config.get("global_context_layers", 4)),
        global_context_heads=int(config.get("global_context_heads", 8)),
        global_context_tokens=int(config.get("global_context_tokens", 64)),
        use_global_fg_guidance=bool(config.get("use_global_fg_guidance", False)),
        global_context_alpha_mode=str(config.get("global_context_alpha_mode", "seed")),
        use_reference_memory=bool(config.get("use_reference_memory", True)),
        reference_tokens_per_frame=int(config.get("reference_tokens_per_frame", 32)),
        num_reference_frames=int(config.get("num_reference_frames", 2)),
        reference_dropout=float(config.get("reference_dropout", 0.25)),
        use_native_refiner=bool(config.get("use_native_refiner", True)),
        native_refiner_blocks=int(refine_head.get("depth", config.get("native_refiner_blocks", 8))),
        native_refiner_hidden=int(refine_head.get("embed_dim", config.get("native_refiner_hidden", 64))),
        native_refiner_chunk_frames=int(config.get("native_refiner_chunk_frames", 2)),
        native_refiner_num_heads=int(refine_head.get("num_heads", config.get("native_refiner_num_heads", 4))),
        native_refiner_head_dim=int(refine_head.get("head_dim", config.get("native_refiner_head_dim", 16))),
        native_refiner_mlp_ratio=float(refine_head.get("mlp_ratio", config.get("native_refiner_mlp_ratio", 2.0))),
        native_refiner_window_sizes=refine_head.get("window_sizes", config.get("native_refiner_window_sizes")),
        native_refiner_shift_every_other_block=bool(refine_head.get("shift_every_other_block", config.get("native_refiner_shift_every_other_block", True))),
        native_refiner_attention_backend=str(refine_head.get("attention_backend", config.get("native_refiner_attention_backend", "linear"))),
        native_refiner_attention_types=refine_head.get("attention_types", config.get("native_refiner_attention_types")),
        native_refiner_linear_feature_map=str(refine_head.get("linear_feature_map", config.get("native_refiner_linear_feature_map", "elu_plus_one"))),
        native_refiner_linear_attention_eps=float(refine_head.get("linear_attention_eps", config.get("native_refiner_linear_attention_eps", 1.0e-6))),
        native_refiner_use_2d_rope=bool(refine_head.get("use_2d_rope", config.get("native_refiner_use_2d_rope", True))),
        native_refiner_use_checkpointing=refine_head.get("use_checkpointing", config.get("native_refiner_use_checkpointing")),
        max_alpha_delta=float(refine_head.get("max_alpha_delta", config.get("native_refiner_max_alpha_delta", config.get("max_alpha_delta", 0.03)))),
        max_fg_delta=float(refine_head.get("max_fg_delta", config.get("native_refiner_max_fg_delta", config.get("max_fg_delta", 0.02)))),
        max_spill_delta=float(refine_head.get("max_spill_delta", config.get("native_refiner_max_spill_delta", config.get("max_spill_delta", 0.15)))),
        gradient_checkpointing=bool(config.get("gradient_checkpointing", False)),
        decoder_out_dim=int(config.get("decoder_out_dim", 256)),
        fp8_cfg=config.get("fp8"),
    )

def build_memory_guided_video_matting_model(config: Dict[str, Any]) -> V3HybridVideoMattingModel:
    return build_v3_hybrid_video_matting_model(config)
