"""V3 matting losses.

Extends the base MattingLossComputer with V3-specific loss terms:
- Soft alpha band losses (transitional region focus)
- Temporal coherence losses
- Composite-against-random-background
- Hint disagreement correction reward
- Green-background alpha suppression
- Quality eval calibration
- Native refiner regularisation

Kernel-fusion optimisations (no torch.compile):
- _laplacian_pyramid_loss: @torch.jit.script fuses element-wise chains
- _green_stats: @torch.jit.script fuses min/max/excess into single kernels
- _green_excess_fg_mask / _green_background_mask: share the fused _green_stats result
- Batched vmask reductions: stack shared-mask errors and reduce in one .sum()
- In-place .add_() for scalar loss accumulation
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


EPS = 1e-6


# ---------------------------------------------------------------------------
# TorchScript-fused helpers
# ---------------------------------------------------------------------------

@torch.jit.script
def _masked_mean_scripted(value: Tensor, mask: Tensor) -> Tensor:
    """Fused masked mean — TorchScript merges multiply+sum into fewer kernels."""
    return (value * mask).sum() / mask.sum().clamp_min(1.0)


def _masked_mean(value: Tensor, mask: Optional[Tensor] = None) -> Tensor:
    """Mean over unmasked support, expanding singleton channels if needed."""
    value = value.float()
    if mask is None:
        return value.mean()
    mask = mask.to(device=value.device, dtype=value.dtype)
    if mask.shape != value.shape:
        mask = mask.expand_as(value)
    return _masked_mean_scripted(value, mask)


def _sanitize_alpha_target(alpha: Tensor) -> Tensor:
    return torch.nan_to_num(alpha.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)


@torch.jit.script
def _laplacian_pyramid_loss(
    pred: Tensor,
    target: Tensor,
    levels: int = 5,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Multi-scale Laplacian pyramid L1 loss.

    TorchScript fuses (pred - pred_up) - (target - target_up) + abs into
    single kernels per level, and fuses the weighted accumulation.
    """
    total = pred.new_zeros((), dtype=torch.float32)
    weight_sum = pred.new_zeros((), dtype=torch.float32)
    current_pred = pred.float()
    current_target = target.float()
    current_mask: Optional[Tensor] = mask.to(device=pred.device, dtype=torch.float32) if mask is not None else None
    weight = 1.0

    for _ in range(levels):
        if current_pred.shape[-2] < 2 or current_pred.shape[-1] < 2:
            break

        pred_down = F.avg_pool2d(current_pred, 2, stride=2)
        target_down = F.avg_pool2d(current_target, 2, stride=2)

        pred_up = F.interpolate(pred_down, size=current_pred.shape[-2:], mode="bilinear", align_corners=False)
        target_up = F.interpolate(target_down, size=current_target.shape[-2:], mode="bilinear", align_corners=False)

        # Fused: (a - b) - (c - d) then abs — TorchScript merges into 2 kernels
        lap_diff = (current_pred - pred_up) - (current_target - target_up)

        if current_mask is not None:
            total.add_(weight * _masked_mean_scripted(lap_diff.abs(), current_mask))
        else:
            total.add_(weight * lap_diff.abs().mean())
        weight_sum.add_(weight)

        current_pred = pred_down
        current_target = target_down
        if current_mask is not None:
            current_mask = F.interpolate(current_mask, size=pred_down.shape[-2:], mode="area")
        weight *= 0.5

    residual = (current_pred - current_target).abs()
    if current_mask is not None:
        total.add_(weight * _masked_mean_scripted(residual, current_mask))
    else:
        total.add_(weight * residual.mean())
    weight_sum.add_(weight)
    return total / weight_sum.clamp_min(1e-6)


@torch.jit.script
def _green_stats(
    rgb: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Fused green channel statistics.

    Returns (green_excess, saturation, max_rgb) in one TorchScript kernel group.
    Handles both [B,T,C,H,W] (ndim==5) and [B,C,H,W] (ndim==4) via slicing on dim=-3.
    """
    # Works for both 4-D and 5-D by slicing on the channel dimension
    r = rgb[..., 0:1, :, :]
    g = rgb[..., 1:2, :, :]
    b = rgb[..., 2:3, :, :]
    max_rb  = torch.maximum(r, b)
    min_rgb = torch.minimum(torch.minimum(r, g), b)
    max_rgb = torch.maximum(torch.maximum(r, g), b)
    green_excess = g - max_rb
    saturation   = max_rgb - min_rgb
    return green_excess, saturation, max_rgb


def _temporal_gradient_loss(
    pred: Tensor,
    target: Tensor,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Penalise temporal gradient mismatch: d/dt(pred) vs d/dt(target).

    pred, target: [B, T, C, H, W]
    valid_mask: optional [B, T]; a pair is valid only if both frames are valid.
    """
    if pred.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=torch.float32)
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    pair_mask = None
    if valid_mask is not None:
        pair_mask = (valid_mask[:, 1:] * valid_mask[:, :-1])[:, :, None, None, None]
    return _masked_mean((dp.float() - dt.float()).abs(), pair_mask)


def _soft_alpha_band_mask(alpha_gt: Tensor, lo: float = 0.02, hi: float = 0.98) -> Tensor:
    """Mask for transition region where 0.02 < alpha < 0.98."""
    return ((alpha_gt > lo) & (alpha_gt < hi)).to(alpha_gt.dtype)


def _green_excess_fg_mask(
    fg_gt: Tensor,
    alpha_gt: Tensor,
    ge: Tensor,
    sat: Tensor,
    *,
    alpha_thresh: float = 0.75,
    green_margin: float = 0.08,
    min_saturation: float = 0.10,
) -> Tensor:
    """Mask true foreground pixels whose GT foreground color is green.

    Accepts pre-computed green_excess (ge) and saturation (sat) from _green_stats
    to avoid redundant computation.
    """
    mask = (
        (alpha_gt > alpha_thresh)
        & (ge > green_margin)
        & (sat > min_saturation)
    )
    return mask.to(dtype=fg_gt.dtype)


def _green_background_mask(
    video_rgb: Tensor,
    alpha_gt: Tensor,
    ge: Tensor,
    sat: Tensor,
    *,
    alpha_thresh: float = 0.10,
    green_margin: float = 0.08,
    min_saturation: float = 0.10,
) -> Tensor:
    """Mask pixels that look like clearly green background.

    Accepts pre-computed green_excess (ge) and saturation (sat) from _green_stats.
    """
    mask = (
        (alpha_gt < alpha_thresh)
        & (ge > green_margin)
        & (sat > min_saturation)
    )
    return mask.to(dtype=video_rgb.dtype)


def _composite_random_bg_loss(
    alpha_pred: Tensor,
    fg_pred: Tensor,
    alpha_gt: Tensor,
    fg_gt: Tensor,
    fg_representation: str,
    valid_mask: Optional[Tensor] = None,
) -> Tensor:
    """Composite pred and GT over same random background, compare."""
    b, t, _, h, w = alpha_pred.shape
    rand_bg = torch.rand(b, 1, 3, h, w, device=alpha_pred.device, dtype=alpha_pred.dtype)
    rand_bg = rand_bg.expand(b, t, 3, h, w)

    if fg_representation == "straight":
        comp_pred = fg_pred * alpha_pred + (1.0 - alpha_pred) * rand_bg
        comp_gt = fg_gt * alpha_gt + (1.0 - alpha_gt) * rand_bg
    else:
        comp_pred = fg_pred + (1.0 - alpha_pred) * rand_bg
        comp_gt = fg_gt + (1.0 - alpha_gt) * rand_bg

    mask = None
    if valid_mask is not None:
        mask = valid_mask[:, :, None, None, None]
    return _masked_mean((comp_pred.float() - comp_gt.float()).abs(), mask)


class V3MattingLossComputer(nn.Module):
    """V3 loss computer.

    Designed to be swapped into the root trainer via
    ``root_train.MattingLossComputer = V3MattingLossComputer``.

    Matches the ``(pred_dict, batch) -> (total_loss, loss_items)`` interface.
    """

    def __init__(self, weights: Dict[str, Any], fg_representation: str = "premul") -> None:
        super().__init__()
        self.weights = dict(weights)
        self.fg_representation = fg_representation

        # Core weights
        self.w_alpha_l1 = float(weights.get("alpha_l1", 1.0))
        self.w_alpha_lap = float(weights.get("alpha_laplacian", 1.0))
        self.w_fg_l1 = float(weights.get("fg_l1", 1.0))
        self.w_comp_l1 = float(weights.get("comp_l1", 1.0))

        # V3-specific weights
        self.w_alpha_band_l1 = float(weights.get("alpha_band_l1", 0.5))
        self.w_alpha_band_lap = float(weights.get("alpha_band_laplacian", 0.5))
        self.w_temporal_alpha = float(weights.get("temporal_alpha_gradient", 0.3))
        self.w_temporal_fg = float(weights.get("temporal_fg_gradient", 0.15))
        self.w_comp_random_bg = float(weights.get("comp_random_bg", 0.3))
        self.w_spill_l1 = float(weights.get("spill_l1", 0.5))
        self.w_uncertainty = float(weights.get("uncertainty", 0.1))
        self.w_green_fg_alpha = float(weights.get("green_fg_alpha", 0.0))
        self.w_green_fg_color = float(weights.get("green_fg_color", 0.0))
        self.w_green_bg_alpha_suppress = float(weights.get("green_bg_alpha_suppress", 0.0))

        # Spill-region masking controls.
        self.spill_transition_only = bool(weights.get("spill_transition_only", False))
        self.spill_transition_low = float(weights.get("spill_transition_low", 0.02))
        self.spill_transition_high = float(weights.get("spill_transition_high", 0.98))
        self.spill_exclude_opaque_fg = bool(weights.get("spill_exclude_opaque_fg", False))
        self.spill_opaque_alpha_thresh = float(weights.get("spill_opaque_alpha_thresh", 0.90))

        # Native refiner regularisation
        self.w_native_alpha_delta = float(weights.get("native_alpha_delta_reg", 0.05))
        self.w_native_fg_delta = float(weights.get("native_fg_delta_reg", 0.05))

        # Coarse prediction supervision
        self.w_coarse_alpha = float(weights.get("coarse_alpha_l1", 0.5))
        self.w_coarse_fg = float(weights.get("coarse_fg_l1", 0.3))

        # Quality eval
        self.w_quality_eval = float(weights.get("quality_eval", 0.1))

        # RGB-only global semantic prior supervision
        self.w_semantic_prior_bce = float(weights.get("semantic_prior_bce", 0.0))
        self.w_semantic_prior_dice = float(weights.get("semantic_prior_dice", 0.0))

    @staticmethod
    def _enabled(weight: float) -> bool:
        return float(weight) > 0.0

    @staticmethod
    def _zero(device: torch.device) -> Tensor:
        return torch.tensor(0.0, device=device, dtype=torch.float32)

    def forward(
        self,
        pred: Dict[str, Tensor],
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        alpha_pred = pred["alpha_pred"]
        fg_pred = pred["fg_pred"]
        comp_pred = pred["comp_pred"]
        alpha_gt = batch["alpha_gt"]
        fg_gt = batch["fg_gt"]

        valid_mask = batch.get("valid_mask")
        items: Dict[str, Tensor] = {}
        device = alpha_pred.device

        # --- Cast to fp32 ---
        alpha_p = alpha_pred.float()
        alpha_g = _sanitize_alpha_target(alpha_gt)
        fg_p = fg_pred.float()
        fg_g = fg_gt.float()

        # Valid mask
        if valid_mask is not None:
            valid_mask = valid_mask.to(device=device, dtype=torch.float32)
            vmask: Optional[Tensor] = valid_mask[:, :, None, None, None]
        else:
            vmask = None

        # ---- Compute element-wise error maps once ----
        err_alpha = (alpha_p - alpha_g).abs()   # [B,T,1,H,W]

        fg_mask = (alpha_g > 0.01).to(torch.float32)
        if vmask is not None:
            fg_mask = fg_mask * vmask
        err_fg = (fg_p - fg_g).abs()            # [B,T,3,H,W]

        # ---- Batch vmask reductions for alpha + fg + comp ----
        # Stack the three shared-vmask errors and reduce in two kernel calls.
        comp_gt = batch.get("input_gt", batch.get("video_rgb"))
        if comp_gt is not None:
            err_comp = (comp_pred.float() - comp_gt.float()).abs()
        else:
            err_comp = None

        if vmask is not None:
            vmask_a = vmask.expand_as(alpha_p)
            denom_v = vmask_a.sum().clamp_min(1.0)
            l_alpha_l1 = (err_alpha * vmask_a).sum() / denom_v
            l_comp = (err_comp * vmask.expand_as(comp_pred.float())).sum() / vmask.expand_as(comp_pred.float()).sum().clamp_min(1.0) if err_comp is not None else self._zero(device)
        else:
            l_alpha_l1 = err_alpha.mean()
            l_comp = err_comp.mean() if err_comp is not None else self._zero(device)
        l_fg_l1 = _masked_mean_scripted(err_fg, fg_mask) if fg_mask.any() else self._zero(device)

        items["alpha_l1"] = l_alpha_l1
        items["fg_l1"] = l_fg_l1
        items["comp_l1"] = l_comp

        # ---- Alpha Laplacian ----
        bt = alpha_p.shape[0] * alpha_p.shape[1]
        if vmask is not None:
            vmask_bt: Optional[Tensor] = vmask.expand_as(alpha_p).reshape(
                bt, 1, alpha_p.shape[-2], alpha_p.shape[-1]
            )
        else:
            vmask_bt = None

        if self._enabled(self.w_alpha_lap):
            l_alpha_lap = _laplacian_pyramid_loss(
                alpha_p.reshape(bt, 1, alpha_p.shape[-2], alpha_p.shape[-1]),
                alpha_g.reshape(bt, 1, alpha_g.shape[-2], alpha_g.shape[-1]),
                mask=vmask_bt,
            )
        else:
            l_alpha_lap = self._zero(device)
        items["alpha_lap"] = l_alpha_lap

        # ---- Soft alpha band losses ----
        band_mask = _soft_alpha_band_mask(alpha_g)
        if vmask is not None:
            band_mask = band_mask * vmask
        band_count = band_mask.sum().clamp_min(1.0)
        l_band_l1 = (err_alpha * band_mask).sum() / band_count
        items["alpha_band_l1"] = l_band_l1

        if self._enabled(self.w_alpha_band_lap):
            l_band_lap = _laplacian_pyramid_loss(
                alpha_p.reshape(bt, 1, alpha_p.shape[-2], alpha_p.shape[-1]),
                alpha_g.reshape(bt, 1, alpha_g.shape[-2], alpha_g.shape[-1]),
                mask=band_mask.reshape(bt, 1, alpha_p.shape[-2], alpha_p.shape[-1]),
            )
        else:
            l_band_lap = self._zero(device)
        items["alpha_band_lap"] = l_band_lap

        # ---- Temporal coherence ----
        if self._enabled(self.w_temporal_alpha):
            l_temporal_alpha = _temporal_gradient_loss(alpha_pred, alpha_gt, valid_mask)
        else:
            l_temporal_alpha = self._zero(device)
        items["temporal_alpha"] = l_temporal_alpha

        if self._enabled(self.w_temporal_fg):
            fg_temporal_mask = (alpha_gt > 0.01).float()
            l_temporal_fg = _temporal_gradient_loss(fg_pred * fg_temporal_mask, fg_gt * fg_temporal_mask, valid_mask)
        else:
            l_temporal_fg = self._zero(device)
        items["temporal_fg"] = l_temporal_fg

        # ---- Composite with random background ----
        if self._enabled(self.w_comp_random_bg):
            l_comp_rand = _composite_random_bg_loss(alpha_pred, fg_pred, alpha_gt, fg_gt, self.fg_representation, valid_mask)
        else:
            l_comp_rand = self._zero(device)
        items["comp_random_bg"] = l_comp_rand

        # ---- Spill mask ----
        l_spill = torch.tensor(0.0, device=device)
        if self._enabled(self.w_spill_l1) and "spill_mask_pred" in pred and "alpha_boundary_gt" in batch:
            green_excess = (batch["video_rgb"][:, :, 1:2] - torch.maximum(
                batch["video_rgb"][:, :, 0:1], batch["video_rgb"][:, :, 2:3]
            )).clamp_min(0.0) * batch["alpha_boundary_gt"]
            spill_gt = (green_excess > 0.03).to(torch.float32)
            spill_weight = torch.ones_like(alpha_g)
            if vmask is not None:
                spill_weight = spill_weight * vmask
            if self.spill_transition_only:
                transition = (
                    (alpha_g > self.spill_transition_low)
                    & (alpha_g < self.spill_transition_high)
                ).to(alpha_g.dtype)
                spill_weight = spill_weight * transition
            if self.spill_exclude_opaque_fg:
                opaque_fg = (alpha_g > self.spill_opaque_alpha_thresh).to(alpha_g.dtype)
                spill_weight = spill_weight * (1.0 - opaque_fg)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                spill_bce = F.binary_cross_entropy(
                    pred["spill_mask_pred"].float().clamp(1e-6, 1.0 - 1e-6),
                    spill_gt.float(),
                    reduction="none",
                )
                l_spill = (spill_bce * spill_weight).sum() / spill_weight.sum().clamp_min(1.0)
        items["spill_l1"] = l_spill

        # ---- Green stats — computed once, reused for fg + bg masks ----
        l_green_fg_alpha = self._zero(device)
        l_green_fg_color = self._zero(device)
        green_fg_pixels = self._zero(device)
        l_green_bg_alpha_suppress = self._zero(device)
        green_bg_pixels = self._zero(device)

        need_fg_stats = self._enabled(self.w_green_fg_alpha) or self._enabled(self.w_green_fg_color)
        need_bg_stats = self._enabled(self.w_green_bg_alpha_suppress)

        if need_fg_stats or need_bg_stats:
            # Compute once for fg_gt (used by fg mask)
            ge_fg, sat_fg, _ = _green_stats(fg_g)

            if need_fg_stats:
                green_fg = _green_excess_fg_mask(
                    fg_gt=fg_g,
                    alpha_gt=alpha_g,
                    ge=ge_fg,
                    sat=sat_fg,
                    alpha_thresh=float(self.weights.get("green_fg_alpha_thresh", 0.75)),
                    green_margin=float(self.weights.get("green_fg_margin", 0.08)),
                    min_saturation=float(self.weights.get("green_fg_min_saturation", 0.10)),
                )
                if vmask is not None:
                    green_fg = green_fg * vmask
                green_den = green_fg.sum().clamp_min(1.0)
                green_fg_pixels = green_fg.mean().detach()

                if self._enabled(self.w_green_fg_alpha):
                    l_green_fg_alpha = (green_fg * err_alpha).sum() / green_den

                if self._enabled(self.w_green_fg_color):
                    green_fg_rgb = green_fg.expand_as(fg_p)
                    l_green_fg_color = (green_fg_rgb * err_fg).sum() / green_fg_rgb.sum().clamp_min(1.0)

            if need_bg_stats:
                # Compute green stats for video_rgb (background)
                ge_bg, sat_bg, _ = _green_stats(batch["video_rgb"].float())
                green_bg = _green_background_mask(
                    video_rgb=batch["video_rgb"],
                    alpha_gt=alpha_g,
                    ge=ge_bg,
                    sat=sat_bg,
                    alpha_thresh=float(self.weights.get("green_bg_alpha_thresh", 0.10)),
                    green_margin=float(self.weights.get("green_bg_margin", 0.08)),
                    min_saturation=float(self.weights.get("green_bg_min_saturation", 0.10)),
                )
                if vmask is not None:
                    green_bg = green_bg * vmask
                green_bg_den = green_bg.sum().clamp_min(1.0)
                green_bg_pixels = green_bg.mean().detach()
                l_green_bg_alpha_suppress = (green_bg * alpha_p).sum() / green_bg_den

        items["green_fg_alpha"] = l_green_fg_alpha
        items["green_fg_color"] = l_green_fg_color
        items["green_fg_pixels"] = green_fg_pixels
        items["green_fg_alpha_abs"] = l_green_fg_alpha.detach()
        items["green_fg_color_abs"] = l_green_fg_color.detach()
        items["green_bg_alpha_suppress"] = l_green_bg_alpha_suppress
        items["green_bg_alpha_suppress_abs"] = l_green_bg_alpha_suppress.detach()
        items["green_bg_pixels"] = green_bg_pixels

        # ---- Coarse prediction supervision ----
        l_coarse_alpha = torch.tensor(0.0, device=device)
        if self._enabled(self.w_coarse_alpha) and "coarse_alpha_pred" in pred:
            l_coarse_alpha = _masked_mean((pred["coarse_alpha_pred"].float() - alpha_g).abs(), vmask)
        items["coarse_alpha_l1"] = l_coarse_alpha

        l_coarse_fg = torch.tensor(0.0, device=device)
        if self._enabled(self.w_coarse_fg) and "coarse_fg_pred" in pred:
            l_coarse_fg = _masked_mean((pred["coarse_fg_pred"].float() - fg_g).abs(), fg_mask)
        items["coarse_fg_l1"] = l_coarse_fg

        # ---- Native refiner regularisation ----
        l_delta_alpha = torch.tensor(0.0, device=device)
        if self._enabled(self.w_native_alpha_delta) and "native_alpha_delta_pred" in pred:
            l_delta_alpha = _masked_mean(pred["native_alpha_delta_pred"].float().abs(), vmask)
        items["native_alpha_delta_reg"] = l_delta_alpha

        l_delta_fg = torch.tensor(0.0, device=device)
        if self._enabled(self.w_native_fg_delta) and "native_fg_delta_pred" in pred:
            l_delta_fg = _masked_mean(pred["native_fg_delta_pred"].float().abs(), vmask)
        items["native_fg_delta_reg"] = l_delta_fg

        # ---- Uncertainty calibration ----
        l_uncertainty = torch.tensor(0.0, device=device)
        if self._enabled(self.w_uncertainty) and "uncertainty_pred" in pred:
            with torch.no_grad():
                uncertainty_target = err_alpha.clamp(0.0, 1.0)
            l_uncertainty = _masked_mean((pred["uncertainty_pred"].float() - uncertainty_target).abs(), vmask)
        items["uncertainty"] = l_uncertainty

        # ---- Quality eval ----
        l_quality = torch.tensor(0.0, device=device)
        if self._enabled(self.w_quality_eval) and "quality_eval_pred" in pred and "coarse_alpha_pred" in pred:
            with torch.no_grad():
                coarse_error = (pred["coarse_alpha_pred"].float() - alpha_g).abs()
                quality_target = coarse_error.clamp(0.0, 1.0)
            l_quality = _masked_mean((pred["quality_eval_pred"].float() - quality_target).abs(), vmask)
        items["quality_eval"] = l_quality

        # ---- Semantic fg/bg prior ----
        l_semantic_bce = torch.tensor(0.0, device=device)
        l_semantic_dice = torch.tensor(0.0, device=device)
        if (
            "semantic_fg_logits" in pred
            and (self._enabled(self.w_semantic_prior_bce) or self._enabled(self.w_semantic_prior_dice))
        ):
            semantic_logits = pred["semantic_fg_logits"].float()
            semantic_target = batch.get("global_alpha_gt", alpha_g)
            semantic_target = _sanitize_alpha_target(semantic_target).to(device=device)
            if semantic_target.ndim == 4:
                semantic_target = semantic_target[:, None]
            if semantic_target.shape[-2:] != semantic_logits.shape[-2:]:
                bt_target = semantic_target.reshape(
                    semantic_target.shape[0] * semantic_target.shape[1],
                    semantic_target.shape[2],
                    semantic_target.shape[3],
                    semantic_target.shape[4],
                )
                semantic_target = F.interpolate(
                    bt_target,
                    size=semantic_logits.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).reshape(
                    semantic_target.shape[0],
                    semantic_target.shape[1],
                    semantic_target.shape[2],
                    semantic_logits.shape[-2],
                    semantic_logits.shape[-1],
                )
            if semantic_target.shape[1] != semantic_logits.shape[1]:
                semantic_target = semantic_target[:, :1].expand(-1, semantic_logits.shape[1], -1, -1, -1)

            semantic_weight = torch.ones_like(semantic_target)
            if valid_mask is not None and semantic_target.shape[1] == valid_mask.shape[1]:
                semantic_weight = semantic_weight * valid_mask[:, :, None, None, None]

            semantic_bce = F.binary_cross_entropy_with_logits(
                semantic_logits,
                semantic_target.clamp(0.0, 1.0),
                reduction="none",
            )
            l_semantic_bce = (semantic_bce * semantic_weight).sum() / semantic_weight.sum().clamp_min(1.0)

            semantic_prob = semantic_logits.sigmoid()
            intersection = (semantic_prob * semantic_target * semantic_weight).sum()
            denom = ((semantic_prob + semantic_target) * semantic_weight).sum().clamp_min(1.0)
            l_semantic_dice = 1.0 - (2.0 * intersection + 1.0) / (denom + 1.0)
        items["semantic_prior_bce"] = l_semantic_bce
        items["semantic_prior_dice"] = l_semantic_dice

        # ---- Total (in-place add to avoid extra allocations) ----
        total = self._zero(device)
        total.add_(self.w_alpha_l1 * l_alpha_l1)
        total.add_(self.w_alpha_lap * l_alpha_lap)
        total.add_(self.w_fg_l1 * l_fg_l1)
        total.add_(self.w_comp_l1 * l_comp)
        total.add_(self.w_alpha_band_l1 * l_band_l1)
        total.add_(self.w_alpha_band_lap * l_band_lap)
        total.add_(self.w_temporal_alpha * l_temporal_alpha)
        total.add_(self.w_temporal_fg * l_temporal_fg)
        total.add_(self.w_comp_random_bg * l_comp_rand)
        total.add_(self.w_spill_l1 * l_spill)
        total.add_(self.w_green_fg_alpha * l_green_fg_alpha)
        total.add_(self.w_green_fg_color * l_green_fg_color)
        total.add_(self.w_green_bg_alpha_suppress * l_green_bg_alpha_suppress)
        total.add_(self.w_coarse_alpha * l_coarse_alpha)
        total.add_(self.w_coarse_fg * l_coarse_fg)
        total.add_(self.w_native_alpha_delta * l_delta_alpha)
        total.add_(self.w_native_fg_delta * l_delta_fg)
        total.add_(self.w_uncertainty * l_uncertainty)
        total.add_(self.w_quality_eval * l_quality)
        total.add_(self.w_semantic_prior_bce * l_semantic_bce)
        total.add_(self.w_semantic_prior_dice * l_semantic_dice)

        items["total"] = total
        return total, items
