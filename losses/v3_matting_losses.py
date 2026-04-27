"""V3 matting losses.

Extends the base MattingLossComputer with V3-specific loss terms:
- Soft alpha band losses (transitional region focus)
- Temporal coherence losses
- Composite-against-random-background
- Hint disagreement correction reward
- Green-background alpha suppression
- Quality eval calibration
- Native refiner regularisation
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


EPS = 1e-6


def _laplacian_pyramid_loss(
    pred: Tensor,
    target: Tensor,
    levels: int = 5,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Multi-scale Laplacian pyramid L1 loss."""
    total = torch.tensor(0.0, device=pred.device, dtype=torch.float32)
    current_pred = pred.float()
    current_target = target.float()
    current_mask = mask.float() if mask is not None else None
    weight = 1.0
    for _ in range(levels):
        # Cannot downsample further once a spatial dimension reaches 1.
        if current_pred.shape[-2] < 2 or current_pred.shape[-1] < 2:
            break
        # Downsample
        pred_down = F.avg_pool2d(current_pred, 2, stride=2)
        target_down = F.avg_pool2d(current_target, 2, stride=2)
        # Upsample back
        pred_up = F.interpolate(pred_down, size=current_pred.shape[-2:], mode="bilinear", align_corners=False)
        target_up = F.interpolate(target_down, size=current_target.shape[-2:], mode="bilinear", align_corners=False)
        # Laplacian = current - upsampled(downsampled)
        lap_pred = current_pred - pred_up
        lap_target = current_target - target_up
        if current_mask is None:
            total = total + weight * (lap_pred - lap_target).abs().mean()
        else:
            diff = (lap_pred - lap_target).abs() * current_mask
            total = total + weight * (diff.sum() / current_mask.sum().clamp_min(1.0))
            current_mask = F.interpolate(current_mask, size=pred_down.shape[-2:], mode="nearest")
        current_pred = pred_down
        current_target = target_down
        weight *= 2.0
    # Coarsest level L1
    if current_mask is None:
        total = total + weight * (current_pred - current_target).abs().mean()
    else:
        coarse_diff = (current_pred - current_target).abs() * current_mask
        total = total + weight * (coarse_diff.sum() / current_mask.sum().clamp_min(1.0))
    return total


def _temporal_gradient_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Penalise temporal gradient mismatch: d/dt(pred) vs d/dt(target).

    pred, target: [B, T, C, H, W]
    """
    if pred.shape[1] < 2:
        return torch.tensor(0.0, device=pred.device, dtype=torch.float32)
    dp = pred[:, 1:] - pred[:, :-1]
    dt = target[:, 1:] - target[:, :-1]
    return (dp.float() - dt.float()).abs().mean()


def _soft_alpha_band_mask(alpha_gt: Tensor, lo: float = 0.02, hi: float = 0.98) -> Tensor:
    """Mask for transition region where 0.02 < alpha < 0.98."""
    return ((alpha_gt > lo) & (alpha_gt < hi)).to(alpha_gt.dtype)


def _green_excess_fg_mask(
    fg_gt: Tensor,
    alpha_gt: Tensor,
    *,
    alpha_thresh: float = 0.75,
    green_margin: float = 0.08,
    min_saturation: float = 0.10,
) -> Tensor:
    """Mask true foreground pixels whose GT foreground color is green.

    Supports either [B,T,C,H,W] or [B,C,H,W].
    Returns a float mask with one channel.
    """
    if fg_gt.ndim == 5:
        r = fg_gt[:, :, 0:1]
        g = fg_gt[:, :, 1:2]
        b = fg_gt[:, :, 2:3]
    elif fg_gt.ndim == 4:
        r = fg_gt[:, 0:1]
        g = fg_gt[:, 1:2]
        b = fg_gt[:, 2:3]
    else:
        raise ValueError(f"Expected FG tensor with ndim 4 or 5, got {fg_gt.ndim}")

    max_rb = torch.maximum(r, b)
    min_rgb = torch.minimum(torch.minimum(r, g), b)
    max_rgb = torch.maximum(torch.maximum(r, g), b)

    green_excess = g - max_rb
    saturation = max_rgb - min_rgb

    mask = (
        (alpha_gt > alpha_thresh)
        & (green_excess > green_margin)
        & (saturation > min_saturation)
    )
    return mask.to(dtype=fg_gt.dtype)


def _green_background_mask(
    video_rgb: Tensor,
    alpha_gt: Tensor,
    *,
    alpha_thresh: float = 0.10,
    green_margin: float = 0.08,
    min_saturation: float = 0.10,
) -> Tensor:
    """Mask pixels that look like clearly green background.

    Supports either [B,T,C,H,W] or [B,C,H,W].
    Returns a float mask with one channel.
    """
    if video_rgb.ndim == 5:
        r = video_rgb[:, :, 0:1]
        g = video_rgb[:, :, 1:2]
        b = video_rgb[:, :, 2:3]
    elif video_rgb.ndim == 4:
        r = video_rgb[:, 0:1]
        g = video_rgb[:, 1:2]
        b = video_rgb[:, 2:3]
    else:
        raise ValueError(f"Expected RGB tensor with ndim 4 or 5, got {video_rgb.ndim}")

    max_rb = torch.maximum(r, b)
    min_rgb = torch.minimum(torch.minimum(r, g), b)
    max_rgb = torch.maximum(torch.maximum(r, g), b)

    green_excess = g - max_rb
    saturation = max_rgb - min_rgb

    mask = (
        (alpha_gt < alpha_thresh)
        & (green_excess > green_margin)
        & (saturation > min_saturation)
    )
    return mask.to(dtype=video_rgb.dtype)


def _composite_random_bg_loss(
    alpha_pred: Tensor,
    fg_pred: Tensor,
    alpha_gt: Tensor,
    fg_gt: Tensor,
    fg_representation: str,
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

    return (comp_pred.float() - comp_gt.float()).abs().mean()


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

        # --- Core losses (fp32 for precision) ---
        alpha_p = alpha_pred.float()
        alpha_g = alpha_gt.float()
        fg_p = fg_pred.float()
        fg_g = fg_gt.float()

        # Valid mask weighting
        if valid_mask is not None:
            vmask = valid_mask[:, :, None, None, None].to(dtype=torch.float32)
        else:
            vmask = 1.0

        # Alpha L1
        l_alpha_l1 = ((alpha_p - alpha_g).abs() * vmask).mean()
        items["alpha_l1"] = l_alpha_l1

        # Alpha Laplacian
        bt = alpha_p.shape[0] * alpha_p.shape[1]
        if valid_mask is not None:
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

        # FG L1 (only where alpha > 0)
        fg_mask = (alpha_g > 0.01).to(torch.float32) * vmask
        l_fg_l1 = ((fg_p - fg_g).abs() * fg_mask).sum() / fg_mask.sum().clamp_min(1.0)
        items["fg_l1"] = l_fg_l1

        # Composite L1
        comp_gt = batch.get("input_gt", batch.get("video_rgb"))
        if comp_gt is not None:
            l_comp = ((comp_pred.float() - comp_gt.float()).abs() * vmask).mean()
        else:
            l_comp = torch.tensor(0.0, device=device)
        items["comp_l1"] = l_comp

        # --- V3-specific losses ---

        # Soft alpha band losses
        band_mask = _soft_alpha_band_mask(alpha_g) * vmask
        band_count = band_mask.sum().clamp_min(1.0)
        l_band_l1 = ((alpha_p - alpha_g).abs() * band_mask).sum() / band_count
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

        # Temporal coherence
        if self._enabled(self.w_temporal_alpha):
            l_temporal_alpha = _temporal_gradient_loss(alpha_pred, alpha_gt)
        else:
            l_temporal_alpha = self._zero(device)
        items["temporal_alpha"] = l_temporal_alpha

        if self._enabled(self.w_temporal_fg):
            l_temporal_fg = _temporal_gradient_loss(fg_pred * (alpha_gt > 0.01).float(), fg_gt * (alpha_gt > 0.01).float())
        else:
            l_temporal_fg = self._zero(device)
        items["temporal_fg"] = l_temporal_fg

        # Composite with random background
        if self._enabled(self.w_comp_random_bg):
            l_comp_rand = _composite_random_bg_loss(alpha_pred, fg_pred, alpha_gt, fg_gt, self.fg_representation)
        else:
            l_comp_rand = self._zero(device)
        items["comp_random_bg"] = l_comp_rand

        # Spill mask
        l_spill = torch.tensor(0.0, device=device)
        if self._enabled(self.w_spill_l1) and "spill_mask_pred" in pred and "alpha_boundary_gt" in batch:
            # Heuristic spill target: green excess in boundary region
            green_excess = (batch["video_rgb"][:, :, 1:2] - torch.maximum(
                batch["video_rgb"][:, :, 0:1], batch["video_rgb"][:, :, 2:3]
            )).clamp_min(0.0) * batch["alpha_boundary_gt"]
            spill_gt = (green_excess > 0.03).to(torch.float32)
            spill_weight = torch.ones_like(alpha_g)
            if self.spill_transition_only:
                transition = (
                    (alpha_g > self.spill_transition_low)
                    & (alpha_g < self.spill_transition_high)
                ).to(alpha_g.dtype)
                spill_weight = spill_weight * transition
            if self.spill_exclude_opaque_fg:
                opaque_fg = (alpha_g > self.spill_opaque_alpha_thresh).to(alpha_g.dtype)
                spill_weight = spill_weight * (1.0 - opaque_fg)
            # Use a non-autocast context for BCE because it's considered unsafe under FP8 autocast
            with torch.amp.autocast(device_type="cuda", enabled=False):
                spill_bce = F.binary_cross_entropy(
                    pred["spill_mask_pred"].float().clamp(1e-6, 1.0 - 1e-6),
                    spill_gt.float(),
                    reduction="none",
                )
                l_spill = (spill_bce * spill_weight).sum() / spill_weight.sum().clamp_min(1.0)
        items["spill_l1"] = l_spill

        # Green foreground protection losses and diagnostics.
        l_green_fg_alpha = self._zero(device)
        l_green_fg_color = self._zero(device)
        green_fg_pixels = self._zero(device)
        if self._enabled(self.w_green_fg_alpha) or self._enabled(self.w_green_fg_color):
            green_fg = _green_excess_fg_mask(
                fg_gt=fg_g,
                alpha_gt=alpha_g,
                alpha_thresh=float(self.weights.get("green_fg_alpha_thresh", 0.75)),
                green_margin=float(self.weights.get("green_fg_margin", 0.08)),
                min_saturation=float(self.weights.get("green_fg_min_saturation", 0.10)),
            )
            green_den = green_fg.sum().clamp_min(1.0)
            green_fg_pixels = green_fg.mean().detach()

            if self._enabled(self.w_green_fg_alpha):
                l_green_fg_alpha = (green_fg * (alpha_p - alpha_g).abs()).sum() / green_den

            if self._enabled(self.w_green_fg_color):
                green_fg_rgb = green_fg.expand_as(fg_p)
                l_green_fg_color = (
                    green_fg_rgb * (fg_p - fg_g).abs()
                ).sum() / green_fg_rgb.sum().clamp_min(1.0)

        items["green_fg_alpha"] = l_green_fg_alpha
        items["green_fg_color"] = l_green_fg_color
        items["green_fg_pixels"] = green_fg_pixels
        items["green_fg_alpha_abs"] = l_green_fg_alpha.detach()
        items["green_fg_color_abs"] = l_green_fg_color.detach()

        # Green background alpha suppression.
        l_green_bg_alpha_suppress = self._zero(device)
        green_bg_pixels = self._zero(device)
        if self._enabled(self.w_green_bg_alpha_suppress):
            green_bg = _green_background_mask(
                video_rgb=batch["video_rgb"],
                alpha_gt=alpha_g,
                alpha_thresh=float(self.weights.get("green_bg_alpha_thresh", 0.10)),
                green_margin=float(self.weights.get("green_bg_margin", 0.08)),
                min_saturation=float(self.weights.get("green_bg_min_saturation", 0.10)),
            )
            green_bg_den = green_bg.sum().clamp_min(1.0)
            green_bg_pixels = green_bg.mean().detach()
            l_green_bg_alpha_suppress = (green_bg * alpha_p).sum() / green_bg_den

        items["green_bg_alpha_suppress"] = l_green_bg_alpha_suppress
        items["green_bg_alpha_suppress_abs"] = l_green_bg_alpha_suppress.detach()
        items["green_bg_pixels"] = green_bg_pixels

        # Coarse prediction supervision
        l_coarse_alpha = torch.tensor(0.0, device=device)
        if self._enabled(self.w_coarse_alpha) and "coarse_alpha_pred" in pred:
            l_coarse_alpha = ((pred["coarse_alpha_pred"].float() - alpha_g).abs() * vmask).mean()
        items["coarse_alpha_l1"] = l_coarse_alpha

        l_coarse_fg = torch.tensor(0.0, device=device)
        if self._enabled(self.w_coarse_fg) and "coarse_fg_pred" in pred:
            l_coarse_fg = ((pred["coarse_fg_pred"].float() - fg_g).abs() * fg_mask).sum() / fg_mask.sum().clamp_min(1.0)
        items["coarse_fg_l1"] = l_coarse_fg

        # Native refiner regularisation (penalise large deltas)
        l_delta_alpha = torch.tensor(0.0, device=device)
        if self._enabled(self.w_native_alpha_delta) and "native_alpha_delta_pred" in pred:
            l_delta_alpha = pred["native_alpha_delta_pred"].float().abs().mean()
        items["native_alpha_delta_reg"] = l_delta_alpha

        l_delta_fg = torch.tensor(0.0, device=device)
        if self._enabled(self.w_native_fg_delta) and "native_fg_delta_pred" in pred:
            l_delta_fg = pred["native_fg_delta_pred"].float().abs().mean()
        items["native_fg_delta_reg"] = l_delta_fg

        # Quality eval
        l_quality = torch.tensor(0.0, device=device)
        if self._enabled(self.w_quality_eval) and "quality_eval_pred" in pred:
            # Target: per-pixel error of the coarse alpha prediction
            with torch.no_grad():
                coarse_error = (pred["coarse_alpha_pred"].float() - alpha_g).abs()
                quality_target = coarse_error.clamp(0.0, 1.0)
            l_quality = (pred["quality_eval_pred"].float() - quality_target).abs().mean()
        items["quality_eval"] = l_quality

        # --- Total ---
        total = (
            self.w_alpha_l1 * l_alpha_l1
            + self.w_alpha_lap * l_alpha_lap
            + self.w_fg_l1 * l_fg_l1
            + self.w_comp_l1 * l_comp
            + self.w_alpha_band_l1 * l_band_l1
            + self.w_alpha_band_lap * l_band_lap
            + self.w_temporal_alpha * l_temporal_alpha
            + self.w_temporal_fg * l_temporal_fg
            + self.w_comp_random_bg * l_comp_rand
            + self.w_spill_l1 * l_spill
            + self.w_green_fg_alpha * l_green_fg_alpha
            + self.w_green_fg_color * l_green_fg_color
            + self.w_green_bg_alpha_suppress * l_green_bg_alpha_suppress
            + self.w_coarse_alpha * l_coarse_alpha
            + self.w_coarse_fg * l_coarse_fg
            + self.w_native_alpha_delta * l_delta_alpha
            + self.w_native_fg_delta * l_delta_fg
            + self.w_quality_eval * l_quality
        )

        items["total"] = total
        return total, items
