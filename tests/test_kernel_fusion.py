"""Kernel fusion equivalence tests.

Verifies that every refactored (TorchScript / fused) function produces
numerically identical results to the original reference implementations.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch
import torch.nn.functional as F

# ---- functions under test ----
from losses.v3_matting_losses import (
    _laplacian_pyramid_loss,
    _green_stats,
    _masked_mean,
)
from models.v3_hybrid_matting import (
    _chroma_distance_map,
    _green_excess_map,
    _boundary_band,
)


# ---------------------------------------------------------------------------
# Reference implementations (copied verbatim from originals, not imported)
# ---------------------------------------------------------------------------

def _ref_laplacian_pyramid_loss(pred, target, levels=5, mask=None):
    """Original non-scripted implementation for comparison."""
    EPS = 1e-6
    total = pred.new_tensor(0.0, dtype=torch.float32)
    weight_sum = pred.new_tensor(0.0, dtype=torch.float32)
    current_pred = pred.float()
    current_target = target.float()
    current_mask = mask.to(device=pred.device, dtype=torch.float32) if mask is not None else None
    weight = 1.0
    for _ in range(levels):
        if current_pred.shape[-2] < 2 or current_pred.shape[-1] < 2:
            break
        pred_down = F.avg_pool2d(current_pred, 2, stride=2)
        target_down = F.avg_pool2d(current_target, 2, stride=2)
        pred_up = F.interpolate(pred_down, size=current_pred.shape[-2:], mode="bilinear", align_corners=False)
        target_up = F.interpolate(target_down, size=current_target.shape[-2:], mode="bilinear", align_corners=False)
        lap_diff = (current_pred - pred_up) - (current_target - target_up)
        if current_mask is not None:
            val = (lap_diff.abs() * current_mask).sum() / current_mask.sum().clamp_min(1.0)
        else:
            val = lap_diff.abs().mean()
        total = total + weight * val
        weight_sum = weight_sum + weight
        current_pred = pred_down
        current_target = target_down
        if current_mask is not None:
            current_mask = F.interpolate(current_mask, size=pred_down.shape[-2:], mode="area")
        weight *= 0.5
    residual = (current_pred - current_target).abs()
    if current_mask is not None:
        val = (residual * current_mask).sum() / current_mask.sum().clamp_min(1.0)
    else:
        val = residual.mean()
    total = total + weight * val
    weight_sum = weight_sum + weight
    return total / weight_sum.clamp_min(EPS)


def _ref_chroma_distance_map(rgb, key_color=(0.0, 1.0, 0.0)):
    luma = 0.299 * rgb[:, :, 0:1] + 0.587 * rgb[:, :, 1:2] + 0.114 * rgb[:, :, 2:3]
    chroma = rgb - luma
    key = torch.tensor(key_color, device=rgb.device, dtype=rgb.dtype).view(1, 1, 3, 1, 1)
    key_luma = 0.299 * key_color[0] + 0.587 * key_color[1] + 0.114 * key_color[2]
    key_chroma = key - key_luma
    diff = chroma - key_chroma
    dist = (diff ** 2).sum(dim=2, keepdim=True).sqrt()
    return dist.clamp(0.0, 1.0)


def _ref_boundary_band(alpha, kernel_size=3):
    bt = alpha.shape[0] * alpha.shape[1]
    a = alpha.reshape(bt, 1, alpha.shape[-2], alpha.shape[-1])
    dil = F.max_pool2d(a, kernel_size, stride=1, padding=kernel_size // 2)
    ero = -F.max_pool2d(-a, kernel_size, stride=1, padding=kernel_size // 2)
    band = (dil - ero).clamp(0.0, 1.0)
    return band.reshape(alpha.shape[0], alpha.shape[1], 1, alpha.shape[-2], alpha.shape[-1])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("levels", [3, 5])
@pytest.mark.parametrize("use_mask", [False, True])
def test_laplacian_pyramid_loss_matches_reference(levels, use_mask):
    torch.manual_seed(0)
    pred   = torch.rand(2, 1, 64, 64)
    target = torch.rand(2, 1, 64, 64)
    mask   = torch.rand(2, 1, 64, 64) if use_mask else None

    ref = _ref_laplacian_pyramid_loss(pred, target, levels=levels, mask=mask)
    got = _laplacian_pyramid_loss(pred, target, levels=levels, mask=mask)

    assert torch.allclose(ref, got, atol=1e-5), (
        f"Laplacian mismatch: ref={ref.item():.6f} got={got.item():.6f}"
    )


def test_laplacian_handles_tiny_spatial():
    for hw in [1, 2, 4]:
        x = torch.rand(1, 1, hw, hw)
        y = torch.rand(1, 1, hw, hw)
        loss = _laplacian_pyramid_loss(x, y, levels=5)
        assert torch.isfinite(loss), f"Non-finite at hw={hw}"


@pytest.mark.parametrize("ndim", [4, 5])
def test_green_stats_values(ndim):
    torch.manual_seed(1)
    if ndim == 5:
        rgb = torch.rand(2, 3, 3, 8, 8)
    else:
        rgb = torch.rand(2, 3, 8, 8)

    ge, sat, max_rgb = _green_stats(rgb)

    r = rgb[..., 0:1, :, :]
    g = rgb[..., 1:2, :, :]
    b = rgb[..., 2:3, :, :]
    expected_ge  = g - torch.maximum(r, b)
    expected_sat = torch.maximum(torch.maximum(r, g), b) - torch.minimum(torch.minimum(r, g), b)

    assert torch.allclose(ge,  expected_ge,  atol=1e-6), "green_excess mismatch"
    assert torch.allclose(sat, expected_sat, atol=1e-6), "saturation mismatch"


def test_chroma_distance_matches_reference():
    torch.manual_seed(2)
    rgb = torch.rand(2, 4, 3, 16, 16)

    ref = _ref_chroma_distance_map(rgb, key_color=(0.0, 1.0, 0.0))
    got = _chroma_distance_map(rgb, key_r=0.0, key_g=1.0, key_b=0.0)

    assert torch.allclose(ref, got, atol=1e-5), (
        f"Chroma distance mismatch: max_diff={( ref - got).abs().max().item():.2e}"
    )


def test_boundary_band_matches_reference():
    torch.manual_seed(3)
    alpha = torch.rand(2, 4, 1, 32, 32)

    ref = _ref_boundary_band(alpha, kernel_size=3)
    got = _boundary_band(alpha, kernel_size=3)

    assert torch.allclose(ref, got, atol=1e-6), (
        f"Boundary band mismatch: max_diff={(ref - got).abs().max().item():.2e}"
    )


def test_boundary_band_range():
    alpha = torch.rand(1, 2, 1, 16, 16)
    band = _boundary_band(alpha)
    assert band.min() >= 0.0, "Band has negative values"
    assert band.max() <= 1.0, "Band exceeds 1.0"


def test_masked_mean_no_mask():
    x = torch.ones(2, 3, 4, 4) * 0.5
    result = _masked_mean(x)
    assert torch.isclose(result, torch.tensor(0.5)), f"Expected 0.5 got {result}"


def test_masked_mean_with_mask():
    x = torch.ones(1, 1, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)
    mask[:, :, :2, :] = 1.0   # top half
    result = _masked_mean(x, mask)
    assert torch.isclose(result, torch.tensor(1.0)), f"Expected 1.0 got {result}"


def test_green_excess_map_range():
    rgb = torch.rand(2, 3, 3, 16, 16)
    ge = _green_excess_map(rgb)
    assert ge.min() >= 0.0
    assert ge.max() <= 1.0
