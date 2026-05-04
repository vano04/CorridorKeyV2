"""V3 loss smoke test.

Verifies: synthetic batch → forward → loss → backward → no NaN/Inf.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
import torch

from models import build_v3_hybrid_video_matting_model
from losses import V3MattingLossComputer
from losses.v3_matting_losses import _laplacian_pyramid_loss


@pytest.fixture
def model_and_device():
    config = {
        "patch_size": 8,
        "embed_dims": [64, 128, 192, 256],
        "depths": [1, 1, 2, 1],
        "num_heads": [2, 4, 6, 8],
        "window_sizes": [8, 8, 4, 4],
        "memory_tokens": 16,
        "temporal_window": 3,
        "gradient_checkpointing": False,
        "decoder_out_dim": 128,
        "global_context_dim": 128,
        "global_context_layers": 2,
        "global_context_heads": 4,
        "global_context_tokens": 16,
        "use_reference_memory": True,
        "reference_tokens_per_frame": 16,
        "num_reference_frames": 1,
        "reference_dropout": 0.0,
        "use_native_refiner": True,
        "native_refiner_blocks": 2,
        "native_refiner_hidden": 32,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_v3_hybrid_video_matting_model(config).to(device)
    return model, device


def test_loss_forward_backward(model_and_device):
    model, device = model_and_device
    model.train()

    B, T, H, W = 1, 2, 256, 256
    video = torch.randn(B, T, 3, H, W, device=device)
    coarse = torch.rand(B, 1, H, W, device=device)
    alpha_gt = torch.rand(B, T, 1, H, W, device=device)
    fg_gt = torch.rand(B, T, 3, H, W, device=device)
    bg_gt = torch.rand(B, T, 3, H, W, device=device)

    out = model(
        video=video,
        coarse_alpha_init=coarse,
        bg_for_comp=bg_gt,
    )

    loss_cfg = {
        "alpha_l1": 1.0,
        "alpha_laplacian": 1.0,
        "fg_l1": 1.0,
        "comp_l1": 1.0,
        "alpha_band_l1": 0.5,
        "temporal_alpha_gradient": 0.3,
    }
    criterion = V3MattingLossComputer(weights=loss_cfg, fg_representation="premul").to(device)
    batch = {
        "alpha_gt": alpha_gt,
        "fg_gt": fg_gt,
        "bg_gt": bg_gt,
        "video_rgb": video,
        "input_gt": video,
    }

    total_loss, loss_items = criterion(out, batch)

    # Check loss is finite
    assert torch.isfinite(total_loss), f"Loss is not finite: {total_loss.item()}"
    assert total_loss.item() > 0, f"Loss should be positive: {total_loss.item()}"

    # Check all named components
    for k, v in loss_items.items():
        assert torch.isfinite(v), f"Loss component {k} is not finite: {v.item()}"

    # Backward
    total_loss.backward()

    # Check for NaN gradients
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(p.grad).any(), f"Inf gradient in {name}"


def test_all_loss_components_present(model_and_device):
    model, device = model_and_device
    model.train()

    B, T, H, W = 1, 2, 256, 256
    video = torch.randn(B, T, 3, H, W, device=device)
    coarse = torch.rand(B, 1, H, W, device=device)

    out = model(video=video, coarse_alpha_init=coarse, bg_for_comp=video)

    criterion = V3MattingLossComputer(
        weights={"alpha_l1": 1.0},
        fg_representation="premul",
    ).to(device)
    batch = {
        "alpha_gt": torch.rand(B, T, 1, H, W, device=device),
        "fg_gt": torch.rand(B, T, 3, H, W, device=device),
        "video_rgb": video,
        "input_gt": video,
    }

    _, items = criterion(out, batch)

    expected_keys = {
        "total", "alpha_l1", "alpha_lap", "fg_l1", "comp_l1",
        "alpha_band_l1", "alpha_band_lap", "temporal_alpha", "temporal_fg",
        "comp_random_bg", "coarse_alpha_l1", "coarse_fg_l1",
        "native_alpha_delta_reg", "native_fg_delta_reg", "uncertainty", "quality_eval",
        "green_bg_alpha_suppress", "green_bg_alpha_suppress_abs", "green_bg_pixels",
        "semantic_prior_bce", "semantic_prior_dice",
    }
    for k in expected_keys:
        assert k in items, f"Missing loss component: {k}"


def test_semantic_prior_loss_uses_global_alpha_target():
    criterion = V3MattingLossComputer(
        weights={"semantic_prior_bce": 1.0, "semantic_prior_dice": 1.0},
        fg_representation="premul",
    )
    pred = {
        "alpha_pred": torch.zeros(1, 1, 1, 8, 8),
        "fg_pred": torch.zeros(1, 1, 3, 8, 8),
        "comp_pred": torch.zeros(1, 1, 3, 8, 8),
        "semantic_fg_logits": torch.zeros(1, 1, 1, 16, 16),
    }
    batch = {
        "alpha_gt": torch.zeros(1, 1, 1, 8, 8),
        "fg_gt": torch.zeros(1, 1, 3, 8, 8),
        "video_rgb": torch.zeros(1, 1, 3, 8, 8),
        "input_gt": torch.zeros(1, 1, 3, 8, 8),
        "global_alpha_gt": torch.ones(1, 1, 1, 16, 16),
    }

    total, items = criterion(pred, batch)

    assert torch.isfinite(total)
    assert items["semantic_prior_bce"] > 0
    assert items["semantic_prior_dice"] > 0


def test_green_background_alpha_suppression_targets_zero_alpha(model_and_device):
    _, device = model_and_device

    criterion = V3MattingLossComputer(
        weights={"green_bg_alpha_suppress": 1.0},
        fg_representation="premul",
    ).to(device)

    alpha_pred = torch.tensor([[[[[0.75, 0.75], [0.75, 0.75]]]]], device=device)
    fg_pred = torch.zeros(1, 1, 3, 2, 2, device=device)
    comp_pred = torch.zeros(1, 1, 3, 2, 2, device=device)

    green_bg_video = torch.tensor(
        [[[[[0.05, 0.08], [0.06, 0.04]],
           [[0.85, 0.92], [0.90, 0.88]],
           [[0.04, 0.05], [0.03, 0.04]]]]],
        device=device,
    )
    nongreen_video = torch.tensor(
        [[[[[0.70, 0.72], [0.68, 0.74]],
           [[0.20, 0.18], [0.24, 0.22]],
           [[0.10, 0.12], [0.08, 0.11]]]]],
        device=device,
    )
    fg_gt = torch.zeros(1, 1, 3, 2, 2, device=device)

    green_bg_batch = {
        "alpha_gt": torch.zeros(1, 1, 1, 2, 2, device=device),
        "fg_gt": fg_gt,
        "video_rgb": green_bg_video,
        "input_gt": green_bg_video,
    }
    nongreen_batch = {
        "alpha_gt": torch.zeros(1, 1, 1, 2, 2, device=device),
        "fg_gt": fg_gt,
        "video_rgb": nongreen_video,
        "input_gt": nongreen_video,
    }
    pred = {"alpha_pred": alpha_pred, "fg_pred": fg_pred, "comp_pred": comp_pred}

    green_bg_loss, green_bg_items = criterion(pred, green_bg_batch)
    nongreen_loss, nongreen_items = criterion(pred, nongreen_batch)

    assert green_bg_items["green_bg_pixels"] > nongreen_items["green_bg_pixels"]
    assert green_bg_items["green_bg_alpha_suppress"] > nongreen_items["green_bg_alpha_suppress"]
    assert torch.isclose(nongreen_items["green_bg_alpha_suppress"], torch.zeros_like(nongreen_items["green_bg_alpha_suppress"]))
    assert green_bg_items["green_bg_alpha_suppress"] > 0


def test_laplacian_loss_handles_small_spatial_sizes():
    x_16 = torch.rand(2, 1, 16, 16)
    y_16 = torch.rand(2, 1, 16, 16)
    l_16 = _laplacian_pyramid_loss(x_16, y_16, levels=5)
    assert torch.isfinite(l_16)

    x_1 = torch.rand(1, 1, 1, 1)
    y_1 = torch.rand(1, 1, 1, 1)
    l_1 = _laplacian_pyramid_loss(x_1, y_1, levels=5)
    assert torch.isfinite(l_1)




def test_laplacian_loss_does_not_amplify_constant_offsets():
    x = torch.zeros(2, 1, 64, 64)
    y = torch.ones(2, 1, 64, 64)

    loss = _laplacian_pyramid_loss(x, y, levels=5)

    assert torch.isfinite(loss)
    assert loss <= 1.0


def test_laplacian_loss_preserves_thin_soft_mask_support():
    x = torch.zeros(1, 1, 64, 64)
    y = torch.zeros_like(x)
    y[:, :, 31:33, :] = 1.0
    mask = torch.zeros_like(x)
    mask[:, :, 31:33, :] = 1.0

    loss = _laplacian_pyramid_loss(x, y, levels=5, mask=mask)

    assert torch.isfinite(loss)
    assert loss > 0


def test_alpha_lap_valid_mask_is_normalized_over_valid_support():
    criterion = V3MattingLossComputer(
        weights={
            "alpha_l1": 0.0,
            "alpha_laplacian": 1.0,
            "fg_l1": 0.0,
            "comp_l1": 0.0,
            "alpha_band_l1": 0.0,
            "alpha_band_laplacian": 0.0,
            "temporal_alpha_gradient": 0.0,
            "temporal_fg_gradient": 0.0,
            "comp_random_bg": 0.0,
            "spill_l1": 0.0,
            "green_fg_alpha": 0.0,
            "green_fg_color": 0.0,
            "green_bg_alpha_suppress": 0.0,
            "coarse_alpha_l1": 0.0,
            "coarse_fg_l1": 0.0,
            "native_alpha_delta_reg": 0.0,
            "native_fg_delta_reg": 0.0,
            "quality_eval": 0.0,
        },
        fg_representation="premul",
    )

    b, t, h, w = 1, 4, 64, 64
    alpha_pred = torch.ones(b, t, 1, h, w)
    alpha_gt = torch.zeros(b, t, 1, h, w)
    fg = torch.zeros(b, t, 3, h, w)
    comp = torch.zeros(b, t, 3, h, w)

    pred = {"alpha_pred": alpha_pred, "fg_pred": fg, "comp_pred": comp}
    base_batch = {
        "alpha_gt": alpha_gt,
        "fg_gt": fg,
        "video_rgb": comp,
        "input_gt": comp,
    }

    _, items_all = criterion(
        pred,
        {
            **base_batch,
            "valid_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]]),
        },
    )
    _, items_half = criterion(
        pred,
        {
            **base_batch,
            "valid_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0]]),
        },
    )

    assert torch.isfinite(items_all["alpha_lap"])
    assert torch.isfinite(items_half["alpha_lap"])
    assert torch.isclose(items_all["alpha_lap"], items_half["alpha_lap"], rtol=1e-5, atol=1e-6)


def test_alpha_lap_sanitizes_invalid_alpha_targets():
    criterion = V3MattingLossComputer(
        weights={
            "alpha_l1": 1.0,
            "alpha_laplacian": 1.0,
            "fg_l1": 0.0,
            "comp_l1": 0.0,
            "alpha_band_l1": 0.0,
            "alpha_band_laplacian": 0.0,
            "temporal_alpha_gradient": 0.0,
            "temporal_fg_gradient": 0.0,
            "comp_random_bg": 0.0,
            "spill_l1": 0.0,
            "green_fg_alpha": 0.0,
            "green_fg_color": 0.0,
            "green_bg_alpha_suppress": 0.0,
            "coarse_alpha_l1": 0.0,
            "coarse_fg_l1": 0.0,
            "native_alpha_delta_reg": 0.0,
            "native_fg_delta_reg": 0.0,
            "uncertainty": 0.0,
            "quality_eval": 0.0,
        },
        fg_representation="premul",
    )

    b, t, h, w = 1, 1, 8, 8
    alpha_pred = torch.zeros(b, t, 1, h, w)
    alpha_gt = torch.rand(b, t, 1, h, w)
    alpha_gt[..., 0, 0] = float("nan")
    alpha_gt[..., 0, 1] = float("inf")
    alpha_gt[..., 0, 2] = -0.25
    alpha_gt[..., 0, 3] = 1.078125
    fg = torch.zeros(b, t, 3, h, w)
    comp = torch.zeros(b, t, 3, h, w)

    total, items = criterion(
        {"alpha_pred": alpha_pred, "fg_pred": fg, "comp_pred": comp},
        {"alpha_gt": alpha_gt, "fg_gt": fg, "video_rgb": comp, "input_gt": comp},
    )

    assert torch.isfinite(total)
    assert torch.isfinite(items["alpha_l1"])
    assert torch.isfinite(items["alpha_lap"])


def test_masked_losses_are_normalized_over_valid_frames():
    criterion = V3MattingLossComputer(
        weights={
            "alpha_l1": 1.0,
            "alpha_laplacian": 0.0,
            "fg_l1": 1.0,
            "comp_l1": 1.0,
            "alpha_band_l1": 0.0,
            "alpha_band_laplacian": 0.0,
            "temporal_alpha_gradient": 0.0,
            "temporal_fg_gradient": 0.0,
            "comp_random_bg": 0.0,
            "spill_l1": 0.0,
            "green_fg_alpha": 0.0,
            "green_fg_color": 0.0,
            "green_bg_alpha_suppress": 0.0,
            "coarse_alpha_l1": 0.0,
            "coarse_fg_l1": 0.0,
            "native_alpha_delta_reg": 0.0,
            "native_fg_delta_reg": 0.0,
            "uncertainty": 0.0,
            "quality_eval": 0.0,
        },
        fg_representation="premul",
    )

    b, t, h, w = 1, 4, 8, 8
    alpha_pred = torch.ones(b, t, 1, h, w)
    alpha_gt = torch.zeros(b, t, 1, h, w)
    fg_pred = torch.ones(b, t, 3, h, w)
    fg_gt = torch.zeros(b, t, 3, h, w)
    comp_pred = torch.ones(b, t, 3, h, w)
    comp_gt = torch.zeros(b, t, 3, h, w)
    pred = {"alpha_pred": alpha_pred, "fg_pred": fg_pred, "comp_pred": comp_pred}
    base_batch = {"alpha_gt": alpha_gt, "fg_gt": fg_gt, "video_rgb": comp_gt, "input_gt": comp_gt}

    _, items_all = criterion(pred, {**base_batch, "valid_mask": torch.tensor([[1.0, 1.0, 1.0, 1.0]])})
    _, items_half = criterion(pred, {**base_batch, "valid_mask": torch.tensor([[1.0, 1.0, 0.0, 0.0]])})

    for key in ("alpha_l1", "fg_l1", "comp_l1"):
        assert torch.isclose(items_all[key], items_half[key], rtol=1e-5, atol=1e-6)


def test_uncertainty_weight_contributes_to_total_when_prediction_exists():
    criterion = V3MattingLossComputer(
        weights={
            "alpha_l1": 0.0,
            "alpha_laplacian": 0.0,
            "fg_l1": 0.0,
            "comp_l1": 0.0,
            "alpha_band_l1": 0.0,
            "alpha_band_laplacian": 0.0,
            "temporal_alpha_gradient": 0.0,
            "temporal_fg_gradient": 0.0,
            "comp_random_bg": 0.0,
            "spill_l1": 0.0,
            "green_fg_alpha": 0.0,
            "green_fg_color": 0.0,
            "green_bg_alpha_suppress": 0.0,
            "coarse_alpha_l1": 0.0,
            "coarse_fg_l1": 0.0,
            "native_alpha_delta_reg": 0.0,
            "native_fg_delta_reg": 0.0,
            "uncertainty": 1.0,
            "quality_eval": 0.0,
        },
        fg_representation="premul",
    )

    b, t, h, w = 1, 2, 4, 4
    alpha_pred = torch.ones(b, t, 1, h, w)
    alpha_gt = torch.zeros(b, t, 1, h, w)
    fg = torch.zeros(b, t, 3, h, w)
    pred = {
        "alpha_pred": alpha_pred,
        "fg_pred": fg,
        "comp_pred": fg,
        "uncertainty_pred": torch.zeros(b, t, 1, h, w),
    }
    batch = {"alpha_gt": alpha_gt, "fg_gt": fg, "video_rgb": fg, "input_gt": fg}

    total, items = criterion(pred, batch)

    assert items["uncertainty"] > 0
    assert torch.isclose(total, items["uncertainty"])
