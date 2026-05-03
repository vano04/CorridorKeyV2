"""V3 model shape test.

Verifies output shapes match expected dimensions for all prediction heads.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_PROJECT_ROOT = str(Path(__file__).resolve().parents[1])
if _PROJECT_ROOT in sys.path:
    sys.path.remove(_PROJECT_ROOT)
sys.path.insert(0, _PROJECT_ROOT)

from models import build_v3_hybrid_video_matting_model
from models.v3_hybrid_matting import _alpha_only_edge_refine, _input_residual_fg_base


@pytest.fixture
def model():
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
    m = build_v3_hybrid_video_matting_model(config)
    if torch.cuda.is_available():
        m = m.cuda()
    return m


@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize("h,w", [(256, 256), (384, 384)])
def test_output_shapes(model, device, h, w):
    B, T = 1, 4
    video = torch.randn(B, T, 3, h, w, device=device)
    coarse = torch.rand(B, 1, h, w, device=device)
    global_video = torch.randn(B, T, 3, h // 2, w // 2, device=device)
    global_coarse = torch.rand(B, 1, h // 2, w // 2, device=device)

    model.eval()
    with torch.no_grad():
        out = model(
            video=video,
            coarse_alpha_init=coarse,
            global_video=global_video,
            global_coarse_alpha_init=global_coarse,
        )

    assert out["alpha_pred"].shape == (B, T, 1, h, w)
    assert out["fg_pred"].shape == (B, T, 3, h, w)
    assert out["comp_pred"].shape == (B, T, 3, h, w)
    assert out["coarse_alpha_pred"].shape == (B, T, 1, h, w)
    assert out["coarse_fg_pred"].shape == (B, T, 3, h, w)

    if "uncertainty_pred" in out:
        assert out["uncertainty_pred"].shape == (B, T, 1, h, w)
    if "spill_mask_pred" in out:
        assert out["spill_mask_pred"].shape == (B, T, 1, h, w)
    if "refine_mask" in out:
        assert out["refine_mask"].shape == (B, T, 1, h, w)


def test_no_quality_eval_at_inference(model, device):
    """Quality eval head should not appear in eval mode."""
    B, T, H, W = 1, 2, 256, 256
    video = torch.randn(B, T, 3, H, W, device=device)
    coarse = torch.rand(B, 1, H, W, device=device)

    model.eval()
    with torch.no_grad():
        out = model(video=video, coarse_alpha_init=coarse)
    assert "quality_eval_pred" not in out


def test_global_context_alpha_mode_zero_masks_seed():
    config = {
        "patch_size": 8,
        "embed_dims": [32, 64, 96, 128],
        "depths": [1, 1, 1, 1],
        "num_heads": [1, 2, 3, 4],
        "global_context_dim": 64,
        "global_context_layers": 1,
        "global_context_heads": 4,
        "global_context_tokens": 4,
        "decoder_out_dim": 64,
        "use_reference_memory": False,
        "use_native_refiner": False,
        "global_context_alpha_mode": "zero",
    }
    model = build_v3_hybrid_video_matting_model(config)
    global_video = torch.randn(1, 2, 3, 32, 32)
    seed = torch.ones(1, 1, 32, 32)

    cond, _, _ = model._build_global_input(global_video, seed)

    assert torch.count_nonzero(cond) == 0


def test_input_residual_fg_mode_uses_native_input_base():
    device = torch.device("cpu")
    config = {
        "patch_size": 8,
        "embed_dims": [32, 48, 64, 80],
        "depths": [1, 1, 1, 1],
        "num_heads": [1, 2, 4, 4],
        "window_sizes": [4, 4, 4, 4],
        "decoder_out_dim": 32,
        "global_context_dim": 32,
        "global_context_layers": 1,
        "global_context_heads": 1,
        "global_context_tokens": 4,
        "use_reference_memory": False,
        "use_native_refiner": False,
        "predict_fg": True,
        "fg_prediction_mode": "input_residual",
    }
    m = build_v3_hybrid_video_matting_model(config).to(device).eval()
    video = torch.rand(1, 2, 3, 64, 64, device=device)
    coarse = torch.rand(1, 1, 64, 64, device=device)

    with torch.no_grad():
        out = m(video=video, coarse_alpha_init=coarse)

    assert "decoder_fg_pred" in out
    assert "input_fg_base_pred" in out
    assert torch.allclose(out["fg_pred"], out["input_fg_base_pred"], atol=1e-6)


def test_input_residual_despill_base_reduces_green_edge_halo():
    device = torch.device("cpu")
    alpha = torch.full((1, 1, 1, 64, 64), 0.5, device=device)
    video = torch.zeros(1, 1, 3, 64, 64, device=device)
    video[:, :, 1:2] = 1.0

    base = _input_residual_fg_base(
        video,
        alpha,
        despill=True,
        strength=1.0,
    )

    naive_green = video[:, :, 1:2] * 0.5
    assert base[:, :, 1:2].mean() < naive_green.mean()
    assert torch.allclose(base[:, :, 0:1], torch.zeros_like(base[:, :, 0:1]))
    assert torch.allclose(base[:, :, 2:3], torch.zeros_like(base[:, :, 2:3]))


def test_alpha_only_edge_refine_sharpens_uncertain_transition():
    step = torch.zeros(1, 1, 1, 16, 16)
    step[..., 8:] = 1.0
    flat = step.reshape(1, 1, 16, 16)
    alpha = torch.nn.functional.avg_pool2d(
        torch.nn.functional.pad(flat, (2, 2, 2, 2), mode="replicate"),
        kernel_size=5,
        stride=1,
    ).reshape_as(step)

    refined, delta = _alpha_only_edge_refine(
        alpha,
        refine_mask=torch.ones_like(alpha),
        strength=1.0,
        kernel_size=5,
    )

    assert refined[..., 7].mean() < alpha[..., 7].mean()
    assert refined[..., 8].mean() > alpha[..., 8].mean()
    assert delta.abs().sum() > 0


def test_quality_eval_at_train(model, device):
    """Quality eval head should appear in train mode."""
    B, T, H, W = 1, 2, 256, 256
    video = torch.randn(B, T, 3, H, W, device=device)
    coarse = torch.rand(B, 1, H, W, device=device)

    model.train()
    out = model(video=video, coarse_alpha_init=coarse)
    assert "quality_eval_pred" in out
    assert out["quality_eval_pred"].shape == (B, T, 1, H, W)
