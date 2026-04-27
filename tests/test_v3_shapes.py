"""V3 model shape test.

Verifies output shapes match expected dimensions for all prediction heads.
"""
from __future__ import annotations

import pytest
import torch

from models import build_v3_hybrid_video_matting_model


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
    global_video = torch.randn(B, T, 3, h, w, device=device)
    global_coarse = torch.rand(B, 1, h, w, device=device)

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


def test_quality_eval_at_train(model, device):
    """Quality eval head should appear in train mode."""
    B, T, H, W = 1, 2, 256, 256
    video = torch.randn(B, T, 3, H, W, device=device)
    coarse = torch.rand(B, 1, H, W, device=device)

    model.train()
    out = model(video=video, coarse_alpha_init=coarse)
    assert "quality_eval_pred" in out
    assert out["quality_eval_pred"].shape == (B, T, 1, H, W)
