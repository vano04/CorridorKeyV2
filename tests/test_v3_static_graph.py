"""V3 DDP static graph test.

Verifies that all model parameters receive gradients every forward pass,
even when refine_mask is all zeros (no refinement needed).
"""
from __future__ import annotations

import pytest
import torch
from torch import Tensor

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return build_v3_hybrid_video_matting_model(config).to(device), device


def test_all_params_get_gradients(model):
    """Every parameter should receive a gradient after forward+backward."""
    m, device = model
    m.train()

    B, T, H, W = 1, 2, 256, 256
    video = torch.randn(B, T, 3, H, W, device=device)
    coarse = torch.rand(B, 1, H, W, device=device)
    alpha_gt = torch.rand(B, T, 1, H, W, device=device)

    out = m(video=video, coarse_alpha_init=coarse)
    # Backprop through ALL outputs so every parameter receives a gradient
    loss = torch.tensor(0.0, device=device)
    for k, v in out.items():
        if isinstance(v, Tensor) and v.requires_grad:
            loss = loss + v.sum()
    loss.backward()

    no_grad_params = []
    for name, p in m.named_parameters():
        if p.requires_grad and p.grad is None:
            no_grad_params.append(name)

    assert len(no_grad_params) == 0, f"Parameters without gradients (would break DDP static_graph): {no_grad_params}"


def test_same_output_keys_with_zero_refine_mask(model):
    """Output dict keys should be the same regardless of refine_mask values."""
    m, device = model
    m.train()

    B, T, H, W = 1, 2, 256, 256
    video = torch.randn(B, T, 3, H, W, device=device)
    coarse = torch.rand(B, 1, H, W, device=device)

    out1 = m(video=video, coarse_alpha_init=coarse)
    out2 = m(video=video, coarse_alpha_init=torch.zeros_like(coarse))

    assert set(out1.keys()) == set(out2.keys()), f"Key mismatch: {set(out1.keys())} vs {set(out2.keys())}"
