"""V3 inference tile merge test.

Verifies that splitting a 2048 frame into 1024 tiles with overlap and
merging them back produces a coherent result.
"""
from __future__ import annotations

import pytest
import torch

from inference import compute_tile_positions, merge_tile_outputs


def test_tile_positions_cover_frame():
    """All pixels should be covered by at least one tile."""
    frame_h, frame_w = 2048, 2048
    tiles = compute_tile_positions(frame_h, frame_w, tile_size=1024, overlap=64)

    coverage = torch.zeros(frame_h, frame_w)
    for y0, x0, y1, x1 in tiles:
        coverage[y0:y1, x0:x1] += 1

    assert (coverage > 0).all(), f"Some pixels not covered: {(coverage == 0).sum().item()} uncovered"


def test_tile_positions_non_power_of_two():
    """Test coverage for non-power-of-2 frame sizes."""
    frame_h, frame_w = 1920, 1080
    tiles = compute_tile_positions(frame_h, frame_w, tile_size=1024, overlap=64)

    coverage = torch.zeros(frame_h, frame_w)
    for y0, x0, y1, x1 in tiles:
        assert y1 - y0 <= 1024
        assert x1 - x0 <= 1024
        coverage[y0:y1, x0:x1] += 1

    assert (coverage > 0).all()


def test_merge_constant_tiles():
    """If all tiles output the same constant, merged result should be that constant."""
    frame_h, frame_w = 2048, 2048
    tiles = compute_tile_positions(frame_h, frame_w, tile_size=1024, overlap=64)

    device = torch.device("cpu")
    constant_val = 0.42
    tile_outputs = []
    for y0, x0, y1, x1 in tiles:
        th, tw = y1 - y0, x1 - x0
        tile_out = {
            "alpha_pred": torch.full((1, 1, 1, th, tw), constant_val, device=device),
        }
        tile_outputs.append(tile_out)

    merged = merge_tile_outputs(
        tile_outputs, tiles, frame_h, frame_w,
        overlap=64, output_keys=("alpha_pred",),
    )

    result = merged["alpha_pred"]
    assert result.shape == (1, 1, 1, frame_h, frame_w)
    assert (result - constant_val).abs().max() < 1e-4, f"Max deviation: {(result - constant_val).abs().max()}"
