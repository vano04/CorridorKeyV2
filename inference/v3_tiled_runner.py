"""V3 tiled inference runner.

Processes a full video frame-by-frame or window-by-window using the V3
three-branch architecture. Global context is computed once per temporal
window and cached; tiles at 1024x1024 are processed with overlap blending.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


def _tile_blend_weight(
    tile_h: int,
    tile_w: int,
    overlap: int,
    device: torch.device,
    dtype: torch.dtype,
    *,
    ramp_top: bool = True,
    ramp_bottom: bool = True,
    ramp_left: bool = True,
    ramp_right: bool = True,
) -> Tensor:
    """Generate a soft blending weight map for overlap regions.

    Only applies ramps on edges where another tile overlaps. Frame-boundary
    edges keep weight=1 so the final result has no darkened borders.
    """
    weight = torch.ones(tile_h, tile_w, device=device, dtype=dtype)
    if overlap > 0:
        ramp = torch.linspace(0.0, 1.0, overlap, device=device, dtype=dtype)
        if ramp_left:
            weight[:, :overlap] *= ramp[None, :]
        if ramp_right:
            weight[:, -overlap:] *= ramp.flip(0)[None, :]
        if ramp_top:
            weight[:overlap, :] *= ramp[:, None]
        if ramp_bottom:
            weight[-overlap:, :] *= ramp.flip(0)[:, None]
    return weight.unsqueeze(0).unsqueeze(0)


def compute_tile_positions(
    frame_h: int,
    frame_w: int,
    tile_size: int = 1024,
    overlap: int = 64,
) -> List[Tuple[int, int, int, int]]:
    """Compute (y0, x0, y1, x1) tile positions to cover a frame with overlap."""
    tiles = []
    stride = tile_size - overlap

    y = 0
    while y < frame_h:
        y1 = min(y + tile_size, frame_h)
        if y1 - y < tile_size and y > 0:
            y = max(0, frame_h - tile_size)
            y1 = frame_h

        x = 0
        while x < frame_w:
            x1 = min(x + tile_size, frame_w)
            if x1 - x < tile_size and x > 0:
                x = max(0, frame_w - tile_size)
                x1 = frame_w

            tiles.append((y, x, y1, x1))

            if x1 >= frame_w:
                break
            x += stride

        if y1 >= frame_h:
            break
        y += stride

    return tiles


def merge_tile_outputs(
    tile_outputs: List[Dict[str, Tensor]],
    tile_positions: List[Tuple[int, int, int, int]],
    frame_h: int,
    frame_w: int,
    overlap: int = 64,
    output_keys: Tuple[str, ...] = ("alpha_pred", "fg_pred"),
) -> Dict[str, Tensor]:
    """Merge tiled outputs into a full frame using overlap blending.

    Args:
        tile_outputs:    list of model output dicts (one per tile)
        tile_positions:  list of (y0, x0, y1, x1)
        frame_h, frame_w: target frame size
        overlap:         overlap in pixels
        output_keys:     which output keys to merge

    Returns:
        dict with merged tensors of shape [1, 1, C, frame_h, frame_w]
    """
    device = tile_outputs[0][output_keys[0]].device
    dtype = tile_outputs[0][output_keys[0]].dtype

    result: Dict[str, Tensor] = {}
    for key in output_keys:
        c = tile_outputs[0][key].shape[2]
        canvas = torch.zeros(1, 1, c, frame_h, frame_w, device=device, dtype=dtype)
        weight_canvas = torch.zeros(1, 1, 1, frame_h, frame_w, device=device, dtype=dtype)

        for tile_out, (y0, x0, y1, x1) in zip(tile_outputs, tile_positions):
            th, tw = y1 - y0, x1 - x0
            blend_w = _tile_blend_weight(
                th, tw, overlap, device, dtype,
                ramp_top=(y0 > 0),
                ramp_bottom=(y1 < frame_h),
                ramp_left=(x0 > 0),
                ramp_right=(x1 < frame_w),
            )

            tile_val = tile_out[key][:, :1]  # [B=1, T=1, C, H, W]
            canvas[:, :, :, y0:y1, x0:x1] += tile_val * blend_w
            weight_canvas[:, :, :, y0:y1, x0:x1] += blend_w

        result[key] = canvas / weight_canvas.clamp_min(1e-6)

    return result
