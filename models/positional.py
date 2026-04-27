"""Tile position embedding and coordinate channel utilities for V3."""
from __future__ import annotations

import torch
from torch import Tensor


def make_tile_coordinate_channels(
    tile_coords: Tensor,
    source_hw: Tensor,
    tile_h: int,
    tile_w: int,
) -> Tensor:
    """Build normalised coordinate channels for a batch of tiles.

    Args:
        tile_coords: [B, 4] with (y0, y1, x0, x1) in source-frame pixels.
        source_hw:   [B, 2] with (source_H, source_W).
        tile_h:      spatial height of the tile.
        tile_w:      spatial width of the tile.

    Returns:
        [B, 5, tile_h, tile_w] with channels:
            0: x0 / W   (left edge normalised)
            1: y0 / H   (top edge normalised)
            2: x1 / W   (right edge normalised)
            3: y1 / H   (bottom edge normalised)
            4: distance_to_frame_edge  (min normalised distance to any edge)
    """
    b = tile_coords.shape[0]
    device = tile_coords.device
    dtype = tile_coords.dtype
    source_hw = source_hw.to(device=device, dtype=dtype)

    src_h = source_hw[:, 0].clamp_min(1.0)
    src_w = source_hw[:, 1].clamp_min(1.0)

    y0_norm = tile_coords[:, 0] / src_h
    y1_norm = tile_coords[:, 1] / src_h
    x0_norm = tile_coords[:, 2] / src_w
    x1_norm = tile_coords[:, 3] / src_w

    # Per-pixel interpolation within the tile
    gy = torch.linspace(0.0, 1.0, tile_h, device=device, dtype=dtype)
    gx = torch.linspace(0.0, 1.0, tile_w, device=device, dtype=dtype)

    # Pixel-level normalised y and x within the source frame
    py = y0_norm[:, None] + (y1_norm - y0_norm)[:, None] * gy[None, :]  # [B, tile_h]
    px = x0_norm[:, None] + (x1_norm - x0_norm)[:, None] * gx[None, :]  # [B, tile_w]

    # Distance to nearest frame edge (min of 4 distances), per pixel
    dist_top = py                        # [B, tile_h]
    dist_bot = 1.0 - py                  # [B, tile_h]
    dist_left = px                       # [B, tile_w]
    dist_right = 1.0 - px               # [B, tile_w]

    min_vert = torch.minimum(dist_top, dist_bot)           # [B, tile_h]
    min_horiz = torch.minimum(dist_left, dist_right)       # [B, tile_w]
    edge_dist = torch.minimum(
        min_vert[:, :, None].expand(b, tile_h, tile_w),
        min_horiz[:, None, :].expand(b, tile_h, tile_w),
    )  # [B, tile_h, tile_w]

    # Constant channels (same value across spatial dims)
    ch_x0 = x0_norm[:, None, None].expand(b, tile_h, tile_w)
    ch_y0 = y0_norm[:, None, None].expand(b, tile_h, tile_w)
    ch_x1 = x1_norm[:, None, None].expand(b, tile_h, tile_w)
    ch_y1 = y1_norm[:, None, None].expand(b, tile_h, tile_w)

    return torch.stack([ch_x0, ch_y0, ch_x1, ch_y1, edge_dist], dim=1)


def make_default_tile_coords(b: int, h: int, w: int, device: torch.device, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
    """Create identity tile coords when no crop is applied (full frame = tile)."""
    tile_coords = torch.tensor(
        [[0.0, float(h), 0.0, float(w)]] * b,
        device=device, dtype=dtype,
    )
    source_hw = torch.tensor(
        [[float(h), float(w)]] * b,
        device=device, dtype=dtype,
    )
    return tile_coords, source_hw
