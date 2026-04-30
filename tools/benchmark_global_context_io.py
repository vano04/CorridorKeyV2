#!/usr/bin/env python3
"""Benchmark EXR load/downscale choices for V3 global context.

Examples:
  .venv/bin/python tools/benchmark_global_context_io.py \
    --exr Infer/corridor_greenscreen_demo_dwab1024/Input/00001.exr

  .venv/bin/python tools/benchmark_global_context_io.py \
    --tar CorridorKeyDataset/shard_00000.tar \
    --member clip_001.input.00001.exr
"""

from __future__ import annotations

import argparse
import contextlib
import os
from pathlib import Path
import statistics
import tarfile
import tempfile
import time
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import OpenEXR  # type: ignore[import-not-found]
import torch
import torch.nn.functional as F


TileGrid = Tuple[int, int, int, int]


def _compose_channels(channels: Dict[str, np.ndarray]) -> np.ndarray:
    names = set(channels)
    if {"R", "G", "B"}.issubset(names):
        parts = [channels["R"], channels["G"], channels["B"]]
        if "A" in names:
            parts.append(channels["A"])
        return np.stack(parts, axis=2)
    if "Y" in names:
        return channels["Y"][..., None]
    if "A" in names and len(names) == 1:
        return channels["A"][..., None]
    ordered = [channels[name] for name in sorted(channels)]
    if len(ordered) == 1:
        return ordered[0][..., None]
    return np.stack(ordered, axis=2)


@contextlib.contextmanager
def _source_path(args: argparse.Namespace) -> Iterator[str]:
    if args.exr is not None:
        yield str(args.exr)
        return

    assert args.tar is not None and args.member is not None
    with tarfile.open(args.tar, mode="r:") as archive:
        extracted = archive.extractfile(args.member)
        if extracted is None:
            raise FileNotFoundError(f"missing tar member: {args.member}")
        payload = extracted.read()

    tmp_dir = "/dev/shm" if os.path.isdir("/dev/shm") else None
    fd, tmp_path = tempfile.mkstemp(suffix=".exr", dir=tmp_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        yield tmp_path
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _read_header(path: str) -> Tuple[int, int, int, int]:
    with OpenEXR.File(path, header_only=True) as exr:
        header = dict(exr.header())
    data_window = header.get("dataWindow")
    if data_window is None:
        raise RuntimeError("EXR is missing dataWindow")
    dw_min, dw_max = data_window
    width = int(dw_max[0] - dw_min[0] + 1)
    height = int(dw_max[1] - dw_min[1] + 1)
    tiles = header.get("tiles")
    if tiles is None:
        raise RuntimeError("EXR is not tiled; this benchmark expects tiled DWAB EXRs")
    return height, width, int(tiles.ySize), int(tiles.xSize)


def _read_exr_hwc(path: str, tile_grid: Optional[TileGrid]) -> torch.Tensor:
    if tile_grid is None:
        with OpenEXR.File(path, separate_channels=True) as exr:
            channels = {name: ch.pixels for name, ch in exr.parts[0].channels.items()}
    else:
        tx0, tx1, ty0, ty1 = tile_grid
        with OpenEXR.File(path, header_only=True) as exr:
            channels = {
                name: ch.pixels
                for name, ch in exr.readTiles(tx0, tx1, ty0, ty1, separate_channels=True).items()
            }
    array = _compose_channels(channels)
    return torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.float16)


def _read_exr_hwc_tiles(path: str, tile_grids: Sequence[TileGrid]) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    with OpenEXR.File(path, header_only=True) as exr:
        for tx0, tx1, ty0, ty1 in tile_grids:
            channels = {
                name: ch.pixels
                for name, ch in exr.readTiles(tx0, tx1, ty0, ty1, separate_channels=True).items()
            }
            array = _compose_channels(channels)
            out.append(torch.from_numpy(np.ascontiguousarray(array)).to(dtype=torch.float16))
    return out


def _parse_grid(raw: str) -> TileGrid:
    parts = [int(p.strip()) for p in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--tile-grid must be tx0,tx1,ty0,ty1")
    tx0, tx1, ty0, ty1 = parts
    if tx1 < tx0 or ty1 < ty0:
        raise ValueError("--tile-grid max values must be >= min values")
    return tx0, tx1, ty0, ty1


def _full_grid(height: int, width: int, tile_h: int, tile_w: int) -> TileGrid:
    tiles_y = (height + tile_h - 1) // tile_h
    tiles_x = (width + tile_w - 1) // tile_w
    return 0, tiles_x - 1, 0, tiles_y - 1


def _neighbor_grids(tile_grid: TileGrid, full_grid: TileGrid) -> List[TileGrid]:
    tx0, tx1, ty0, ty1 = tile_grid
    span_x = tx1 - tx0 + 1
    span_y = ty1 - ty0 + 1
    fx0, fx1, fy0, fy1 = full_grid
    candidates = [
        (tx0 + span_x, tx1 + span_x, ty0, ty1),
        (tx0, tx1, ty0 + span_y, ty1 + span_y),
        (tx0 + span_x, tx1 + span_x, ty0 + span_y, ty1 + span_y),
    ]
    return [
        (max(fx0, a), min(fx1, b), max(fy0, c), min(fy1, d))
        for a, b, c, d in candidates
        if a <= fx1 and c <= fy1
    ]


def _bilinear_to_1024(img_hwc_fp16: torch.Tensor) -> torch.Tensor:
    x = img_hwc_fp16.permute(2, 0, 1).unsqueeze(0).float()
    out = F.interpolate(x, size=(1024, 1024), mode="bilinear", align_corners=False)
    return out.squeeze(0).permute(1, 2, 0).half()


def _box2x(img_hwc_fp16: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    h_even = (int(img_hwc_fp16.shape[0]) // 2) * 2
    w_even = (int(img_hwc_fp16.shape[1]) // 2) * 2
    x = img_hwc_fp16[:h_even, :w_even].to(dtype=dtype)
    out = (
        x[0::2, 0::2]
        + x[1::2, 0::2]
        + x[0::2, 1::2]
        + x[1::2, 1::2]
    ) * 0.25
    return out.half()


def _stitched_2x_context(path: str, tile_grid: TileGrid, source_h: int, source_w: int, tile_h: int, tile_w: int) -> torch.Tensor:
    target_long = 1024
    source_long = max(source_h, source_w)
    if source_long != target_long * 2 or tile_h != tile_w:
        raise ValueError("stitched 2x context benchmark currently expects a 2048-long source and square native tiles")

    target_h = int(round(source_h * (target_long / float(source_long))))
    target_w = int(round(source_w * (target_long / float(source_long))))
    context_h = target_h * 2
    context_w = target_w * 2
    tx0, _, ty0, _ = tile_grid
    context_y0 = min(max(0, ty0 * tile_h), max(0, source_h - context_h))
    context_x0 = min(max(0, tx0 * tile_w), max(0, source_w - context_w))
    half_h = context_h // 2
    half_w = context_w // 2

    def grid_for(y0: int, x0: int, y1: int, x1: int) -> TileGrid:
        return (
            x0 // tile_w,
            max(x0 // tile_w, (x1 - 1) // tile_w),
            y0 // tile_h,
            max(y0 // tile_h, (y1 - 1) // tile_h),
        )

    specs = []
    grids: List[TileGrid] = []
    for y_off in (0, half_h):
        for x_off in (0, half_w):
            y0 = context_y0 + y_off
            x0 = context_x0 + x_off
            y1 = min(source_h, y0 + half_h)
            x1 = min(source_w, x0 + half_w)
            local_y0 = y0 - (y0 // tile_h) * tile_h
            local_x0 = x0 - (x0 // tile_w) * tile_w
            specs.append((y_off, x_off, y1 - y0, x1 - x0, local_y0, local_x0, len(grids)))
            grids.append(grid_for(y0, x0, y1, x1))

    pieces = _read_exr_hwc_tiles(path, grids)
    canvas: Optional[torch.Tensor] = None
    for y_off, x_off, crop_h, crop_w, local_y0, local_x0, piece_idx in specs:
        piece = pieces[piece_idx]
        crop = piece[local_y0 : local_y0 + crop_h, local_x0 : local_x0 + crop_w]
        if canvas is None:
            canvas = torch.empty((context_h, context_w, int(crop.shape[-1])), dtype=crop.dtype)
        canvas[y_off : y_off + crop.shape[0], x_off : x_off + crop.shape[1]] = crop

    assert canvas is not None
    return _box2x(canvas, torch.float16)


def _timeit(label: str, repeats: int, fn) -> Tuple[str, List[float]]:
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        if isinstance(result, torch.Tensor):
            _ = result.shape
        times.append((time.perf_counter() - t0) * 1000.0)
    return label, times


def _summary(label: str, times: Sequence[float]) -> str:
    mean = statistics.fmean(times)
    best = min(times)
    worst = max(times)
    return f"{label:30s} mean={mean:8.2f}ms best={best:8.2f}ms worst={worst:8.2f}ms"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--exr", type=Path)
    source.add_argument("--tar", type=Path)
    parser.add_argument("--member", help="EXR member name inside --tar")
    parser.add_argument("--downscaled-exr", type=Path, help="Optional real 1024 EXR sidecar to time.")
    parser.add_argument("--tile-grid", help="tx0,tx1,ty0,ty1. Overrides --tile-pixels.")
    parser.add_argument("--tile-pixels", type=int, default=1024, help="Pixel span for the tile-load timing.")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--openexr-threads", type=int, default=0)
    args = parser.parse_args()

    if args.tar is not None and not args.member:
        parser.error("--member is required with --tar")
    if hasattr(OpenEXR, "setGlobalThreadCount"):
        OpenEXR.setGlobalThreadCount(max(0, int(args.openexr_threads)))

    repeats = max(1, int(args.repeats))
    warmup = max(0, int(args.warmup))

    with _source_path(args) as header_path:
        h, w, tile_h, tile_w = _read_header(header_path)
    all_tiles = _full_grid(h, w, tile_h, tile_w)
    if args.tile_grid:
        tile_grid = _parse_grid(args.tile_grid)
    else:
        span_y = max(1, (max(1, int(args.tile_pixels)) + tile_h - 1) // tile_h)
        span_x = max(1, (max(1, int(args.tile_pixels)) + tile_w - 1) // tile_w)
        tile_grid = (0, min(all_tiles[1], span_x - 1), 0, min(all_tiles[3], span_y - 1))
    neighbor_tiles = _neighbor_grids(tile_grid, all_tiles)

    def load_tile() -> torch.Tensor:
        with _source_path(args) as path:
            return _read_exr_hwc(path, tile_grid)

    def load_full() -> torch.Tensor:
        with _source_path(args) as path:
            return _read_exr_hwc(path, all_tiles)

    def load_neighbors() -> List[torch.Tensor]:
        out = []
        for grid in neighbor_tiles:
            with _source_path(args) as path:
                out.append(_read_exr_hwc(path, grid))
        return out

    def load_stitched_2x() -> torch.Tensor:
        with _source_path(args) as path:
            return _stitched_2x_context(path, tile_grid, h, w, tile_h, tile_w)

    for _ in range(warmup):
        full = load_full()
        _ = load_tile()
        _ = _bilinear_to_1024(full)
        _ = _box2x(full, torch.float32)
        _ = _box2x(full, torch.float16)

    full_img = load_full()
    benches = [
        _timeit("load 1024 tile", repeats, load_tile),
        _timeit("load full tiled EXR", repeats, load_full),
        _timeit("bilinear full -> 1024", repeats, lambda: _bilinear_to_1024(full_img)),
        _timeit("box2x CPU fp32 accum", repeats, lambda: _box2x(full_img, torch.float32)),
        _timeit("box2x CPU fp16 accum", repeats, lambda: _box2x(full_img, torch.float16)),
    ]
    if max(h, w) == 2048:
        benches.append(_timeit("stitched 2x2 load+box", repeats, load_stitched_2x))
    if neighbor_tiles:
        benches.append((_timeit(f"load {len(neighbor_tiles)} neighbor tiles", repeats, load_neighbors)))
    if args.downscaled_exr is not None:
        sidecar_args = argparse.Namespace(exr=args.downscaled_exr, tar=None, member=None)
        benches.append(_timeit("load real 1024 sidecar", repeats, lambda: _read_exr_hwc(str(sidecar_args.exr), None)))

    print(
        f"source_hw={h}x{w} native_tile={tile_h}x{tile_w} "
        f"tile_grid={tile_grid} full_grid={all_tiles} neighbor_grids={neighbor_tiles}"
    )
    for label, times in benches:
        print(_summary(label, times))


if __name__ == "__main__":
    main()
