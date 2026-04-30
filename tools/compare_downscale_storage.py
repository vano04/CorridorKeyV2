#!/usr/bin/env python3
"""Compare original EXR member sizes with saved 1024 downscale sidecars."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tarfile
import tempfile
from typing import Dict, Sequence

import numpy as np
import OpenEXR  # type: ignore[import-not-found]
import torch


def _read_channels(path: Path) -> Dict[str, np.ndarray]:
    with OpenEXR.File(str(path), separate_channels=True) as exr:
        return {name: np.ascontiguousarray(ch.pixels) for name, ch in exr.parts[0].channels.items()}


def _tile_description(tile_size: int):
    td = OpenEXR.TileDescription()
    td.xSize = int(tile_size)
    td.ySize = int(tile_size)
    td.mode = OpenEXR.LevelMode.ONE_LEVEL
    td.roundingMode = OpenEXR.LevelRoundingMode.ROUND_DOWN
    return td


def _write_dwab(path: Path, channels: Dict[str, np.ndarray], tile_size: int) -> None:
    header = {
        "type": OpenEXR.tiledimage,
        "tiles": _tile_description(tile_size),
        "compression": OpenEXR.Compression.DWAB_COMPRESSION,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    OpenEXR.File([OpenEXR.Part(header, channels)]).write(str(path))


def _box2x_channel(array: np.ndarray, dtype: torch.dtype) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    h_even = (int(tensor.shape[0]) // 2) * 2
    w_even = (int(tensor.shape[1]) // 2) * 2
    x = tensor[:h_even, :w_even].to(dtype=dtype)
    out = (
        x[0::2, 0::2]
        + x[1::2, 0::2]
        + x[0::2, 1::2]
        + x[1::2, 1::2]
    ) * 0.25
    return np.ascontiguousarray(out.to(dtype=tensor.dtype).numpy())


def _member_name(clip: str, modality: str, frame: str) -> str:
    return f"{clip}.{modality.lower()}.{frame}.exr"


def _fmt_bytes(n: int) -> str:
    return f"{n / (1024 * 1024):8.2f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tar", type=Path, required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--frame", default="01")
    parser.add_argument("--modalities", nargs="+", default=["input", "fg", "alpha"])
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--output-dir", type=Path, default=Path(tempfile.gettempdir()) / "corridorkey_downscale_storage")
    args = parser.parse_args()

    accum_dtype = torch.float32 if args.dtype == "fp32" else torch.float16
    tmp_dir = args.output_dir / f"{args.clip}.{args.frame}.{args.dtype}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with tarfile.open(args.tar, mode="r:") as archive:
        for modality in args.modalities:
            member_name = _member_name(args.clip, modality, args.frame)
            info = archive.getmember(member_name)
            extracted = archive.extractfile(info)
            if extracted is None:
                raise FileNotFoundError(member_name)
            src_path = tmp_dir / member_name
            with src_path.open("wb") as handle:
                handle.write(extracted.read())

            channels = _read_channels(src_path)
            downscaled = {
                name: _box2x_channel(array, accum_dtype)
                for name, array in channels.items()
            }
            dst_path = tmp_dir / f"global_1024.{modality.lower()}.{args.frame}.exr"
            _write_dwab(dst_path, downscaled, tile_size=args.tile_size)
            rows.append((modality, int(info.size), int(dst_path.stat().st_size), len(channels), next(iter(channels.values())).shape))

    orig_total = sum(row[1] for row in rows)
    down_total = sum(row[2] for row in rows)
    print(f"tar={args.tar}")
    print(f"clip={args.clip} frame={args.frame} accum={args.dtype} output_dir={tmp_dir}")
    print("modality channels source_hw     original      down1024       ratio")
    for modality, orig, down, channels, shape in rows:
        ratio = down / float(orig)
        print(f"{modality:8s} {channels:8d} {shape[0]}x{shape[1]:<8d} {_fmt_bytes(orig)} {_fmt_bytes(down)} {ratio:8.3f}")
    print(f"{'TOTAL':8s} {'':8s} {'':13s} {_fmt_bytes(orig_total)} {_fmt_bytes(down_total)} {down_total / float(orig_total):8.3f}")
    print(f"dataset_growth_if_added={down_total / float(orig_total):.3%} for these modalities")


if __name__ == "__main__":
    main()
