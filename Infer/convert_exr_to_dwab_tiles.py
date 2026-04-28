from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

try:
    import OpenEXR  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - import error is surfaced in main.
    OpenEXR = None  # type: ignore[assignment]
    _OPENEXR_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _OPENEXR_IMPORT_ERROR = None


_HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ConversionResult:
    src: Path
    dst: Path
    channels: int
    width: int
    height: int
    bytes_written: int


def _require_openexr() -> None:
    if OpenEXR is None:
        raise RuntimeError(
            "OpenEXR is required for EXR conversion. Run with the project venv "
            "(.venv/bin/python) or install the OpenEXR Python bindings."
        ) from _OPENEXR_IMPORT_ERROR


def _set_openexr_threads(n_threads: int) -> int:
    _require_openexr()
    n = max(0, int(n_threads))
    if hasattr(OpenEXR, "setGlobalThreadCount") and hasattr(OpenEXR, "globalThreadCount"):
        OpenEXR.setGlobalThreadCount(n)
        return int(OpenEXR.globalThreadCount())
    return 0


def _tile_description(tile_size: int):
    td = OpenEXR.TileDescription()
    td.xSize = int(tile_size)
    td.ySize = int(tile_size)
    td.mode = OpenEXR.LevelMode.ONE_LEVEL
    td.roundingMode = OpenEXR.LevelRoundingMode.ROUND_DOWN
    return td


def _read_channels(path: Path) -> Dict[str, np.ndarray]:
    with OpenEXR.File(str(path), separate_channels=True) as exr:
        part = exr.parts[0]
        return {name: np.ascontiguousarray(ch.pixels) for name, ch in part.channels.items()}


def _base_header(tile_size: int) -> Dict[str, object]:
    return {
        "type": OpenEXR.tiledimage,
        "tiles": _tile_description(tile_size),
        "compression": OpenEXR.Compression.DWAB_COMPRESSION,
    }


def _verify_output(path: Path, tile_size: int) -> None:
    with OpenEXR.File(str(path), header_only=True) as exr:
        header = exr.header()
        if header.get("type") != OpenEXR.tiledimage:
            raise RuntimeError(f"{path} was not written as OpenEXR.tiledimage")
        if header.get("compression") != OpenEXR.Compression.DWAB_COMPRESSION:
            raise RuntimeError(f"{path} was not written with DWAB compression")
        tiles = header.get("tiles")
        if tiles is None or int(tiles.xSize) != tile_size or int(tiles.ySize) != tile_size:
            raise RuntimeError(f"{path} tile size is not {tile_size}x{tile_size}")


def convert_one(src: Path, dst: Path, tile_size: int, overwrite: bool, verify: bool) -> ConversionResult:
    _require_openexr()
    if src.suffix.lower() != ".exr":
        raise ValueError(f"Expected .exr input, got {src}")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Output exists, pass --overwrite to replace: {dst}")

    channels = _read_channels(src)
    if not channels:
        raise RuntimeError(f"{src} has no readable channels")

    first = next(iter(channels.values()))
    if first.ndim != 2:
        raise ValueError(f"{src} channel arrays must be 2D, got {first.shape}")
    height, width = int(first.shape[0]), int(first.shape[1])
    for name, pixels in channels.items():
        if pixels.shape != first.shape:
            raise ValueError(f"{src} channel {name} has shape {pixels.shape}, expected {first.shape}")

    dst.parent.mkdir(parents=True, exist_ok=True)
    part = OpenEXR.Part(_base_header(tile_size), channels)
    OpenEXR.File([part]).write(str(dst))
    if verify:
        _verify_output(dst, tile_size)

    return ConversionResult(
        src=src,
        dst=dst,
        channels=len(channels),
        width=width,
        height=height,
        bytes_written=dst.stat().st_size,
    )


def _find_inputs(input_path: Path, recursive: bool, glob_pattern: str) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".exr":
            raise ValueError(f"Input file is not .exr: {input_path}")
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    iterator = input_path.rglob(glob_pattern) if recursive else input_path.glob(glob_pattern)
    files = sorted(p for p in iterator if p.is_file() and p.suffix.lower() == ".exr")
    if not files:
        raise FileNotFoundError(f"No .exr files found under {input_path}")
    return files


def _output_for(src: Path, input_root: Path, output_root: Path, preserve_tree: bool) -> Path:
    if input_root.is_file():
        if output_root.suffix.lower() == ".exr":
            return output_root
        return output_root / src.name
    if preserve_tree:
        return output_root / src.relative_to(input_root)
    return output_root / src.name


def _default_output(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path.with_name(f"{input_path.stem}.dwab1024.exr")
    return input_path.with_name(f"{input_path.name}_dwab1024")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert EXR files to native 1024x1024 tiled DWAB OpenEXR files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_HERE / "corridor_greenscreen_demo",
        help="Input .exr file or directory. Defaults to Infer/corridor_greenscreen_demo.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output .exr file or directory.")
    parser.add_argument("--glob", default="*.exr", help="Glob used when --input is a directory.")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--preserve-tree", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) // 2)))
    parser.add_argument("--exr-internal-threads", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=0, help="Optional number of files for smoke tests.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.tile_size <= 0:
        raise ValueError("--tile-size must be positive")

    _set_openexr_threads(args.exr_internal_threads)
    input_path = args.input.resolve()
    output_root = (args.output or _default_output(input_path)).resolve()
    inputs = _find_inputs(input_path, recursive=args.recursive, glob_pattern=args.glob)
    if args.limit > 0:
        inputs = inputs[: args.limit]

    print(
        f"[convert] files={len(inputs)} tile={args.tile_size} workers={args.workers} "
        f"output={output_root}",
        flush=True,
    )

    results: List[ConversionResult] = []
    if args.workers <= 1:
        for src in inputs:
            dst = _output_for(src, input_path, output_root, args.preserve_tree)
            result = convert_one(src, dst, args.tile_size, args.overwrite, args.verify)
            results.append(result)
            print(f"[ok] {src} -> {dst}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=max(1, int(args.workers))) as pool:
            futures = {}
            for src in inputs:
                dst = _output_for(src, input_path, output_root, args.preserve_tree)
                futures[pool.submit(convert_one, src, dst, args.tile_size, args.overwrite, args.verify)] = (src, dst)
            for future in as_completed(futures):
                src, dst = futures[future]
                result = future.result()
                results.append(result)
                print(f"[ok] {src} -> {dst}", flush=True)

    total_bytes = sum(item.bytes_written for item in results)
    print(f"[done] converted={len(results)} bytes={total_bytes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
