"""CorridorKeyV2 inference CLI.

Wraps the V3 three-branch model with the tiled temporal inference engine.
Replaces the V1/V2 ``Infer/inference.py`` model builder with
``build_v3_hybrid_video_matting_model`` while reusing the existing EXR
I/O, tile management, temporal windowing, and output-writing infrastructure.

Usage:
    python infer.py \
        --checkpoint runs/v3_hybrid_1024/checkpoint_best.pt \
        --input-dir Infer/corridor_greenscreen_demo_dwab1024/Input \
        --alpha-dir Infer/corridor_greenscreen_demo_dwab1024/Alpha \
        --output-dir V3_output \
        --tile-size 1024 --temporal-frames 4

The model's ``build_global_context`` method is expected to be adapted for V3
inference. For now, V3 inference runs in the "no global context" fallback path
unless ``--global-context`` is enabled and the checkpoint exposes
``build_global_context()``.
"""
from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE
_REPO_ROOT = _HERE.parent
for _path in (str(_PROJECT_ROOT), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Import from the existing inference engine — reuse all the I/O infrastructure
from Infer.inference import (
    EngineKnobs,
    GlobalContextWindow,
    Tile,
    TiledExrSequence,
    _assert_same_tiled_layout,
    _autocast_context,
    _axis_starts,
    _checkpoint_state_dict,
    _default_alpha_dir,
    _default_input_dir,
    _global_context_hw,
    _iter_window_tiles_sync,
    _normalize_compile_prefix_for_target,
    _read_alpha_tile,
    _read_tiled_exr_info,
    _resolve_config,
    _resolve_dtype,
    _resize_alpha_to,
    _resize_chw_to,
    _save_outputs,
    _set_openexr_threads,
    _sorted_exr_files,
    _temporal_starts,
    _tile_weight,
    _write_comp_video,
    WindowTilePrefetcher,
    run_tiled_temporal_inference,
)

from models import build_v3_hybrid_video_matting_model
from models.v3_hybrid_matting import V3InferenceOptions


_EPS = 1e-6


def _load_v3_model(
    checkpoint_path: Path,
    cfg: Dict[str, Any],
    device: torch.device,
    checkpoint: Optional[Any] = None,
) -> torch.nn.Module:
    """Load a V3 model from a checkpoint."""
    model_cfg = dict(cfg.get("model", {}))
    model = build_v3_hybrid_video_matting_model(model_cfg).to(device)
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _normalize_compile_prefix_for_target(
        _checkpoint_state_dict(checkpoint), model
    )
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _run_v3_global_context(
    *,
    model: torch.nn.Module,
    sequence: TiledExrSequence,
    window_start: int,
    seed_alpha_cpu: Tensor,
    temporal_frames: int,
    global_long_side_cap: int,
    amp: bool,
    amp_dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    """Compute global context tokens for a temporal window.

    Returns: [B=1, M, C] global tokens.
    """
    global_hw = _global_context_hw(sequence.info, global_long_side_cap)
    full_frame = Tile(y0=0, y1=sequence.info.height, x0=0, x1=sequence.info.width)
    video_full, _, _ = sequence.read_window_tile(
        start=window_start,
        window=temporal_frames,
        tile=full_frame,
    )
    video_global = torch.stack(
        [_resize_chw_to(frame, global_hw) for frame in video_full], dim=0
    )
    seed_global = _resize_alpha_to(seed_alpha_cpu, global_hw)

    video_dev = video_global.unsqueeze(0).to(device=device, dtype=torch.float32)
    seed_dev = seed_global.unsqueeze(0).to(device=device, dtype=torch.float32)

    with _autocast_context(device, amp, amp_dtype):
        coarse_bc = torch.zeros(
            1, temporal_frames, 1, global_hw[0], global_hw[1],
            device=device, dtype=video_dev.dtype,
        )
        coarse_bc[:, 0] = seed_dev

        green_priors = None
        if hasattr(model, "use_green_priors") and model.use_green_priors:
            from models.v3_hybrid_matting import _green_excess_map, _chroma_distance_map
            ge = _green_excess_map(video_dev)
            cd = _chroma_distance_map(video_dev)
            green_priors = torch.cat([ge, cd], dim=2)

        global_tokens, _ = model.global_context(
            video_rgb_global=video_dev,
            coarse_alpha_global=coarse_bc,
            green_priors_global=green_priors,
        )

    return global_tokens


@torch.inference_mode()
def run_v3_tiled_inference(
    *,
    model: torch.nn.Module,
    sequence: TiledExrSequence,
    initial_alpha_cpu: Tensor,
    tile_size: int = 1024,
    tile_overlap: int = 64,
    temporal_frames: int = 4,
    temporal_stride: int = 1,
    use_global_context: bool = True,
    global_long_side_cap: int = 512,
    amp: bool = True,
    amp_dtype: torch.dtype = torch.bfloat16,
    device: torch.device = torch.device("cuda"),
) -> Tuple[Tensor, Tensor]:
    """Run V3 tiled temporal inference.

    Returns: (alpha_pred, fg_pred) both [N_frames, C, H, W] on CPU.
    """
    n_frames = len(sequence.paths)
    h = sequence.info.height
    w = sequence.info.width
    alpha_hint = _resize_alpha_to(initial_alpha_cpu, (h, w))
    starts = _temporal_starts(n_frames, temporal_frames, temporal_stride)
    if not starts:
        raise ValueError("No frames to process")

    h_pad = max(h, tile_size)
    w_pad = max(w, tile_size)
    alpha_sum = torch.zeros((n_frames, 1, h, w), dtype=torch.float32)
    fg_sum = torch.zeros((n_frames, 3, h, w), dtype=torch.float32)
    count = torch.zeros((n_frames, 1, 1, 1), dtype=torch.float32)

    carry_seed = alpha_hint

    y_starts = _axis_starts(h_pad, tile_size, tile_overlap)
    x_starts = _axis_starts(w_pad, tile_size, tile_overlap)
    tiles = [
        Tile(y0=y, y1=y + tile_size, x0=x, x1=x + tile_size)
        for y in y_starts for x in x_starts
    ]

    for start_i, start in enumerate(starts):
        actual = min(n_frames - start, temporal_frames)

        # Compute global context if enabled
        global_tokens = None
        if use_global_context:
            t0 = time.perf_counter()
            global_tokens = _run_v3_global_context(
                model=model,
                sequence=sequence,
                window_start=start,
                seed_alpha_cpu=carry_seed,
                temporal_frames=temporal_frames,
                global_long_side_cap=global_long_side_cap,
                amp=amp,
                amp_dtype=amp_dtype,
                device=device,
            )
            dt_global = time.perf_counter() - t0
        else:
            dt_global = 0.0

        # Process tiles
        t0_tiles = time.perf_counter()
        alpha_acc = torch.zeros((temporal_frames, 1, h_pad, w_pad), dtype=torch.float32, device="cpu")
        fg_acc = torch.zeros((temporal_frames, 3, h_pad, w_pad), dtype=torch.float32, device="cpu")
        weight_acc = torch.zeros((temporal_frames, 1, h_pad, w_pad), dtype=torch.float32, device="cpu")

        for tile in tiles:
            video_tile, _, valid_mask_cpu = sequence.read_window_tile(
                start=start, window=temporal_frames, tile=tile,
            )
            seed_tile = initial_alpha_cpu[:1, :tile.y1 - tile.y0, :tile.x1 - tile.x0]
            if seed_tile.shape[-2:] != (tile.y1 - tile.y0, tile.x1 - tile.x0):
                from Infer.inference import _crop_alpha_tile
                seed_tile = _crop_alpha_tile(carry_seed, tile)

            video_dev = video_tile.unsqueeze(0).to(device=device, dtype=torch.float32)
            seed_dev = seed_tile.unsqueeze(0).to(device=device, dtype=torch.float32)

            tile_coords = torch.tensor(
                [[float(tile.y0), float(tile.y1), float(tile.x0), float(tile.x1)]],
                device=device, dtype=torch.float32,
            )
            source_hw = torch.tensor(
                [[float(h), float(w)]],
                device=device, dtype=torch.float32,
            )

            with _autocast_context(device, amp, amp_dtype):
                out = model(
                    video=video_dev,
                    coarse_alpha_init=seed_dev,
                    global_tokens=global_tokens,
                    tile_coords=tile_coords,
                    source_hw=source_hw,
                )

            alpha_tile = out["alpha_pred"][0].detach().cpu().clamp(0.0, 1.0).to(torch.float32)
            fg_tile = out["fg_pred"][0].detach().cpu().to(torch.float32)
            weight = _tile_weight(tile, h_pad, w_pad, tile_overlap, torch.device("cpu"))

            alpha_acc[:, :, tile.y0:tile.y1, tile.x0:tile.x1] += alpha_tile * weight
            fg_acc[:, :, tile.y0:tile.y1, tile.x0:tile.x1] += fg_tile * weight
            weight_acc[:, :, tile.y0:tile.y1, tile.x0:tile.x1] += weight

            del video_dev, seed_dev, out

        dt_tiles = time.perf_counter() - t0_tiles

        weight_acc = weight_acc.clamp_min(_EPS)
        alpha_win = (alpha_acc / weight_acc)[:actual, :, :h, :w]
        fg_win = (fg_acc / weight_acc)[:actual, :, :h, :w]

        for local_t in range(actual):
            global_t = start + local_t
            alpha_sum[global_t] += alpha_win[local_t]
            fg_sum[global_t] += fg_win[local_t]
            count[global_t] += 1.0

        if start_i + 1 < len(starts):
            next_start = starts[start_i + 1]
            if start <= next_start < start + actual:
                carry_seed = alpha_win[next_start - start].clone()
            else:
                carry_seed = alpha_win[actual - 1].clone()

        print(
            f"[v3-infer] window {start_i + 1}/{len(starts)} "
            f"frames {start:05d}-{start + actual - 1:05d} "
            f"global={dt_global:.2f}s tiles={dt_tiles:.2f}s "
            f"({len(tiles)} tiles)",
            flush=True,
        )

    count = count.clamp_min(1.0)
    return (alpha_sum / count).clamp(0.0, 1.0), fg_sum / count


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="V3 hybrid video matting inference with tiled temporal windows."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="V3 model checkpoint .pt")
    parser.add_argument("--config", type=Path, default=None, help="YAML config override")
    parser.add_argument("--input-dir", type=Path, default=_default_input_dir())
    parser.add_argument("--alpha-dir", type=Path, default=_default_alpha_dir())
    parser.add_argument("--output-dir", type=Path, default=_HERE / "v3_output")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--tile-overlap", type=int, default=64)
    parser.add_argument("--temporal-frames", type=int, default=4)
    parser.add_argument("--temporal-stride", type=int, default=1)
    parser.add_argument("--global-context", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--global-long-side-cap", type=int, default=512)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fg-source", choices=("model", "input"), default="model")
    parser.add_argument("--make-video", action="store_true")
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--exr-decode-threads", type=int, default=4)
    parser.add_argument("--exr-internal-threads", type=int, default=0)
    parser.add_argument("--write-threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _set_openexr_threads(args.exr_internal_threads)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    print(f"[v3-model] loading checkpoint {args.checkpoint}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg, cfg_source = _resolve_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
    )
    print(f"[v3-model] config: {cfg_source}", flush=True)

    model = _load_v3_model(args.checkpoint, cfg, device, checkpoint=checkpoint)
    del checkpoint
    if args.compile_model:
        model = torch.compile(model)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    input_paths = _sorted_exr_files(args.input_dir)
    if args.limit > 0:
        input_paths = input_paths[:args.limit]
    alpha_paths = _sorted_exr_files(args.alpha_dir)
    alpha_path = alpha_paths[0]

    print(f"[v3-load] frames={len(input_paths)} input_dir={args.input_dir}", flush=True)
    sequence = TiledExrSequence.from_paths(input_paths, decode_workers=args.exr_decode_threads)
    _assert_same_tiled_layout(alpha_path, sequence.info)
    full_frame = Tile(y0=0, y1=sequence.info.height, x0=0, x1=sequence.info.width)
    alpha_cpu = _read_alpha_tile(alpha_path, sequence.info, full_frame)

    amp_dtype = _resolve_dtype(args.amp_dtype)
    amp = bool(args.amp and amp_dtype is not torch.float32)

    data_cfg = dict(cfg.get("data", {}))
    fg_representation = str(
        cfg.get("model", {}).get("fg_representation",
        data_cfg.get("fg_representation", "premul"))
    )

    print(
        f"[v3-infer] tile={args.tile_size} overlap={args.tile_overlap} "
        f"temporal={args.temporal_frames} stride={args.temporal_stride} "
        f"global_context={args.global_context} "
        f"global_cap={args.global_long_side_cap} "
        f"amp={amp_dtype if amp else 'off'}",
        flush=True,
    )

    alpha_pred, fg_pred = run_v3_tiled_inference(
        model=model,
        sequence=sequence,
        initial_alpha_cpu=alpha_cpu,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        temporal_frames=args.temporal_frames,
        temporal_stride=args.temporal_stride,
        use_global_context=args.global_context,
        global_long_side_cap=args.global_long_side_cap,
        amp=amp,
        amp_dtype=amp_dtype,
        device=device,
    )

    visible_rgb_paths = input_paths
    visible_rgb_info = sequence.info

    print(f"[v3-write] {args.output_dir}", flush=True)
    _save_outputs(
        alpha_pred=alpha_pred,
        fg_pred=fg_pred,
        input_paths=visible_rgb_paths,
        input_info=visible_rgb_info,
        output_dir=args.output_dir,
        fg_representation=fg_representation,
        fg_source=args.fg_source,
        workers=args.write_threads,
    )

    if args.make_video:
        video_path = args.output_dir / "comp.mp4"
        _write_comp_video(args.output_dir / "Comp", video_path, fps=args.fps)
        print(f"[v3-write] {video_path}", flush=True)

    print("[v3-infer] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
