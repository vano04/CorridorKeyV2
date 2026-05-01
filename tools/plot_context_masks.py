#!/usr/bin/env python
"""Plot the full-resolution V3 global Alpha/FG context tensors.

This visualizes the tensors used by ``infer.py`` to build global context:

    alpha_context_mask = seed_global
    fg_context_rgb     = global FG guidance tensor

With the default ``--global-fg-guidance masked-input`` mode, the FG guidance is
``video_global * seed_global``. The RGB tensor is frame-dependent; the seed mask
may be reused across frames during inference. No local tile/refiner inference is
run here.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Infer.inference import (  # noqa: E402
    Tile,
    TiledExrSequence,
    _assert_same_tiled_layout,
    _default_alpha_dir,
    _default_input_dir,
    _read_alpha_tile,
    _resolve_config,
    _resize_alpha_to,
    _set_openexr_threads,
    _sorted_exr_files,
)
from infer import (  # noqa: E402
    _build_pseudo_global_fg_guidance,
    _global_context_square_side,
    _letterbox_chw_to_square,
)


def _plot_mask(ax: plt.Axes, mask: Tensor, title: str, cmap: str) -> None:
    im = ax.imshow(mask.squeeze().cpu().numpy(), vmin=0.0, vmax=1.0, cmap=cmap)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _save_single(mask: Tensor, path: Path, cmap: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    _plot_mask(ax, mask, path.stem, cmap)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _show_rgb(ax: plt.Axes, rgb_chw: Tensor, title: str) -> None:
    rgb = rgb_chw.detach().float().clamp(0.0, 1.0).permute(1, 2, 0).cpu().numpy()
    ax.imshow(rgb)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _save_rgb(rgb_chw: Tensor, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    _show_rgb(ax, rgb_chw, path.stem)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot full-res global Alpha/FG context tensors."
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint_interrupted_step015137.pt"))
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--input-dir", type=Path, default=_default_input_dir())
    parser.add_argument("--alpha-dir", type=Path, default=_default_alpha_dir())
    parser.add_argument("--output", type=Path, default=Path("context_masks_step015137.png"))
    parser.add_argument("--save-separate", action="store_true")
    parser.add_argument(
        "--fg-context-only",
        action="store_true",
        help="Only plot/save the actual global FG guidance RGB tensor.",
    )
    parser.add_argument(
        "--absolute-frame",
        type=int,
        default=None,
        help="Absolute input frame index to visualize. Overrides --window-start/--frame-index.",
    )
    parser.add_argument("--window-start", type=int, default=0)
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--temporal-frames", type=int, default=4)
    parser.add_argument("--global-long-side-cap", type=int, default=0)
    parser.add_argument(
        "--global-fg-guidance",
        choices=("masked-input", "input", "none"),
        default="masked-input",
    )
    parser.add_argument("--exr-decode-threads", type=int, default=4)
    parser.add_argument("--exr-internal-threads", type=int, default=0)
    return parser.parse_args(argv)


@torch.inference_mode()
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _set_openexr_threads(args.exr_internal_threads)
    if args.absolute_frame is not None:
        args.window_start = int(args.absolute_frame)
        args.frame_index = 0

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg, cfg_source = _resolve_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
    )

    input_paths = _sorted_exr_files(args.input_dir)
    alpha_paths = _sorted_exr_files(args.alpha_dir)
    if not input_paths:
        raise FileNotFoundError(f"No EXR inputs found in {args.input_dir}")
    if not alpha_paths:
        raise FileNotFoundError(f"No EXR alpha hints found in {args.alpha_dir}")

    sequence = TiledExrSequence.from_paths(input_paths, decode_workers=args.exr_decode_threads)
    _assert_same_tiled_layout(alpha_paths[0], sequence.info)
    full_frame = Tile(y0=0, y1=sequence.info.height, x0=0, x1=sequence.info.width)
    alpha_hint = _resize_alpha_to(
        _read_alpha_tile(alpha_paths[0], sequence.info, full_frame),
        (sequence.info.height, sequence.info.width),
    )

    data_cfg = dict(cfg.get("data", {}))
    train_cfg = dict(cfg.get("train", {}))
    inference_cfg = dict(cfg.get("inference", {}))
    global_long_side = int(
        args.global_long_side_cap
        or data_cfg.get(
            "global_context_long_side",
            train_cfg.get("global_long_side_cap", inference_cfg.get("global_long_side_cap", 1024)),
        )
    )
    global_side = _global_context_square_side(sequence, global_long_side)
    video_full, _, _ = sequence.read_window_tile(
        start=args.window_start,
        window=args.temporal_frames,
        tile=full_frame,
    )
    video_global = torch.stack(
        [_letterbox_chw_to_square(frame, global_side, fill=0.0) for frame in video_full],
        dim=0,
    )
    seed_global = _letterbox_chw_to_square(
        alpha_hint,
        global_side,
        fill=0.0,
        clamp=(0.0, 1.0),
    )
    fg_guidance_global = _build_pseudo_global_fg_guidance(
        video_global,
        seed_global,
        args.global_fg_guidance,
    )

    alpha_context_mask = seed_global.unsqueeze(0).expand(args.temporal_frames, -1, -1, -1)
    if fg_guidance_global is None:
        fg_context_mask = torch.zeros_like(alpha_context_mask)
        fg_context_rgb = torch.zeros_like(video_global)
    elif args.global_fg_guidance == "input":
        fg_context_mask = (video_global.abs().amax(dim=1, keepdim=True) > 0).to(torch.float32)
        fg_context_rgb = fg_guidance_global
    else:
        fg_context_mask = seed_global.unsqueeze(0).expand(args.temporal_frames, -1, -1, -1)
        fg_context_rgb = fg_guidance_global

    frame = int(args.frame_index)
    if frame < 0 or frame >= alpha_context_mask.shape[0]:
        raise IndexError(f"--frame-index must be in [0, {alpha_context_mask.shape[0] - 1}]")
    alpha_mask = alpha_context_mask[frame, 0].cpu()
    fg_mask = fg_context_mask[frame, 0].cpu()
    fg_rgb = fg_context_rgb[frame].cpu()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.fg_context_only:
        fig, axes = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        _show_rgb(axes, fg_rgb, f"FG global context RGB {tuple(fg_rgb.shape[-2:])}")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        _plot_mask(axes[0], alpha_mask, f"Alpha context mask {tuple(alpha_mask.shape)}", "magma")
        _plot_mask(axes[1], fg_mask, f"FG context mask {tuple(fg_mask.shape)}", "viridis")
        _show_rgb(axes[2], fg_rgb, f"FG global context RGB {tuple(fg_rgb.shape[-2:])}")
    fig.suptitle(
        f"{args.checkpoint.name} | {cfg_source} | global_side={global_side} | fg_guidance={args.global_fg_guidance}",
        fontsize=10,
    )
    fig.savefig(args.output, dpi=180)
    plt.close(fig)

    if args.save_separate:
        stem = args.output.with_suffix("")
        suffix = args.output.suffix or ".png"
        if not args.fg_context_only:
            _save_single(alpha_mask, stem.with_name(stem.name + "_alpha" + suffix), "magma")
            _save_single(fg_mask, stem.with_name(stem.name + "_fg_mask" + suffix), "viridis")
        _save_rgb(fg_rgb, stem.with_name(stem.name + "_fg_context_rgb" + suffix))

    print(f"saved {args.output}")
    print(f"alpha_context_mask_shape={tuple(alpha_mask.shape)}")
    print(f"fg_context_mask_shape={tuple(fg_mask.shape)}")
    print(f"fg_context_rgb_shape={tuple(fg_rgb.shape)}")
    print(f"absolute_frame={args.window_start + frame}")
    print(f"global_context_side={global_side}")
    print(f"global_fg_guidance={args.global_fg_guidance}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
