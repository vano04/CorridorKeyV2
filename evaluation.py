"""CorridorKeyV2 evaluation CLI.

Runs the V3 model on the WebDataset dev/validation split, which defaults to
shard 33 when no explicit validation shard list is provided.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
from torch.utils.data import DataLoader

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = str(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from CorridorKeyDataset.dataset import (  # noqa: E402
    DEFAULT_MODALITIES,
    CorridorKeyWebSequenceDataset,
    create_single_gpu_dataloader,
)
from Infer.inference import (  # noqa: E402
    _checkpoint_state_dict,
    _normalize_compile_prefix_for_target,
    _resolve_config,
)
from losses import V3MattingLossComputer  # noqa: E402
from models import build_v3_hybrid_video_matting_model  # noqa: E402
from utils import move_batch_to_device, pad_collate_video  # noqa: E402


def _resolve_data_root(data_cfg: Dict[str, Any]) -> Path:
    root = data_cfg.get("root_dir", data_cfg.get("dataset_root", "CorridorKeyDataset"))
    return Path(root)


def _parse_int_list(value: Any, default: Sequence[int]) -> list[int]:
    if value is None:
        return [int(v) for v in default]
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            return [int(v) for v in default]
        return [int(part) for part in parts]
    if isinstance(value, (list, tuple, set)):
        parsed = [int(v) for v in value]
        return parsed if parsed else [int(v) for v in default]
    return [int(value)]


def _resolve_validation_shards(data_cfg: Dict[str, Any]) -> list[int]:
    shard_cfg = data_cfg.get("validation_shard_indices", data_cfg.get("validation_shards", [33]))
    return _parse_int_list(shard_cfg, default=(33,))


def _resolve_eval_side(data_cfg: Dict[str, Any]) -> int:
    explicit = data_cfg.get("eval_resolution", data_cfg.get("eval_side", None))
    if explicit is not None:
        return max(1, int(explicit))

    fixed_crop_size = int(data_cfg.get("fixed_crop_size", 0))
    if fixed_crop_size > 0:
        return fixed_crop_size

    buckets = data_cfg.get("multi_resolution_buckets", [2048])
    if isinstance(buckets, str):
        parsed = [int(part.strip()) for part in buckets.split(",") if part.strip()]
        if parsed:
            return max(parsed)
    elif isinstance(buckets, (list, tuple, set)):
        parsed = [int(v) for v in buckets]
        if parsed:
            return max(parsed)
    return 2048


def _build_eval_transform(data_cfg: Dict[str, Any]) -> Any:
    from utils.data import CorridorMattingTransform

    eval_side = _resolve_eval_side(data_cfg)
    clip_len = max(1, int(data_cfg.get("clip_len_max", data_cfg.get("clip_len_min", 4))))
    return CorridorMattingTransform(
        clip_len_min=clip_len,
        clip_len_max=clip_len,
        resolution_buckets=(eval_side,),
        resize_scale_min=1.0,
        resize_scale_max=1.0,
        fixed_crop_size=0,
        fg_representation=str(data_cfg.get("fg_representation", "premul")),
        horizontal_flip_p=0.0,
        background_replace_p=0.0,
        spill_augment_p=0.0,
        green_foreground_augment_p=0.0,
        temporal_jitter_p=0.0,
        skip_temporal_sample=True,
        device_offload=False,
        subject_gain_min=1.0,
        subject_gain_max=1.0,
        bg_gain_min=1.0,
        bg_gain_max=1.0,
        wb_jitter_p=0.0,
        color_jitter_p=0.0,
        noise_p=0.0,
        shot_noise_p=0.0,
        blur_p=0.0,
        motion_blur_p=0.0,
        compression_p=0.0,
        host_dtype=torch.float32,
    )


def _resolve_amp_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    return torch.float32


def _all_tensor_mapping(value: Any) -> bool:
    return isinstance(value, dict) and bool(value) and all(torch.is_tensor(v) for v in value.values())


def _ema_state_dict(checkpoint: Any) -> Optional[Dict[str, torch.Tensor]]:
    if not isinstance(checkpoint, dict):
        return None

    value = checkpoint.get("model_ema_state_dict")
    if _all_tensor_mapping(value):
        return value

    value = checkpoint.get("model_ema")
    if _all_tensor_mapping(value):
        return value
    if isinstance(value, dict):
        shadow = value.get("shadow")
        if _all_tensor_mapping(shadow):
            return shadow
    return None


def _checkpoint_state_dict_for_weights(checkpoint: Any, weights: str) -> Dict[str, torch.Tensor]:
    key = str(weights).strip().lower()
    if key in {"ema", "model_ema"}:
        state_dict = _ema_state_dict(checkpoint)
        if state_dict is None:
            raise ValueError("Checkpoint does not contain EMA weights")
        return state_dict
    if key in {"model", "non_ema", "non-ema", "raw"}:
        return _checkpoint_state_dict(checkpoint)
    raise ValueError(f"Unsupported weights variant: {weights!r}")


def _load_v3_model(
    checkpoint_path: Path,
    cfg: Dict[str, Any],
    device: torch.device,
    checkpoint: Optional[Any] = None,
    *,
    weights: str = "model",
) -> torch.nn.Module:
    model_cfg = dict(cfg.get("model", {}))
    model = build_v3_hybrid_video_matting_model(model_cfg).to(device)
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _normalize_compile_prefix_for_target(_checkpoint_state_dict_for_weights(checkpoint, weights), model)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def evaluate(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    loss_fn: V3MattingLossComputer,
    device: torch.device,
    amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    totals: Dict[str, float] = OrderedDict()
    sample_count = 0

    for batch in dataloader:
        batch = move_batch_to_device(batch, device, non_blocking=False)
        batch_size = int(batch["alpha_gt"].shape[0])

        model_kwargs: Dict[str, Any] = {}
        if "global_input_gt" in batch:
            model_kwargs["global_video"] = batch["global_input_gt"]
        if "global_alpha_gt" in batch:
            global_alpha = batch["global_alpha_gt"]
            if global_alpha.ndim == 5:
                global_alpha = global_alpha[:, 0]
            model_kwargs["global_coarse_alpha_init"] = global_alpha
        if "global_fg_gt" in batch:
            model_kwargs["global_fg_guidance"] = batch["global_fg_gt"]
        if "tile_coords" in batch:
            model_kwargs["tile_coords"] = batch["tile_coords"]
        if "source_hw" in batch:
            model_kwargs["source_hw"] = batch["source_hw"]

        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda", dtype=amp_dtype):
            pred = model(
                video=batch["video_rgb"],
                coarse_alpha_init=batch["coarse_alpha_init"],
                valid_mask=batch.get("valid_mask"),
                bg_for_comp=batch["bg_gt"],
                **model_kwargs,
            )
            _, loss_items = loss_fn(pred, batch)

        for key, value in loss_items.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().to(device="cpu")) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("No samples were evaluated")

    return {key: value / sample_count for key, value in totals.items()}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a V3 checkpoint on the dev/validation split.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="V3 checkpoint .pt")
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config override")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split", default="dev", choices=("dev", "eval", "validation", "val", "train", "all"))
    parser.add_argument(
        "--validation-shards",
        default=None,
        help="Optional comma-separated validation shard indices. Defaults to config or shard 33.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=_HERE / "eval_output")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", default="bf16", choices=("bf16", "fp16", "fp32"))
    parser.add_argument(
        "--weights",
        default="both",
        choices=("model", "ema", "both"),
        help="Evaluate raw model weights, EMA weights, or both when the checkpoint contains EMA.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    print(f"[v3-eval] loading checkpoint {args.checkpoint}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg, cfg_source = _resolve_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
    )
    print(f"[v3-eval] config: {cfg_source}", flush=True)

    loss_cfg = cfg.get("loss", {})
    data_cfg = dict(cfg.get("data", {}))

    data_root = _resolve_data_root(data_cfg)
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    validation_shards = _resolve_validation_shards(data_cfg)
    if args.validation_shards is not None:
        validation_shards = _parse_int_list(args.validation_shards, default=validation_shards)

    transform = _build_eval_transform(data_cfg)
    dataset = CorridorKeyWebSequenceDataset(
        root_dir=data_root,
        sequence_length=max(1, int(data_cfg.get("clip_len_max", data_cfg.get("clip_len_min", 4)))),
        frame_stride=int(data_cfg.get("frame_stride", 1)),
        sequence_stride=int(data_cfg.get("sequence_stride", 1)),
        modalities=tuple(str(m) for m in data_cfg.get("modalities", DEFAULT_MODALITIES)),
        transform=transform,
        convert_to_float=bool(data_cfg.get("dataset_convert_to_float", True)),
        shard_glob=str(data_cfg.get("shard_glob", "*.tar")),
        manifest_filename=str(data_cfg.get("manifest_filename", "manifest.json")),
        exr_decode_threads=max(1, int(data_cfg.get("exr_decode_threads", 1))),
        split=str(args.split),
        validation_shard_indices=validation_shards,
        decode_full_frame=bool(data_cfg.get("decode_full_frame", False)),
        local_tile_span=int(data_cfg.get("local_tile_span", 4)),
        exr_tile_size=int(data_cfg.get("exr_tile_size", 256)),
        source_hw=data_cfg.get("source_hw", [2048, 2048]),
        emit_tile_metadata=bool(data_cfg.get("emit_tile_metadata", True)),
        decode_global_context=bool(data_cfg.get("decode_global_context", False)),
        global_context_root_dir=data_cfg.get("global_context_root_dir"),
        global_context_long_side=int(data_cfg.get("global_context_long_side", 0)),
        global_context_modalities=data_cfg.get("global_context_modalities", ["Input", "Alpha"]),
    )

    collate_fn = lambda batch: pad_collate_video(batch, pad_multiple=int(cfg.get("model", {}).get("patch_size", 8)))
    num_workers = args.num_workers if args.num_workers is not None else int(data_cfg.get("num_workers", 4))
    dataloader = create_single_gpu_dataloader(
        dataset=dataset,
        batch_size=max(1, int(args.batch_size)),
        num_workers=max(0, int(num_workers)),
        shuffle=False,
        drop_last=False,
        pin_memory=bool(data_cfg.get("pin_memory", True)) and device.type == "cuda",
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
        seed=int(data_cfg.get("seed", 1337)),
        collate_fn=collate_fn,
        num_torch_threads=int(data_cfg.get("num_torch_threads", 1)),
        exr_internal_threads=int(data_cfg.get("exr_internal_threads", 0)),
    )

    loss_fn = V3MattingLossComputer(
        weights=loss_cfg,
        fg_representation=str(data_cfg.get("fg_representation", cfg.get("model", {}).get("fg_representation", "premul"))),
    ).to(device)

    amp_dtype = _resolve_amp_dtype(args.amp_dtype)
    amp_enabled = bool(args.amp and device.type == "cuda" and amp_dtype is not torch.float32)

    print(
        f"[v3-eval] split={str(args.split)} shards={validation_shards} "
        f"samples={len(dataset)} batch_size={args.batch_size} device={device}",
        flush=True,
    )

    has_ema = _ema_state_dict(checkpoint) is not None
    if args.weights == "both":
        variants = ["model", "ema"] if has_ema else ["model"]
        if not has_ema:
            print("[v3-eval] checkpoint has no EMA weights; evaluating model weights only", flush=True)
    else:
        variants = [str(args.weights)]
        if args.weights == "ema" and not has_ema:
            raise ValueError("Requested --weights=ema, but checkpoint does not contain EMA weights")

    all_metrics: Dict[str, Dict[str, float]] = OrderedDict()
    for variant in variants:
        print(f"[v3-eval] evaluating weights={variant}", flush=True)
        model = _load_v3_model(
            args.checkpoint,
            cfg,
            device,
            checkpoint=checkpoint,
            weights=variant,
        )
        all_metrics[variant] = evaluate(
            model=model,
            dataloader=dataloader,
            loss_fn=loss_fn,
            device=device,
            amp=amp_enabled,
            amp_dtype=amp_dtype,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(all_metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print(f"[v3-eval] metrics saved to {metrics_path}", flush=True)
    for variant, metrics in all_metrics.items():
        variant_path = args.output_dir / f"metrics_{variant}.json"
        with variant_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2, sort_keys=True)
            handle.write("\n")
        for key in sorted(metrics):
            print(f"[v3-eval] {variant}/{key}={metrics[key]:.6f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
