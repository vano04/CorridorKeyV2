#!/usr/bin/env python3
"""Probe one real V3 train step memory footprint."""
from __future__ import annotations

import argparse
import copy
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from losses import V3MattingLossComputer  # noqa: E402
from models import V3InferenceOptions, build_v3_hybrid_video_matting_model  # noqa: E402
from training import (  # noqa: E402
    build_dataloader,
    build_optimizer_parameter_groups,
    build_temporal_chunks,
    select_device,
    set_seed,
    slice_batch_temporal,
)
from utils import build_device_transform_from_data_cfg, load_config, move_batch_to_device  # noqa: E402
from utils.ema import ModelEma  # noqa: E402


def _set_nested(cfg: Dict[str, Any], dotted: str, value: str) -> None:
    target: Dict[str, Any] = cfg
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    old = target.get(parts[-1])
    if isinstance(old, bool):
        parsed: Any = value.strip().lower() in {"1", "true", "yes", "on"}
    elif isinstance(old, int) and not isinstance(old, bool):
        parsed = int(value)
    elif isinstance(old, float):
        parsed = float(value)
    elif value.strip().lower() in {"true", "false"}:
        parsed = value.strip().lower() == "true"
    else:
        parsed = value
    target[parts[-1]] = parsed


def _bytes(obj: Any, device_type: str | None = None) -> int:
    seen: set[tuple[int, int, torch.device]] = set()
    total = 0

    def visit(value: Any) -> None:
        nonlocal total
        if torch.is_tensor(value):
            if device_type is not None and value.device.type != device_type:
                return
            storage = value.untyped_storage()
            ident = (storage.data_ptr(), storage.nbytes(), value.device)
            if ident not in seen:
                seen.add(ident)
                total += ident[1]
        elif isinstance(value, dict):
            for item in value.values():
                visit(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)

    visit(obj)
    return total


def _report(label: str, device: torch.device, batch: Dict[str, Any] | None = None) -> None:
    alloc = torch.cuda.memory_allocated(device) / (1024**3)
    reserved = torch.cuda.memory_reserved(device) / (1024**3)
    peak = torch.cuda.max_memory_allocated(device) / (1024**3)
    extra = ""
    if batch is not None:
        extra = f" batch_cuda={_bytes(batch, 'cuda') / (1024**3):.2f}GiB"
    print(f"{label:24s} alloc={alloc:6.2f}GiB reserved={reserved:6.2f}GiB peak={peak:6.2f}GiB{extra}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/v3_single_5090_fp8_alt_1024.yaml")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--override", action="append", default=[], help="Dotted override, e.g. data.batch_size=2")
    args = parser.parse_args()

    cfg = load_config(args.config)
    for override in args.override:
        key, sep, value = override.partition("=")
        if not sep:
            raise ValueError(f"Invalid override {override!r}; expected key=value")
        _set_nested(cfg, key, value)

    set_seed(args.seed)
    device = select_device(cfg["train"])
    if device.type != "cuda":
        raise RuntimeError("This probe is CUDA-only")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    print(
        "probe config: "
        f"batch_size={cfg['data'].get('batch_size')} cached_four={cfg['data'].get('cached_four_quadrant_batch')} "
        f"clip={cfg['data'].get('clip_len_min')}..{cfg['data'].get('clip_len_max')} "
        f"temporal_batch_size={cfg['train'].get('temporal_batch_size')} "
        f"ema={cfg['train'].get('ema', {}).get('enabled')} ema_device={cfg['train'].get('ema', {}).get('device')} "
        f"workers={cfg['data'].get('num_workers')} decode={cfg['data'].get('exr_decode_threads')}",
        flush=True,
    )

    dataloader = build_dataloader(cfg)
    device_transform = build_device_transform_from_data_cfg(cfg["data"], cfg.get("train", {})).to(device).eval()
    model_cfg = dict(cfg["model"])
    model_cfg.setdefault("fg_representation", str(cfg["data"].get("fg_representation", "premul")))
    model_cfg["gradient_checkpointing"] = bool(cfg["train"].get("rematerialize_activations", model_cfg.get("gradient_checkpointing", True)))
    model = build_v3_hybrid_video_matting_model(model_cfg).to(device).to(memory_format=torch.channels_last)
    _report("after model", device)

    params = [p for p in model.parameters() if p.requires_grad]
    groups, _ = build_optimizer_parameter_groups(
        model,
        cfg["train"],
        float(cfg["train"].get("lr", 1e-4)),
        float(cfg["train"].get("weight_decay", 1e-2)),
    )
    optimizer = torch.optim.AdamW(groups, lr=float(cfg["train"].get("lr", 1e-4)), fused=bool(cfg["train"].get("fused_adamw", False)))
    ema_cfg = cfg["train"].get("ema", {}) or {}
    ema_device_name = str(ema_cfg.get("device", "model")).strip().lower()
    ema_device = torch.device("cpu") if ema_device_name == "cpu" else None
    model_ema = ModelEma(model, decay=float(ema_cfg.get("decay", 0.9999)), device=ema_device) if bool(ema_cfg.get("enabled", False)) else None
    del params
    _report("after optimizer+ema", device)

    host_batch = next(iter(dataloader))
    print(f"host_batch={_bytes(host_batch) / (1024**3):.2f}GiB", flush=True)
    batch = move_batch_to_device(host_batch, device, non_blocking=True)
    torch.cuda.synchronize(device)
    _report("after h2d", device, batch)
    with torch.no_grad():
        batch = device_transform(batch)
    torch.cuda.synchronize(device)
    _report("after device xform", device, batch)

    criterion = V3MattingLossComputer(
        weights=cfg["loss"],
        fg_representation=str(cfg["data"].get("fg_representation", "premul")),
    ).to(device)
    train_opts = V3InferenceOptions(
        mode=str(cfg["train"].get("inference_mode", "hybrid")),
        global_long_side_cap=int(cfg["train"].get("global_long_side_cap", 2048)),
        tile_size=int(cfg["train"].get("tile_size", 1024)),
        tile_overlap=int(cfg["train"].get("tile_overlap", 64)),
    )

    amp_dtype = torch.bfloat16 if str(cfg["train"].get("amp_dtype", "bf16")).lower() == "bf16" else torch.float16
    amp_enabled = bool(cfg["train"].get("amp", True))
    temporal_batch_size = int(cfg["train"].get("temporal_batch_size", 0))
    chunks = build_temporal_chunks(int(batch["video_rgb"].shape[1]), temporal_batch_size)
    total_loss_value = 0.0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    coarse_seed = batch["coarse_alpha_init"]
    for chunk_idx, (start_t, end_t) in enumerate(chunks):
        chunk = slice_batch_temporal(batch, start_t, end_t)
        if chunk_idx > 0:
            chunk["coarse_alpha_init"] = coarse_seed
        with nullcontext():
            with torch.amp.autocast(device.type, enabled=amp_enabled, dtype=amp_dtype):
                kwargs: Dict[str, Any] = {}
                if "global_video_rgb" in chunk:
                    kwargs["global_video"] = chunk["global_video_rgb"]
                if "global_coarse_alpha_init" in chunk:
                    kwargs["global_coarse_alpha_init"] = chunk["global_coarse_alpha_init"]
                if "global_fg_gt" in chunk:
                    kwargs["global_fg_guidance"] = chunk["global_fg_gt"]
                if "tile_coords" in chunk:
                    kwargs["tile_coords"] = chunk["tile_coords"]
                if "source_hw" in chunk:
                    kwargs["source_hw"] = chunk["source_hw"]
                pred = model(
                    video=chunk["video_rgb"],
                    coarse_alpha_init=chunk["coarse_alpha_init"],
                    valid_mask=chunk.get("valid_mask"),
                    bg_for_comp=chunk["bg_gt"],
                    inference_options=train_opts,
                    **kwargs,
                )
                loss, _ = criterion(pred, chunk)
                loss = loss / max(1, len(chunks))
        loss.backward()
        total_loss_value += float(loss.detach().to("cpu"))
        coarse_seed = pred["alpha_pred"][:, -1].detach()
        _report(f"after backward chunk {chunk_idx + 1}", device, batch)

    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg["train"].get("grad_clip", 1.0)))
    _report("after grad clip", device, batch)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if model_ema is not None:
        model_ema.update(model)
    torch.cuda.synchronize(device)
    _report("after optim+ema", device, batch)
    print(f"loss={total_loss_value:.6f}", flush=True)


if __name__ == "__main__":
    main()
