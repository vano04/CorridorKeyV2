#!/usr/bin/env python3
"""Profile V3 dataloader batch size, H2D transfer, and GPU transform memory.

Examples:
  .venv/bin/python tools/profile_v3_batch_io.py --config configs/v3_single_5090_fp8_alt_1024.yaml
  .venv/bin/python tools/profile_v3_batch_io.py --config configs/v3_single_5090_fp8_alt_1024.yaml --four-tiles
"""
from __future__ import annotations

import argparse
import copy
import random
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CorridorKeyDataset import dataset as dataset_mod  # noqa: E402
from training import build_dataloader, set_seed  # noqa: E402
from utils import build_device_transform_from_data_cfg, load_config, move_batch_to_device, pad_collate_video  # noqa: E402


TileGrid = Tuple[int, int, int, int]


def _rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def _iter_tensors(obj: Any) -> Iterable[Tuple[str, torch.Tensor]]:
    if torch.is_tensor(obj):
        yield "", obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if torch.is_tensor(value):
                yield str(key), value
            elif isinstance(value, (dict, list, tuple)):
                for subkey, tensor in _iter_tensors(value):
                    name = str(key) if not subkey else f"{key}.{subkey}"
                    yield name, tensor
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            for subkey, tensor in _iter_tensors(value):
                name = str(i) if not subkey else f"{i}.{subkey}"
                yield name, tensor


def _tensor_storage_id(tensor: torch.Tensor) -> Tuple[int, int, torch.device]:
    storage = tensor.untyped_storage()
    return (storage.data_ptr(), storage.nbytes(), tensor.device)


def tensor_bytes(obj: Any, *, unique_storage: bool = True, device_type: Optional[str] = None) -> int:
    total = 0
    seen: set[Tuple[int, int, torch.device]] = set()
    for _, tensor in _iter_tensors(obj):
        if device_type is not None and tensor.device.type != device_type:
            continue
        if unique_storage:
            ident = _tensor_storage_id(tensor)
            if ident in seen:
                continue
            seen.add(ident)
            total += ident[1]
        else:
            total += tensor.numel() * tensor.element_size()
    return int(total)


def tensor_summary(batch: Dict[str, Any]) -> List[str]:
    rows = []
    for key, tensor in sorted(_iter_tensors(batch), key=lambda item: item[0]):
        rows.append(
            f"{key:24s} shape={tuple(tensor.shape)!s:24s} dtype={str(tensor.dtype).replace('torch.', ''):8s} "
            f"device={str(tensor.device):8s} bytes={tensor.numel() * tensor.element_size() / (1024**2):8.2f} MiB"
        )
    return rows


def _time_cuda_copy(batch: Dict[str, Any], device: torch.device, repeat: int) -> Tuple[float, Dict[str, Any], int, int]:
    if device.type != "cuda":
        t0 = time.perf_counter()
        moved = move_batch_to_device(batch, device, non_blocking=False)
        return time.perf_counter() - t0, moved, tensor_bytes(batch, device_type="cpu"), 0

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    copied: Dict[str, Any] = {}
    times: List[float] = []
    copy_bytes = tensor_bytes(batch, device_type="cpu")
    for _ in range(max(1, repeat)):
        t0 = time.perf_counter()
        copied = move_batch_to_device(batch, device, non_blocking=True)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
        del copied
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    copied = move_batch_to_device(batch, device, non_blocking=True)
    torch.cuda.synchronize(device)
    final_dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated(device)
    return statistics.median(times) if times else final_dt, copied, copy_bytes, peak


def _time_device_transform(
    batch: Dict[str, Any],
    cfg: Dict[str, Any],
    device: torch.device,
    repeat: int,
) -> Tuple[Optional[float], Optional[Dict[str, Any]], int]:
    if device.type != "cuda" or not bool(cfg["data"].get("device_augment", False)):
        return None, None, 0
    local_b = None
    if torch.is_tensor(batch.get("fg_gt")):
        local_b = int(batch["fg_gt"].shape[0])
    for key in ("global_input_gt", "global_alpha_gt", "global_fg_gt", "global_bg_gt"):
        value = batch.get(key)
        if local_b is not None and torch.is_tensor(value) and int(value.shape[0]) not in (local_b,):
            return None, None, 0
    transform = build_device_transform_from_data_cfg(cfg["data"], cfg.get("train", {})).to(device)
    transform.eval()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    out: Optional[Dict[str, Any]] = None
    times: List[float] = []
    with torch.no_grad():
        for _ in range(max(1, repeat)):
            torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            out = transform(batch)
            torch.cuda.synchronize(device)
            times.append(time.perf_counter() - t0)
            del out
            torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        out = transform(batch)
        torch.cuda.synchronize(device)
    return (statistics.median(times) if times else None), out, int(torch.cuda.max_memory_allocated(device))


def _first_batch_from_loader(cfg: Dict[str, Any], warmup: int) -> Tuple[Dict[str, Any], List[float]]:
    loader = build_dataloader(cfg)
    it = iter(loader)
    times: List[float] = []
    batch: Optional[Dict[str, Any]] = None
    for i in range(max(1, warmup + 1)):
        t0 = time.perf_counter()
        batch = next(it)
        dt = time.perf_counter() - t0
        if i >= warmup:
            times.append(dt)
    assert batch is not None
    return batch, times


def _sample_for_tile(dataset: Any, index: int, tile_grid: TileGrid) -> Dict[str, Any]:
    clip_idx, start = dataset.samples[index]
    clip = dataset.clips[clip_idx]
    positions = [start + (i * dataset.frame_stride) for i in range(dataset.sequence_length)]
    if dataset.clip_len_range is not None:
        lo, hi = dataset.clip_len_range
        clip_len = min(len(positions), max(1, min(int(lo), int(hi))))
        positions = positions[:clip_len]
    frame_numbers = [clip.frame_numbers[pos] for pos in positions]
    tile_size = int(dataset.exr_tile_size)
    source_h, source_w = dataset.source_hw
    tx0, tx1, ty0, ty1 = tile_grid
    tile_coords = torch.tensor(
        [
            float(ty0 * tile_size),
            float(min(source_h, (ty1 + 1) * tile_size)),
            float(tx0 * tile_size),
            float(min(source_w, (tx1 + 1) * tile_size)),
        ],
        dtype=torch.float32,
    )
    sample: Dict[str, Any] = {
        "clip_name": clip.name,
        "frame_numbers": torch.tensor(frame_numbers, dtype=torch.long),
        "tile_coords": tile_coords,
        "source_hw": torch.tensor([float(source_h), float(source_w)], dtype=torch.float32),
        "tile_grid": torch.tensor([tx0, tx1, ty0, ty1], dtype=torch.long),
    }
    sample.update(dataset._load_all_modalities(clip, frame_numbers, tile_grid))
    sample.update(dataset._load_global_modalities(clip, frame_numbers, tile_grid))
    if dataset.transform is not None:
        sample = dataset.transform(sample)
    return sample


def _four_tile_setup(cfg: Dict[str, Any]) -> Tuple[Any, List[TileGrid]]:
    cfg = copy.deepcopy(cfg)
    cfg["data"]["shuffle"] = False
    cfg["data"]["num_workers"] = 0
    cfg["data"]["persistent_workers"] = False
    cfg["data"]["decode_global_context"] = True
    loader = build_dataloader(cfg)
    dataset = loader.dataset
    source_h, source_w = dataset.source_hw
    tile = int(dataset.exr_tile_size)
    span = int(dataset.local_tile_span)
    if source_h != source_w or source_h != tile * span * 2:
        raise ValueError(
            f"four-tile mode expects a 2x2 grid of local_tile_span regions; got source={dataset.source_hw}, "
            f"tile={tile}, span={span}"
        )
    grids: List[TileGrid] = [
        (0, span - 1, 0, span - 1),
        (span, (span * 2) - 1, 0, span - 1),
        (0, span - 1, span, (span * 2) - 1),
        (span, (span * 2) - 1, span, (span * 2) - 1),
    ]
    return dataset, grids


def _dedupe_global_batch(batch: Dict[str, Any], *, duplicate_global: bool) -> Dict[str, Any]:
    if duplicate_global:
        return batch
    for key in ("global_input_gt", "global_alpha_gt", "global_fg_gt", "global_bg_gt"):
        value = batch.get(key)
        if torch.is_tensor(value) and value.shape[0] == 4:
            batch[key] = value[:1].clone()
    return batch


def _four_tile_batch_naive(cfg: Dict[str, Any], *, duplicate_global: bool) -> Tuple[Dict[str, Any], float]:
    dataset, grids = _four_tile_setup(cfg)
    t0 = time.perf_counter()
    samples = [_sample_for_tile(dataset, 0, grid) for grid in grids]
    batch = pad_collate_video(samples, pad_multiple=int(cfg["model"].get("patch_size", 8)))
    return _dedupe_global_batch(batch, duplicate_global=duplicate_global), time.perf_counter() - t0


def _sample_frame_numbers(dataset: Any, index: int) -> Tuple[Any, List[int]]:
    clip_idx, start = dataset.samples[index]
    clip = dataset.clips[clip_idx]
    positions = [start + (i * dataset.frame_stride) for i in range(dataset.sequence_length)]
    if dataset.clip_len_range is not None:
        lo, hi = dataset.clip_len_range
        clip_len = min(len(positions), max(1, min(int(lo), int(hi))))
        positions = positions[:clip_len]
    return clip, [clip.frame_numbers[pos] for pos in positions]


def _decode_quadrant_tensor(dataset: Any, clip: Any, modality: str, frame_number: int, grid: TileGrid) -> torch.Tensor:
    member = dataset._resolve_member(clip, modality, frame_number)
    return dataset_mod._load_member_tensor(
        clip.shard_path,
        member,
        modality=modality,
        dtype=dataset.dtype,
        tile_grid=grid,
        source_hw=dataset.source_hw,
        tile_size=dataset.exr_tile_size,
    )


def _assemble_full_frame_from_quadrants(
    dataset: Any,
    quadrant_tensors: List[torch.Tensor],
    grids: List[TileGrid],
) -> torch.Tensor:
    source_h, source_w = dataset.source_hw
    tile = int(dataset.exr_tile_size)
    c = int(quadrant_tensors[0].shape[0])
    canvas = torch.empty((c, source_h, source_w), dtype=quadrant_tensors[0].dtype)
    for tensor, grid in zip(quadrant_tensors, grids):
        tx0, tx1, ty0, ty1 = grid
        y0 = ty0 * tile
        x0 = tx0 * tile
        y1 = min(source_h, (ty1 + 1) * tile)
        x1 = min(source_w, (tx1 + 1) * tile)
        canvas[:, y0:y1, x0:x1] = tensor[:, : y1 - y0, : x1 - x0]
    return canvas


def _downscale_global_frame(dataset: Any, frame_chw: torch.Tensor) -> torch.Tensor:
    source_h, source_w = dataset.source_hw
    source_long = max(source_h, source_w)
    target_long = int(dataset.global_context_long_side)
    if target_long > 0 and source_long == target_long * 2:
        return dataset_mod._downscale_2x_box_chw(frame_chw)
    raise ValueError(
        "cached four-tile global assembly currently expects source_long == 2 * global_context_long_side; "
        f"got source={dataset.source_hw}, global_context_long_side={target_long}"
    )


def _four_tile_batch_cached(cfg: Dict[str, Any], *, duplicate_global: bool) -> Tuple[Dict[str, Any], float]:
    dataset, grids = _four_tile_setup(cfg)
    clip, frame_numbers = _sample_frame_numbers(dataset, 0)
    local_modalities = tuple(str(m) for m in dataset.modalities)
    global_modalities = tuple(str(m) for m in dataset.global_context_modalities)
    needed_modalities = tuple(dict.fromkeys([*local_modalities, *global_modalities]))

    t0 = time.perf_counter()
    cache: Dict[Tuple[str, int, int], torch.Tensor] = {}
    decode_tasks: List[Tuple[str, int, int, TileGrid]] = []
    for frame_number in frame_numbers:
        for grid_idx, grid in enumerate(grids):
            for modality in needed_modalities:
                decode_tasks.append((modality, int(frame_number), grid_idx, grid))

    def decode_task(task: Tuple[str, int, int, TileGrid]) -> Tuple[Tuple[str, int, int], torch.Tensor]:
        modality, frame_number, grid_idx, grid = task
        return (
            (modality, frame_number, grid_idx),
            _decode_quadrant_tensor(dataset, clip, modality, frame_number, grid),
        )

    pool = dataset._get_decode_pool()
    decoded = [decode_task(task) for task in decode_tasks] if pool is None else list(pool.map(decode_task, decode_tasks))
    cache.update(decoded)

    global_by_modality: Dict[str, torch.Tensor] = {}
    for modality in global_modalities:
        frames = []
        for frame_number in frame_numbers:
            quadrants = [cache[(modality, int(frame_number), grid_idx)] for grid_idx in range(len(grids))]
            full = _assemble_full_frame_from_quadrants(dataset, quadrants, grids)
            frames.append(_downscale_global_frame(dataset, full))
        global_by_modality[dataset_mod._global_key(modality)] = torch.stack(frames, dim=0)

    samples: List[Dict[str, Any]] = []
    source_h, source_w = dataset.source_hw
    tile = int(dataset.exr_tile_size)
    for grid_idx, grid in enumerate(grids):
        tx0, tx1, ty0, ty1 = grid
        sample: Dict[str, Any] = {
            "clip_name": clip.name,
            "frame_numbers": torch.tensor(frame_numbers, dtype=torch.long),
            "tile_coords": torch.tensor(
                [
                    float(ty0 * tile),
                    float(min(source_h, (ty1 + 1) * tile)),
                    float(tx0 * tile),
                    float(min(source_w, (tx1 + 1) * tile)),
                ],
                dtype=torch.float32,
            ),
            "source_hw": torch.tensor([float(source_h), float(source_w)], dtype=torch.float32),
            "tile_grid": torch.tensor([tx0, tx1, ty0, ty1], dtype=torch.long),
        }
        for modality in local_modalities:
            sample[modality] = torch.stack(
                [cache[(modality, int(frame_number), grid_idx)] for frame_number in frame_numbers],
                dim=0,
            )
        sample.update(global_by_modality)
        if dataset.transform is not None:
            sample = dataset.transform(sample)
        samples.append(sample)

    batch = pad_collate_video(samples, pad_multiple=int(cfg["model"].get("patch_size", 8)))
    return _dedupe_global_batch(batch, duplicate_global=duplicate_global), time.perf_counter() - t0


def _cached_sweep(config: Dict[str, Any], specs: List[Tuple[int, int, int]], repeats: int) -> None:
    rows: List[Dict[str, Any]] = []
    for num_workers, exr_decode_threads, exr_internal_threads in specs:
        cfg = copy.deepcopy(config)
        cfg["data"]["num_workers"] = int(num_workers)
        cfg["data"]["exr_decode_threads"] = int(exr_decode_threads)
        cfg["data"]["exr_internal_threads"] = int(exr_internal_threads)
        try:
            dataset_mod.apply_openexr_internal_threads(int(exr_internal_threads))
        except Exception:
            pass

        times = []
        host_mib = 0.0
        rss_mib = 0.0
        ok = True
        err = ""
        for _ in range(max(1, repeats)):
            try:
                batch, dt = _four_tile_batch_cached(cfg, duplicate_global=False)
                times.append(dt * 1000.0)
                host_mib = tensor_bytes(batch, device_type="cpu") / (1024**2)
                rss_mib = _rss_mb()
                del batch
            except Exception as exc:
                ok = False
                err = str(exc).splitlines()[0]
                break
        row: Dict[str, Any] = {
            "workers": num_workers,
            "decode": exr_decode_threads,
            "internal": exr_internal_threads,
            "host_mib": host_mib,
            "rss_mib": rss_mib,
            "ok": ok,
            "error": err,
        }
        if times:
            row["median_ms"] = statistics.median(times)
            row["min_ms"] = min(times)
            row["max_ms"] = max(times)
        rows.append(row)

    print("\n== cached four-quadrant sweep ==")
    print("Note: cached four-quadrant construction is direct/in-process; num_workers is shown for config parity,")
    print("but DataLoader worker processes are not used by this synthetic one-batch cached path.")
    print(
        f"{'workers':>7s} {'decode':>6s} {'internal':>8s} {'median_ms':>10s} "
        f"{'min_ms':>9s} {'max_ms':>9s} {'host_MiB':>9s} {'RSS_MiB':>9s} status"
    )
    for row in rows:
        if row["ok"] and "median_ms" in row:
            print(
                f"{row['workers']:7d} {row['decode']:6d} {row['internal']:8d} "
                f"{row['median_ms']:10.2f} {row['min_ms']:9.2f} {row['max_ms']:9.2f} "
                f"{row['host_mib']:9.2f} {row['rss_mib']:9.2f} ok"
            )
        else:
            print(
                f"{row['workers']:7d} {row['decode']:6d} {row['internal']:8d} "
                f"{'':>10s} {'':>9s} {'':>9s} {'':>9s} {'':>9s} error: {row['error']}"
            )


def _pipeline_sweep(config: Dict[str, Any], specs: List[Tuple[int, int, int, int]], device: torch.device, warmup: int, repeats: int) -> None:
    rows: List[Dict[str, Any]] = []
    for num_workers, exr_decode_threads, exr_internal_threads, prefetch_factor in specs:
        cfg = copy.deepcopy(config)
        cfg["data"]["num_workers"] = int(num_workers)
        cfg["data"]["exr_decode_threads"] = int(exr_decode_threads)
        cfg["data"]["exr_internal_threads"] = int(exr_internal_threads)
        cfg["data"]["prefetch_factor"] = int(prefetch_factor)
        cfg["data"]["persistent_workers"] = int(num_workers) > 0
        try:
            dataset_mod.apply_openexr_internal_threads(int(exr_internal_threads))
        except Exception:
            pass

        row: Dict[str, Any] = {
            "workers": num_workers,
            "decode": exr_decode_threads,
            "internal": exr_internal_threads,
            "prefetch": prefetch_factor,
            "ok": True,
            "error": "",
        }
        loader_times: List[float] = []
        h2d_times: List[float] = []
        transform_times: List[float] = []
        gbps_values: List[float] = []
        host_mib = 0.0
        copy_mib = 0.0
        post_mib = 0.0
        copy_peak_mib = 0.0
        transform_peak_mib = 0.0
        rss_mib = 0.0
        try:
            loader = build_dataloader(cfg)
            it = iter(loader)
            for _ in range(max(0, warmup)):
                _ = next(it)
            for _ in range(max(1, repeats)):
                t0 = time.perf_counter()
                host_batch = next(it)
                loader_dt = time.perf_counter() - t0
                loader_times.append(loader_dt * 1000.0)
                host_mib = tensor_bytes(host_batch, device_type="cpu") / (1024**2)
                h2d_dt, gpu_batch, copy_bytes, copy_peak = _time_cuda_copy(host_batch, device, repeat=1)
                h2d_times.append(h2d_dt * 1000.0)
                copy_mib = copy_bytes / (1024**2)
                copy_peak_mib = copy_peak / (1024**2)
                gbps_values.append((copy_bytes / h2d_dt / (1000**3)) if h2d_dt > 0 else 0.0)
                transform_dt, out, transform_peak = _time_device_transform(gpu_batch, cfg, device, repeat=1)
                if transform_dt is not None:
                    transform_times.append(transform_dt * 1000.0)
                if out is not None:
                    post_mib = tensor_bytes(out, device_type="cuda" if device.type == "cuda" else None) / (1024**2)
                transform_peak_mib = transform_peak / (1024**2)
                rss_mib = _rss_mb()
                del host_batch, gpu_batch, out
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        except Exception as exc:
            row["ok"] = False
            row["error"] = str(exc).splitlines()[0]

        if row["ok"]:
            row.update(
                {
                    "loader_ms": statistics.median(loader_times),
                    "h2d_ms": statistics.median(h2d_times),
                    "transform_ms": statistics.median(transform_times) if transform_times else 0.0,
                    "total_ms": statistics.median([a + b + (transform_times[i] if i < len(transform_times) else 0.0) for i, (a, b) in enumerate(zip(loader_times, h2d_times))]),
                    "gbps": statistics.median(gbps_values),
                    "host_mib": host_mib,
                    "copy_mib": copy_mib,
                    "post_mib": post_mib,
                    "copy_peak_mib": copy_peak_mib,
                    "transform_peak_mib": transform_peak_mib,
                    "rss_mib": rss_mib,
                }
            )
        rows.append(row)

    print("\n== actual DataLoader -> H2D -> device transform sweep ==")
    print(
        f"{'workers':>7s} {'decode':>6s} {'internal':>8s} {'pref':>4s} "
        f"{'loader':>9s} {'h2d':>8s} {'rx_GB/s':>8s} {'xform':>9s} {'total':>9s} "
        f"{'host':>8s} {'gpu_out':>8s} {'cuda_peak':>10s} {'RSS':>8s} status"
    )
    for row in rows:
        if row["ok"]:
            print(
                f"{row['workers']:7d} {row['decode']:6d} {row['internal']:8d} {row['prefetch']:4d} "
                f"{row['loader_ms']:9.2f} {row['h2d_ms']:8.2f} {row['gbps']:8.2f} {row['transform_ms']:9.2f} {row['total_ms']:9.2f} "
                f"{row['host_mib']:8.2f} {row['post_mib']:8.2f} {row['transform_peak_mib']:10.2f} {row['rss_mib']:8.2f} ok"
            )
        else:
            print(
                f"{row['workers']:7d} {row['decode']:6d} {row['internal']:8d} {row['prefetch']:4d} "
                f"{'':>9s} {'':>8s} {'':>8s} {'':>9s} {'':>9s} {'':>8s} {'':>8s} {'':>10s} {'':>8s} "
                f"error: {row['error']}"
            )


def _parse_sweep_specs(raw: str) -> List[Tuple[int, int, int]]:
    specs: List[Tuple[int, int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 3:
            raise ValueError(f"Invalid sweep spec {item!r}; expected workers:exr_decode_threads:exr_internal_threads")
        specs.append((int(parts[0]), int(parts[1]), int(parts[2])))
    if not specs:
        raise ValueError("Sweep spec is empty")
    return specs


def _parse_pipeline_specs(raw: str) -> List[Tuple[int, int, int, int]]:
    specs: List[Tuple[int, int, int, int]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split(":")]
        if len(parts) != 4:
            raise ValueError(
                f"Invalid pipeline sweep spec {item!r}; expected workers:exr_decode_threads:exr_internal_threads:prefetch_factor"
            )
        specs.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])))
    if not specs:
        raise ValueError("Pipeline sweep spec is empty")
    return specs


def _report(name: str, batch: Dict[str, Any], cfg: Dict[str, Any], device: torch.device, copy_repeat: int, transform_repeat: int) -> None:
    host_bytes = tensor_bytes(batch, device_type="cpu")
    print(f"\n== {name} ==")
    print(f"host tensors: {host_bytes / (1024**2):.2f} MiB unique storage")
    print(f"process RSS:  {_rss_mb():.2f} MiB")
    for row in tensor_summary(batch):
        print("  " + row)

    copy_dt, gpu_batch, copy_bytes, copy_peak = _time_cuda_copy(batch, device, copy_repeat)
    gbps = (copy_bytes / copy_dt / (1000**3)) if copy_dt > 0 else 0.0
    print(f"H2D copy:     {copy_dt * 1000.0:.2f} ms for {copy_bytes / (1024**2):.2f} MiB = {gbps:.2f} GB/s")
    if device.type == "cuda":
        print(f"CUDA alloc after copy: {tensor_bytes(gpu_batch, device_type='cuda') / (1024**2):.2f} MiB tensors, peak {copy_peak / (1024**2):.2f} MiB")

    transform_dt, out, transform_peak = _time_device_transform(gpu_batch, cfg, device, transform_repeat)
    if transform_dt is not None and out is not None:
        print(f"device transform: {transform_dt * 1000.0:.2f} ms")
        print(f"post-transform CUDA tensors: {tensor_bytes(out, device_type='cuda') / (1024**2):.2f} MiB unique storage")
        print(f"CUDA peak through transform: {transform_peak / (1024**2):.2f} MiB")
        for row in tensor_summary(out):
            if "global_" in row or row.startswith("video_rgb") or row.startswith("fg_gt") or row.startswith("alpha_gt"):
                print("  " + row)
    elif bool(cfg["data"].get("device_augment", False)) and device.type == "cuda":
        print("device transform: skipped because shared-global batch dims do not match local tile batch dims")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/v3_single_5090_fp8_alt_1024.yaml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--loader-warmup", type=int, default=1)
    parser.add_argument("--copy-repeat", type=int, default=5)
    parser.add_argument("--transform-repeat", type=int, default=3)
    parser.add_argument("--four-tiles", action="store_true", help="Also profile a synthetic 2x2 quadrant batch.")
    parser.add_argument("--four-tiles-duplicate-global", action="store_true", help="Keep one global context per local tile.")
    parser.add_argument(
        "--sweep-cached",
        action="store_true",
        help="Run only the cached four-quadrant one-batch build sweep.",
    )
    parser.add_argument(
        "--sweep-pipeline",
        action="store_true",
        help="Run only an actual DataLoader -> H2D -> device-transform one-batch sweep.",
    )
    parser.add_argument(
        "--sweep-specs",
        default="0:1:0,0:2:0,0:3:0,0:4:0,0:6:0,0:8:0,0:2:1,0:3:1,0:4:1,0:2:2,0:3:2,2:3:0,4:3:0,6:3:0",
        help="Comma-separated workers:exr_decode_threads:exr_internal_threads specs for --sweep-cached.",
    )
    parser.add_argument("--sweep-repeats", type=int, default=1)
    parser.add_argument(
        "--pipeline-specs",
        default=(
            "2:4:0:2,4:2:0:2,4:3:0:2,5:2:0:2,6:2:0:2,"
            "8:1:0:2,3:4:0:2,2:6:0:2,4:4:0:2,4:2:1:2,2:4:1:2,6:2:0:3"
        ),
        help="Comma-separated workers:exr_decode_threads:exr_internal_threads:prefetch_factor specs.",
    )
    parser.add_argument("--pipeline-warmup", type=int, default=1)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"config: {args.config}")
    print(
        "data: "
        f"batch_size={cfg['data'].get('batch_size')} clip_len={cfg['data'].get('clip_len_min')}..{cfg['data'].get('clip_len_max')} "
        f"exr_tile_size={cfg['data'].get('exr_tile_size')} local_tile_span={cfg['data'].get('local_tile_span')} "
        f"decode_global_context={cfg['data'].get('decode_global_context')} global_long={cfg['data'].get('global_context_long_side')} "
        f"host_dtype={cfg['data'].get('host_dtype')} read_dtype={cfg['data'].get('read_dtype')} "
        f"workers={cfg['data'].get('num_workers')} prefetch={cfg['data'].get('prefetch_factor')} pin={cfg['data'].get('pin_memory')}"
    )
    if args.sweep_cached:
        _cached_sweep(cfg, _parse_sweep_specs(args.sweep_specs), repeats=max(1, int(args.sweep_repeats)))
        return
    if args.sweep_pipeline:
        _pipeline_sweep(
            cfg,
            _parse_pipeline_specs(args.pipeline_specs),
            device=device,
            warmup=max(0, int(args.pipeline_warmup)),
            repeats=max(1, int(args.sweep_repeats)),
        )
        return

    batch, loader_times = _first_batch_from_loader(cfg, args.loader_warmup)
    if loader_times:
        print(f"loader next(): {statistics.median(loader_times) * 1000.0:.2f} ms after {args.loader_warmup} warmup batch(es)")
    _report("configured dataloader batch", batch, cfg, device, args.copy_repeat, args.transform_repeat)

    if args.four_tiles:
        four, build_dt = _four_tile_batch_naive(cfg, duplicate_global=bool(args.four_tiles_duplicate_global))
        suffix = "duplicated global" if args.four_tiles_duplicate_global else "shared global"
        print(f"\nfour-tile naive build: {build_dt * 1000.0:.2f} ms")
        _report(f"four 1024 quadrants naive + {suffix}", four, cfg, device, args.copy_repeat, args.transform_repeat)

        cached, cached_build_dt = _four_tile_batch_cached(cfg, duplicate_global=bool(args.four_tiles_duplicate_global))
        print(f"\nfour-tile cached-quadrant build: {cached_build_dt * 1000.0:.2f} ms")
        _report(
            f"four 1024 quadrants cached + {suffix}",
            cached,
            cfg,
            device,
            args.copy_repeat,
            args.transform_repeat,
        )


if __name__ == "__main__":
    main()
