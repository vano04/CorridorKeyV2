"""CorridorKeyV2 standalone training entrypoint."""
from __future__ import annotations

try:
    import transformer_engine.pytorch as te  # noqa: F401  # Fix library loading order for cublasLt when available
except Exception:
    te = None  # type: ignore[assignment]
import sys
from pathlib import Path

# Make the project root importable regardless of the working directory, and
# also keep the mono-repo root available for shared helpers.
_project_root = str(Path(__file__).resolve().parent)
_repo_root = str(Path(__file__).resolve().parent.parent)
for _path in (_project_root, _repo_root):
    if _path not in sys.path:
        sys.path.insert(0, _path)


import argparse
import importlib
import math
import os
import random
import signal
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# Process-wide environment defaults, applied BEFORE torch import so they
# influence threading-library initialisation, the CUDA caching allocator
# layout, and NCCL's transport selection. ``setdefault`` so any user-set
# value (e.g. NCCL_DEBUG=INFO from torchrun) is preserved.
#
#   CUDA_VISIBLE_DEVICES=0,1,2,3
#       Default to the four local training GPUs when the launcher does not
#       explicitly scope devices. User-provided CUDA_VISIBLE_DEVICES still wins.
#   OMP_NUM_THREADS=2 and MKL/NUMEXPR_NUM_THREADS=1
#       Each DataLoader worker can spawn its own OpenMP/MKL pool. Without
#       this cap, ``num_workers`` workers each grabbing N threads
#       oversubscribes CPU (the typical failure mode is ~30-50% of host
#       throughput lost to context switches). The workers individually
#       call ``torch.set_num_threads(num_torch_threads)`` for the few
#       transform ops that actually benefit from parallel CPU.
#   TORCH_NCCL_ASYNC_ERROR_HANDLING=1
#       Surfaces NCCL collective failures as Python exceptions instead of
#       silently hanging the rank. Cheap insurance for multi-GPU runs;
#       no-op on single-GPU hosts.
#   NCCL_IB_DISABLE=1
#       Single-node setup: skip InfiniBand probe and use NVLink/PCIe
#       directly. Avoids ~3-5 s startup latency on hosts where IB is
#       compiled in but not configured.
#   NCCL_P2P_DISABLE=0
#       Keep direct GPU peer-to-peer transfers enabled for the 4x local GPU
#       training path.
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#       Lets the caching allocator grow a single contiguous segment instead
#       of fragmenting into many fixed-size blocks. Big win for variable
#       per-step shapes (multi-resolution buckets, dynamic temporal chunks)
#       and for Blackwell (sm_120) where peak VRAM utilisation matters more
#       than allocation locality.
#   CUBLAS_WORKSPACE_CONFIG=:4096:8
#       Larger cuBLAS workspace lets it pick higher-throughput GEMM
#       algorithms (especially the bf16 split-k variants used heavily by
#       the transformer stages). 32 MB extra VRAM on a 32 GB card is
#       trivial; the kernel speed-up is not.
for _key, _val in (
    ("CUDA_VISIBLE_DEVICES", "0,1,2,3"),
    ("OMP_NUM_THREADS", "2"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1"),
    ("NCCL_IB_DISABLE", "1"),
    ("NCCL_P2P_DISABLE", "0"),
    ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
):
    os.environ.setdefault(_key, _val)

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.profiler import ProfilerActivity, schedule, tensorboard_trace_handler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from losses import V3MattingLossComputer
from models import V3InferenceOptions, build_v3_hybrid_video_matting_model
from utils import (
    AsyncDevicePrefetcher,
    CorridorMattingTransform,
    DeviceMattingTransform,
    build_device_transform_from_data_cfg,
    cleanup_distributed,
    init_distributed,
    load_config,
    move_batch_to_device,
    pad_collate_video,
)
from utils.ddp import is_rank0, reduce_scalar
from utils.nvtx_utils import nvtx_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train V3 hybrid video matting model")
    parser.add_argument("--config", type=str, default="configs/memory_vmatte.yaml")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument(
        "--resume-model-only",
        action="store_true",
        help=(
            "Load only model weights from --resume and reset optimizer/scheduler/scaler "
            "to start a fresh schedule."
        ),
    )
    parser.add_argument("--output-dir", type=str, default="runs/memory_vmatte")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--profile", action="store_true", help="Enable CUDA profiling for a few steps then exit")
    parser.add_argument("--profile-wait", type=int, default=0, help="Profiler warmup iterations (no recording)")
    parser.add_argument("--profile-warmup", type=int, default=0, help="Profiler warmup iterations (tracing on, results discarded)")
    parser.add_argument("--profile-active", type=int, default=1, help="Profiler active iterations (recorded)")
    parser.add_argument("--profile-dir", type=str, default="profile_traces", help="Directory for profiler output")
    parser.add_argument("--profile-stacks", action="store_true", help="Capture Python stacks (very slow, off by default)")
    parser.add_argument(
        "--profile-include-compile",
        action="store_true",
        help=(
            "Include the first compile-heavy iteration in the profiler active window. "
            "By default, when train.compile=true and both --profile-wait/--profile-warmup are 0, "
            "we auto-set wait=1 so traces focus on steady-state iterations."
        ),
    )
    parser.add_argument(
        "--profile-print-tables",
        action="store_true",
        help="Print profiler key_averages tables at profile end (can be very slow)",
    )
    parser.add_argument(
        "--debug-console",
        action="store_true",
        help=(
            "Enable verbose train.py debug/info diagnostics on stdout."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int, rank: int = 0) -> None:
    full_seed = seed + rank
    random.seed(full_seed)
    torch.manual_seed(full_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(full_seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    # PyTorch 2.x canonical TF32 knob. The legacy ``allow_tf32`` flags only
    # cover cuDNN conv and some cuBLAS paths; ``set_float32_matmul_precision``
    # also enables TF32 (and bf16-accumulate) for the dispatcher-level
    # ``torch.matmul`` calls used by attention QKV projections, the loss-side
    # Laplacian / boundary ops, and AdamW's fused fp32 master-weight step.
    # On Blackwell (sm_120) "high" picks TF32 for fp32 inputs and leaves
    # bf16 inputs untouched -- so this is purely additive on top of the
    # bf16 autocast region.
    torch.set_float32_matmul_precision("high")


def iterate_device_batches(
    dataloader: DataLoader,
    device: torch.device,
    use_cuda_prefetch: bool = True,
    prefetch_queue: int = 2,
    post_move_fn: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield batches already resident on *device*.

    On CUDA, this uses :class:`AsyncDevicePrefetcher` so the host-side
    ``next(data_iter)`` call runs on a background thread and genuinely
    overlaps with GPU compute. The previous implementation ran that call on
    the training thread, so the "prefetch stream" only hid the (tiny) H2D
    memcpy while the (large) DataLoader dequeue still blocked training.
    """
    if device.type != "cuda" or not use_cuda_prefetch:
        data_iter = iter(dataloader)
        while True:
            with nvtx_range("dataloader_next"):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
            with nvtx_range("h2d_move"):
                batch = move_batch_to_device(batch, device, non_blocking=False)
            if post_move_fn is not None:
                with nvtx_range("device_transform_main"):
                    with torch.no_grad():
                        batch = post_move_fn(batch)
            yield batch
        return

    try:
        max_batches: Optional[int] = len(dataloader)
    except TypeError:
        max_batches = None

    prefetcher = AsyncDevicePrefetcher(
        dataloader=dataloader,
        device=device,
        queue_size=max(1, prefetch_queue),
        max_batches=max_batches,
        post_move_fn=post_move_fn,
    )
    try:
        prefetch_iter = iter(prefetcher)
        while True:
            with nvtx_range("prefetch_queue_get_wait"):
                try:
                    batch = next(prefetch_iter)
                except StopIteration:
                    break
            yield batch
    finally:
        prefetcher.close()


TEMPORAL_BATCH_KEYS = {
    "video_rgb",
    "global_video_rgb",
    "alpha_gt",
    "global_alpha_gt",
    "fg_gt",
    "global_fg_gt",
    "bg_gt",
    "global_bg_gt",
    "input_gt",
    "global_input_gt",
    "alpha_boundary_gt",
    "valid_mask",
    "frame_indices",
}


@dataclass(frozen=True)
class DatasetRuntime:
    module_name: str
    web_sequence_dataset_cls: Any
    sequence_dataset_cls: Any
    create_ddp_dataloader: Callable[..., DataLoader]
    set_dataloader_epoch: Callable[[DataLoader, int], None]


@lru_cache(maxsize=16)
def _import_dataset_runtime(module_name: str) -> DatasetRuntime:
    module = importlib.import_module(module_name)

    required = {
        "CorridorKeyWebSequenceDataset",
        "CorridorKeySequenceDataset",
        "create_ddp_dataloader",
        "set_dataloader_epoch",
    }
    missing = [symbol for symbol in sorted(required) if not hasattr(module, symbol)]
    if missing:
        raise ImportError(
            f"Module '{module_name}' is missing required symbols: {', '.join(missing)}"
        )

    return DatasetRuntime(
        module_name=module_name,
        web_sequence_dataset_cls=getattr(module, "CorridorKeyWebSequenceDataset"),
        sequence_dataset_cls=getattr(module, "CorridorKeySequenceDataset"),
        create_ddp_dataloader=getattr(module, "create_ddp_dataloader"),
        set_dataloader_epoch=getattr(module, "set_dataloader_epoch"),
    )


def resolve_dataset_runtime(data_cfg: Dict[str, Any]) -> DatasetRuntime:
    import dataset_web
    return DatasetRuntime(
        module_name="dataset_web",
        web_sequence_dataset_cls=dataset_web.CorridorKeyWebSequenceDataset,
        sequence_dataset_cls=dataset_web.CorridorKeySequenceDataset,
        create_ddp_dataloader=dataset_web.create_ddp_dataloader,
        set_dataloader_epoch=dataset_web.set_dataloader_epoch,
    )


def resolve_webdataset_root(data_cfg: Dict[str, Any], default_root: Path, shard_glob: str) -> Path | None:
    explicit_web_root = str(data_cfg.get("webdataset_root_dir", "")).strip()

    candidates: List[Path] = []
    if explicit_web_root:
        candidates.append(Path(explicit_web_root))

    candidates.extend([
        default_root,
        Path("CorridorKeyWebDataset"),
        default_root.parent / "CorridorKeyWebDataset",
        default_root / "CorridorKeyWebDataset",
        # Support both historical shard root names used in this repo.
        Path("CorridorKeyDataset"),
        default_root.parent / "CorridorKeyDataset",
        default_root / "CorridorKeyDataset",
    ])

    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    if explicit_web_root:
        explicit_path = Path(explicit_web_root)
        if explicit_path.exists():
            return explicit_path

    for candidate in deduped:
        if candidate.is_dir() and any(candidate.glob(shard_glob)):
            return candidate

    return None


def set_dataloader_epoch_safe(dataloader: DataLoader, epoch: int) -> None:
    set_epoch_fn = getattr(dataloader, "_corridorkey_set_dataloader_epoch", None)
    if callable(set_epoch_fn):
        set_epoch_fn(dataloader, epoch)
        return

    sampler = getattr(dataloader, "sampler", None)
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)


def build_temporal_chunks(total_frames: int, chunk_size: int) -> List[Tuple[int, int]]:
    if total_frames < 1:
        return []
    if chunk_size <= 0 or chunk_size >= total_frames:
        return [(0, total_frames)]

    chunks: List[Tuple[int, int]] = []
    for start in range(0, total_frames, chunk_size):
        end = min(total_frames, start + chunk_size)
        chunks.append((start, end))
    return chunks


def slice_batch_temporal(batch: Dict[str, Any], start_t: int, end_t: int) -> Dict[str, Any]:
    sliced: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v) and k in TEMPORAL_BATCH_KEYS:
            sliced[k] = v[:, start_t:end_t]
        else:
            sliced[k] = v
    return sliced


def cuda_warmup(
    model: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    patch_size: int,
    temporal_batch_size: int,
    batch_size: int = 1,
    input_dtype: torch.dtype = torch.float32,
    resolution_buckets: List[int] | None = None,
    inference_options: V3InferenceOptions | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    verbose: bool = False,
) -> None:
    """Run dummy forward+backward passes to force cuDNN kernel compilation and CUDA memory pool init.

    Iterates over all resolution buckets so cuDNN caches the optimal algorithm for each shape.
    With cudnn.benchmark=True, each unique spatial size triggers a one-time benchmark;
    doing it here moves that cost out of the training loop.

    When *optimizer* is provided, AdamW state tensors (momentum / variance) are
    explicitly pre-allocated and zero-initialised so the first real training
    step does not pay the allocation cost. We deliberately do NOT call
    ``optimizer.step()`` on the warmup gradients: the dummy loss only uses
    means of active outputs to keep gradient magnitudes small. We zero
    gradients after the backward pass instead so cuDNN/CUDA caches and DDP
    reducer buckets are still warmed without poisoning the optimizer state.
    """
    if device.type != "cuda":
        return

    was_training = model.training
    model.train()

    t = max(2, temporal_batch_size)
    b = max(1, int(batch_size))
    if resolution_buckets is None:
        resolution_buckets = [patch_size * 4]

    autocast_device = device.type
    for idx, res in enumerate(resolution_buckets):
        h = w = int(math.ceil(res / patch_size) * patch_size)

        dummy_video = torch.randn(b, t, 3, h, w, device=device, dtype=input_dtype)
        dummy_alpha = torch.rand(b, 1, h, w, device=device, dtype=input_dtype)
        warmup_opts = inference_options if inference_options is not None else V3InferenceOptions(mode="full")
        forward_kwargs: Dict[str, Any] = {}
        if warmup_opts.mode == "hybrid":
            cap = max(1, int(warmup_opts.global_long_side_cap))
            scale = min(1.0, cap / float(max(h, w)))
            gh = max(1, int(round(h * scale)))
            gw = max(1, int(round(w * scale)))
            forward_kwargs["global_video"] = torch.randn(
                b, t, 3, gh, gw, device=device, dtype=input_dtype
            )
            forward_kwargs["global_coarse_alpha_init"] = torch.rand(
                b, 1, gh, gw, device=device, dtype=input_dtype
            )
            coord_dtype = input_dtype if input_dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32
            forward_kwargs["tile_coords"] = torch.tensor(
                [[0.0, float(h), 0.0, float(w)] for _ in range(b)],
                device=device,
                dtype=coord_dtype,
            )
            forward_kwargs["source_hw"] = torch.tensor(
                [[float(h), float(w)] for _ in range(b)],
                device=device,
                dtype=coord_dtype,
            )

        with torch.amp.autocast(autocast_device, enabled=amp_enabled, dtype=amp_dtype):
            out = model(
                video=dummy_video,
                coarse_alpha_init=dummy_alpha,
                valid_mask=None,
                bg_for_comp=dummy_video,
                inference_options=warmup_opts,
                **forward_kwargs,
            )
            # Touch every active output branch so DDP sees stable parameter
            # usage between warmup and real training iterations. Use mean()
            # rather than sum() to keep gradient magnitudes small while still
            # priming kernel selection, cache state, and reducer buckets.
            loss = out["alpha_pred"].mean() + out["fg_pred"].mean()
            uncertainty = out.get("uncertainty_pred")
            if uncertainty is not None:
                loss = loss + uncertainty.mean()
            spill = out.get("spill_mask_pred")
            if spill is not None:
                loss = loss + spill.mean()
            edge_alpha = out.get("edge_alpha_refine")
            if edge_alpha is not None:
                loss = loss + edge_alpha.mean()
            edge_fg = out.get("edge_fg_refine")
            if edge_fg is not None:
                loss = loss + edge_fg.mean()

        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # NEVER call optimizer.step() with warmup gradients (see docstring).
        # The first real training step will allocate AdamW state (which is
        # cheap relative to the rest of the step); skipping that one-time
        # cost is not worth risking poisoned ``exp_avg_sq``.
        for p in model.parameters():
            p.grad = None

        del dummy_video, dummy_alpha, out, loss

    torch.cuda.synchronize(device)

    if not was_training:
        model.eval()

    if verbose and is_rank0():
        print(f"CUDA warmup complete (compiled kernels for {len(resolution_buckets)} resolution buckets, t={t})")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)
    min_lr_ratio = max(0.0, min(1.0, float(min_lr_ratio)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def _checkpoint_value_to_cpu(value: Any) -> Any:
    """Recursively detach tensors and copy them to CPU for safe serialization."""
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)

    if isinstance(value, OrderedDict):
        out = OrderedDict((k, _checkpoint_value_to_cpu(v)) for k, v in value.items())
        metadata = getattr(value, "_metadata", None)
        if metadata is not None:
            out._metadata = metadata
        return out

    if isinstance(value, dict):
        return {k: _checkpoint_value_to_cpu(v) for k, v in value.items()}

    if isinstance(value, list):
        return [_checkpoint_value_to_cpu(v) for v in value]

    if isinstance(value, tuple):
        return tuple(_checkpoint_value_to_cpu(v) for v in value)

    return value


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """Write a torch checkpoint without leaving a corrupt final file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _atomic_write_text(path: Path, text: str) -> None:
    """Write a small text marker atomically for cross-rank filesystem sync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(text)
        os.replace(tmp_path, path)
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass


def _safe_sync_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _init_filesystem_sync_session(output_dir: Path, ddp: Any) -> str:
    """Create a run-unique token shared by all ranks for marker filenames."""
    if not ddp.is_distributed:
        return "single"

    session_obj: List[str | None] = [None]
    if ddp.rank == 0:
        elastic_id = os.environ.get("TORCHELASTIC_RUN_ID", "torchrun")
        session_obj[0] = _safe_sync_name(f"{elastic_id}-{os.getpid()}-{uuid.uuid4().hex[:12]}")
    dist.broadcast_object_list(session_obj, src=0)
    session = session_obj[0]
    if not session:
        raise RuntimeError("failed to initialise distributed filesystem sync session")

    (output_dir / ".ddp_sync" / session).mkdir(parents=True, exist_ok=True)
    return session


def _wait_for_checkpoint_marker(
    done_path: Path,
    failed_path: Path,
    timeout_seconds: float,
    poll_seconds: float,
    description: str,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        if done_path.exists():
            return
        if failed_path.exists():
            detail = failed_path.read_text(errors="replace").strip()
            raise RuntimeError(f"{description} failed on rank 0: {detail}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {description} marker at {done_path}")
        time.sleep(poll_seconds)


def _filesystem_rendezvous(
    sync_dir: Path,
    rank: int,
    world_size: int,
    timeout_seconds: float,
    poll_seconds: float,
    description: str,
) -> None:
    """CPU/filesystem barrier used when a NCCL barrier would wait behind I/O."""
    ready_path = sync_dir / f"rank{rank:05d}.ready"
    _atomic_write_text(ready_path, f"pid={os.getpid()}\ntime={time.time():.6f}\n")

    deadline = time.monotonic() + timeout_seconds
    while True:
        missing = [
            r for r in range(world_size)
            if not (sync_dir / f"rank{r:05d}.ready").exists()
        ]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for ranks {missing} at {description}")
        time.sleep(poll_seconds)


def _save_checkpoint_with_filesystem_rendezvous(
    *,
    ddp: Any,
    output_dir: Path,
    sync_session: str,
    sync_name: str,
    timeout_seconds: float,
    checkpoint_fn: Callable[[], None],
) -> None:
    """Run a rank-0 checkpoint save without parking peers in a NCCL barrier.

    Rank 0 can spend a long time copying optimizer/model tensors to CPU and
    flushing the checkpoint to disk. If peers enter ``dist.barrier()`` during
    that window, the NCCL watchdog can abort the whole job even though nothing
    is wrong with training. Marker files give us a slow but safe CPU rendezvous.
    """
    if not ddp.is_distributed:
        checkpoint_fn()
        return

    sync_dir = output_dir / ".ddp_sync" / sync_session / _safe_sync_name(sync_name)
    done_path = sync_dir / "rank0_checkpoint.done"
    failed_path = sync_dir / "rank0_checkpoint.failed"
    poll_seconds = 1.0

    if ddp.rank == 0:
        try:
            checkpoint_fn()
        except BaseException as exc:
            _atomic_write_text(failed_path, f"{type(exc).__name__}: {exc}\n")
            raise
        else:
            _atomic_write_text(done_path, "ok\n")
    else:
        _wait_for_checkpoint_marker(
            done_path=done_path,
            failed_path=failed_path,
            timeout_seconds=timeout_seconds,
            poll_seconds=poll_seconds,
            description=sync_name,
        )

    _filesystem_rendezvous(
        sync_dir=sync_dir,
        rank=ddp.rank,
        world_size=ddp.world_size,
        timeout_seconds=timeout_seconds,
        poll_seconds=poll_seconds,
        description=sync_name,
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    config: Dict[str, Any],
) -> None:
    target_model = model.module if hasattr(model, "module") else model
    model_state = _checkpoint_value_to_cpu(target_model.state_dict())
    optimizer_state = _checkpoint_value_to_cpu(optimizer.state_dict())
    scheduler_state = _checkpoint_value_to_cpu(scheduler.state_dict())
    ckpt = {
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler_state,
        "epoch": epoch,
        "global_step": global_step,
        "config": config,
        "torch_rng_state": torch.get_rng_state().clone(),
    }

    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_device = torch.cuda.current_device()
        ckpt["cuda_rng_device"] = int(cuda_device)
        ckpt["cuda_rng_state"] = torch.cuda.get_rng_state(cuda_device).detach().cpu()

    if scaler is not None:
        ckpt["scaler"] = _checkpoint_value_to_cpu(scaler.state_dict())

    _atomic_torch_save(ckpt, path)


def _resolve_checkpoint_path(path: str | Path) -> Path:
    ckpt_path = Path(path).expanduser()
    if ckpt_path.exists():
        return ckpt_path

    project_root = Path(__file__).resolve().parent
    candidates: List[Path] = []

    if not ckpt_path.is_absolute():
        candidates.append(project_root / ckpt_path)
        candidates.append(project_root / "runs" / ckpt_path.name)

        runs_dir = project_root / "runs"
        if runs_dir.exists():
            matches = sorted(runs_dir.glob(f"**/{ckpt_path.name}"))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                listed = ", ".join(str(m.relative_to(project_root)) for m in matches[:8])
                raise FileNotFoundError(
                    f"Ambiguous checkpoint path '{path}'. Multiple matches under runs/: {listed}. "
                    "Please pass an explicit checkpoint path."
                )

    for cand in candidates:
        if cand.exists():
            return cand

    raise FileNotFoundError(
        f"Checkpoint not found: {path}. Tried current working directory, project root, "
        "and runs/ under the repository root."
    )


def _normalize_compile_prefix_for_target(
    state_dict: Dict[str, torch.Tensor],
    target: nn.Module,
) -> Dict[str, torch.Tensor]:
    """Adapt ``torch.compile`` wrapper prefixes when loading checkpoints.

    ``torch.compile`` stores the wrapped module under ``_orig_mod``. That means
    compiled-model checkpoints use ``_orig_mod.foo`` keys, while eager models
    use ``foo`` keys. Normalize either direction so ``train.compile`` can be
    changed between runs without breaking strict checkpoint loads.
    """
    prefix = "_orig_mod."
    target_keys = set(target.state_dict().keys())
    source_keys = set(state_dict.keys())
    if source_keys == target_keys:
        return state_dict

    stripped = {
        (k[len(prefix) :] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }
    if set(stripped.keys()) == target_keys:
        return stripped

    added = {
        (k if k.startswith(prefix) else f"{prefix}{k}"): v
        for k, v in state_dict.items()
    }
    if set(added.keys()) == target_keys:
        return added

    def strip_compile_wrappers(key: str) -> str:
        return ".".join(part for part in key.split(".") if part != "_orig_mod")

    target_by_unwrapped_key: Dict[str, str] = {}
    for key in target_keys:
        unwrapped = strip_compile_wrappers(key)
        if unwrapped in target_by_unwrapped_key:
            target_by_unwrapped_key = {}
            break
        target_by_unwrapped_key[unwrapped] = key

    if target_by_unwrapped_key:
        remapped: Dict[str, torch.Tensor] = {}
        for key, value in state_dict.items():
            target_key = target_by_unwrapped_key.get(strip_compile_wrappers(key))
            if target_key is None:
                remapped = {}
                break
            remapped[target_key] = value
        if set(remapped.keys()) == target_keys:
            return remapped

    return state_dict


def maybe_compile_boundary_refine(
    model: nn.Module,
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    rank0_debug_console: bool,
) -> nn.Module:
    if not bool(model_cfg.get("compile_boundary_refine", False)):
        return model

    if bool(train_cfg.get("compile", False)):
        _maybe_print(
            rank0_debug_console,
            "compile_boundary_refine ignored because train.compile=true already wraps the full model.",
        )
        return model

    boundary_refine = getattr(model, "boundary_refine", None)
    if boundary_refine is None:
        _maybe_print(
            rank0_debug_console,
            "compile_boundary_refine requested, but this model has no boundary_refine module.",
        )
        return model

    dynamic_default = bool(train_cfg.get("compile_dynamic", False))
    dynamic = bool(model_cfg.get("compile_boundary_refine_dynamic", dynamic_default))
    model.boundary_refine = torch.compile(boundary_refine, dynamic=dynamic)
    _maybe_print(rank0_debug_console, f"torch.compile enabled for boundary_refine (dynamic={dynamic})")
    return model


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
) -> Tuple[int, int]:
    ckpt = torch.load(_resolve_checkpoint_path(path), map_location=device)

    target = model.module if hasattr(model, "module") else model
    state_dict = _normalize_compile_prefix_for_target(ckpt["model"], target)
    target.load_state_dict(state_dict, strict=True)

    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])
    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    global_step = int(ckpt.get("global_step", 0))
    return start_epoch, global_step


def load_model_weights_only(path: str, model: nn.Module, device: torch.device) -> Tuple[int, int]:
    ckpt = torch.load(_resolve_checkpoint_path(path), map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        ckpt_epoch = int(ckpt.get("epoch", -1))
        ckpt_global_step = int(ckpt.get("global_step", 0))
    else:
        state_dict = ckpt
        ckpt_epoch = -1
        ckpt_global_step = 0

    target = model.module if hasattr(model, "module") else model
    state_dict = _normalize_compile_prefix_for_target(state_dict, target)
    target.load_state_dict(state_dict, strict=True)
    return ckpt_epoch, ckpt_global_step


_HOST_DTYPE_ALIASES: Dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "f32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "half": torch.float16,
}


def _resolve_host_dtype(value: Any) -> Optional[torch.dtype]:
    """Parse the ``data.host_dtype`` config value into a ``torch.dtype``.

    Accepts ``None`` / empty string / "float32"/"fp32" -> the safe fp32
    path (returned as ``None`` so the transform can no-op the cast).
    Accepts "bfloat16"/"bf16"/"float16"/"fp16"/"half" for the H2D-bytes
    halving paths. Anything else raises so config typos fail loudly
    instead of silently shipping the wrong precision.
    """
    if value is None:
        return None
    if isinstance(value, torch.dtype):
        return None if value == torch.float32 else value
    if not isinstance(value, str):
        raise ValueError(f"data.host_dtype must be a string or torch.dtype, got {type(value)!r}")
    key = value.strip().lower()
    if not key:
        return None
    if key not in _HOST_DTYPE_ALIASES:
        raise ValueError(
            f"data.host_dtype={value!r} is not recognised. "
            f"Supported: {sorted(set(_HOST_DTYPE_ALIASES))}"
        )
    resolved = _HOST_DTYPE_ALIASES[key]
    return None if resolved == torch.float32 else resolved


def _resolve_bool_cfg(value: Any, *, key_name: str, default: bool) -> bool:
    """Parse a flexible bool config value and fail loudly on invalid strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(
            f"{key_name}={value!r} is not recognised. "
            "Use true/false (or 1/0, yes/no, on/off)."
        )
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"{key_name} must be a boolean-like value, got {type(value)!r}")


def _resolve_debug_console_enabled(
    config: Dict[str, Any],
    args: argparse.Namespace | None = None,
) -> bool:
    if args is not None and bool(getattr(args, "debug_console", False)):
        return True

    env_value = os.environ.get("CORRIDORKEY_TRAIN_DEBUG_CONSOLE")
    if env_value is not None:
        return _resolve_bool_cfg(
            env_value,
            key_name="CORRIDORKEY_TRAIN_DEBUG_CONSOLE",
            default=False,
        )

    train_cfg = config.get("train", {})
    if isinstance(train_cfg, dict):
        return _resolve_bool_cfg(
            train_cfg.get("debug_console"),
            key_name="train.debug_console",
            default=True,
        )

    return True


def _maybe_print(enabled: bool, *args: Any, **kwargs: Any) -> None:
    if enabled:
        print(*args, **kwargs)


_NO_WEIGHT_DECAY_MODULE_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.SyncBatchNorm,
    nn.GroupNorm,
    nn.LayerNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.Embedding,
)


def _as_string_tuple(value: Any, default: Tuple[str, ...] = ()) -> Tuple[str, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        return tuple(part.strip() for part in value.split(",") if part.strip())
    if isinstance(value, (list, tuple, set)):
        return tuple(str(part).strip() for part in value if str(part).strip())
    raise ValueError(f"Expected string or sequence of strings, got {type(value)!r}")


def _optimizer_group_cfg(train_cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    cfg = train_cfg.get("parameter_groups", train_cfg.get("optimizer_parameter_groups", False))
    if isinstance(cfg, bool):
        return cfg, {}
    if isinstance(cfg, dict):
        return bool(cfg.get("enabled", True)), cfg
    raise ValueError("train.parameter_groups must be a boolean or mapping when provided.")


def build_optimizer_parameter_groups(
    model: nn.Module,
    train_cfg: Dict[str, Any],
    base_lr: float,
    weight_decay: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Build AdamW groups inspired by MatAnyone2's parameter-group utility."""
    enabled, group_cfg = _optimizer_group_cfg(train_cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not enabled:
        return [{"params": trainable, "lr": base_lr, "weight_decay": weight_decay}], {
            "default": sum(p.numel() for p in trainable),
        }

    no_decay_keywords = _as_string_tuple(
        group_cfg.get("no_weight_decay_keywords"),
        default=(
            "summary_pos",
            "query_init",
            "query_emb",
            "obj_pe",
            "pos_embed",
            "absolute_pos_embed",
            "relative_position_bias",
        ),
    )
    backbone_prefixes = _as_string_tuple(group_cfg.get("backbone_lr_prefixes"))
    backbone_lr_ratio = float(group_cfg.get("backbone_lr_ratio", 1.0))
    embed_weight_decay = group_cfg.get("embed_weight_decay", None)

    grouped: Dict[Tuple[str, float, float], List[nn.Parameter]] = {}
    stats: Dict[str, int] = {}
    seen: set[int] = set()

    def add_param(label: str, lr: float, wd: float, param: nn.Parameter) -> None:
        grouped.setdefault((label, lr, wd), []).append(param)
        stats[label] = stats.get(label, 0) + int(param.numel())

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)

            full_name = f"{module_name}.{param_name}" if module_name else param_name
            canonical_name = full_name.removeprefix("module.")
            in_backbone = any(
                canonical_name == prefix or canonical_name.startswith(f"{prefix}.")
                for prefix in backbone_prefixes
            )
            is_embedding_name = any(keyword in canonical_name for keyword in no_decay_keywords)
            use_no_decay = (
                param_name == "bias"
                or param.ndim <= 1
                or isinstance(module, _NO_WEIGHT_DECAY_MODULE_TYPES)
                or is_embedding_name
            )

            lr = base_lr * backbone_lr_ratio if in_backbone else base_lr
            wd = 0.0 if use_no_decay else weight_decay
            if is_embedding_name and embed_weight_decay is not None:
                wd = float(embed_weight_decay)

            label_parts = ["backbone" if in_backbone else "main"]
            label_parts.append("no_decay" if wd == 0.0 else "decay")
            if is_embedding_name and embed_weight_decay is not None:
                label_parts.append("embed")
            add_param("/".join(label_parts), lr, wd, param)

    if len(seen) != len(trainable):
        missing = len(trainable) - len(seen)
        if missing > 0:
            fallback = [p for p in trainable if id(p) not in seen]
            for param in fallback:
                add_param("main/decay/fallback", base_lr, weight_decay, param)

    param_groups = [
        {"params": params, "lr": lr, "weight_decay": wd}
        for (label, lr, wd), params in grouped.items()
        if params
    ]
    return param_groups, stats


def build_dataloader(config: Dict[str, Any], rank: int, world_size: int) -> DataLoader:
    data_cfg = config["data"]
    model_cfg = config["model"]
    debug_console = _resolve_debug_console_enabled(config)

    dataset_runtime = resolve_dataset_runtime(data_cfg)

    tar_cache_max = data_cfg.get("tar_member_cache_max_shards")
    if tar_cache_max is not None:
        os.environ["CORRIDORKEY_TAR_MEMBER_MAP_CACHE_MAX_SHARDS"] = str(
            max(0, int(tar_cache_max))
        )

    clip_len_min = int(data_cfg.get("clip_len_min", 4))
    clip_len_max = int(data_cfg.get("clip_len_max", 12))

    modalities_cfg = data_cfg.get("modalities", ["Input", "FG", "BG", "Alpha"])
    if isinstance(modalities_cfg, str):
        modalities = tuple(part.strip() for part in modalities_cfg.split(",") if part.strip())
    else:
        modalities = tuple(str(part) for part in modalities_cfg)
    if not modalities:
        raise ValueError("data.modalities must define at least one modality")

    # Refuse to silently consume the deprecated crop knobs. Tile-mode
    # inference always sees a fixed tile_size and the lowres global pass
    # always sees a fixed long-side cap, so size-randomized crops produced
    # ~3x average step-time variance for scale invariance the inference
    # path never used. Surfaced as an error rather than ignored so
    # checked-in YAMLs that still set these get fixed instead of silently
    # behaving differently from before.
    deprecated_crop_keys = {"unknown_crop_p", "crop_scale_min", "crop_scale_max"}
    stale_crop_keys = sorted(deprecated_crop_keys.intersection(data_cfg.keys()))
    if stale_crop_keys:
        raise ValueError(
            "data.* keys "
            + ", ".join(stale_crop_keys)
            + " are no longer supported. Random spatial crops were removed; the "
            "transform now resizes each clip to long-side==target_side and "
            "preserves source aspect. Remove these keys from your config."
        )

    # When ``data.device_augment=true``, the CPU transform short-circuits
    # after channel-norm + temporal-sample and emits FG/BG/Alpha at native
    # source resolution; the spatial crop, all photometric/sensor/codec
    # augments, the representation-specific composite, and ``coarse_alpha_init`` then
    # run on the GPU via ``DeviceMattingTransform`` (built in ``train()``
    # and applied right after ``iterate_device_batches`` yield). This is the
    # right setting whenever the dataloader workers are the bottleneck and
    # the GPU has spare VRAM/SMs -- see ``utils/device_transform.py``.
    device_offload = bool(data_cfg.get("device_augment", False))

    # Optional dtype downcast for the device-offload H2D path. Accepts
    # "float32" / "bfloat16" / "float16" (and the usual aliases). When set
    # to a smaller-than-fp32 dtype the worker casts FG/BG/Alpha right
    # before returning, which halves the bytes shipped through IPC, the
    # pin_memory daemon, and PCIe; ``DeviceMattingTransform`` then runs
    # in that dtype natively because it inherits ``dtype=x.dtype`` from
    # its inputs. Only honoured when ``device_augment=true``.
    host_dtype = _resolve_host_dtype(data_cfg.get("host_dtype"))

    transform = CorridorMattingTransform(
        clip_len_min=clip_len_min,
        clip_len_max=clip_len_max,
        resolution_buckets=tuple(data_cfg.get("multi_resolution_buckets", [512])),
        resize_scale_min=float(data_cfg.get("resize_scale_min", 1.0)),
        resize_scale_max=float(data_cfg.get("resize_scale_max", 1.0)),
        # Fixed-size square crop for tiling-model training. >0 means
        # "resize so short side == this, then take a position-randomized
        # crop of this size". Set to model's tile_size to make every
        # training sample shape-identical to a single inference tile.
        fixed_crop_size=int(data_cfg.get("fixed_crop_size", 0)),
        fg_representation=str(data_cfg.get("fg_representation", "premul")),
        horizontal_flip_p=float(data_cfg.get("horizontal_flip_p", 0.5)),
        background_replace_p=float(data_cfg.get("background_replace_p", 0.3)),
        spill_augment_p=float(data_cfg.get("spill_augment_p", 0.4)),
        green_foreground_augment_p=float(data_cfg.get("green_foreground_augment_p", 0.0)),
        green_foreground_strength_min=float(data_cfg.get("green_foreground_strength_min", 0.6)),
        green_foreground_strength_max=float(data_cfg.get("green_foreground_strength_max", 1.0)),
        temporal_jitter_p=float(data_cfg.get("temporal_jitter_p", 0.1)),
        skip_temporal_sample=True,
        device_offload=device_offload,
        # Photometric: subject / background gain.
        subject_gain_min=float(data_cfg.get("subject_gain_min", 0.7)),
        subject_gain_max=float(data_cfg.get("subject_gain_max", 1.3)),
        bg_gain_min=float(data_cfg.get("bg_gain_min", 0.65)),
        bg_gain_max=float(data_cfg.get("bg_gain_max", 1.35)),
        # White-balance jitter.
        wb_jitter_p=float(data_cfg.get("wb_jitter_p", 0.7)),
        wb_jitter_strength=float(data_cfg.get("wb_jitter_strength", 0.15)),
        # HSV-style chroma jitter (saturation + hue rotation).
        color_jitter_p=float(data_cfg.get("color_jitter_p", 0.4)),
        saturation_min=float(data_cfg.get("saturation_min", 0.7)),
        saturation_max=float(data_cfg.get("saturation_max", 1.3)),
        hue_jitter_max=float(data_cfg.get("hue_jitter_max", 0.15)),
        # Sensor noise.
        noise_p=float(data_cfg.get("noise_p", 0.8)),
        noise_sigma_min=float(data_cfg.get("noise_sigma_min", 0.005)),
        noise_sigma_max=float(data_cfg.get("noise_sigma_max", 0.04)),
        shot_noise_p=float(data_cfg.get("shot_noise_p", 0.3)),
        shot_noise_strength=float(data_cfg.get("shot_noise_strength", 0.025)),
        # Lens softness gaussian blur.
        blur_p=float(data_cfg.get("blur_p", 0.2)),
        blur_sigma_min=float(data_cfg.get("blur_sigma_min", 0.3)),
        blur_sigma_max=float(data_cfg.get("blur_sigma_max", 0.8)),
        # Per-clip directional motion blur.
        motion_blur_p=float(data_cfg.get("motion_blur_p", 0.25)),
        motion_blur_kernel_min=int(data_cfg.get("motion_blur_kernel_min", 3)),
        motion_blur_kernel_max=int(data_cfg.get("motion_blur_kernel_max", 7)),
        # Codec / compression artefact proxy.
        compression_p=float(data_cfg.get("compression_p", 0.2)),
        compression_downscale_min=float(data_cfg.get("compression_downscale_min", 1.05)),
        compression_downscale_max=float(data_cfg.get("compression_downscale_max", 1.35)),
        host_dtype=host_dtype,
    )

    sequence_length = int(max(data_cfg.get("clip_len_max", 12), data_cfg.get("dataset_sequence_length", 12)))

    dataset_root = Path(data_cfg.get("root_dir", "."))
    shard_glob = str(data_cfg.get("shard_glob", "*.tar"))
    
    use_webdataset = True


    dataset_convert_to_float = _resolve_bool_cfg(
        data_cfg.get("dataset_convert_to_float"),
        key_name="data.dataset_convert_to_float",
        default=True,
    )

    dataset_kwargs = {
        "root_dir": dataset_root,
        "sequence_length": sequence_length,
        "frame_stride": int(data_cfg.get("frame_stride", 1)),
        "sequence_stride": int(data_cfg.get("sequence_stride", 1)),
        "modalities": modalities,
        "transform": transform,
        "convert_to_float": dataset_convert_to_float,
        "clip_len_range": (clip_len_min, clip_len_max),
        "exr_decode_threads": int(data_cfg.get("exr_decode_threads", 1)),
        "decode_full_frame": bool(data_cfg.get("decode_full_frame", False)),
        "local_tile_span": int(data_cfg.get("local_tile_span", 4)),
        "exr_tile_size": int(data_cfg.get("exr_tile_size", 256)),
        "source_hw": data_cfg.get("source_hw", [2048, 2048]),
        "emit_tile_metadata": bool(data_cfg.get("emit_tile_metadata", True)),
        "decode_global_context": bool(data_cfg.get("decode_global_context", False)),
        "global_context_root_dir": data_cfg.get("global_context_root_dir"),
        "global_context_long_side": int(data_cfg.get("global_context_long_side", 0)),
        "global_context_modalities": data_cfg.get("global_context_modalities", ["Input", "Alpha"]),
    }

    if use_webdataset:
        validation_shards_cfg = data_cfg.get(
            "validation_shard_indices",
            data_cfg.get("validation_shards", [33]),
        )
        if validation_shards_cfg is None:
            validation_shard_indices = []
        elif isinstance(validation_shards_cfg, str):
            validation_shard_indices = [
                int(part.strip())
                for part in validation_shards_cfg.split(",")
                if part.strip()
            ]
        elif isinstance(validation_shards_cfg, (list, tuple, set)):
            validation_shard_indices = [int(v) for v in validation_shards_cfg]
        else:
            validation_shard_indices = [int(validation_shards_cfg)]

        dataset = dataset_runtime.web_sequence_dataset_cls(
            **dataset_kwargs,
            shard_glob=shard_glob,
            manifest_filename=str(data_cfg.get("manifest_filename", "clips_manifest.jsonl")),
            split=str(data_cfg.get("webdataset_split", "train")),
            validation_shard_indices=validation_shard_indices,
        )
    else:
        dataset = dataset_runtime.sequence_dataset_cls(**dataset_kwargs)

    pad_multiple = int(model_cfg.get("patch_size", 8))
    collate_fn = lambda batch: pad_collate_video(batch, pad_multiple=pad_multiple)

    loader = dataset_runtime.create_ddp_dataloader(
        dataset=dataset,
        batch_size=int(data_cfg.get("batch_size_per_gpu", 1)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        shuffle=bool(data_cfg.get("shuffle", True)),
        drop_last=bool(data_cfg.get("drop_last", False)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
        seed=int(data_cfg.get("seed", 1337)),
        rank=rank,
        world_size=world_size,
        collate_fn=collate_fn,
        num_torch_threads=int(data_cfg.get("num_torch_threads", 1)),
        exr_internal_threads=int(data_cfg.get("exr_internal_threads", 0)),
    )

    setattr(loader, "_corridorkey_set_dataloader_epoch", dataset_runtime.set_dataloader_epoch)

    if rank == 0 and debug_console:
        print(
            f"Dataset module={dataset_runtime.module_name} "
            f"mode={'webdataset' if use_webdataset else 'filesystem'} root={dataset_root} "
            f"modalities={modalities} convert_to_float={dataset_convert_to_float}"
        )

    return loader


def train() -> None:
    args = parse_args()
    if args.resume_model_only and not args.resume:
        raise ValueError("--resume-model-only requires --resume <checkpoint_path>.")

    cfg = load_config(args.config)
    debug_console = _resolve_debug_console_enabled(cfg, args=args)

    ddp_cfg = cfg.get("ddp", {})
    ddp_timeout_minutes_cfg = ddp_cfg.get("timeout_minutes")
    ddp_timeout_minutes = (
        float(ddp_timeout_minutes_cfg) if ddp_timeout_minutes_cfg is not None else None
    )
    ddp = init_distributed(
        backend=str(ddp_cfg.get("backend", "nccl")),
        timeout_minutes=ddp_timeout_minutes,
    )
    rank0_debug_console = bool(debug_console and ddp.rank == 0)

    set_seed(args.seed, rank=ddp.rank)

    train_cfg = cfg["train"]
    model_cfg = dict(cfg["model"])
    model_cfg.setdefault("fg_representation", str(cfg["data"].get("fg_representation", "premul")))
    cfg["model"] = model_cfg
    loss_cfg = cfg["loss"]

    train_inference_mode = str(train_cfg.get("inference_mode", "full")).strip().lower()
    if train_inference_mode not in {"full", "lowres", "tiled", "hybrid"}:
        raise ValueError(
            f"Unsupported train.inference_mode='{train_inference_mode}'. "
            "Expected one of: full, lowres, tiled, hybrid."
        )
    enable_boundary_refine_train = bool(train_cfg.get("enable_boundary_refine", False))

    dataloader = build_dataloader(cfg, rank=ddp.rank, world_size=ddp.world_size)

    # Mirror the ``device_augment`` flag into a GPU-side transform module.
    # When set, ``CorridorMattingTransform`` (constructed inside
    # ``build_dataloader``) returns FG/BG/Alpha at native resolution and
    # leaves the spatial crop + photometric augments + composite +
    # ``coarse_alpha_init`` for ``device_transform`` to materialise after
    # H2D. Eager mode -- runs outside the compiled model graph and uses
    # default CUDA RNG (already seeded per-rank by ``set_seed``).
    device_augment = bool(cfg["data"].get("device_augment", False))
    device_transform: DeviceMattingTransform | None = None
    if train_inference_mode == "hybrid":
        if not device_augment:
            raise ValueError("train.inference_mode='hybrid' currently requires data.device_augment=true.")
        # Hybrid no longer requires full-frame FG/BG/Alpha decode. The fast V3
        # path decodes a local OpenEXR tile ROI and optionally consumes
        # precomputed global Input/Alpha sidecars for full-frame context. If no
        # sidecar is configured, DeviceMattingTransform falls back to building
        # global context from the local ROI, which is less ideal but keeps the
        # run usable while sidecars are being generated.
    if device_augment:
        if ddp.device.type != "cuda":
            raise ValueError(
                "data.device_augment=true requires a CUDA device. "
                "Either run on GPU or unset device_augment in the config."
            )
        device_transform = build_device_transform_from_data_cfg(
            cfg["data"], cfg.get("train", {})
        ).to(ddp.device).eval()
        _maybe_print(
            rank0_debug_console,
            (
                "GPU augment enabled: CPU pipeline emits "
                + ("native-res" if bool(cfg["data"].get("decode_full_frame", False)) else "OpenEXR tile-ROI")
                + " FG/BG/Alpha; spatial crop + photometric/sensor/codec augments "
                + "+ composite + coarse_alpha_init run on device after H2D."
            ),
        )

    model = build_v3_hybrid_video_matting_model(model_cfg).to(ddp.device)

    if ddp.device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if train_inference_mode == "full" and not enable_boundary_refine_train:
        boundary_refine = getattr(model, "boundary_refine", None)
        if boundary_refine is not None:
            frozen_tensors = 0
            frozen_numel = 0
            for p in boundary_refine.parameters():
                if p.requires_grad:
                    p.requires_grad_(False)
                    frozen_tensors += 1
                    frozen_numel += p.numel()
            if frozen_tensors > 0:
                _maybe_print(
                    rank0_debug_console,
                    "Training notice: boundary_refine is inactive for mode='full'; "
                    f"froze {frozen_tensors} tensors ({frozen_numel:,} params) "
                    "to reduce DDP communication overhead.",
                )

    model = maybe_compile_boundary_refine(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        rank0_debug_console=rank0_debug_console,
    )

    if bool(train_cfg.get("compile", False)):
        # ``compile_dynamic`` controls Dynamo's shape specialization:
        #   true  -> trace once with symbolic shapes (handles
        #            multi_resolution_buckets / variable clip_len at
        #            the cost of 5-15 min cold compile and shape-guard
        #            overhead on every step). Right default for the
        #            general-purpose / production configs.
        #   false -> trace once per concrete shape (~30-90 s cold
        #            compile, no shape guards on the hot path). Right
        #            choice for tile-mode configs that pin
        #            fixed_crop_size and clip_len_min==clip_len_max.
        # Default true preserves the behaviour every other compile=true
        # config in the repo was written against.
        compile_dynamic = bool(train_cfg.get("compile_dynamic", True))
        model = torch.compile(model, dynamic=compile_dynamic)
        _maybe_print(rank0_debug_console, f"torch.compile enabled (dynamic={compile_dynamic})")

    if ddp.is_distributed and bool(ddp_cfg.get("sync_bn", False)):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if ddp.is_distributed:
        configured_find_unused = bool(ddp_cfg.get("find_unused_parameters", False))
        boundary_refine = getattr(model, "boundary_refine", None)
        has_trainable_boundary_refine = (
            boundary_refine is not None
            and any(p.requires_grad for p in boundary_refine.parameters())
        )
        effective_find_unused = configured_find_unused
        if not effective_find_unused and train_inference_mode == "full" and has_trainable_boundary_refine:
            effective_find_unused = True
            _maybe_print(
                rank0_debug_console,
                "DDP notice: enabling find_unused_parameters because trainable "
                "boundary_refine parameters are inactive in mode='full'.",
            )
        # Tiled mode can skip boundary-refine tiles when the data-dependent
        # refine mask is empty, which may leave boundary_refine without grad on
        # that step. Lowres mode uses the full-frame refiner path and calls the
        # refiner every iteration, so it can keep find_unused_parameters=false.
        if (
            not effective_find_unused
            and train_inference_mode == "tiled"
            and has_trainable_boundary_refine
        ):
            effective_find_unused = True
            _maybe_print(
                rank0_debug_console,
                "DDP notice: enabling find_unused_parameters because the "
                "boundary refiner only receives gradient on steps whose GT "
                "alpha contains soft edges; some batches will leave its "
                "params without grad.",
            )

        static_graph_cfg = ddp_cfg.get("static_graph", None)
        use_static_graph = (not effective_find_unused) if static_graph_cfg is None else bool(static_graph_cfg)
        if effective_find_unused and use_static_graph:
            use_static_graph = False
            _maybe_print(
                rank0_debug_console,
                "DDP notice: disabling static_graph because find_unused_parameters=True.",
            )
        # Boundary/refiner training can have a dynamic autograd graph across
        # temporal chunks or data-dependent tiled refine masks, so static_graph
        # -- which asserts an identical graph every iteration -- can throw
        # mid-training. Default it off whenever the refiner is live, unless the
        # user explicitly opts back in.
        refiner_active = (
            train_inference_mode in {"lowres", "tiled", "hybrid"} and has_trainable_boundary_refine
        )
        if refiner_active and use_static_graph and static_graph_cfg is None:
            use_static_graph = False
            _maybe_print(
                rank0_debug_console,
                "DDP notice: disabling static_graph because the boundary "
                f"refiner is active (inference_mode='{train_inference_mode}'); "
                "its data-dependent gating is not compatible with a static graph.",
            )

        ddp_kwargs = {
            "module": model,
            "device_ids": [ddp.local_rank] if ddp.device.type == "cuda" else None,
            "output_device": ddp.local_rank if ddp.device.type == "cuda" else None,
            "find_unused_parameters": effective_find_unused,
            "broadcast_buffers": False,
            "gradient_as_bucket_view": True,
            "bucket_cap_mb": int(ddp_cfg.get("bucket_cap_mb", 100)),
        }
        if use_static_graph:
            ddp_kwargs["static_graph"] = True

        try:
            model = DDP(**ddp_kwargs)
        except TypeError:
            ddp_kwargs.pop("static_graph", None)
            model = DDP(**ddp_kwargs)

        _maybe_print(
            rank0_debug_console,
            "DDP config: "
            f"find_unused_parameters={effective_find_unused}, "
            f"static_graph={use_static_graph}, "
            f"bucket_cap_mb={ddp_kwargs['bucket_cap_mb']}",
        )

        grad_compression = str(ddp_cfg.get("grad_compression", "")).strip().lower()
        if grad_compression in {"fp16", "bf16"}:
            try:
                from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as ddp_default_hooks

                hook = (
                    ddp_default_hooks.bf16_compress_hook
                    if grad_compression == "bf16"
                    else ddp_default_hooks.fp16_compress_hook
                )
                model.register_comm_hook(state=None, hook=hook)
                _maybe_print(
                    rank0_debug_console,
                    f"DDP comm hook enabled: {grad_compression} gradient compression",
                )
            except Exception as exc:
                _maybe_print(
                    rank0_debug_console,
                    f"DDP comm hook warning: unable to enable {grad_compression} compression ({exc})",
                )

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after applying training-time freezes.")

    if rank0_debug_console:
        total_numel = sum(p.numel() for p in model.parameters())
        trainable_numel = sum(p.numel() for p in trainable_parameters)
        print(f"Trainable parameters: {trainable_numel:,} / {total_numel:,}")

    fused_adamw_cfg = bool(train_cfg.get("fused_adamw", False))
    fused_adamw = fused_adamw_cfg and ddp.device.type == "cuda"
    if fused_adamw_cfg and not fused_adamw:
        _maybe_print(rank0_debug_console, "Training notice: fused_adamw=true ignored on non-CUDA device.")

    base_lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 1e-2))
    optimizer_param_groups, optimizer_group_stats = build_optimizer_parameter_groups(
        model=model,
        train_cfg=train_cfg,
        base_lr=base_lr,
        weight_decay=weight_decay,
    )

    optimizer = AdamW(
        optimizer_param_groups,
        lr=base_lr,
        betas=tuple(train_cfg.get("betas", [0.9, 0.999])),
        weight_decay=weight_decay,
        fused=fused_adamw,
    )
    if rank0_debug_console:
        group_desc = ", ".join(f"{k}={v:,}" for k, v in sorted(optimizer_group_stats.items()))
        print(f"Optimizer: AdamW(fused={fused_adamw}, groups={len(optimizer_param_groups)}) {group_desc}")

    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    temporal_batch_size = int(train_cfg.get("temporal_batch_size", 0))
    epoch_steps = math.ceil(len(dataloader) / max(1, grad_accum_steps))
    epochs_total_steps = int(train_cfg.get("epochs", 40)) * epoch_steps
    # Optional hard cap on optimizer steps. Useful for short smoke runs where
    # we want a real save artefact without grinding through a whole epoch.
    max_steps_cfg = train_cfg.get("max_steps", None)
    max_steps = int(max_steps_cfg) if max_steps_cfg is not None else 0

    # The LR schedule's "total_steps" should reflect how many optimizer
    # updates we ACTUALLY plan to take, i.e. ``min(epochs*epoch_steps,
    # max_steps)`` when both are set, falling back to whichever is non-zero.
    # The previous formulation set total_steps=min(epoch_steps, max_steps)
    # unconditionally, which on a single-epoch run made the cosine bottom out
    # (at min_lr_ratio * base_lr) by end-of-epoch even when ``max_steps`` was
    # nominally much larger than one epoch -- because ``epoch_steps`` itself
    # was the real cap. We now keep both numbers honest.
    if max_steps > 0:
        total_steps = min(epochs_total_steps, max_steps) if epochs_total_steps > 0 else max_steps
    else:
        total_steps = epochs_total_steps
    scheduler = build_scheduler(
        optimizer=optimizer,
        warmup_steps=int(train_cfg.get("warmup_steps", 5000)),
        total_steps=total_steps,
        min_lr_ratio=float(train_cfg.get("min_lr_ratio", 0.1)),
    )
    _maybe_print(
        rank0_debug_console,
        f"LR schedule: warmup={int(train_cfg.get('warmup_steps', 5000))}, "
        f"total={total_steps}, base_lr={float(train_cfg.get('lr', 1e-4)):.2e}, "
        f"min_lr_ratio={float(train_cfg.get('min_lr_ratio', 0.1))}",
    )
    _maybe_print(
        rank0_debug_console,
        f"Step accounting: dataloader_iters_per_epoch={len(dataloader)}, "
        f"grad_accum_steps={grad_accum_steps}, optimizer_steps_per_epoch={epoch_steps}",
    )

    amp_enabled = bool(train_cfg.get("amp", True)) and ddp.device.type == "cuda"
    amp_dtype_name = str(train_cfg.get("amp_dtype", "fp16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler_device = ddp.device.type if ddp.device.type in {"cuda", "cpu"} else "cuda"
    scaler = torch.amp.GradScaler(scaler_device, enabled=use_scaler)

    criterion = V3MattingLossComputer(
        weights=loss_cfg,
        fg_representation=str(cfg["data"].get("fg_representation", "premul")),
    ).to(ddp.device)

    start_epoch = 0
    global_step = 0
    if args.resume:
        if args.resume_model_only:
            ckpt_epoch, ckpt_global_step = load_model_weights_only(
                path=args.resume,
                model=model,
                device=ddp.device,
            )
            _maybe_print(
                rank0_debug_console,
                f"Loaded model weights from {args.resume} "
                f"(checkpoint epoch={ckpt_epoch}, global_step={ckpt_global_step}); "
                "optimizer/scheduler/scaler reset for fresh schedule.",
            )
        else:
            start_epoch, global_step = load_checkpoint(
                path=args.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if use_scaler else None,
                device=ddp.device,
            )
            _maybe_print(
                rank0_debug_console,
                f"Resumed from {args.resume}, start_epoch={start_epoch}, global_step={global_step}",
            )

    # Training can run the full, lowres, or tiled inference path; build the
    # options before CUDA warmup so warmup exercises the same active branches.
    inference_cfg = cfg.get("inference", {})
    train_inference_opts = V3InferenceOptions(
        mode=train_inference_mode,
        global_long_side_cap=int(train_cfg.get("global_long_side_cap", inference_cfg.get("global_long_side_cap", 2048))),
        tile_size=int(train_cfg.get("tile_size", inference_cfg.get("tile_size", 1024))),
        tile_overlap=int(train_cfg.get("tile_overlap", inference_cfg.get("tile_overlap", 64))),
        refine_on_uncertainty=bool(
            train_cfg.get("refine_on_uncertainty", inference_cfg.get("refine_on_uncertainty", True))
        ),
        refine_on_edges=bool(train_cfg.get("refine_on_edges", inference_cfg.get("refine_on_edges", True))),
        refine_on_spill_regions=bool(
            train_cfg.get("refine_on_spill_regions", inference_cfg.get("refine_on_spill_regions", True))
        ),
    )
    _maybe_print(rank0_debug_console, f"Training inference mode: {train_inference_opts.mode}")

    run_cuda_warmup = bool(train_cfg.get("cuda_warmup", True))
    if ddp.device.type == "cuda" and run_cuda_warmup:
        data_cfg = cfg["data"]
        warmup_buckets = list(data_cfg.get("multi_resolution_buckets", [384, 512, 768, 1024]))
        warmup_input_dtype = torch.float32
        if bool(data_cfg.get("device_augment", False)):
            warmup_input_dtype = _resolve_host_dtype(data_cfg.get("host_dtype")) or torch.float32
        _maybe_print(
            rank0_debug_console,
            f"Running CUDA warmup pass for resolution buckets {warmup_buckets}...",
        )
        cuda_warmup(
            model=model,
            device=ddp.device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            patch_size=int(model_cfg.get("patch_size", 8)),
            temporal_batch_size=int(train_cfg.get("temporal_batch_size", 4)),
            batch_size=int(cfg["data"].get("batch_size_per_gpu", 1)),
            input_dtype=warmup_input_dtype,
            resolution_buckets=warmup_buckets,
            inference_options=train_inference_opts,
            optimizer=optimizer,
            scaler=scaler if use_scaler else None,
            verbose=rank0_debug_console,
        )
        if ddp.is_distributed:
            dist.barrier()
    elif ddp.device.type == "cuda":
        _maybe_print(rank0_debug_console, "CUDA warmup skipped (train.cuda_warmup=false).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    filesystem_sync_session = _init_filesystem_sync_session(output_dir, ddp)

    epochs = int(train_cfg.get("epochs", 40))
    log_interval = int(train_cfg.get("log_interval", 20))
    save_interval = int(train_cfg.get("save_interval", 1))
    checkpoint_sync_timeout_minutes = float(
        train_cfg.get(
            "checkpoint_sync_timeout_minutes",
            max(120.0, ddp.timeout.total_seconds() / 60.0 * 2.0),
        )
    )
    checkpoint_sync_timeout_seconds = max(60.0, checkpoint_sync_timeout_minutes * 60.0)
    sync_log_metrics = bool(train_cfg.get("sync_log_metrics", False))
    show_loss_in_pbar = bool(train_cfg.get("show_loss_in_pbar", False))
    profile_log_metrics = bool(train_cfg.get("profile_log_metrics", False))
    ddp_sync_every_iter_cfg = train_cfg.get("ddp_sync_every_iter", None)
    if ddp_sync_every_iter_cfg is None:
        ddp_sync_every_iter = grad_accum_steps <= 1
    else:
        ddp_sync_every_iter = bool(ddp_sync_every_iter_cfg)
    pin_memory_enabled = bool(cfg["data"].get("pin_memory", True))
    cuda_prefetch_cfg = bool(train_cfg.get("cuda_prefetch", False))
    cuda_prefetch = cuda_prefetch_cfg and pin_memory_enabled and ddp.device.type == "cuda"
    cuda_prefetch_queue = max(1, int(train_cfg.get("cuda_prefetch_queue", 1)))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    autocast_device = ddp.device.type
    activation_offload_mode = str(train_cfg.get("activation_offload", "")).strip().lower()
    activation_offload_cpu = (
        activation_offload_mode in {"cpu", "host"}
        or bool(train_cfg.get("activation_offload_cpu", False))
    )
    activation_offload_pin_memory = bool(train_cfg.get("activation_offload_pin_memory", True))
    if activation_offload_cpu and ddp.device.type != "cuda":
        _maybe_print(
            rank0_debug_console,
            "Training notice: activation_offload=cpu ignored on non-CUDA device.",
        )
        activation_offload_cpu = False

    if rank0_debug_console:
        cadence = "every-iteration" if ddp_sync_every_iter else "optimizer-step-only"
        print(f"DDP sync cadence: {cadence} (sync on final temporal chunk)")
        if activation_offload_cpu:
            print(
                "Activation offload: saved tensors for backward are staged on CPU "
                f"(pin_memory={activation_offload_pin_memory})."
            )
        if grad_accum_steps > 1 and ddp_sync_every_iter:
            print(
                "Training notice: grad_accum_steps>1 with ddp_sync_every_iter=true can "
                "increase all-reduce overhead; consider ddp_sync_every_iter=false if communication dominates."
            )
        if cuda_prefetch_cfg and ddp.device.type == "cuda" and not pin_memory_enabled:
            print(
                "Training notice: train.cuda_prefetch=true but data.pin_memory=false; "
                "CUDA prefetch is disabled. Enable pin_memory for host-to-device overlap."
            )

    # Fixed chunk count across all ranks so no rank idles while others still compute.
    # +1 is only needed when temporal jitter can duplicate a frame.
    clip_len_max = int(cfg["data"].get("clip_len_max", 12))
    temporal_jitter_p = float(cfg["data"].get("temporal_jitter_p", 0.1))
    extra_temporal_frame = 1 if temporal_jitter_p > 0 else 0
    fixed_n_chunks = (
        math.ceil((clip_len_max + extra_temporal_frame) / temporal_batch_size)
        if temporal_batch_size > 0
        else 1
    )

    profile_wait = max(0, int(args.profile_wait))
    profile_warmup = max(0, int(args.profile_warmup))
    profile_active = max(1, int(args.profile_active))
    compile_enabled = bool(train_cfg.get("compile", False))
    auto_adjusted_profile_schedule = False

    # Profiling with torch.compile often captures first-call graph compilation
    # when wait/warmup are both 0, which hides steady-state behavior. Unless
    # explicitly requested, move that compile hit into an unrecorded wait step.
    if (
        args.profile
        and compile_enabled
        and not bool(args.profile_include_compile)
        and profile_wait == 0
        and profile_warmup == 0
    ):
        profile_wait = 1
        auto_adjusted_profile_schedule = True

    profiling = args.profile and ddp.device.type == "cuda" and is_rank0()
    prof: torch.profiler.profile | None = None
    profile_total_steps = 0
    if profiling:
        profile_dir = Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_total_steps = profile_wait + profile_warmup + profile_active
        print(
            f"Profiling enabled (rank 0 only): wait={profile_wait}, warmup={profile_warmup}, "
            f"active={profile_active} → {profile_total_steps} iterations then exit"
        )
        if auto_adjusted_profile_schedule:
            print(
                "  auto-adjusted schedule: wait=1 to keep torch.compile startup "
                "out of the recorded active window (disable with --profile-include-compile)"
            )
        if args.profile_stacks:
            print("  with_stack=True (expect ~10x slower iterations)")
        print(f"Traces will be saved to: {profile_dir.resolve()}")
        prof_schedule = schedule(
            wait=profile_wait,
            warmup=profile_warmup,
            active=profile_active,
            repeat=1,
        )
        prof = torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=prof_schedule,
            on_trace_ready=tensorboard_trace_handler(str(profile_dir)),
            record_shapes=True,
            profile_memory=True,
            with_stack=args.profile_stacks,
            with_flops=True,
            acc_events=True,
        )
        prof.start()
    elif args.profile and ddp.device.type == "cuda" and not is_rank0():
        profile_total_steps = profile_wait + profile_warmup + profile_active

    _interrupted = False
    profile_finished = False
    prof_stopped = False
    max_steps_reached = False

    # ------------------------------------------------------------------
    # Interrupt-handling strategy (rewritten after multiple failed attempts
    # at a "polled flag + cooperative break" cleanup).
    #
    # Empirical findings from the runtime logs:
    #   * Python signal handlers run on the main thread *between bytecodes*.
    #     Whichever rank happens to be at a Python boundary when SIGINT/
    #     SIGTERM lands runs its handler immediately; ranks that are inside
    #     a long C-level call (CUDA kernel launch queue, NCCL collective,
    #     DataLoader/queue.get auto-retry under PEP 475, etc.) may never run
    #     their handler before torchrun's elastic agent escalates to SIGKILL.
    #   * Cooperative shutdown via a polled ``_interrupted`` flag is therefore
    #     unreliable: only some ranks see it, the surviving ranks then race
    #     each other through ``cleanup_distributed`` and stall NCCL teardown.
    #
    # New strategy:
    #   1. The signal handler does the emergency checkpoint save *itself*,
    #      synchronously. Doing I/O in a Python signal handler is safe (it
    #      runs in the main thread, not a real signal context). All ranks
    #      hold the same DDP-averaged params + identical optimizer state, so
    #      whichever rank's handler fires first wins a filesystem lock and
    #      writes the canonical checkpoint - we no longer depend on rank 0
    #      specifically being lucky enough to escape its CUDA call.
    #   2. The handler then arms a daemon ``threading.Timer`` that calls
    #      ``os._exit`` after a small grace period. This guarantees the
    #      worker exits even if the cooperative path is wedged. The timer
    #      thread runs independently of the main thread, so a stuck CUDA
    #      call does not block the force-exit.
    #   3. The legacy polled-flag path is kept as a "best effort" soft
    #      shutdown for the lucky case where every rank actually breaks out.
    # ------------------------------------------------------------------

    # Mutable container so the handler closure reads the *latest* loop
    # values at the moment the signal fires (closure cells alone don't help
    # here, because ``epoch`` is a for-loop target that may not be bound
    # yet when the first signal arrives).
    _handler_state = {"epoch": start_epoch, "global_step": int(global_step)}
    _emergency_save_state = {"in_progress": False, "done": False, "path": None}
    _force_exit_state: Dict[str, Any] = {"timer": None}
    _interrupt_lock_path = output_dir / ".interrupted_save.lock"

    # Clear any stale lock from a previous interrupted run so this run can
    # write its own emergency checkpoint when needed.
    if is_rank0():
        try:
            _interrupt_lock_path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass

    _sigint_count = {"n": 0}

    def _emergency_save_in_handler() -> None:
        """Best-effort synchronous checkpoint write from the signal handler.

        Any rank may win the filesystem lock; in DDP every rank holds the
        same averaged parameters and identical optimizer state, so whichever
        one fires its handler first writes the canonical ``checkpoint_
        interrupted_step*.pt``. This is the key reliability win over the
        previous "only rank 0 saves in finally" path, which silently lost
        the run whenever rank 0 was stuck inside a C call when the signal
        arrived.
        """
        if _emergency_save_state["in_progress"] or _emergency_save_state["done"]:
            return
        _emergency_save_state["in_progress"] = True
        try:
            try:
                fd = os.open(
                    str(_interrupt_lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                )
                try:
                    os.write(
                        fd,
                        f"rank={int(os.environ.get('RANK', '0'))} pid={os.getpid()}".encode(),
                    )
                finally:
                    os.close(fd)
            except FileExistsError:
                # Another rank already won the lock and is/has saved.
                return

            gs = int(_handler_state["global_step"])
            ep = int(_handler_state["epoch"])
            ckpt_path = output_dir / f"checkpoint_interrupted_step{gs:06d}.pt"
            save_checkpoint(
                path=ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler if use_scaler else None,
                epoch=ep,
                global_step=gs,
                config=cfg,
            )
            _emergency_save_state["path"] = ckpt_path
            _emergency_save_state["done"] = True
            print(
                f"\n[train] in-handler emergency checkpoint saved "
                f"(rank={int(os.environ.get('RANK', '0'))}): {ckpt_path}",
                flush=True,
            )
        except BaseException as exc:
            try:
                print(
                    f"\n[train] in-handler emergency save FAILED: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            except Exception:
                pass
        finally:
            _emergency_save_state["in_progress"] = False

    def _spawn_kill_backstop(delay_s: float) -> None:
        """Spawn a detached child shell that ``kill -9``s us after ``delay_s``.

        The child runs in its own process, completely independent of our
        Python interpreter and GIL state. Even if every Python thread in
        this rank is wedged inside a C extension call, the kernel will
        deliver SIGKILL to us when the child runs ``kill -9``. SIGKILL
        cannot be caught or ignored, so this is the bullet-proof
        termination path.
        """
        import subprocess as _subp
        pid = os.getpid()
        # Use ``setsid`` so the child survives if the agent kills our
        # process group; ``nohup`` plus ``</dev/null`` detaches from our
        # tty. The shell sleeps then sends SIGKILL.
        _subp.Popen(
            ["sh", "-c", f"sleep {float(delay_s):.1f} && kill -9 {pid}"],
            stdin=_subp.DEVNULL,
            stdout=_subp.DEVNULL,
            stderr=_subp.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )

    def _start_force_exit_timer(delay_s: float) -> None:
        """Arm/re-arm a daemon timer that ``os._exit``s after ``delay_s``.

        Runs on a background thread so it is *independent* of whether the
        main thread is wedged inside a C-level call (CUDA, NCCL, blocking
        queue.get, etc.). This is what guarantees we actually exit.
        """
        existing = _force_exit_state["timer"]
        if existing is not None:
            try:
                existing.cancel()
            except Exception:
                pass

        def _force() -> None:
            try:
                try:
                    print(
                        f"\n[train] force-exit (rank={int(os.environ.get('RANK', '0'))})"
                        f" after {delay_s:.1f}s grace period",
                        flush=True,
                    )
                except Exception:
                    pass
            finally:
                os._exit(0)

        t = threading.Timer(delay_s, _force)
        t.daemon = True
        t.start()
        _force_exit_state["timer"] = t

    def _abort_distributed_for_emergency_save() -> None:
        """Abort in-flight distributed work before checkpoint tensor copies."""
        try:
            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except BaseException:
            pass

    def _sigint_handler(signum: int, frame: object) -> None:
        nonlocal _interrupted
        _sigint_count["n"] += 1
        if _sigint_count["n"] == 1:
            # First interrupt: arm OS-level kill backstop + Python timer,
            # then attempt the lock-arbitrated emergency save. The
            # watchdog thread also runs the same sequence (via the
            # set_wakeup_fd pipe) so whichever path executes first wins.
            try:
                _spawn_kill_backstop(90.0)
            except Exception:
                pass
            _start_force_exit_timer(60.0)
            _abort_distributed_for_emergency_save()
            _emergency_save_in_handler()
        else:
            # Second interrupt: shrink everything to ~1s and restore
            # default handlers so a third signal hard-kills via the OS.
            try:
                _spawn_kill_backstop(2.0)
            except Exception:
                pass
            _start_force_exit_timer(1.0)
            for _sig in (signal.SIGINT, signal.SIGTERM):
                try:
                    signal.signal(_sig, signal.SIG_DFL)
                except (ValueError, OSError):
                    pass
            if is_rank0():
                print(
                    "\n[train] second interrupt - grace period shortened, "
                    "press Ctrl-C again to force-quit immediately.",
                    flush=True,
                )
        _interrupted = True

    prev_sigint = signal.signal(signal.SIGINT, _sigint_handler)
    # Also catch SIGTERM. torchrun's elastic agent reacts to a single Ctrl-C
    # by raising SignalException in its own process and then calling
    # MultiprocessingContext.close(), which sends SIGTERM to every worker.
    prev_sigterm = signal.signal(signal.SIGTERM, _sigint_handler)

    # ------------------------------------------------------------------
    # Out-of-thread signal watchdog.
    #
    # Python signal handlers run on the *main thread* between bytecodes.
    # The runtime logs proved that under DDP, ranks whose main thread is
    # parked inside a long C-level call (CUDA kernel-launch queue, NCCL
    # collective, a DataLoader/queue.get auto-retried by PEP 475, ...)
    # never get to run ``_sigint_handler`` at all - so neither the
    # in-handler emergency save *nor* the ``threading.Timer`` force-exit
    # is armed, and the worker only dies when torchrun's elastic agent
    # escalates to SIGKILL ~30 s later.
    #
    # ``signal.set_wakeup_fd`` makes the OS-level signal trampoline
    # write the signal number to a pipe *before* returning to user
    # code. A background thread blocked on ``os.read`` from that pipe
    # therefore wakes up the instant a signal is delivered, regardless
    # of what the main thread is doing. From that thread we can arm
    # the force-exit timer and attempt the lock-arbitrated save - both
    # of which are idempotent with the in-handler path, so when the
    # main thread *does* run ``_sigint_handler`` the duplicate calls
    # are no-ops.
    # ------------------------------------------------------------------
    _signal_pipe_r, _signal_pipe_w = os.pipe()
    try:
        os.set_blocking(_signal_pipe_w, False)
    except OSError:
        pass
    _prev_wakeup_fd = signal.set_wakeup_fd(_signal_pipe_w)

    def _signal_watchdog() -> None:
        try:
            wake = os.read(_signal_pipe_r, 1)
        except OSError:
            return
        # EOF means our wakeup pipe was closed during normal teardown,
        # not that an actual signal was delivered.
        if not wake:
            return
        signum = int(wake[0])
        if signum not in {signal.SIGINT, signal.SIGTERM}:
            return
        # ----------------------------------------------------------------
        # Spawn an OS-level SIGKILL backstop first. Python timers require
        # the GIL to fire; if another thread on this rank is wedged inside
        # a C extension, the ``threading.Timer`` force-exit can be starved.
        # A detached child shell is immune to our GIL and ``kill -9`` cannot
        # be caught.
        # ----------------------------------------------------------------
        try:
            _spawn_kill_backstop(90.0)
        except BaseException:
            pass
        # Arm the soft Python-level force-exit timer too (cheap belt-and-
        # suspenders for the common case where the GIL is fine).
        try:
            _start_force_exit_timer(60.0)
        except BaseException:
            pass
        # ----------------------------------------------------------------
        # Per the PyTorch docs (Distributed Shutdown section),
        # ``destroy_process_group()`` is what calls ``ncclCommAbort`` on
        # the underlying communicator. Doing this before the save aborts
        # any in-flight collective whose peers have abandoned us, which
        # frees the CUDA stream that ``save_checkpoint``'s GPU->CPU copy
        # would otherwise be queued behind.
        # ----------------------------------------------------------------
        _abort_distributed_for_emergency_save()
        try:
            _emergency_save_in_handler()
        except BaseException:
            pass
        # Direct exit from the watchdog thread - we have nothing left to
        # do and we don't want to depend on the Timer thread acquiring
        # the GIL.
        try:
            print(
                f"\n[train] watchdog exit (rank={int(os.environ.get('RANK', '0'))})"
                f" save_done={_emergency_save_state['done']}",
                flush=True,
            )
        except Exception:
            pass
        os._exit(0)

    _watchdog_thread = threading.Thread(
        target=_signal_watchdog,
        name="signal-watchdog",
        daemon=True,
    )
    _watchdog_thread.start()

    try:
        for epoch in range(start_epoch, epochs):
            if _interrupted or max_steps_reached:
                break

            _handler_state["epoch"] = epoch

            set_dataloader_epoch_safe(dataloader, epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_loss = torch.zeros((), device=ddp.device, dtype=torch.float32)
            pbar: tqdm | None = None
            if is_rank0():
                pbar = tqdm(
                    total=len(dataloader),
                    desc=f"Epoch {epoch + 1}/{epochs}",
                    dynamic_ncols=True,
                    leave=True,
                )

            prefetch_device_transform = device_transform if cuda_prefetch else None

            for it, batch in enumerate(
                iterate_device_batches(
                    dataloader,
                    ddp.device,
                    use_cuda_prefetch=cuda_prefetch,
                    prefetch_queue=cuda_prefetch_queue,
                    post_move_fn=prefetch_device_transform,
                )
            ):
                if _interrupted:
                    break

                if device_transform is not None and prefetch_device_transform is None:
                    # Spatial crop + photometric/sensor/codec augments +
                    # representation-specific composite + coarse_alpha_init. Runs on the
                    # main compute stream when CUDA prefetch is disabled. With
                    # CUDA prefetch enabled, the prefetcher applies this before
                    # queueing so the queue holds cropped training tensors
                    # instead of large native-resolution decode tensors.
                    with nvtx_range("device_transform_main"):
                        with torch.no_grad():
                            batch = device_transform(batch)

                temporal_chunks = build_temporal_chunks(
                    total_frames=int(batch["video_rgb"].shape[1]),
                    chunk_size=temporal_batch_size,
                )
                real_n_chunks = len(temporal_chunks)
                while len(temporal_chunks) < fixed_n_chunks:
                    temporal_chunks.append(temporal_chunks[-1])

                valid_mask = batch.get("valid_mask")
                if valid_mask is not None:
                    total_weight_denom = valid_mask.sum().clamp_min(1.0)
                else:
                    total_weight_denom = batch["video_rgb"].new_tensor(float(batch["video_rgb"].shape[1]))

                coarse_seed = batch["coarse_alpha_init"]
                did_backward = False
                batch_total_loss = batch["video_rgb"].new_tensor(0.0)
                batch_loss_items: Dict[str, torch.Tensor] = {}
                do_step = ((it + 1) % grad_accum_steps == 0) or (it + 1 == len(dataloader))

                # All ranks run the same number of forward/backward passes so they
                # finish compute at roughly the same time.  Padding chunks beyond
                # the real count use zero weight (no gradient contribution).
                # By default, sync DDP on each iteration's final chunk to avoid
                # bursty all-reduce phases when grad_accum_steps > 1.
                no_sync = model.no_sync if ddp.is_distributed else nullcontext

                for chunk_idx, (start_t, end_t) in enumerate(temporal_chunks):
                    is_padding_chunk = chunk_idx >= real_n_chunks
                    with nvtx_range("slice_temporal_chunk"):
                        chunk_batch = slice_batch_temporal(batch, start_t, end_t)
                    if chunk_idx > 0:
                        chunk_batch["coarse_alpha_init"] = coarse_seed

                    if is_padding_chunk:
                        chunk_weight = batch["video_rgb"].new_tensor(0.0)
                    else:
                        chunk_valid_mask = chunk_batch.get("valid_mask")
                        if chunk_valid_mask is not None:
                            chunk_weight = (chunk_valid_mask.sum() / total_weight_denom).to(
                                dtype=batch["video_rgb"].dtype
                            )
                        else:
                            chunk_weight = batch["video_rgb"].new_tensor(
                                (end_t - start_t) / max(1, int(batch["video_rgb"].shape[1]))
                            )

                    if ddp_sync_every_iter:
                        should_sync_ddp = chunk_idx == len(temporal_chunks) - 1
                    else:
                        should_sync_ddp = do_step and (chunk_idx == len(temporal_chunks) - 1)
                    sync_context = nullcontext if (not ddp.is_distributed or should_sync_ddp) else no_sync

                    with sync_context():
                        offload_context = (
                            torch.autograd.graph.save_on_cpu(pin_memory=activation_offload_pin_memory)
                            if activation_offload_cpu
                            else nullcontext()
                        )
                        with offload_context:
                            with torch.amp.autocast(autocast_device, enabled=amp_enabled, dtype=amp_dtype):
                                model_kwargs: Dict[str, Any] = {}
                                if "global_video_rgb" in chunk_batch:
                                    model_kwargs["global_video"] = chunk_batch["global_video_rgb"]
                                if "global_coarse_alpha_init" in chunk_batch:
                                    model_kwargs["global_coarse_alpha_init"] = chunk_batch["global_coarse_alpha_init"]
                                if "global_fg_gt" in chunk_batch:
                                    model_kwargs["global_fg_guidance"] = chunk_batch["global_fg_gt"]
                                if "tile_coords" in chunk_batch:
                                    model_kwargs["tile_coords"] = chunk_batch["tile_coords"]
                                if "source_hw" in chunk_batch:
                                    model_kwargs["source_hw"] = chunk_batch["source_hw"]

                                with nvtx_range("model_forward"):
                                    pred = model(
                                        video=chunk_batch["video_rgb"],
                                        coarse_alpha_init=chunk_batch["coarse_alpha_init"],
                                        valid_mask=chunk_batch.get("valid_mask"),
                                        bg_for_comp=chunk_batch["bg_gt"],
                                        inference_options=train_inference_opts,
                                        **model_kwargs,
                                    )
                                with nvtx_range("loss_compute"):
                                    total_loss, loss_items = criterion(pred, chunk_batch)
                                    weighted_total_loss = total_loss * chunk_weight
                                    loss = weighted_total_loss / max(1, grad_accum_steps)

                        with nvtx_range("backward"):
                            if use_scaler:
                                scaler.scale(loss).backward()
                            else:
                                loss.backward()
                        did_backward = True

                    if not is_padding_chunk:
                        batch_total_loss = batch_total_loss + weighted_total_loss.detach()
                        for k, v in loss_items.items():
                            v_detached = (v * chunk_weight).detach()
                            if k in batch_loss_items:
                                batch_loss_items[k] = batch_loss_items[k] + v_detached
                            else:
                                batch_loss_items[k] = v_detached

                        coarse_seed = pred["alpha_pred"][:, -1].detach()

                if do_step and did_backward:
                    if grad_clip > 0:
                        with nvtx_range("grad_clip"):
                            if use_scaler:
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                    with nvtx_range("optimizer_step"):
                        if use_scaler:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()

                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                    global_step += 1
                    _handler_state["global_step"] = global_step

                    if max_steps > 0 and global_step >= max_steps:
                        ckpt_path = output_dir / f"checkpoint_step_{global_step:06d}.pt"

                        def _save_max_steps_checkpoint() -> None:
                            tqdm.write(
                                f"Reached max_steps={max_steps} at global_step={global_step}; "
                                "saving final checkpoint and stopping."
                            )
                            tqdm.write(f"Saving checkpoint: {ckpt_path}")
                            save_checkpoint(
                                path=ckpt_path,
                                model=model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                scaler=scaler if use_scaler else None,
                                epoch=epoch,
                                global_step=global_step,
                                config=cfg,
                            )
                            tqdm.write(f"Saved checkpoint: {ckpt_path}")

                        _save_checkpoint_with_filesystem_rendezvous(
                            ddp=ddp,
                            output_dir=output_dir,
                            sync_session=filesystem_sync_session,
                            sync_name=f"max_steps_step{global_step:06d}",
                            timeout_seconds=checkpoint_sync_timeout_seconds,
                            checkpoint_fn=_save_max_steps_checkpoint,
                        )
                        max_steps_reached = True
                        break
                elif do_step:
                    optimizer.zero_grad(set_to_none=True)

                should_log = (it + 1) % log_interval == 0
                if args.profile and not profile_log_metrics:
                    should_log = False

                running_loss = running_loss + batch_total_loss.detach().to(dtype=torch.float32)

                # IMPORTANT: any ``reduce_scalar`` call in this block is a
                # collective and MUST be executed by every rank, not just
                # rank 0. Previously the allreduces were nested inside
                # ``if pbar is not None`` / ``if is_rank0()`` guards, which
                # would deadlock the moment ``sync_log_metrics=true`` was
                # set (rank 0 posts an allreduce nobody else does). Do the
                # reduce first on all ranks, then fold the rank-0-only
                # formatting and printing in afterwards.
                pbar_needs_loss = show_loss_in_pbar and should_log
                if pbar_needs_loss:
                    loss_for_bar = batch_total_loss.detach()
                    if sync_log_metrics and ddp.is_distributed:
                        loss_for_bar = reduce_scalar(loss_for_bar, average=True)
                else:
                    loss_for_bar = None

                reduced_items_t: Dict[str, torch.Tensor] | None = None
                if should_log:
                    if sync_log_metrics and ddp.is_distributed:
                        reduced_items_t = {
                            k: reduce_scalar(v, average=True) for k, v in batch_loss_items.items()
                        }
                    elif is_rank0():
                        reduced_items_t = {k: v for k, v in batch_loss_items.items()}

                if pbar is not None:
                    pbar.update(1)
                    lr_str = f"{optimizer.param_groups[0]['lr']:.2e}"
                    if pbar_needs_loss:
                        pbar.set_postfix(loss=f"{float(loss_for_bar):.4f}", lr=lr_str, step=global_step)
                    else:
                        pbar.set_postfix(lr=lr_str, step=global_step)

                if should_log and is_rank0() and reduced_items_t is not None:
                    reduced_items = {k: float(v.detach().to(device="cpu")) for k, v in reduced_items_t.items()}
                    reduced_items_str = " ".join(f"{k}={v:.4f}" for k, v in reduced_items.items())
                    lr = optimizer.param_groups[0]["lr"]
                    tqdm.write(
                        f"epoch={epoch:03d} iter={it + 1:05d}/{len(dataloader):05d} "
                        f"step={global_step:06d} lr={lr:.2e} {reduced_items_str}"
                    )

                if prof is not None:
                    prof.step()

                if args.profile and profile_total_steps > 0 and it + 1 >= profile_total_steps:
                    if prof is not None:
                        if ddp.device.type == "cuda":
                            torch.cuda.synchronize(ddp.device)
                        try:
                            prof.stop()
                        except RuntimeError as exc:
                            tqdm.write(f"Profiler stop warning: {exc}")
                        prof_stopped = True

                        chrome_trace_path = profile_dir / f"chrome_trace_rank{ddp.rank}.json"
                        chrome_trace_msg = str(chrome_trace_path)
                        try:
                            prof.export_chrome_trace(str(chrome_trace_path))
                        except Exception as exc:
                            chrome_trace_msg = (
                                "manual export failed; TensorBoard trace files may still have "
                                f"been written by on_trace_ready ({type(exc).__name__}: {exc})"
                            )
                        print(f"\nProfiler traces saved to {profile_dir.resolve()}/")
                        print(f"  Chrome trace:     {chrome_trace_msg}")
                        print(f"  TensorBoard dir:  {profile_dir.resolve()}/")
                        print(f"\nView with:")
                        print(f"  tensorboard --logdir {profile_dir} --port 6006")
                        if chrome_trace_msg.endswith(".json") or chrome_trace_msg.endswith(".json.gz"):
                            print(f"  or open {chrome_trace_msg} in chrome://tracing")
                        if args.profile_print_tables:
                            print(f"\n--- Top CUDA kernels by total time ---")
                            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
                            print(f"\n--- Top operators by GPU memory ---")
                            print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))
                        else:
                            print("Profiler tables skipped. Use --profile-print-tables to include them.")
                    # Rank 0 may spend seconds flushing profiler output. Keep
                    # peers alive until the trace is on disk, then let every
                    # rank leave the loop together and destroy DDP cleanly.
                    if ddp.is_distributed:
                        if ddp.device.type == "cuda" and dist.get_backend() == "nccl":
                            dist.barrier(device_ids=[ddp.local_rank])
                        else:
                            dist.barrier()
                    profile_finished = True
                    break

            if pbar is not None:
                pbar.close()

            if profile_finished:
                break

            # NOTE: do NOT skip ``reduce_scalar`` on interrupt. It is the only
            # rendezvous between the inner training loop and the cleanup path;
            # if a subset of ranks bails out here while peers are still inside
            # a backward()/NCCL collective, those peers can no longer complete
            # their allreduce and the whole world hangs until the NCCL watchdog
            # aborts it (~30s+). We only skip the *disk* checkpoint write on
            # rank 0 below — the finally block writes an emergency checkpoint
            # with the same state, so the regular per-epoch save is redundant
            # on interrupt and was a multi-second stall that other ranks then
            # waited out in cleanup.
            should_save_epoch_checkpoint = (
                not max_steps_reached
                and not _interrupted
                and ((epoch + 1) % save_interval == 0 or (epoch + 1) == epochs)
            )

            # Save before the epoch-loss allreduce. In DDP, rank 0 has
            # completed the same optimizer steps as its peers once it exits
            # the inner loop, but peers can still be draining DataLoader /
            # CUDA-prefetch work. The checkpoint helper uses marker files
            # instead of a NCCL barrier so long disk writes do not trip the
            # watchdog while peers wait.
            if max_steps_reached:
                pass  # Final checkpoint already saved at the max_steps cap.
            elif _interrupted:
                # Skip the regular per-epoch disk write on interrupt: the
                # ``finally`` block writes an emergency checkpoint with the
                # exact same state, and the back-to-back saves were a
                # multi-second shutdown stall that other ranks waited out
                # inside ``cleanup_distributed``.
                pass
            elif should_save_epoch_checkpoint:
                ckpt_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pt"

                def _save_epoch_checkpoint() -> None:
                    tqdm.write(f"Saving checkpoint: {ckpt_path}")
                    save_checkpoint(
                        path=ckpt_path,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler if use_scaler else None,
                        epoch=epoch,
                        global_step=global_step,
                        config=cfg,
                    )
                    tqdm.write(f"Saved checkpoint: {ckpt_path}")

                _save_checkpoint_with_filesystem_rendezvous(
                    ddp=ddp,
                    output_dir=output_dir,
                    sync_session=filesystem_sync_session,
                    sync_name=f"epoch{epoch:03d}_step{global_step:06d}",
                    timeout_seconds=checkpoint_sync_timeout_seconds,
                    checkpoint_fn=_save_epoch_checkpoint,
                )

            epoch_loss = running_loss / max(1, len(dataloader))
            if ddp.is_distributed:
                epoch_loss = reduce_scalar(epoch_loss, average=True)
            if is_rank0():
                tqdm.write(f"[epoch {epoch:03d}] avg_total_loss={float(epoch_loss.detach().to(device='cpu')):.6f}")

    except KeyboardInterrupt:
        _interrupted = True
    except Exception:
        raise

    finally:
        signal.signal(signal.SIGINT, prev_sigint)
        try:
            signal.signal(signal.SIGTERM, prev_sigterm)
        except (ValueError, OSError):
            pass
        try:
            signal.set_wakeup_fd(_prev_wakeup_fd)
        except (ValueError, OSError):
            pass
        for _fd in (_signal_pipe_w, _signal_pipe_r):
            try:
                os.close(_fd)
            except OSError:
                pass

        if prof is not None:
            if not prof_stopped:
                try:
                    prof.stop()
                except RuntimeError:
                    pass

        if _interrupted and is_rank0() and not _emergency_save_state["done"]:
            # Fallback: the in-handler save did not run on this rank (e.g.
            # we got here via KeyboardInterrupt without our handler firing,
            # or another rank's handler also missed). Best-effort save.
            print("\n\nInterrupted — saving emergency checkpoint (finally fallback)...")
            ckpt_path = output_dir / f"checkpoint_interrupted_step{global_step:06d}.pt"
            try:
                save_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if use_scaler else None,
                    epoch=epoch if "epoch" in locals() else start_epoch,
                    global_step=global_step,
                    config=cfg,
                )
                print(f"Saved: {ckpt_path}")
            except Exception as exc:
                print(f"Failed to save emergency checkpoint: {exc}")

        if profile_finished and is_rank0():
            tqdm.write("Profile capture completed.")
        elif not _interrupted and is_rank0():
            tqdm.write("Training completed.")

        cleanup_distributed()


if __name__ == "__main__":
    train()
