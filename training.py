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
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


import argparse
import importlib
import math
import os
import random
import signal
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

# Process-wide environment defaults, applied BEFORE torch import so they
# influence threading-library initialisation and the CUDA caching allocator.
# ``setdefault`` keeps any user-provided environment overrides intact.
#
#   OMP_NUM_THREADS=2 and MKL/NUMEXPR_NUM_THREADS=1
#       Each DataLoader worker can spawn its own OpenMP/MKL pool. Without
#       this cap, ``num_workers`` workers each grabbing N threads
#       oversubscribes CPU (the typical failure mode is ~30-50% of host
#       throughput lost to context switches). The workers individually
#       call ``torch.set_num_threads(num_torch_threads)`` for the few
#       transform ops that actually benefit from parallel CPU.
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
    ("OMP_NUM_THREADS", "2"),
    ("MKL_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
):
    os.environ.setdefault(_key, _val)

import torch
from torch import nn
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
    load_config,
    move_batch_to_device,
    pad_collate_video,
)
from utils.ema import ModelEma
from utils.nvtx_utils import nvtx_range
from utils.wandb_logging import init_wandb_logger


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
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


def select_device(train_cfg: Dict[str, Any]) -> torch.device:
    requested = str(train_cfg.get("device", "")).strip().lower()
    if requested in {"cpu"}:
        return torch.device("cpu")
    if requested and requested not in {"cuda", "gpu"} and not requested.startswith("cuda:"):
        raise ValueError("train.device must be 'cuda', 'cuda:N', or 'cpu' when provided.")
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"train.device={requested!r} requested CUDA, but CUDA is unavailable.")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
    create_dataloader: Callable[..., DataLoader]
    set_dataloader_epoch: Callable[[DataLoader, int], None]


@lru_cache(maxsize=16)
def _import_dataset_runtime(module_name: str) -> DatasetRuntime:
    module = importlib.import_module(module_name)

    required = {
        "CorridorKeyWebSequenceDataset",
        "CorridorKeySequenceDataset",
        "create_single_gpu_dataloader",
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
        create_dataloader=getattr(module, "create_single_gpu_dataloader"),
        set_dataloader_epoch=getattr(module, "set_dataloader_epoch"),
    )


def resolve_dataset_runtime(data_cfg: Dict[str, Any]) -> DatasetRuntime:
    preferred_module = str(data_cfg.get("dataset_module", "")).strip()
    candidates = [
        preferred_module,
        "CorridorKeyDataset.dataset",
    ]

    seen = set()
    unique_candidates: List[str] = []
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        unique_candidates.append(candidate)

    errors: List[str] = []
    for module_name in unique_candidates:
        try:
            return _import_dataset_runtime(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc}")

    raise ImportError(
        "Unable to resolve dataset module. Tried: "
        + ", ".join(unique_candidates)
        + ". Errors: "
        + " | ".join(errors)
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

    When *optimizer* is provided we deliberately do NOT call
    ``optimizer.step()`` on the warmup gradients: the dummy loss only uses
    means of active outputs to keep gradient magnitudes small. We zero
    gradients after the backward pass instead so cuDNN/CUDA caches are
    warmed without poisoning the optimizer state.
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
            # Touch every active output branch. Use mean() rather than sum()
            # to keep gradient magnitudes small while still priming kernel
            # selection and cache state.
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

    if verbose:
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


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler | None,
    model_ema: ModelEma | None,
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
    if model_ema is not None:
        ckpt["model_ema"] = _checkpoint_value_to_cpu(model_ema.state_dict())
        ckpt["model_ema_state_dict"] = _checkpoint_value_to_cpu(model_ema.model_state_dict())

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
    debug_console_enabled: bool,
) -> nn.Module:
    if not bool(model_cfg.get("compile_boundary_refine", False)):
        return model

    if bool(train_cfg.get("compile", False)):
        _maybe_print(
            debug_console_enabled,
            "compile_boundary_refine ignored because train.compile=true already wraps the full model.",
        )
        return model

    boundary_refine = getattr(model, "boundary_refine", None)
    if boundary_refine is None:
        _maybe_print(
            debug_console_enabled,
            "compile_boundary_refine requested, but this model has no boundary_refine module.",
        )
        return model

    dynamic_default = bool(train_cfg.get("compile_dynamic", False))
    dynamic = bool(model_cfg.get("compile_boundary_refine_dynamic", dynamic_default))
    model.boundary_refine = torch.compile(boundary_refine, dynamic=dynamic)
    _maybe_print(debug_console_enabled, f"torch.compile enabled for boundary_refine (dynamic={dynamic})")
    return model


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    scaler: torch.amp.GradScaler | None,
    model_ema: ModelEma | None,
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
    if model_ema is not None:
        ema_state = ckpt.get("model_ema", ckpt.get("model_ema_state_dict"))
        if ema_state is not None:
            model_ema.load_state_dict(ema_state)
        else:
            model_ema.reset(model)

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


def build_dataloader(config: Dict[str, Any]) -> DataLoader:
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
    stream_temporal_batches = bool(data_cfg.get("stream_temporal_batches", False))
    temporal_stream_chunk_size = int(
        data_cfg.get(
            "temporal_stream_chunk_size",
            config.get("train", {}).get("temporal_batch_size", clip_len_max),
        )
    )

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
    if stream_temporal_batches:
        if temporal_stream_chunk_size < 1:
            raise ValueError("data.temporal_stream_chunk_size must be >= 1 when streaming is enabled")
        if not device_offload:
            raise ValueError("data.stream_temporal_batches requires data.device_augment=true")
        if clip_len_min != clip_len_max:
            raise ValueError("data.stream_temporal_batches currently requires clip_len_min == clip_len_max")
        if float(data_cfg.get("temporal_jitter_p", 0.1)) != 0.0:
            raise ValueError("data.stream_temporal_batches currently requires temporal_jitter_p: 0.0")

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
        green_foreground_augment_p=float(data_cfg.get("green_foreground_prob", data_cfg.get("green_foreground_augment_p", 0.0))),
        green_foreground_strength_min=float(data_cfg.get("green_foreground_strength_min", 0.25)),
        green_foreground_strength_max=float(data_cfg.get("green_foreground_strength_max", 0.75)),
        temporal_jitter_p=float(data_cfg.get("temporal_jitter_p", 0.1)),
        skip_temporal_sample=True,
        disable_temporal_augment=stream_temporal_batches,
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

    dataset_root = Path(data_cfg.get("root_dir", data_cfg.get("dataset_root", "CorridorKeyDataset")))
    shard_glob = str(data_cfg.get("shard_glob", "*.tar"))
    webdataset_root = resolve_webdataset_root(data_cfg=data_cfg, default_root=dataset_root, shard_glob=shard_glob)

    dataset_source = str(data_cfg.get("dataset", data_cfg.get("dataset_type", "local"))).strip().lower()
    if dataset_source in {"", "auto", "local", "filesystem", "fs", "webdataset", "wds", "tar"}:
        dataset_source = "local"
    elif dataset_source in {"web", "hf", "huggingface", "remote"}:
        dataset_source = "web"
    else:
        raise ValueError(
            f"data.dataset={data_cfg.get('dataset')!r} is not supported. "
            "Use 'local' or 'web'."
        )

    if dataset_source == "local" and webdataset_root is not None:
        dataset_root = webdataset_root

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
        "cached_four_quadrant_batch": bool(data_cfg.get("cached_four_quadrant_batch", False)),
        "global_context_root_dir": data_cfg.get("global_context_root_dir"),
        "global_context_long_side": int(data_cfg.get("global_context_long_side", 0)),
        "global_context_modalities": data_cfg.get("global_context_modalities", ["Input", "Alpha"]),
        "webdataset_repo_id": str(data_cfg.get("webdataset_repo_id", "vano04/CorridorKeyDataset_Custom")),
        "web_shard_cache_dir": data_cfg.get("web_shard_cache_dir"),
        "web_shard_cache_max_shards": int(data_cfg.get("web_shard_cache_max_shards", 3)),
        "web_shard_delete_after_use": bool(data_cfg.get("web_shard_delete_after_use", False)),
        "web_shard_download_retries": int(data_cfg.get("web_shard_download_retries", 6)),
        "web_shard_download_timeout_s": int(data_cfg.get("web_shard_download_timeout_s", 120)),
        "dataset_source": "web" if dataset_source == "web" else "local",
        "dtype": data_cfg.get("read_dtype", data_cfg.get("dataset_dtype", "fp16")),
        "preindex_shards": bool(data_cfg.get("preindex_shards", True)),
    }

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
        manifest_filename=str(data_cfg.get("manifest_filename", "manifest.json")),
        split=str(data_cfg.get("webdataset_split", data_cfg.get("split", "train"))),
        validation_shard_indices=validation_shard_indices,
    )

    pad_multiple = int(model_cfg.get("patch_size", 8))
    collate_fn = lambda batch: pad_collate_video(batch, pad_multiple=pad_multiple)

    loader = dataset_runtime.create_dataloader(
        dataset=dataset,
        batch_size=int(data_cfg.get("batch_size", 1)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        shuffle=bool(data_cfg.get("shuffle", True)),
        drop_last=bool(data_cfg.get("drop_last", False)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
        persistent_workers=bool(data_cfg.get("persistent_workers", True)),
        prefetch_factor=int(data_cfg.get("prefetch_factor", 2)),
        seed=int(data_cfg.get("seed", 1337)),
        collate_fn=collate_fn,
        num_torch_threads=int(data_cfg.get("num_torch_threads", 1)),
        exr_internal_threads=int(data_cfg.get("exr_internal_threads", 0)),
        temporal_stream_chunk_size=temporal_stream_chunk_size if stream_temporal_batches else 0,
    )

    setattr(loader, "_corridorkey_set_dataloader_epoch", dataset_runtime.set_dataloader_epoch)
    if stream_temporal_batches:
        setattr(loader, "_corridorkey_temporal_stream", True)
        logical_len = getattr(getattr(loader, "batch_sampler", None), "logical_len", len(loader))
        setattr(loader, "_corridorkey_logical_len", int(logical_len))

    if debug_console:
        print(
            f"Dataset module={dataset_runtime.module_name} "
            f"mode={dataset_source} root={dataset_root} "
            f"modalities={modalities} convert_to_float={dataset_convert_to_float}"
        )

    return loader


def train() -> None:
    args = parse_args()
    if args.resume_model_only and not args.resume:
        raise ValueError("--resume-model-only requires --resume <checkpoint_path>.")

    cfg = load_config(args.config)
    debug_console = _resolve_debug_console_enabled(cfg, args=args)

    train_cfg = cfg["train"]
    device = select_device(train_cfg)
    debug_console_enabled = bool(debug_console)

    set_seed(args.seed)

    model_cfg = dict(cfg["model"])
    model_cfg.setdefault("fg_representation", str(cfg["data"].get("fg_representation", "premul")))
    if "rematerialize_activations" in train_cfg:
        model_cfg["gradient_checkpointing"] = bool(train_cfg["rematerialize_activations"])
    elif "activation_rematerialization" in train_cfg:
        model_cfg["gradient_checkpointing"] = bool(train_cfg["activation_rematerialization"])
    cfg["model"] = model_cfg
    loss_cfg = cfg["loss"]

    train_inference_mode = str(train_cfg.get("inference_mode", "full")).strip().lower()
    if train_inference_mode not in {"full", "lowres", "tiled", "hybrid"}:
        raise ValueError(
            f"Unsupported train.inference_mode='{train_inference_mode}'. "
            "Expected one of: full, lowres, tiled, hybrid."
        )
    enable_boundary_refine_train = bool(train_cfg.get("enable_boundary_refine", False))

    dataloader = build_dataloader(cfg)

    # Mirror the ``device_augment`` flag into a GPU-side transform module.
    # When set, ``CorridorMattingTransform`` (constructed inside
    # ``build_dataloader``) returns FG/BG/Alpha at native resolution and
    # leaves the spatial crop + photometric augments + composite +
    # ``coarse_alpha_init`` for ``device_transform`` to materialise after
    # H2D. Eager mode -- runs outside the compiled model graph and uses
    # default CUDA RNG (already seeded by ``set_seed``).
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
        if device.type != "cuda":
            raise ValueError(
                "data.device_augment=true requires a CUDA device. "
                "Either run on GPU or unset device_augment in the config."
            )
        device_transform = build_device_transform_from_data_cfg(
            cfg["data"], cfg.get("train", {})
        ).to(device).eval()
        _maybe_print(
            debug_console_enabled,
            (
                "GPU augment enabled: CPU pipeline emits "
                + ("native-res" if bool(cfg["data"].get("decode_full_frame", False)) else "OpenEXR tile-ROI")
                + " FG/BG/Alpha; spatial crop + photometric/sensor/codec augments "
                + "+ composite + coarse_alpha_init run on device after H2D."
            ),
        )

    model = build_v3_hybrid_video_matting_model(model_cfg).to(device)

    if device.type == "cuda":
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
                    debug_console_enabled,
                    "Training notice: boundary_refine is inactive for mode='full'; "
                    f"froze {frozen_tensors} tensors ({frozen_numel:,} params) "
                    "to skip unused optimizer work.",
                )

    model = maybe_compile_boundary_refine(
        model=model,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        debug_console_enabled=debug_console_enabled,
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
        _maybe_print(debug_console_enabled, f"torch.compile enabled (dynamic={compile_dynamic})")

    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain after applying training-time freezes.")

    if debug_console_enabled:
        total_numel = sum(p.numel() for p in model.parameters())
        trainable_numel = sum(p.numel() for p in trainable_parameters)
        print(f"Trainable parameters: {trainable_numel:,} / {total_numel:,}")

    fused_adamw_cfg = bool(train_cfg.get("fused_adamw", False))
    fused_adamw = fused_adamw_cfg and device.type == "cuda"
    if fused_adamw_cfg and not fused_adamw:
        _maybe_print(debug_console_enabled, "Training notice: fused_adamw=true ignored on non-CUDA device.")

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
    if debug_console_enabled:
        group_desc = ", ".join(f"{k}={v:,}" for k, v in sorted(optimizer_group_stats.items()))
        print(f"Optimizer: AdamW(fused={fused_adamw}, groups={len(optimizer_param_groups)}) {group_desc}")

    grad_accum_steps = int(train_cfg.get("grad_accum_steps", 1))
    temporal_batch_size = int(train_cfg.get("temporal_batch_size", 0))
    dataloader_iter_len = len(dataloader)
    dataloader_logical_len = int(getattr(dataloader, "_corridorkey_logical_len", dataloader_iter_len))
    epoch_steps = math.ceil(dataloader_logical_len / max(1, grad_accum_steps))
    epochs_total_steps = int(train_cfg.get("epochs", 40)) * epoch_steps
    # Optional hard cap on optimizer steps. Useful for short smoke runs where
    # we want a real save artefact without grinding through a whole epoch.
    max_steps_cfg = train_cfg.get("max_steps", None)
    max_steps = int(max_steps_cfg) if max_steps_cfg is not None else 0
    max_epoch_batches = max(0, int(train_cfg.get("max_epoch_batches", 0)))

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
        debug_console_enabled,
        f"LR schedule: warmup={int(train_cfg.get('warmup_steps', 5000))}, "
        f"total={total_steps}, base_lr={float(train_cfg.get('lr', 1e-4)):.2e}, "
        f"min_lr_ratio={float(train_cfg.get('min_lr_ratio', 0.1))}",
    )
    _maybe_print(
        debug_console_enabled,
        f"Step accounting: dataloader_iters_per_epoch={dataloader_iter_len}, "
        f"logical_batches_per_epoch={dataloader_logical_len}, "
        f"grad_accum_steps={grad_accum_steps}, optimizer_steps_per_epoch={epoch_steps}",
    )
    if max_epoch_batches > 0:
        _maybe_print(
            debug_console_enabled,
            f"Epoch batch cap enabled: max_epoch_batches={max_epoch_batches}",
        )

    amp_enabled = bool(train_cfg.get("amp", True)) and device.type == "cuda"
    amp_dtype_name = str(train_cfg.get("amp_dtype", "fp16")).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name == "bf16" else torch.float16
    use_scaler = amp_enabled and amp_dtype == torch.float16
    scaler_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    scaler = torch.amp.GradScaler(scaler_device, enabled=use_scaler)

    ema_cfg = dict(train_cfg.get("ema", {}) or {})
    ema_enabled = bool(ema_cfg.get("enabled", train_cfg.get("ema_enabled", False)))
    ema_decay = float(ema_cfg.get("decay", train_cfg.get("ema_decay", 0.9999)))
    ema_update_after_step = int(ema_cfg.get("update_after_step", train_cfg.get("ema_update_after_step", 0)))
    ema_update_every = max(1, int(ema_cfg.get("update_every", train_cfg.get("ema_update_every", 1))))
    ema_device_name = str(ema_cfg.get("device", train_cfg.get("ema_device", "model"))).strip().lower()
    if ema_device_name in {"model", "same", ""}:
        ema_device: torch.device | None = None
    elif ema_device_name == "cpu":
        ema_device = torch.device("cpu")
    elif ema_device_name in {"cuda", "gpu"}:
        ema_device = device if device.type == "cuda" else None
        if ema_device is None:
            _maybe_print(debug_console_enabled, "Training notice: train.ema.device=cuda ignored on non-CUDA device.")
    else:
        raise ValueError("train.ema.device must be one of: model, cpu, cuda")
    model_ema = ModelEma(model, decay=ema_decay, device=ema_device) if ema_enabled else None
    if model_ema is not None:
        _maybe_print(
            debug_console_enabled,
            f"Model EMA enabled: decay={ema_decay}, update_after_step={ema_update_after_step}, "
            f"update_every={ema_update_every}, device={ema_device_name or 'model'}",
        )

    criterion = V3MattingLossComputer(
        weights=loss_cfg,
        fg_representation=str(cfg["data"].get("fg_representation", "premul")),
    ).to(device)

    start_epoch = 0
    global_step = 0
    if args.resume:
        if args.resume_model_only:
            ckpt_epoch, ckpt_global_step = load_model_weights_only(
                path=args.resume,
                model=model,
                device=device,
            )
            if model_ema is not None:
                model_ema.reset(model)
            _maybe_print(
                debug_console_enabled,
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
                model_ema=model_ema,
                device=device,
            )
            _maybe_print(
                debug_console_enabled,
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
    _maybe_print(debug_console_enabled, f"Training inference mode: {train_inference_opts.mode}")

    run_cuda_warmup = bool(train_cfg.get("cuda_warmup", True))
    if device.type == "cuda" and run_cuda_warmup:
        data_cfg = cfg["data"]
        warmup_buckets = list(data_cfg.get("multi_resolution_buckets", [384, 512, 768, 1024]))
        warmup_input_dtype = torch.float32
        if bool(data_cfg.get("device_augment", False)):
            warmup_input_dtype = _resolve_host_dtype(data_cfg.get("host_dtype")) or torch.float32
        _maybe_print(
            debug_console_enabled,
            f"Running CUDA warmup pass for resolution buckets {warmup_buckets}...",
        )
        cuda_warmup(
            model=model,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            patch_size=int(model_cfg.get("patch_size", 8)),
            temporal_batch_size=int(train_cfg.get("temporal_batch_size", 4)),
            batch_size=int(cfg["data"].get("batch_size", 1)),
            input_dtype=warmup_input_dtype,
            resolution_buckets=warmup_buckets,
            inference_options=train_inference_opts,
            optimizer=optimizer,
            scaler=scaler if use_scaler else None,
            verbose=debug_console_enabled,
        )
    elif device.type == "cuda":
        _maybe_print(debug_console_enabled, "CUDA warmup skipped (train.cuda_warmup=false).")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wandb_logger = init_wandb_logger(
        cfg,
        output_dir=output_dir,
        enabled_for_process=True,
        model=model,
        debug_console=debug_console_enabled,
    )

    epochs = int(train_cfg.get("epochs", 40))
    log_interval = int(train_cfg.get("log_interval", 20))
    wandb_cfg = dict(train_cfg.get("wandb", {}) or {})
    wandb_log_interval = max(1, int(wandb_cfg.get("log_interval", log_interval)))
    wandb_loss_ema_decay = float(wandb_cfg.get("loss_ema_decay", 0.98))
    if not 0.0 <= wandb_loss_ema_decay < 1.0:
        raise ValueError("train.wandb.loss_ema_decay must be >= 0.0 and < 1.0")
    wandb_loss_ema: float | None = None
    save_interval = int(train_cfg.get("save_interval", 1))
    show_loss_in_pbar = bool(train_cfg.get("show_loss_in_pbar", False))
    profile_log_metrics = bool(train_cfg.get("profile_log_metrics", False))
    pin_memory_enabled = bool(cfg["data"].get("pin_memory", True))
    cuda_prefetch_cfg = bool(train_cfg.get("cuda_prefetch", False))
    cuda_prefetch = cuda_prefetch_cfg and pin_memory_enabled and device.type == "cuda"
    cuda_prefetch_queue = max(1, int(train_cfg.get("cuda_prefetch_queue", 1)))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))
    autocast_device = device.type
    activation_offload_mode = str(train_cfg.get("activation_offload", "")).strip().lower()
    activation_offload_cpu = (
        activation_offload_mode in {"cpu", "host"}
        or bool(train_cfg.get("activation_offload_cpu", False))
    )
    activation_offload_pin_memory = bool(train_cfg.get("activation_offload_pin_memory", True))
    if activation_offload_cpu and device.type != "cuda":
        _maybe_print(
            debug_console_enabled,
            "Training notice: activation_offload=cpu ignored on non-CUDA device.",
        )
        activation_offload_cpu = False

    if debug_console_enabled:
        if activation_offload_cpu:
            print(
                "Activation offload: saved tensors for backward are staged on CPU "
                f"(pin_memory={activation_offload_pin_memory})."
            )
        if cuda_prefetch_cfg and device.type == "cuda" and not pin_memory_enabled:
            print(
                "Training notice: train.cuda_prefetch=true but data.pin_memory=false; "
                "CUDA prefetch is disabled. Enable pin_memory for host-to-device overlap."
            )

    # Fixed chunk count keeps shape/control flow stable across temporal batches.
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

    profiling = args.profile and device.type == "cuda"
    prof: torch.profiler.profile | None = None
    profile_total_steps = 0
    if profiling:
        profile_dir = Path(args.profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        profile_total_steps = profile_wait + profile_warmup + profile_active
        print(
            f"Profiling enabled: wait={profile_wait}, warmup={profile_warmup}, "
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
    _interrupted = False
    profile_finished = False
    prof_stopped = False
    max_steps_reached = False

    def _sigint_handler(signum: int, frame: object) -> None:
        del signum, frame
        nonlocal _interrupted
        _interrupted = True
        raise KeyboardInterrupt

    prev_sigint = signal.signal(signal.SIGINT, _sigint_handler)
    prev_sigterm = signal.signal(signal.SIGTERM, _sigint_handler)

    try:
        for epoch in range(start_epoch, epochs):
            if _interrupted or max_steps_reached:
                break

            set_dataloader_epoch_safe(dataloader, epoch)

            model.train()
            optimizer.zero_grad(set_to_none=True)

            running_loss = torch.zeros((), device=device, dtype=torch.float32)
            pbar = tqdm(
                total=dataloader_iter_len,
                desc=f"Epoch {epoch + 1}/{epochs}",
                dynamic_ncols=True,
                leave=True,
            )

            prefetch_device_transform = device_transform if cuda_prefetch else None
            stream_coarse_seed: torch.Tensor | None = None
            stream_current_logical_batch: int | None = None
            stream_total_loss: torch.Tensor | None = None
            stream_loss_items: Dict[str, torch.Tensor] = {}

            for it, batch in enumerate(
                iterate_device_batches(
                    dataloader,
                    device,
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

                is_stream_batch = "temporal_stream_chunk_index" in batch
                logical_it = it
                stream_is_final_chunk = True
                if is_stream_batch:
                    logical_it = int(batch["temporal_stream_logical_batch"][0].detach().to(device="cpu"))
                    stream_chunk_idx = int(batch["temporal_stream_chunk_index"][0].detach().to(device="cpu"))
                    stream_num_chunks = int(batch["temporal_stream_num_chunks"][0].detach().to(device="cpu"))
                    stream_is_final_chunk = stream_chunk_idx + 1 >= stream_num_chunks
                    if stream_current_logical_batch != logical_it or stream_chunk_idx == 0:
                        stream_current_logical_batch = logical_it
                        stream_coarse_seed = batch["coarse_alpha_init"]
                        stream_total_loss = batch["video_rgb"].new_tensor(0.0)
                        stream_loss_items = {}
                    elif stream_coarse_seed is not None:
                        batch["coarse_alpha_init"] = stream_coarse_seed
                    temporal_chunks = [(0, int(batch["video_rgb"].shape[1]))]
                    real_n_chunks = 1
                else:
                    temporal_chunks = build_temporal_chunks(
                        total_frames=int(batch["video_rgb"].shape[1]),
                        chunk_size=temporal_batch_size,
                    )
                    real_n_chunks = len(temporal_chunks)
                    while len(temporal_chunks) < fixed_n_chunks:
                        temporal_chunks.append(temporal_chunks[-1])

                valid_mask = batch.get("valid_mask")
                if valid_mask is not None:
                    if is_stream_batch:
                        total_weight_denom = batch["temporal_stream_full_frames"].sum().clamp_min(1).to(
                            dtype=batch["video_rgb"].dtype
                        )
                    else:
                        total_weight_denom = valid_mask.sum().clamp_min(1.0)
                elif is_stream_batch:
                    total_weight_denom = batch["temporal_stream_full_frames"].sum().clamp_min(1).to(
                        dtype=batch["video_rgb"].dtype
                    )
                else:
                    total_weight_denom = batch["video_rgb"].new_tensor(float(batch["video_rgb"].shape[1]))

                coarse_seed = batch["coarse_alpha_init"]
                did_backward = False
                batch_total_loss = stream_total_loss if is_stream_batch and stream_total_loss is not None else batch["video_rgb"].new_tensor(0.0)
                batch_loss_items = stream_loss_items if is_stream_batch else {}
                do_step = stream_is_final_chunk and (
                    ((logical_it + 1) % grad_accum_steps == 0)
                    or (logical_it + 1 == dataloader_logical_len)
                )

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
                            if is_stream_batch:
                                chunk_weight = (
                                    batch["video_rgb"].new_tensor(float(batch["video_rgb"].shape[0] * (end_t - start_t)))
                                    / total_weight_denom
                                ).to(dtype=batch["video_rgb"].dtype)
                            else:
                                chunk_weight = batch["video_rgb"].new_tensor(
                                    (end_t - start_t) / max(1, int(batch["video_rgb"].shape[1]))
                                )

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

                if is_stream_batch:
                    stream_coarse_seed = coarse_seed
                    stream_total_loss = batch_total_loss.detach()
                    stream_loss_items = batch_loss_items

                if do_step and did_backward:
                    if grad_clip > 0:
                        with nvtx_range("grad_clip"):
                            if use_scaler:
                                scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                    with nvtx_range("optimizer_step"):
                        optimizer_step_ran = True
                        if use_scaler:
                            old_scale = float(scaler.get_scale())
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer_step_ran = float(scaler.get_scale()) >= old_scale
                        else:
                            optimizer.step()

                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                    global_step += 1
                    if (
                        model_ema is not None
                        and optimizer_step_ran
                        and global_step > ema_update_after_step
                        and global_step % ema_update_every == 0
                    ):
                        with nvtx_range("ema_update"):
                            model_ema.update(model)

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
                                model_ema=model_ema,
                                epoch=epoch,
                                global_step=global_step,
                                config=cfg,
                            )
                            tqdm.write(f"Saved checkpoint: {ckpt_path}")
                            wandb_logger.log_checkpoint(
                                ckpt_path,
                                epoch=epoch,
                                global_step=global_step,
                                aliases=["latest", f"step-{global_step:06d}"],
                            )

                        _save_max_steps_checkpoint()
                        max_steps_reached = True
                        break
                elif do_step:
                    optimizer.zero_grad(set_to_none=True)

                should_log = stream_is_final_chunk and ((logical_it + 1) % log_interval == 0)
                should_wandb_log = (
                    stream_is_final_chunk
                    and wandb_logger.enabled
                    and (global_step % wandb_log_interval == 0)
                )
                if args.profile and not profile_log_metrics:
                    should_log = False
                    should_wandb_log = False
                should_metric_log = should_log or should_wandb_log

                if stream_is_final_chunk:
                    running_loss = running_loss + batch_total_loss.detach().to(dtype=torch.float32)

                pbar_needs_loss = show_loss_in_pbar and should_log
                loss_for_bar = batch_total_loss.detach() if pbar_needs_loss else None

                reduced_items_t: Dict[str, torch.Tensor] | None = None
                reduced_total_loss_t: torch.Tensor | None = None
                if should_metric_log:
                    reduced_total_loss_t = batch_total_loss.detach()
                    reduced_items_t = {k: v for k, v in batch_loss_items.items()}

                if pbar is not None:
                    pbar.update(1)
                    lr_str = f"{optimizer.param_groups[0]['lr']:.2e}"
                    if pbar_needs_loss:
                        pbar.set_postfix(loss=f"{float(loss_for_bar):.4f}", lr=lr_str, step=global_step)
                    else:
                        pbar.set_postfix(lr=lr_str, step=global_step)

                if should_metric_log and reduced_items_t is not None:
                    reduced_items = {k: float(v.detach().to(device="cpu")) for k, v in reduced_items_t.items()}
                    reduced_total_loss = (
                        float(reduced_total_loss_t.detach().to(device="cpu"))
                        if reduced_total_loss_t is not None
                        else float(batch_total_loss.detach().to(device="cpu"))
                    )
                    lr = optimizer.param_groups[0]["lr"]
                    if should_log:
                        reduced_items_str = " ".join(f"{k}={v:.4f}" for k, v in reduced_items.items())
                        tqdm.write(
                            f"epoch={epoch:03d} iter={logical_it + 1:05d}/{dataloader_logical_len:05d} "
                            f"step={global_step:06d} lr={lr:.2e} {reduced_items_str}"
                        )
                    if should_wandb_log:
                        if wandb_loss_ema is None:
                            wandb_loss_ema = reduced_total_loss
                        else:
                            wandb_loss_ema = (
                                wandb_loss_ema_decay * wandb_loss_ema
                                + (1.0 - wandb_loss_ema_decay) * reduced_total_loss
                            )
                        wandb_logger.log(
                            {
                                "train/loss": reduced_total_loss,
                                "train/loss_ema": wandb_loss_ema,
                                "train/lr": float(lr),
                                "train/epoch": int(epoch),
                                "train/iter": int(logical_it + 1),
                                **{f"train/{k}": v for k, v in reduced_items.items()},
                            },
                            step=global_step,
                        )

                if prof is not None:
                    prof.step()

                if args.profile and profile_total_steps > 0 and it + 1 >= profile_total_steps:
                    if prof is not None:
                        if device.type == "cuda":
                            torch.cuda.synchronize(device)
                        try:
                            prof.stop()
                        except RuntimeError as exc:
                            tqdm.write(f"Profiler stop warning: {exc}")
                        prof_stopped = True

                        chrome_trace_path = profile_dir / "chrome_trace.json"
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
                    profile_finished = True
                    break

                if max_epoch_batches > 0 and stream_is_final_chunk and logical_it + 1 >= max_epoch_batches:
                    tqdm.write(
                        f"Reached max_epoch_batches={max_epoch_batches} at epoch iter={logical_it + 1}; "
                        "ending epoch early."
                    )
                    break

            if pbar is not None:
                pbar.close()

            if profile_finished:
                break

            should_save_epoch_checkpoint = (
                not max_steps_reached
                and not _interrupted
                and ((epoch + 1) % save_interval == 0 or (epoch + 1) == epochs)
            )

            if max_steps_reached:
                pass  # Final checkpoint already saved at the max_steps cap.
            elif _interrupted:
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
                        model_ema=model_ema,
                        epoch=epoch,
                        global_step=global_step,
                        config=cfg,
                    )
                    tqdm.write(f"Saved checkpoint: {ckpt_path}")
                    wandb_logger.log_checkpoint(
                        ckpt_path,
                        epoch=epoch,
                        global_step=global_step,
                        aliases=["latest", f"epoch-{epoch:03d}", f"step-{global_step:06d}"],
                    )

                _save_epoch_checkpoint()

            epoch_loss = running_loss / max(1, dataloader_logical_len)
            epoch_loss_value = float(epoch_loss.detach().to(device="cpu"))
            tqdm.write(f"[epoch {epoch:03d}] avg_total_loss={epoch_loss_value:.6f}")
            wandb_logger.log(
                {"epoch/avg_total_loss": epoch_loss_value, "epoch": int(epoch)},
                step=global_step,
            )

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

        if prof is not None:
            if not prof_stopped:
                try:
                    prof.stop()
                except RuntimeError:
                    pass

        if _interrupted:
            print("\n\nInterrupted - saving emergency checkpoint...")
            ckpt_path = output_dir / f"checkpoint_interrupted_step{global_step:06d}.pt"
            try:
                save_checkpoint(
                    path=ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler if use_scaler else None,
                    model_ema=model_ema,
                    epoch=epoch if "epoch" in locals() else start_epoch,
                    global_step=global_step,
                    config=cfg,
                )
                print(f"Saved: {ckpt_path}")
            except Exception as exc:
                print(f"Failed to save emergency checkpoint: {exc}")

        if profile_finished:
            tqdm.write("Profile capture completed.")
        elif not _interrupted:
            tqdm.write("Training completed.")

        wandb_logger.finish()


if __name__ == "__main__":
    train()
