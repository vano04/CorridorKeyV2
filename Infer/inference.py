from __future__ import annotations

import argparse
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

# Let PyTorch fall back to CPU for MPS kernels it does not implement yet. This
# must be set before importing torch.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torch import Tensor

try:
    import OpenEXR  # type: ignore[import-not-found]
except Exception as exc:  # pragma: no cover - import error is surfaced in main.
    OpenEXR = None  # type: ignore[assignment]
    _OPENEXR_IMPORT_ERROR: Optional[BaseException] = exc
else:
    _OPENEXR_IMPORT_ERROR = None


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from models import V3InferenceOptions, build_memory_guided_video_matting_model


_FRAME_RE = re.compile(r"(\d+)(?!.*\d)")
_EPS = 1e-6
_PREFETCH_SENTINEL = object()
_DEFAULT_CHECKPOINT = (
    _PROJECT_ROOT
    / "runs"
    / "real_hybrid_feature_fusion_1024_from_scratch_memsafe"
    / "checkpoint_interrupted_step004884.pt"
)
_FALLBACK_CONFIG = _PROJECT_ROOT / "configs" / "real.yaml"


@dataclass(frozen=True)
class EngineKnobs:
    temporal_frames: int
    temporal_stride: int
    external_tile_size: int
    external_tile_overlap: int
    use_global_context: bool
    fg_representation: str
    amp: bool
    amp_dtype: torch.dtype
    gpu_prefetch: bool
    gpu_prefetch_queue: int
    window_accum_device: str
    gpu_accum_max_mib: int
    model_inference_options: V3InferenceOptions


@dataclass(frozen=True)
class Tile:
    y0: int
    y1: int
    x0: int
    x1: int


@dataclass(frozen=True)
class TiledExrInfo:
    width: int
    height: int
    tile_width: int
    tile_height: int
    num_x_tiles: int
    num_y_tiles: int


@dataclass(frozen=True)
class GlobalContextWindow:
    features: Tuple[Tensor, Tensor, Tensor, Tensor]
    memory: Tensor
    source_hw: Tensor


@dataclass(frozen=True)
class PrefetchedTile:
    tile: Tile
    video: Tensor
    seed_alpha: Tensor
    valid_mask: Tensor
    event: Optional[torch.cuda.Event] = None


@dataclass(frozen=True)
class TiledExrSequence:
    paths: Sequence[Path]
    info: TiledExrInfo
    decode_workers: int

    @classmethod
    def from_paths(cls, paths: Sequence[Path], decode_workers: int) -> "TiledExrSequence":
        if not paths:
            raise ValueError("No input frames were provided")
        info = _read_tiled_exr_info(paths[0])
        return cls(paths=paths, info=info, decode_workers=max(1, int(decode_workers)))

    def read_window_tile(self, start: int, window: int, tile: Tile) -> Tuple[Tensor, int, Tensor]:
        n = len(self.paths)
        end = min(n, start + window)
        actual = max(0, end - start)
        if actual <= 0:
            raise ValueError("Cannot build an empty temporal window")

        indices = [min(start + i, n - 1) for i in range(window)]

        def read_index(frame_index: int) -> Tensor:
            return _read_input_tile(self.paths[frame_index], self.info, tile)

        if self.decode_workers <= 1 or len(indices) <= 1:
            frames = [read_index(i) for i in indices]
        else:
            frames_by_pos: Dict[int, Tensor] = {}
            with ThreadPoolExecutor(max_workers=self.decode_workers) as pool:
                futures = {pool.submit(read_index, frame_index): pos for pos, frame_index in enumerate(indices)}
                for future in as_completed(futures):
                    frames_by_pos[futures[future]] = future.result()
            frames = [frames_by_pos[i] for i in range(len(indices))]

        valid_mask = torch.zeros((1, window), dtype=torch.float32)
        valid_mask[:, :actual] = 1.0
        return torch.stack(frames, dim=0), actual, valid_mask


def _require_openexr() -> None:
    if OpenEXR is None:
        raise RuntimeError(
            "OpenEXR is required for EXR inference I/O. Run with the project venv "
            "(.venv/bin/python) or install the OpenEXR Python bindings."
        ) from _OPENEXR_IMPORT_ERROR


def _set_openexr_threads(n_threads: int) -> int:
    _require_openexr()
    n = max(0, int(n_threads))
    if hasattr(OpenEXR, "setGlobalThreadCount") and hasattr(OpenEXR, "globalThreadCount"):
        OpenEXR.setGlobalThreadCount(n)
        return int(OpenEXR.globalThreadCount())
    return 0


def _extract_frame_index(path: Path) -> int:
    match = _FRAME_RE.search(path.stem)
    if match is None:
        return -1
    return int(match.group(1))


def _sorted_exr_files(path: Path) -> List[Path]:
    files = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() == ".exr"]
    if not files:
        raise FileNotFoundError(f"No .exr files in {path}")
    return sorted(files, key=lambda p: (_extract_frame_index(p), p.name))


def _read_tiled_exr_info(path: Path) -> TiledExrInfo:
    _require_openexr()
    with OpenEXR.File(str(path), header_only=True) as exr:
        header = dict(exr.header())

    if header.get("type") != OpenEXR.tiledimage:
        raise RuntimeError(
            f"{path} is not a tiled OpenEXR image. Convert it first with "
            "Infer/convert_exr_to_dwab_tiles.py so inference can use readTiles()."
        )

    tiles = header.get("tiles")
    if tiles is None:
        raise RuntimeError(f"{path} is tiled but has no tiles header attribute")

    data_window = header.get("dataWindow")
    if data_window is None:
        raise RuntimeError(f"{path} has no dataWindow header")
    dw_min, dw_max = data_window
    width = int(dw_max[0] - dw_min[0] + 1)
    height = int(dw_max[1] - dw_min[1] + 1)
    tile_width = int(tiles.xSize)
    tile_height = int(tiles.ySize)
    if width <= 0 or height <= 0 or tile_width <= 0 or tile_height <= 0:
        raise RuntimeError(f"{path} has invalid image/tile dimensions")

    return TiledExrInfo(
        width=width,
        height=height,
        tile_width=tile_width,
        tile_height=tile_height,
        num_x_tiles=(width + tile_width - 1) // tile_width,
        num_y_tiles=(height + tile_height - 1) // tile_height,
    )


def _assert_same_tiled_layout(path: Path, expected: TiledExrInfo) -> None:
    actual = _read_tiled_exr_info(path)
    if actual != expected:
        raise ValueError(
            f"{path} has tiled layout {actual}, expected {expected}. "
            "All inference frames and the alpha hint must share dimensions and native tile size."
        )


def _compose_exr_channels(channels: Dict[str, np.ndarray]) -> np.ndarray:
    if not channels:
        raise RuntimeError("EXR file has no readable channels")

    names = set(channels)
    if {"R", "G", "B"}.issubset(names):
        stack = [channels["R"], channels["G"], channels["B"]]
        if "A" in names:
            stack.append(channels["A"])
        return np.stack(stack, axis=-1)
    if "Y" in names:
        return channels["Y"]
    if "A" in names and len(names) == 1:
        return channels["A"]

    ordered = [channels[name] for name in sorted(channels)]
    if len(ordered) == 1:
        return ordered[0]
    return np.stack(ordered, axis=-1)


def _read_exr_array(path: Path) -> np.ndarray:
    array = _compose_exr_channels(_channel_pixels_from_file(path))
    return np.asarray(array)


def _array_to_chw_float(array: np.ndarray) -> Tensor:
    if array.ndim == 2:
        array = array[:, :, None]
    if array.ndim != 3:
        raise ValueError(f"Expected HxW or HxWxC EXR data, got shape {array.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(torch.float32) / 255.0
    elif tensor.dtype == torch.uint16:
        tensor = tensor.to(torch.float32) / 65535.0
    else:
        tensor = tensor.to(torch.float32)
    return tensor.permute(2, 0, 1).contiguous()


def _read_exr_native_tiles(path: Path, info: TiledExrInfo, tile: Tile) -> Tuple[Tensor, Tile]:
    read_x0 = max(0, min(tile.x0, info.width - 1))
    read_y0 = max(0, min(tile.y0, info.height - 1))
    read_x1 = max(read_x0 + 1, min(tile.x1, info.width))
    read_y1 = max(read_y0 + 1, min(tile.y1, info.height))

    tx0 = read_x0 // info.tile_width
    ty0 = read_y0 // info.tile_height
    tx1 = (read_x1 - 1) // info.tile_width
    ty1 = (read_y1 - 1) // info.tile_height
    tx1 = min(tx1, info.num_x_tiles - 1)
    ty1 = min(ty1, info.num_y_tiles - 1)

    with OpenEXR.File(str(path), header_only=True) as exr:
        channels = {
            name: np.asarray(ch.pixels)
            for name, ch in exr.readTiles(tx0, tx1, ty0, ty1, separate_channels=True).items()
        }

    tensor = _array_to_chw_float(_compose_exr_channels(channels))
    x_offset = read_x0 - tx0 * info.tile_width
    y_offset = read_y0 - ty0 * info.tile_height
    crop_w = read_x1 - read_x0
    crop_h = read_y1 - read_y0
    tensor = tensor[:, y_offset : y_offset + crop_h, x_offset : x_offset + crop_w]
    return tensor.contiguous(), Tile(y0=read_y0, y1=read_y1, x0=read_x0, x1=read_x1)


def _pad_tile_tensor(tensor: Tensor, requested: Tile, actual: Tile) -> Tensor:
    out_h = requested.y1 - requested.y0
    out_w = requested.x1 - requested.x0
    pad_top = actual.y0 - requested.y0
    pad_left = actual.x0 - requested.x0
    pad_bottom = requested.y1 - actual.y1
    pad_right = requested.x1 - actual.x1
    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        return tensor
    padded = F.pad(
        tensor.unsqueeze(0),
        (max(0, pad_left), max(0, pad_right), max(0, pad_top), max(0, pad_bottom)),
        mode="replicate",
    )[0]
    return padded[:, :out_h, :out_w].contiguous()


def _read_input_tile(path: Path, info: TiledExrInfo, tile: Tile) -> Tensor:
    tensor, actual = _read_exr_native_tiles(path, info, tile)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(3, 1, 1)
    elif tensor.shape[0] >= 3:
        tensor = tensor[:3]
    else:
        raise ValueError(f"Input frame {path} has unsupported channel count {tensor.shape[0]}")
    return _pad_tile_tensor(tensor, tile, actual)


def _read_alpha_tile(path: Path, info: TiledExrInfo, tile: Tile) -> Tensor:
    read_x0 = max(0, min(tile.x0, info.width - 1))
    read_y0 = max(0, min(tile.y0, info.height - 1))
    read_x1 = max(read_x0 + 1, min(tile.x1, info.width))
    read_y1 = max(read_y0 + 1, min(tile.y1, info.height))

    tx0 = read_x0 // info.tile_width
    ty0 = read_y0 // info.tile_height
    tx1 = (read_x1 - 1) // info.tile_width
    ty1 = (read_y1 - 1) // info.tile_height
    tx1 = min(tx1, info.num_x_tiles - 1)
    ty1 = min(ty1, info.num_y_tiles - 1)

    with OpenEXR.File(str(path), header_only=True) as exr:
        channels = {
            name: np.asarray(ch.pixels)
            for name, ch in exr.readTiles(tx0, tx1, ty0, ty1, separate_channels=True).items()
        }

    if not channels:
        raise ValueError(f"Alpha frame {path} has no channels")

    selected = None
    for name in ("A", "a", "alpha", "Alpha", "Y", "y", "R", "r"):
        if name in channels:
            selected = channels[name]
            break
    if selected is None:
        # Fall back to deterministic order if no standard alpha/luma names exist.
        selected = channels[sorted(channels)[0]]

    tensor = _array_to_chw_float(np.asarray(selected))
    x_offset = read_x0 - tx0 * info.tile_width
    y_offset = read_y0 - ty0 * info.tile_height
    crop_w = read_x1 - read_x0
    crop_h = read_y1 - read_y0
    tensor = tensor[:, y_offset : y_offset + crop_h, x_offset : x_offset + crop_w]
    actual = Tile(y0=read_y0, y1=read_y1, x0=read_x0, x1=read_x1)
    return _pad_tile_tensor(tensor[:1].contiguous(), tile, actual).clamp(0.0, 1.0)


def _resize_alpha_to(alpha: Tensor, hw: Tuple[int, int]) -> Tensor:
    if tuple(alpha.shape[-2:]) == tuple(hw):
        return alpha
    x = alpha.unsqueeze(0)
    x = F.interpolate(x, size=hw, mode="bilinear", align_corners=False)
    return x[0].clamp(0.0, 1.0)


def _global_context_hw(info: TiledExrInfo, long_side_cap: int) -> Tuple[int, int]:
    cap = max(1, int(long_side_cap))
    long_side = max(info.height, info.width)
    if long_side <= cap:
        return info.height, info.width
    scale = cap / float(long_side)
    return max(1, int(round(info.height * scale))), max(1, int(round(info.width * scale)))


def _resize_chw_to(tensor: Tensor, hw: Tuple[int, int]) -> Tensor:
    if tuple(tensor.shape[-2:]) == tuple(hw):
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=hw,
        mode="bilinear",
        align_corners=False,
    )[0]


def _sample_context_to_tile(
    context: Tensor,
    tile: Tile,
    full_hw: Tuple[int, int],
    out_hw: Tuple[int, int],
) -> Tensor:
    # context: [T, C, Hg, Wg] -> [T, C, out_h, out_w]
    t, c, _, _ = context.shape
    out_h, out_w = out_hw
    full_h, full_w = full_hw
    yy = (torch.arange(out_h, dtype=context.dtype, device=context.device) + 0.5)
    xx = (torch.arange(out_w, dtype=context.dtype, device=context.device) + 0.5)
    y = float(tile.y0) + yy * (float(tile.y1 - tile.y0) / float(max(1, out_h)))
    x = float(tile.x0) + xx * (float(tile.x1 - tile.x0) / float(max(1, out_w)))
    gy = (2.0 * y / float(max(1, full_h))) - 1.0
    gx = (2.0 * x / float(max(1, full_w))) - 1.0
    grid_y = gy[:, None].expand(out_h, out_w)
    grid_x = gx[None, :].expand(out_h, out_w)
    grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0).expand(t, out_h, out_w, 2)
    return F.grid_sample(
        context,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).reshape(t, c, out_h, out_w)


def _resolve_dtype(name: str) -> torch.dtype:
    key = str(name).strip().lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32", "full"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype {name!r}; expected bf16, fp16, or fp32")


def _mps_backend_available() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())


def _mps_backend_built() -> bool:
    return bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_built())


def _best_available_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_backend_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_device(name: str) -> torch.device:
    requested = str(name or "auto").strip().lower()
    if requested in {"", "auto", "best"}:
        return _best_available_device()

    if requested.startswith("mlx"):
        requested = "mps" + requested[len("mlx") :]
        print("[device] --device mlx requested; using PyTorch MPS backend.", flush=True)

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
    if device.type == "mps":
        if not _mps_backend_built():
            raise RuntimeError("MPS was requested, but this PyTorch build has no MPS support")
        if not _mps_backend_available():
            raise RuntimeError(
                "MPS was requested, but torch.backends.mps.is_available() is false. "
                "Use --device cpu on non-Apple-GPU systems."
            )
    return device


def _adapt_knobs_for_device(knobs: EngineKnobs, device: torch.device) -> EngineKnobs:
    if device.type == "mps" and knobs.amp and knobs.amp_dtype is torch.bfloat16:
        print(
            "[amp] MPS bf16 autocast is not broadly supported; using fp16. "
            "Pass --amp-dtype fp32 or --no-amp for full precision.",
            flush=True,
        )
        return replace(knobs, amp_dtype=torch.float16)
    if device.type == "cpu" and knobs.gpu_prefetch:
        return replace(knobs, gpu_prefetch=False)
    return knobs


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} did not parse to a dictionary")
    return data


def _checkpoint_config(checkpoint: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(checkpoint, dict):
        return None
    config = checkpoint.get("config")
    if isinstance(config, dict):
        return config
    return None


def _resolve_config(
    *,
    config_path: Optional[Path],
    checkpoint_path: Path,
    checkpoint: Any,
) -> Tuple[Dict[str, Any], str]:
    if config_path is not None:
        return _load_yaml(config_path), str(config_path)

    embedded = _checkpoint_config(checkpoint)
    if embedded is not None:
        return embedded, f"{checkpoint_path}:config"

    return _load_yaml(_FALLBACK_CONFIG), str(_FALLBACK_CONFIG)


def _normalize_compile_prefix_for_target(state_dict: Dict[str, Tensor], target: torch.nn.Module) -> Dict[str, Tensor]:
    prefix = "_orig_mod."
    target_keys = set(target.state_dict().keys())
    source_keys = set(state_dict.keys())
    if source_keys == target_keys:
        return state_dict

    stripped = {(k[len(prefix) :] if k.startswith(prefix) else k): v for k, v in state_dict.items()}
    if set(stripped.keys()) == target_keys:
        return stripped

    added = {(k if k.startswith(prefix) else f"{prefix}{k}"): v for k, v in state_dict.items()}
    if set(added.keys()) == target_keys:
        return added
    return state_dict


def _checkpoint_state_dict(checkpoint: Any) -> Dict[str, Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        return checkpoint
    raise ValueError("Checkpoint must be a state dict or contain a model/model_state_dict/state_dict entry")


def _load_model(
    checkpoint_path: Path,
    cfg: Dict[str, Any],
    device: torch.device,
    checkpoint: Optional[Any] = None,
) -> torch.nn.Module:
    model_cfg = dict(cfg.get("model", {}))
    model = build_memory_guided_video_matting_model(model_cfg).to(device)
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = _normalize_compile_prefix_for_target(_checkpoint_state_dict(checkpoint), model)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _build_knobs(args: argparse.Namespace, cfg: Dict[str, Any]) -> EngineKnobs:
    model_cfg = dict(cfg.get("model", {}))
    data_cfg = dict(cfg.get("data", {}))
    train_cfg = dict(cfg.get("train", {}))
    inference_cfg = dict(cfg.get("inference", {}))

    temporal_frames = int(args.temporal_frames or train_cfg.get("temporal_batch_size") or data_cfg.get("clip_len_max", 4))
    temporal_frames = max(1, temporal_frames)
    temporal_stride = int(args.temporal_stride or data_cfg.get("sequence_stride", 1))
    temporal_stride = max(1, temporal_stride)

    external_tile_size = int(args.tile_size or data_cfg.get("fixed_crop_size") or 1024)
    external_tile_overlap = int(args.tile_overlap)
    if args.tile_overlap < 0:
        external_tile_overlap = int(inference_cfg.get("tile_overlap", train_cfg.get("tile_overlap", 32)))
    external_tile_overlap = max(0, min(external_tile_overlap, external_tile_size - 1))

    mode = str(args.model_inference_mode or train_cfg.get("inference_mode", "full")).strip().lower()
    if mode not in {"full", "lowres", "tiled", "hybrid"}:
        raise ValueError(f"Unsupported model inference mode {mode!r}")
    if mode == "hybrid" and not bool(args.global_context):
        raise ValueError("--model-inference-mode=hybrid requires --global-context.")

    opts = V3InferenceOptions(
        mode=mode,
        global_long_side_cap=int(
            args.global_long_side_cap
            or train_cfg.get("global_long_side_cap", inference_cfg.get("global_long_side_cap", 2048))
        ),
        tile_size=int(args.refine_tile_size or train_cfg.get("tile_size", inference_cfg.get("tile_size", 1024))),
        tile_overlap=int(args.refine_tile_overlap if args.refine_tile_overlap >= 0 else train_cfg.get("tile_overlap", inference_cfg.get("tile_overlap", 64))),
        refine_on_uncertainty=bool(
            train_cfg.get("refine_on_uncertainty", inference_cfg.get("refine_on_uncertainty", True))
        ),
        refine_on_edges=bool(train_cfg.get("refine_on_edges", inference_cfg.get("refine_on_edges", True))),
        refine_on_spill_regions=bool(
            train_cfg.get("refine_on_spill_regions", inference_cfg.get("refine_on_spill_regions", True))
        ),
    )

    amp_dtype = _resolve_dtype(args.amp_dtype or train_cfg.get("amp_dtype", "bf16"))
    amp = bool(args.amp and amp_dtype is not torch.float32)

    return EngineKnobs(
        temporal_frames=temporal_frames,
        temporal_stride=temporal_stride,
        external_tile_size=external_tile_size,
        external_tile_overlap=external_tile_overlap,
        use_global_context=bool(args.global_context),
        fg_representation=str(model_cfg.get("fg_representation", data_cfg.get("fg_representation", "straight"))).lower(),
        amp=amp,
        amp_dtype=amp_dtype,
        gpu_prefetch=bool(args.gpu_prefetch),
        gpu_prefetch_queue=max(1, int(args.gpu_prefetch_queue)),
        window_accum_device=str(args.window_accum_device).strip().lower(),
        gpu_accum_max_mib=max(0, int(args.gpu_accum_max_mib)),
        model_inference_options=opts,
    )


def _axis_starts(length: int, tile_size: int, overlap: int) -> List[int]:
    if length <= tile_size:
        return [0]
    stride = max(1, tile_size - overlap)
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def _spatial_tiles(h: int, w: int, tile_size: int, overlap: int) -> List[Tile]:
    return [
        Tile(y0=y, y1=y + tile_size, x0=x, x1=x + tile_size)
        for y in _axis_starts(h, tile_size, overlap)
        for x in _axis_starts(w, tile_size, overlap)
    ]


def _temporal_starts(n_frames: int, window: int, stride: int) -> List[int]:
    if n_frames <= 0:
        return []
    if n_frames <= window:
        return [0]
    starts = list(range(0, n_frames - window + 1, stride))
    last = n_frames - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def _pad_to_min_tile(x: Tensor, tile_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    h, w = x.shape[-2:]
    pad_h = max(0, tile_size - h)
    pad_w = max(0, tile_size - w)
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    flat = x.reshape(-1, 1, h, w) if x.ndim == 3 else x.reshape(-1, x.shape[-3], h, w)
    padded = F.pad(flat, (0, pad_w, 0, pad_h), mode="replicate")
    return padded.reshape(*x.shape[:-2], h + pad_h, w + pad_w), (pad_h, pad_w)


def _crop_alpha_tile(alpha: Tensor, tile: Tile) -> Tensor:
    h, w = alpha.shape[-2:]
    actual = Tile(
        y0=max(0, min(tile.y0, h - 1)),
        y1=max(1, min(tile.y1, h)),
        x0=max(0, min(tile.x0, w - 1)),
        x1=max(1, min(tile.x1, w)),
    )
    cropped = alpha[:, actual.y0 : actual.y1, actual.x0 : actual.x1]
    return _pad_tile_tensor(cropped, tile, actual)


def _tile_weight(tile: Tile, full_h: int, full_w: int, overlap: int, device: torch.device) -> Tensor:
    h = tile.y1 - tile.y0
    w = tile.x1 - tile.x0
    weight = torch.ones((1, h, w), dtype=torch.float32, device=device)
    edge = min(max(0, overlap // 2), h // 2, w // 2)
    if edge <= 0:
        return weight

    ramp = torch.linspace(0.0, 1.0, edge + 1, dtype=torch.float32, device=device)[1:]
    if tile.y0 > 0:
        weight[:, :edge, :] *= ramp.view(1, edge, 1)
    if tile.y1 < full_h:
        weight[:, -edge:, :] *= ramp.flip(0).view(1, edge, 1)
    if tile.x0 > 0:
        weight[:, :, :edge] *= ramp.view(1, 1, edge)
    if tile.x1 < full_w:
        weight[:, :, -edge:] *= ramp.flip(0).view(1, 1, edge)
    return weight


def _window_frames(video: Tensor, start: int, window: int) -> Tuple[Tensor, int, Tensor]:
    n = int(video.shape[0])
    end = min(n, start + window)
    actual = max(0, end - start)
    frames = [video[i] for i in range(start, end)]
    if not frames:
        raise ValueError("Cannot build an empty temporal window")
    while len(frames) < window:
        frames.append(frames[-1])
    valid_mask = torch.zeros((1, window), dtype=torch.float32)
    valid_mask[:, :actual] = 1.0
    return torch.stack(frames, dim=0), actual, valid_mask


def _autocast_context(device: torch.device, enabled: bool, dtype: torch.dtype):
    if not enabled or dtype is torch.float32:
        return nullcontext()
    if device.type in {"cuda", "mps", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=dtype, enabled=True)
    return nullcontext()


def _record_tile_stream(item: PrefetchedTile, stream: torch.cuda.Stream) -> None:
    for tensor in (item.video, item.seed_alpha, item.valid_mask):
        if tensor.device.type == "cuda":
            tensor.record_stream(stream)


def _pin_cpu_tensor(tensor: Tensor) -> Tensor:
    return tensor.contiguous().pin_memory()


class WindowTilePrefetcher:
    def __init__(
        self,
        *,
        sequence: TiledExrSequence,
        tiles: Sequence[Tile],
        window_start: int,
        seed_alpha_cpu: Tensor,
        knobs: EngineKnobs,
        device: torch.device,
    ) -> None:
        self._sequence = sequence
        self._tiles = tiles
        self._window_start = window_start
        self._seed_alpha_cpu = seed_alpha_cpu
        self._knobs = knobs
        self._device = device
        self._queue: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, knobs.gpu_prefetch_queue))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._use_cuda = device.type == "cuda"
        self._stream: Optional[torch.cuda.Stream] = (
            torch.cuda.Stream(device=device) if self._use_cuda else None
        )

    def _load_host_tile(self, tile: Tile) -> Tuple[Tensor, Tensor, Tensor]:
        video_tile, _, valid_mask_cpu = self._sequence.read_window_tile(
            start=self._window_start,
            window=self._knobs.temporal_frames,
            tile=tile,
        )
        seed_tile = _crop_alpha_tile(self._seed_alpha_cpu, tile)
        return video_tile.unsqueeze(0), seed_tile.unsqueeze(0), valid_mask_cpu

    def _build_payload(self, tile: Tile) -> PrefetchedTile:
        video_host, seed_host, valid_host = self._load_host_tile(tile)
        if self._use_cuda and self._stream is not None:
            video_host = _pin_cpu_tensor(video_host)
            seed_host = _pin_cpu_tensor(seed_host)
            valid_host = _pin_cpu_tensor(valid_host)
            with torch.cuda.stream(self._stream), torch.inference_mode():
                video_dev = video_host.to(device=self._device, dtype=torch.float32, non_blocking=True)
                seed_dev = seed_host.to(device=self._device, dtype=torch.float32, non_blocking=True)
                valid_dev = valid_host.to(device=self._device, dtype=torch.float32, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self._stream)
            return PrefetchedTile(
                tile=tile,
                video=video_dev,
                seed_alpha=seed_dev,
                valid_mask=valid_dev,
                event=event,
            )

        return PrefetchedTile(
            tile=tile,
            video=video_host.to(device=self._device, dtype=torch.float32, non_blocking=False),
            seed_alpha=seed_host.to(device=self._device, dtype=torch.float32, non_blocking=False),
            valid_mask=valid_host.to(device=self._device, non_blocking=False),
            event=None,
        )

    def _put(self, item: Any) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return
            except queue.Full:
                continue

    def _produce(self) -> None:
        try:
            for tile in self._tiles:
                if self._stop.is_set():
                    return
                self._put(self._build_payload(tile))
        except BaseException as exc:
            self._put(exc)
        finally:
            self._put(_PREFETCH_SENTINEL)

    def __iter__(self) -> Iterator[PrefetchedTile]:
        if self._thread is not None:
            raise RuntimeError("WindowTilePrefetcher is not re-entrant")

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._produce,
            name="WindowTilePrefetcher",
            daemon=True,
        )
        self._thread.start()

        try:
            while True:
                item = self._queue.get()
                if item is _PREFETCH_SENTINEL:
                    break
                if isinstance(item, BaseException):
                    raise item
                if item.event is not None:
                    current_stream = torch.cuda.current_stream(self._device)
                    item.event.wait(current_stream)
                    _record_tile_stream(item, current_stream)
                yield item
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=0.2)


def _iter_window_tiles_sync(
    *,
    sequence: TiledExrSequence,
    tiles: Sequence[Tile],
    window_start: int,
    seed_alpha_cpu: Tensor,
    knobs: EngineKnobs,
    device: torch.device,
) -> Iterator[PrefetchedTile]:
    for tile in tiles:
        video_tile, _, valid_mask_cpu = sequence.read_window_tile(
            start=window_start,
            window=knobs.temporal_frames,
            tile=tile,
        )
        seed_tile = _crop_alpha_tile(seed_alpha_cpu, tile)
        yield PrefetchedTile(
            tile=tile,
            video=video_tile.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=False),
            seed_alpha=seed_tile.unsqueeze(0).to(device=device, dtype=torch.float32, non_blocking=False),
            valid_mask=valid_mask_cpu.to(device=device, non_blocking=False),
            event=None,
        )


def _run_global_context_window(
    *,
    model: torch.nn.Module,
    sequence: TiledExrSequence,
    window_start: int,
    seed_alpha_cpu: Tensor,
    knobs: EngineKnobs,
    device: torch.device,
) -> GlobalContextWindow:
    global_hw = _global_context_hw(sequence.info, knobs.model_inference_options.global_long_side_cap)
    full_frame = Tile(y0=0, y1=sequence.info.height, x0=0, x1=sequence.info.width)
    video_full, _, valid_mask_cpu = sequence.read_window_tile(
        start=window_start,
        window=knobs.temporal_frames,
        tile=full_frame,
    )
    video_global = torch.stack([_resize_chw_to(frame, global_hw) for frame in video_full], dim=0)
    seed_global = _resize_alpha_to(seed_alpha_cpu, global_hw)

    video_host = video_global.unsqueeze(0)
    seed_host = seed_global.unsqueeze(0)
    valid_host = valid_mask_cpu
    if device.type == "cuda":
        video_host = _pin_cpu_tensor(video_host)
        seed_host = _pin_cpu_tensor(seed_host)
        valid_host = _pin_cpu_tensor(valid_host)

    valid_mask = valid_host.to(device=device, non_blocking=device.type == "cuda")
    video_dev = video_host.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    seed_dev = seed_host.to(device=device, dtype=torch.float32, non_blocking=device.type == "cuda")
    source_hw = torch.tensor(
        [[float(sequence.info.height), float(sequence.info.width)]],
        device=device,
        dtype=torch.float32,
    )

    with _autocast_context(device, knobs.amp, knobs.amp_dtype):
        build_global_context = getattr(model, "build_global_context", None)
        if build_global_context is None and hasattr(model, "_orig_mod"):
            build_global_context = getattr(model._orig_mod, "build_global_context", None)
        if build_global_context is None:
            raise AttributeError("Loaded model does not expose build_global_context().")
        features, memory, context_source_hw = build_global_context(
            video=video_dev,
            coarse_alpha_init=seed_dev,
            valid_mask=valid_mask,
            source_hw=source_hw,
        )

    return GlobalContextWindow(
        features=features,
        memory=memory,
        source_hw=context_source_hw,
    )


def _estimate_window_accum_mib(temporal_frames: int, h: int, w: int) -> float:
    # alpha + fg + blend weight, all float32: (1 + 3 + 1) channels per frame.
    n_bytes = int(temporal_frames) * int(h) * int(w) * 5 * 4
    return n_bytes / float(1024 * 1024)


def _resolve_window_accum_device(knobs: EngineKnobs, model_device: torch.device, h: int, w: int) -> torch.device:
    requested = knobs.window_accum_device
    accelerator_types = {"cuda", "mps"}
    if requested not in {"auto", "cpu", "cuda", "mps", "gpu"}:
        raise ValueError("--window-accum-device must be one of: auto, cpu, cuda, mps, gpu")
    if requested == "cpu":
        return torch.device("cpu")

    estimated_mib = _estimate_window_accum_mib(knobs.temporal_frames, h, w)
    if requested == "gpu":
        requested = model_device.type if model_device.type in accelerator_types else "cpu"
    if requested in accelerator_types:
        if model_device.type != requested:
            raise RuntimeError(
                f"--window-accum-device {requested} was requested, but the model is on {model_device.type}."
            )
        if knobs.gpu_accum_max_mib > 0 and estimated_mib > float(knobs.gpu_accum_max_mib):
            raise RuntimeError(
                f"{requested.upper()} window accumulation needs about {estimated_mib:.0f} MiB, "
                f"above --gpu-accum-max-mib={knobs.gpu_accum_max_mib}. "
                "Raise the cap or use --window-accum-device cpu."
            )
        return model_device
    if model_device.type not in accelerator_types:
        return torch.device("cpu")

    if knobs.gpu_accum_max_mib <= 0 or estimated_mib <= float(knobs.gpu_accum_max_mib):
        return model_device
    return torch.device("cpu")


def _run_window(
    *,
    model: torch.nn.Module,
    sequence: TiledExrSequence,
    window_start: int,
    seed_alpha_cpu: Tensor,
    knobs: EngineKnobs,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    tile_size = knobs.external_tile_size
    overlap = knobs.external_tile_overlap

    h_pad = max(sequence.info.height, tile_size)
    w_pad = max(sequence.info.width, tile_size)
    accum_device = _resolve_window_accum_device(knobs, device, h_pad, w_pad)

    alpha_acc = torch.zeros((knobs.temporal_frames, 1, h_pad, w_pad), dtype=torch.float32, device=accum_device)
    fg_acc = torch.zeros((knobs.temporal_frames, 3, h_pad, w_pad), dtype=torch.float32, device=accum_device)
    weight_acc = torch.zeros((knobs.temporal_frames, 1, h_pad, w_pad), dtype=torch.float32, device=accum_device)

    global_context: Optional[GlobalContextWindow] = None
    if knobs.use_global_context:
        global_context = _run_global_context_window(
            model=model,
            sequence=sequence,
            window_start=window_start,
            seed_alpha_cpu=seed_alpha_cpu,
            knobs=knobs,
            device=device,
        )

    tiles = _spatial_tiles(h_pad, w_pad, tile_size, overlap)
    if device.type == "cuda" and knobs.gpu_prefetch:
        tile_iter: Iterator[PrefetchedTile] = iter(
            WindowTilePrefetcher(
                sequence=sequence,
                tiles=tiles,
                window_start=window_start,
                seed_alpha_cpu=seed_alpha_cpu,
                knobs=knobs,
                device=device,
            )
        )
    else:
        tile_iter = _iter_window_tiles_sync(
            sequence=sequence,
            tiles=tiles,
            window_start=window_start,
            seed_alpha_cpu=seed_alpha_cpu,
            knobs=knobs,
            device=device,
        )

    for batch in tile_iter:
        tile = batch.tile
        video_dev = batch.video
        seed_dev = batch.seed_alpha
        valid_mask = batch.valid_mask
        if global_context is None:
            with _autocast_context(device, knobs.amp, knobs.amp_dtype):
                out = model(
                    video_dev,
                    seed_dev,
                    valid_mask=valid_mask,
                    bg_for_comp=None,
                    inference_options=knobs.model_inference_options,
                )
        else:
            tile_coords = torch.tensor(
                [[float(tile.y0), float(tile.y1), float(tile.x0), float(tile.x1)]],
                device=device,
                dtype=torch.float32,
            )
            source_hw = torch.tensor(
                [[float(sequence.info.height), float(sequence.info.width)]],
                device=device,
                dtype=torch.float32,
            )
            with _autocast_context(device, knobs.amp, knobs.amp_dtype):
                out = model(
                    video=video_dev,
                    coarse_alpha_init=seed_dev,
                    valid_mask=valid_mask,
                    bg_for_comp=None,
                    inference_options=knobs.model_inference_options,
                    global_features=global_context.features,
                    global_memory=global_context.memory,
                    global_source_hw=global_context.source_hw,
                    tile_coords=tile_coords,
                    source_hw=source_hw,
                )

        alpha_tile = out["alpha_pred"][0].detach().to(device=accum_device, dtype=torch.float32).clamp(0.0, 1.0)
        fg_tile = out["fg_pred"][0].detach().to(device=accum_device, dtype=torch.float32)
        weight = _tile_weight(tile, h_pad, w_pad, overlap, device=accum_device)

        alpha_acc[:, :, tile.y0 : tile.y1, tile.x0 : tile.x1] += alpha_tile * weight
        fg_acc[:, :, tile.y0 : tile.y1, tile.x0 : tile.x1] += fg_tile * weight
        weight_acc[:, :, tile.y0 : tile.y1, tile.x0 : tile.x1] += weight

        del batch, video_dev, seed_dev, valid_mask, out, alpha_tile, fg_tile

    weight_acc = weight_acc.clamp_min(_EPS)
    alpha_out = alpha_acc / weight_acc
    fg_out = fg_acc / weight_acc
    if alpha_out.device.type != "cpu":
        alpha_out = alpha_out.cpu()
        fg_out = fg_out.cpu()
    return alpha_out, fg_out


@torch.inference_mode()
def run_tiled_temporal_inference(
    *,
    model: torch.nn.Module,
    sequence: TiledExrSequence,
    initial_alpha_cpu: Tensor,
    knobs: EngineKnobs,
    device: torch.device,
) -> Tuple[Tensor, Tensor]:
    n_frames = len(sequence.paths)
    h = sequence.info.height
    w = sequence.info.width
    alpha_hint = _resize_alpha_to(initial_alpha_cpu, (h, w))
    starts = _temporal_starts(n_frames, knobs.temporal_frames, knobs.temporal_stride)
    if not starts:
        raise ValueError("No frames to process")

    alpha_sum = torch.zeros((n_frames, 1, h, w), dtype=torch.float32)
    fg_sum = torch.zeros((n_frames, 3, h, w), dtype=torch.float32)
    count = torch.zeros((n_frames, 1, 1, 1), dtype=torch.float32)

    carry_seed = alpha_hint
    for start_i, start in enumerate(starts):
        actual = min(n_frames - start, knobs.temporal_frames)
        alpha_win, fg_win = _run_window(
            model=model,
            sequence=sequence,
            window_start=start,
            seed_alpha_cpu=carry_seed,
            knobs=knobs,
            device=device,
        )

        alpha_win = alpha_win[:actual, :, :h, :w].contiguous()
        fg_win = fg_win[:actual, :, :h, :w].contiguous()
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
            f"[infer] window {start_i + 1}/{len(starts)} "
            f"frames {start:05d}-{start + actual - 1:05d}",
            flush=True,
        )

    count = count.clamp_min(1.0)
    return (alpha_sum / count).clamp(0.0, 1.0), fg_sum / count


def _linear_to_srgb_tensor(x: Tensor) -> Tensor:
    x = x.clamp(0.0, 1.0)
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * torch.pow(x, 1.0 / 2.4) - 0.055).clamp(0.0, 1.0)


def _checkerboard_linear(h: int, w: int, size: int = 128, dark_srgb: float = 0.15, light_srgb: float = 0.55) -> Tensor:
    yy = torch.arange(h).view(h, 1)
    xx = torch.arange(w).view(1, w)
    mask = ((yy // size + xx // size) % 2).to(torch.float32)
    srgb = torch.where(mask > 0.0, torch.full_like(mask, light_srgb), torch.full_like(mask, dark_srgb))
    linear = torch.where(srgb <= 0.04045, srgb / 12.92, torch.pow((srgb + 0.055) / 1.055, 2.4))
    return linear.unsqueeze(0).repeat(3, 1, 1)


def _fg_to_straight(fg: Tensor, alpha: Tensor, representation: str) -> Tensor:
    if representation == "straight":
        return fg
    if representation == "premul":
        denom = alpha.clamp_min(_EPS)
        return torch.where(alpha > _EPS, fg / denom, torch.zeros_like(fg))
    raise ValueError(f"Unsupported fg representation {representation!r}")


def _composite_over_checker(fg: Tensor, alpha: Tensor, representation: str) -> Tensor:
    checker = _checkerboard_linear(alpha.shape[-2], alpha.shape[-1]).to(fg.dtype)
    if representation == "straight":
        return fg * alpha + checker * (1.0 - alpha)
    if representation == "premul":
        return fg + checker * (1.0 - alpha)
    raise ValueError(f"Unsupported fg representation {representation!r}")


def _to_uint8_image(chw: Tensor) -> np.ndarray:
    hwc = chw.detach().cpu().permute(1, 2, 0).numpy()
    return (np.clip(hwc, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _to_uint8_alpha(alpha: Tensor) -> np.ndarray:
    hw = alpha.detach().cpu().squeeze(0).numpy()
    return (np.clip(hw, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _to_uint8_rgba(rgb: Tensor, alpha: Tensor) -> np.ndarray:
    rgb8 = _to_uint8_image(_linear_to_srgb_tensor(rgb))
    alpha8 = _to_uint8_alpha(alpha)
    return np.concatenate([rgb8, alpha8[:, :, None]], axis=2)


def _save_frame_outputs(
    *,
    index: int,
    alpha: Tensor,
    fg: Tensor,
    input_path: Optional[Path],
    input_info: Optional[TiledExrInfo],
    output_dir: Path,
    fg_representation: str,
    fg_source: str,
) -> None:
    matte_dir = output_dir / "Matte"
    fg_dir = output_dir / "FG"
    comp_dir = output_dir / "Comp"
    processed_dir = output_dir / "Processed"
    for path in (matte_dir, fg_dir, comp_dir, processed_dir):
        path.mkdir(parents=True, exist_ok=True)

    alpha = alpha.clamp(0.0, 1.0)
    fg = fg.clamp(0.0, 1.0)
    if fg_source == "model":
        fg_straight = _fg_to_straight(fg, alpha, fg_representation).clamp(0.0, 1.0)
    elif fg_source == "input":
        if input_path is None or input_info is None:
            raise ValueError("fg_source='input' requires input_path and input_info")
        full_frame = Tile(y0=0, y1=input_info.height, x0=0, x1=input_info.width)
        fg_straight = _read_input_tile(input_path, input_info, full_frame).clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported fg_source={fg_source!r}")

    comp = _composite_over_checker(fg_straight, alpha, "straight").clamp(0.0, 1.0)

    stem = f"{index:05d}"

    Image.fromarray(_to_uint8_alpha(alpha), mode="L").save(matte_dir / f"matte_{stem}.png")
    Image.fromarray(_to_uint8_image(_linear_to_srgb_tensor(fg_straight)), mode="RGB").save(fg_dir / f"fg_{stem}.png")
    Image.fromarray(_to_uint8_rgba(fg_straight, alpha), mode="RGBA").save(processed_dir / f"processed_{stem}.png")
    Image.fromarray(_to_uint8_image(_linear_to_srgb_tensor(comp))).save(comp_dir / f"comp_{stem}.png")


def _save_outputs(
    alpha_pred: Tensor,
    fg_pred: Tensor,
    input_paths: Sequence[Path],
    input_info: TiledExrInfo,
    output_dir: Path,
    fg_representation: str,
    fg_source: str,
    workers: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if workers <= 1:
        for i in range(alpha_pred.shape[0]):
            _save_frame_outputs(
                index=i,
                alpha=alpha_pred[i],
                fg=fg_pred[i],
                input_path=input_paths[i] if fg_source == "input" else None,
                input_info=input_info if fg_source == "input" else None,
                output_dir=output_dir,
                fg_representation=fg_representation,
                fg_source=fg_source,
            )
        return

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = [
            pool.submit(
                _save_frame_outputs,
                index=i,
                alpha=alpha_pred[i],
                fg=fg_pred[i],
                input_path=input_paths[i] if fg_source == "input" else None,
                input_info=input_info if fg_source == "input" else None,
                output_dir=output_dir,
                fg_representation=fg_representation,
                fg_source=fg_source,
            )
            for i in range(alpha_pred.shape[0])
        ]
        for future in as_completed(futures):
            future.result()


def _write_comp_video(comp_dir: Path, output_path: Path, fps: float) -> None:
    frames = sorted(comp_dir.glob("comp_*.png"))
    if not frames:
        raise FileNotFoundError(f"No comp_*.png frames in {comp_dir}")

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for --make-video and was not found on PATH")

    pattern = comp_dir / "comp_*.png"
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        f"{float(fps):g}",
        "-pattern_type",
        "glob",
        "-i",
        str(pattern),
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed while writing {output_path}") from exc


def _default_input_dir() -> Path:
    converted = _HERE / "corridor_greenscreen_demo_dwab1024" / "Input"
    if converted.is_dir():
        return converted
    return _HERE / "corridor_greenscreen_demo" / "Input"


def _default_alpha_dir() -> Path:
    converted = _HERE / "corridor_greenscreen_demo_dwab1024" / "Alpha"
    if converted.is_dir():
        return converted
    return _HERE / "corridor_greenscreen_demo" / "Alpha"


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CorridorKey memory-vmatte inference with temporal sliding windows "
            "and external 1024x1024 spatial tiles loaded via OpenEXR.readTiles()."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
        help="Model checkpoint .pt file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Optional YAML config override. By default, inference uses the "
            "config embedded in the checkpoint, falling back to configs/real.yaml."
        ),
    )
    parser.add_argument("--input-dir", type=Path, default=_default_input_dir())
    parser.add_argument("--alpha-dir", type=Path, default=_default_alpha_dir())
    parser.add_argument("--output-dir", type=Path, default=_HERE / "corridor_greenscreen_demo_output")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device. auto chooses CUDA, then MPS, then CPU. 'mlx' is accepted as an alias for PyTorch MPS.",
    )
    parser.add_argument("--exr-decode-threads", type=int, default=4)
    parser.add_argument("--exr-internal-threads", type=int, default=0)
    parser.add_argument("--write-threads", type=int, default=4)
    parser.add_argument("--tile-size", type=int, default=0, help="External spatial tile size. 0 = config data.fixed_crop_size.")
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=-1,
        help="External spatial tile overlap. -1 = config inference/train overlap.",
    )
    parser.add_argument("--temporal-frames", type=int, default=0, help="Frames per model window. 0 = config.")
    parser.add_argument("--temporal-stride", type=int, default=0, help="Temporal sliding stride. 0 = config sequence_stride.")
    parser.add_argument("--model-inference-mode", choices=("full", "lowres", "tiled", "hybrid"), default=None)
    parser.add_argument("--global-long-side-cap", type=int, default=0)
    parser.add_argument(
        "--global-context",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run one full-frame low-res temporal pass per window, then refine "
            "1024 tiles with cropped global features and memory. Disable for the legacy "
            "tile-isolated path."
        ),
    )
    parser.add_argument("--refine-tile-size", type=int, default=0, help="Model internal tiled-refine tile size. 0 = config.")
    parser.add_argument("--refine-tile-overlap", type=int, default=-1, help="Model internal tiled-refine overlap. -1 = config.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--amp-dtype", default=None, choices=("bf16", "bfloat16", "fp16", "float16", "fp32", "float32"))
    parser.add_argument(
        "--gpu-prefetch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On CUDA, decode the next local tile on a background thread, pin it, "
            "and copy it on a side stream while the current tile runs."
        ),
    )
    parser.add_argument(
        "--gpu-prefetch-queue",
        type=int,
        default=2,
        help="Number of local tile batches to stage on GPU. Higher costs VRAM.",
    )
    parser.add_argument(
        "--window-accum-device",
        choices=("auto", "cpu", "cuda", "mps", "gpu"),
        default="auto",
        help=(
            "Where to blend local tile predictions. auto uses the active accelerator only when "
            "the estimated window accumulator fits under --gpu-accum-max-mib."
        ),
    )
    parser.add_argument(
        "--gpu-accum-max-mib",
        type=int,
        default=2048,
        help=(
            "Accelerator memory budget for window accumulation in auto/cuda/mps/gpu mode. "
            "Set 0 for no cap."
        ),
    )
    parser.add_argument(
        "--fg-source",
        choices=("model", "input"),
        default="model",
        help=(
            "Visible RGB source for debug FG/Processed/Comp PNGs. "
            "'model' writes the predicted foreground; 'input' writes the "
            "original smooth plate RGB gated by the predicted alpha."
        ),
    )
    parser.add_argument("--compile-model", action="store_true", help="Apply torch.compile to the loaded model.")
    parser.add_argument("--limit", type=int, default=0, help="Optional frame-count limit for debugging.")
    parser.add_argument("--make-video", action="store_true", help="Stitch Comp/*.png into an mp4 after writing frames.")
    parser.add_argument("--fps", type=float, default=24.0)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    _set_openexr_threads(args.exr_internal_threads)
    device = _resolve_device(args.device)

    print(f"[model] loading checkpoint metadata {args.checkpoint}", flush=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg, cfg_source = _resolve_config(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        checkpoint=checkpoint,
    )
    knobs = _build_knobs(args, cfg)
    knobs = _adapt_knobs_for_device(knobs, device)
    if knobs.fg_representation not in {"straight", "premul"}:
        raise ValueError(f"Unsupported fg_representation={knobs.fg_representation!r}")

    input_paths = _sorted_exr_files(args.input_dir)
    if args.limit > 0:
        input_paths = input_paths[: args.limit]
    alpha_paths = _sorted_exr_files(args.alpha_dir)
    alpha_path = alpha_paths[0]

    print(f"[load] frames={len(input_paths)} input_dir={args.input_dir}", flush=True)
    sequence = TiledExrSequence.from_paths(input_paths, decode_workers=args.exr_decode_threads)
    _assert_same_tiled_layout(alpha_path, sequence.info)
    full_frame = Tile(y0=0, y1=sequence.info.height, x0=0, x1=sequence.info.width)
    alpha_cpu = _read_alpha_tile(alpha_path, sequence.info, full_frame)

    print(f"[model] config {cfg_source}", flush=True)
    print(f"[model] loading weights {args.checkpoint}", flush=True)
    model = _load_model(args.checkpoint, cfg, device, checkpoint=checkpoint)
    del checkpoint
    if args.compile_model:
        if device.type == "mps":
            print("[compile] torch.compile on MPS may fall back or fail depending on PyTorch kernels.", flush=True)
        model = torch.compile(model)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    accum_h = max(sequence.info.height, knobs.external_tile_size)
    accum_w = max(sequence.info.width, knobs.external_tile_size)
    accum_est_mib = _estimate_window_accum_mib(knobs.temporal_frames, accum_h, accum_w)
    accum_device = _resolve_window_accum_device(knobs, device, accum_h, accum_w)

    print(
        "[infer] "
        f"device={device} "
        f"tile={knobs.external_tile_size} overlap={knobs.external_tile_overlap} "
        f"temporal={knobs.temporal_frames} stride={knobs.temporal_stride} "
        f"model_mode={knobs.model_inference_options.mode} "
        f"global_context={knobs.use_global_context} "
        f"global_cap={knobs.model_inference_options.global_long_side_cap} "
        f"gpu_prefetch={knobs.gpu_prefetch if device.type == 'cuda' else False} "
        f"prefetch_queue={knobs.gpu_prefetch_queue} "
        f"window_accum={accum_device.type} "
        f"accum_est={accum_est_mib:.0f}MiB "
        f"accum_cap={knobs.gpu_accum_max_mib}MiB "
        f"amp={knobs.amp_dtype if knobs.amp else 'off'}",
        flush=True,
    )
    alpha_pred, fg_pred = run_tiled_temporal_inference(
        model=model,
        sequence=sequence,
        initial_alpha_cpu=alpha_cpu,
        knobs=knobs,
        device=device,
    )

    print(f"[write] {args.output_dir}", flush=True)
    _save_outputs(
        alpha_pred=alpha_pred,
        fg_pred=fg_pred,
        input_paths=input_paths,
        input_info=sequence.info,
        output_dir=args.output_dir,
        fg_representation=knobs.fg_representation,
        fg_source=args.fg_source,
        workers=args.write_threads,
    )

    if args.make_video:
        video_path = args.output_dir / "comp.mp4"
        _write_comp_video(args.output_dir / "Comp", video_path, fps=args.fps)
        print(f"[write] {video_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
