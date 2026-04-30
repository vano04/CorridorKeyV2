from __future__ import annotations

import atexit
import fnmatch
import json
import math
import os
import random
import re
import tarfile
import tempfile
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import OpenEXR  # type: ignore[import-not-found]
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


DEFAULT_MODALITIES: Tuple[str, ...] = ("Input", "FG", "BG", "Alpha")
DEFAULT_WEB_DATASET_REPO = "vano04/CorridorKeyDataset_Custom"
DEFAULT_MANIFEST = "manifest.json"
SHARD_INDEX_RE = re.compile(r"(\d+)(?!.*\d)")


@dataclass(frozen=True, slots=True)
class TarMemberRef:
    offset_data: int
    size: int


@dataclass(frozen=True, slots=True)
class WebClipIndex:
    key: str
    name: str
    shard_path: Path
    shard_index: int
    frame_numbers: Tuple[int, ...]
    modalities: Tuple[str, ...] = field(default_factory=tuple)


ClipIndex = WebClipIndex

_tar_member_maps: "OrderedDict[str, Dict[str, TarMemberRef]]" = OrderedDict()
_raw_tar_fds: Dict[Tuple[str, int], BinaryIO] = {}
_tmp_paths: set[str] = set()
_tmp_paths_lock = threading.Lock()
_thread_decode_slot = threading.local()

_HAS_EXR_THREAD_API = hasattr(OpenEXR, "setGlobalThreadCount") and hasattr(OpenEXR, "globalThreadCount")
_HAS_EXR_FILE = hasattr(OpenEXR, "File")
_HAS_READ_TILE = _HAS_EXR_FILE and hasattr(OpenEXR.File, "readTile")
_HAS_READ_TILES = _HAS_EXR_FILE and hasattr(OpenEXR.File, "readTiles")
_TMPFS_DIR = "/dev/shm" if os.path.isdir("/dev/shm") else None

if _HAS_EXR_THREAD_API:
    OpenEXR.setGlobalThreadCount(0)


def apply_openexr_internal_threads(n_threads: int) -> int:
    if not _HAS_EXR_THREAD_API:
        return 0
    OpenEXR.setGlobalThreadCount(max(0, int(n_threads)))
    return int(OpenEXR.globalThreadCount())


def _cleanup_tmp_paths() -> None:
    with _tmp_paths_lock:
        paths = list(_tmp_paths)
        _tmp_paths.clear()
    for path in paths:
        try:
            os.unlink(path)
        except OSError:
            pass


atexit.register(_cleanup_tmp_paths)


def _normalise_modality_name(modality: str) -> str:
    aliases = {
        "input": "input",
        "in": "input",
        "fg": "fg",
        "foreground": "fg",
        "bg": "bg",
        "background": "bg",
        "alpha": "alpha",
        "matte": "alpha",
    }
    raw = str(modality).strip().lower()
    return aliases.get(raw, raw)


def _output_modality_name(field: str) -> str:
    return {"input": "Input", "fg": "FG", "bg": "BG", "alpha": "Alpha"}.get(field.lower(), field)


def _global_key(modality: str) -> str:
    return f"global_{_output_modality_name(_normalise_modality_name(modality))}"


def _normalise_global_modalities(modalities: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for modality in modalities:
        item = str(modality).strip()
        if item and item not in out:
            out.append(item)
    return tuple(out)


def _coerce_source_hw(source_hw: Optional[Sequence[int]]) -> Tuple[int, int]:
    if source_hw is None:
        return (2048, 2048)
    if len(source_hw) != 2:
        raise ValueError(f"source_hw must be [H, W], got {source_hw!r}")
    h, w = int(source_hw[0]), int(source_hw[1])
    if h < 1 or w < 1:
        raise ValueError(f"source_hw values must be positive, got {(h, w)!r}")
    return h, w


def _resolve_torch_dtype(value: Any, default: torch.dtype = torch.float16) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    if value is None:
        return default
    aliases = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
        "f32": torch.float32,
    }
    raw = str(value).strip().lower()
    if raw not in aliases:
        raise ValueError(f"Unsupported dataset dtype {value!r}; expected fp16, bf16 or fp32")
    return aliases[raw]


def _full_tile_grid(source_hw: Tuple[int, int], tile_size: int) -> Tuple[int, int, int, int]:
    source_h, source_w = source_hw
    tile = max(1, int(tile_size))
    return (0, max(0, math.ceil(source_w / tile) - 1), 0, max(0, math.ceil(source_h / tile) - 1))


def _sample_tile_selection(
    *,
    decode_full_frame: bool,
    source_hw: Tuple[int, int],
    tile_size: int,
    local_tile_span: int,
) -> Tuple[Optional[Tuple[int, int, int, int]], Tensor, Tensor, Tensor]:
    source_h, source_w = source_hw
    source_hw_t = torch.tensor([float(source_h), float(source_w)], dtype=torch.float32)
    if decode_full_frame:
        return (
            None,
            torch.tensor([0.0, float(source_h), 0.0, float(source_w)], dtype=torch.float32),
            source_hw_t,
            torch.tensor([-1, -1, -1, -1], dtype=torch.long),
        )

    tile = max(1, int(tile_size))
    span = max(1, int(local_tile_span))
    tiles_y = max(1, int(math.ceil(source_h / tile)))
    tiles_x = max(1, int(math.ceil(source_w / tile)))
    span_y = min(span, tiles_y)
    span_x = min(span, tiles_x)
    tx0 = random.randint(0, max(0, tiles_x - span_x))
    ty0 = random.randint(0, max(0, tiles_y - span_y))
    tx1 = tx0 + span_x - 1
    ty1 = ty0 + span_y - 1
    y0 = ty0 * tile
    x0 = tx0 * tile
    y1 = min(source_h, (ty1 + 1) * tile)
    x1 = min(source_w, (tx1 + 1) * tile)
    return (
        (tx0, tx1, ty0, ty1),
        torch.tensor([float(y0), float(y1), float(x0), float(x1)], dtype=torch.float32),
        source_hw_t,
        torch.tensor([tx0, tx1, ty0, ty1], dtype=torch.long),
    )


def _extract_shard_index(path: Path | str) -> Optional[int]:
    match = SHARD_INDEX_RE.search(Path(path).stem)
    return int(match.group(1)) if match else None


def _get_raw_tar_fd(shard_path: Path) -> BinaryIO:
    key = (str(shard_path), threading.get_ident())
    handle = _raw_tar_fds.get(key)
    if handle is None:
        handle = open(shard_path, "rb")
        _raw_tar_fds[key] = handle
    return handle


def _read_member_payload(shard_path: Path, member: TarMemberRef) -> bytes:
    handle = _get_raw_tar_fd(shard_path)
    handle.seek(member.offset_data)
    return handle.read(member.size)


def _tar_member_map_cache_max() -> int:
    raw = os.environ.get("CORRIDORKEY_TAR_MEMBER_MAP_CACHE_MAX_SHARDS", "0").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _close_raw_fds_for_shard(key: str) -> None:
    stale = [fd_key for fd_key in _raw_tar_fds if fd_key[0] == key]
    for fd_key in stale:
        handle = _raw_tar_fds.pop(fd_key, None)
        if handle is not None:
            try:
                handle.close()
            except OSError:
                pass


def _enforce_tar_member_map_cache_limit() -> None:
    max_maps = _tar_member_map_cache_max()
    if max_maps <= 0:
        return
    while len(_tar_member_maps) > max_maps:
        key, _ = _tar_member_maps.popitem(last=False)
        _close_raw_fds_for_shard(key)


def _get_tar_member_map(shard_path: Path) -> Dict[str, TarMemberRef]:
    key = str(shard_path)
    member_map = _tar_member_maps.get(key)
    if member_map is not None:
        _tar_member_maps.move_to_end(key)
        return member_map
    with tarfile.open(key, mode="r:") as archive:
        member_map = {
            member.name: TarMemberRef(offset_data=int(member.offset_data), size=int(member.size))
            for member in archive
            if member.isfile()
        }
    _tar_member_maps[key] = member_map
    _enforce_tar_member_map_cache_limit()
    return member_map


@dataclass(slots=True)
class _DecodeSlot:
    fd: int
    path: str


def _create_decode_slot() -> _DecodeSlot:
    if hasattr(os, "memfd_create") and os.path.isdir("/proc/self/fd"):
        fd = os.memfd_create(f"corridorkey-exr-{os.getpid()}-{threading.get_ident()}")
        return _DecodeSlot(fd=fd, path=f"/proc/self/fd/{fd}")

    fd, path = tempfile.mkstemp(suffix=".exr", dir=_TMPFS_DIR)
    with _tmp_paths_lock:
        _tmp_paths.add(path)
    return _DecodeSlot(fd=fd, path=path)


def _write_payload_to_decode_slot(payload: bytes) -> str:
    if not payload:
        raise RuntimeError("Cannot decode an empty EXR payload")
    slot = getattr(_thread_decode_slot, "slot", None)
    if slot is None:
        slot = _create_decode_slot()
        _thread_decode_slot.slot = slot
    os.ftruncate(slot.fd, 0)
    os.lseek(slot.fd, 0, os.SEEK_SET)
    view = memoryview(payload)
    while view:
        n = os.write(slot.fd, view)
        view = view[n:]
    os.ftruncate(slot.fd, len(payload))
    return slot.path


def _channel_pixels(value: Any) -> np.ndarray:
    return value.pixels if hasattr(value, "pixels") else value


def _compose_array_from_channels(channels: Dict[str, Any], modality: str) -> np.ndarray:
    if not channels:
        raise RuntimeError("EXR file has no readable channels")
    pixels = {name: _channel_pixels(ch) for name, ch in channels.items()}
    names = set(pixels)
    field = _normalise_modality_name(modality)

    if field == "alpha":
        for name in ("A", "Y", "R", "G", "B"):
            if name in pixels:
                return pixels[name]
        return pixels[sorted(names)[0]]

    if {"R", "G", "B"}.issubset(names):
        return np.stack([pixels["R"], pixels["G"], pixels["B"]], axis=2)
    if "Y" in pixels:
        return pixels["Y"]
    if "A" in pixels and len(pixels) == 1:
        return pixels["A"]
    ordered = [pixels[name] for name in sorted(names)]
    return ordered[0] if len(ordered) == 1 else np.stack(ordered[:3], axis=2)


def _crop_tile_array(
    array: np.ndarray,
    tile_grid: Tuple[int, int, int, int],
    source_hw: Tuple[int, int],
    tile_size: int,
) -> np.ndarray:
    tx0, tx1, ty0, ty1 = tile_grid
    source_h, source_w = source_hw
    tile = max(1, int(tile_size))
    expected_h = max(1, min(source_h, (ty1 + 1) * tile) - ty0 * tile)
    expected_w = max(1, min(source_w, (tx1 + 1) * tile) - tx0 * tile)
    return array[:expected_h, :expected_w, ...] if array.ndim == 3 else array[:expected_h, :expected_w]


def _read_exr_tile_array(
    path: str,
    *,
    modality: str,
    tile_grid: Optional[Tuple[int, int, int, int]],
    source_hw: Tuple[int, int],
    tile_size: int,
) -> np.ndarray:
    if not _HAS_EXR_FILE or not _HAS_READ_TILE or not _HAS_READ_TILES:
        raise RuntimeError("The fast CorridorKey loader requires OpenEXR.File.readTile/readTiles")
    read_grid = tile_grid if tile_grid is not None else _full_tile_grid(source_hw, tile_size)
    tx0, tx1, ty0, ty1 = read_grid
    with OpenEXR.File(path, header_only=True) as exr:
        if tx0 == tx1 and ty0 == ty1:
            channels = exr.readTile(tx0, ty0, separate_channels=True)
        else:
            channels = exr.readTiles(tx0, tx1, ty0, ty1, separate_channels=True)
    return _crop_tile_array(_compose_array_from_channels(channels, modality), read_grid, source_hw, tile_size)


def _to_frame_tensor(array: np.ndarray, dtype: torch.dtype) -> Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected EXR array with 2D/3D shape, got {tuple(tensor.shape)}")
    return tensor if tensor.dtype == dtype else tensor.to(dtype=dtype)


def _load_member_tensor(
    shard_path: Path,
    member: TarMemberRef,
    *,
    modality: str,
    dtype: torch.dtype,
    tile_grid: Optional[Tuple[int, int, int, int]],
    source_hw: Tuple[int, int],
    tile_size: int,
) -> Tensor:
    path = _write_payload_to_decode_slot(_read_member_payload(shard_path, member))
    array = _read_exr_tile_array(
        path,
        modality=modality,
        tile_grid=tile_grid,
        source_hw=source_hw,
        tile_size=tile_size,
    )
    return _to_frame_tensor(array, dtype)


def _downscale_2x_box_chw(tensor: Tensor, *, accumulate_dtype: torch.dtype = torch.float16) -> Tensor:
    h_even = (int(tensor.shape[-2]) // 2) * 2
    w_even = (int(tensor.shape[-1]) // 2) * 2
    if h_even < 2 or w_even < 2:
        return tensor
    x = tensor[..., :h_even, :w_even].to(dtype=accumulate_dtype)
    out = (
        x[..., 0::2, 0::2]
        + x[..., 1::2, 0::2]
        + x[..., 0::2, 1::2]
        + x[..., 1::2, 1::2]
    ) * 0.25
    return out.to(dtype=tensor.dtype)


def _read_json_or_jsonl(path: Path) -> Any:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                raw = line.strip()
                if not raw:
                    continue
                try:
                    records.append(json.loads(raw))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"Invalid JSON in {path}:{line_no}") from exc
        return records
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _manifest_candidates(root_dir: Path, requested: str) -> List[Path]:
    names = []
    for name in (requested, DEFAULT_MANIFEST, "clips_manifest.jsonl"):
        if name and name not in names:
            names.append(name)
    return [root_dir / name for name in names]


def _find_manifest(root_dir: Path, requested: str) -> Path:
    for path in _manifest_candidates(root_dir, requested):
        if path.is_file():
            return path
    tried = ", ".join(str(p) for p in _manifest_candidates(root_dir, requested))
    raise FileNotFoundError(f"Could not find dataset manifest. Tried: {tried}")


def _normalise_split(split: str) -> str:
    key = str(split).strip().lower()
    if key in {"", "auto"}:
        return "train"
    if key in {"val", "dev", "eval"}:
        return "validation"
    if key not in {"all", "train", "validation"}:
        raise ValueError(f"Unsupported split={split!r}; expected all, train or validation")
    return key


def _clip_is_selected(clip_name: str, key: str, include: Optional[set[str]], exclude: set[str]) -> bool:
    if include is not None and clip_name not in include and key not in include:
        return False
    return clip_name not in exclude and key not in exclude


def _frame_numbers_from_record(
    record: Dict[str, Any],
    *,
    requested_fields: Sequence[str],
    default_count: int,
) -> Tuple[int, ...]:
    frame_values = record.get("frame_numbers")
    if isinstance(frame_values, list) and frame_values:
        try:
            return tuple(sorted({int(v) for v in frame_values}))
        except (TypeError, ValueError):
            pass
    source_counts = record.get("source_counts")
    if isinstance(source_counts, dict) and source_counts:
        counts = []
        for field in requested_fields:
            value = source_counts.get(field)
            if value is not None:
                try:
                    counts.append(int(value))
                except (TypeError, ValueError):
                    pass
        if counts:
            return tuple(range(1, max(0, min(counts)) + 1))
    try:
        n = int(record.get("num_frames", record.get("frames_per_modality", default_count)))
    except (TypeError, ValueError):
        n = int(default_count)
    return tuple(range(1, max(0, n) + 1))


def _shard_path_from_manifest(root_dir: Path, shard_record: Dict[str, Any], shard_glob: str) -> Optional[Path]:
    candidates: List[str] = []
    raw_path = str(shard_record.get("path", "")).strip()
    if raw_path:
        candidates.append(Path(raw_path).name)
    try:
        idx = int(shard_record.get("shard_index"))
        candidates.extend([f"corridorkey-{idx:06d}.tar", f"shard-{idx:06d}.tar", f"{idx:06d}.tar"])
    except (TypeError, ValueError):
        idx = None
    for name in candidates:
        path = root_dir / name
        if path.is_file() and fnmatch.fnmatch(path.name, shard_glob):
            return path
    if idx is None:
        return None
    for path in sorted(root_dir.glob(shard_glob)):
        if _extract_shard_index(path) == idx:
            return path
    return None


def _download_hf_snapshot(
    *,
    repo_id: str,
    split: str,
    validation_shard_indices: Sequence[int],
    shard_glob: str,
    manifest_filename: str,
    cache_dir: Optional[str | os.PathLike[str]],
) -> Path:
    try:
        import huggingface_hub
    except ImportError as exc:
        raise RuntimeError("data.dataset=web requires huggingface_hub") from exc

    info = huggingface_hub.HfApi().dataset_info(repo_id)
    val_indices = {int(v) for v in validation_shard_indices}
    split_key = _normalise_split(split)
    shard_names = []
    for sibling in info.siblings:
        name = str(sibling.rfilename)
        if not fnmatch.fnmatch(Path(name).name, shard_glob):
            continue
        shard_idx = _extract_shard_index(name)
        is_val = shard_idx is not None and shard_idx in val_indices
        if split_key == "train" and is_val:
            continue
        if split_key == "validation" and not is_val:
            continue
        shard_names.append(name)
    if not shard_names:
        raise RuntimeError(f"No HF shards matched split={split_key!r} in {repo_id}")
    patterns = sorted(set([manifest_filename, DEFAULT_MANIFEST, *shard_names]))
    return Path(
        huggingface_hub.snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            allow_patterns=patterns,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )


class CorridorKeyWebSequenceDataset(Dataset):
    """Single-GPU shard loader optimized for CorridorKey WebDataset tars."""

    def __init__(
        self,
        root_dir: Path | str,
        sequence_length: int = 1,
        frame_stride: int = 1,
        sequence_stride: int = 1,
        modalities: Sequence[str] = DEFAULT_MODALITIES,
        transform: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
        convert_to_float: bool = True,
        include_clips: Optional[Sequence[str]] = None,
        exclude_clips: Optional[Sequence[str]] = None,
        clip_len_range: Optional[Tuple[int, int]] = None,
        shard_glob: str = "*.tar",
        manifest_filename: str = DEFAULT_MANIFEST,
        exr_decode_threads: int = 1,
        split: str = "train",
        validation_shard_indices: Optional[Sequence[int]] = None,
        decode_full_frame: bool = False,
        local_tile_span: int = 4,
        exr_tile_size: int = 256,
        source_hw: Optional[Sequence[int]] = None,
        emit_tile_metadata: bool = True,
        decode_global_context: bool = False,
        cached_four_quadrant_batch: bool = False,
        global_context_root_dir: Path | str | None = None,
        global_context_long_side: int = 0,
        global_context_modalities: Sequence[str] = ("FG", "Alpha"),
        dtype: Any = torch.float16,
        read_dtype: Any = None,
        dataset_source: str = "local",
        webdataset_repo_id: str = DEFAULT_WEB_DATASET_REPO,
        web_shard_cache_dir: Optional[str | os.PathLike[str]] = None,
        preindex_shards: bool = True,
        **_compat_kwargs: object,
    ) -> None:
        super().__init__()
        del convert_to_float, global_context_root_dir
        if not _HAS_READ_TILE or not _HAS_READ_TILES:
            raise RuntimeError("This loader requires the patched OpenEXR readTile/readTiles methods")
        if sequence_length < 1 or frame_stride < 1 or sequence_stride < 1:
            raise ValueError("sequence_length, frame_stride and sequence_stride must be >= 1")

        self.sequence_length = int(sequence_length)
        self.frame_stride = int(frame_stride)
        self.sequence_stride = int(sequence_stride)
        self.modalities = tuple(str(m) for m in modalities)
        self.modality_fields = tuple(_normalise_modality_name(m) for m in self.modalities)
        self.transform = transform
        self.clip_len_range = clip_len_range
        self.shard_glob = str(shard_glob)
        self.manifest_filename = str(manifest_filename or DEFAULT_MANIFEST)
        self.exr_decode_threads = max(1, int(exr_decode_threads))
        self.decode_full_frame = bool(decode_full_frame)
        self.local_tile_span = max(1, int(local_tile_span))
        self.exr_tile_size = max(1, int(exr_tile_size))
        self.source_hw = _coerce_source_hw(source_hw)
        self.emit_tile_metadata = bool(emit_tile_metadata)
        self.decode_global_context = bool(decode_global_context)
        self.cached_four_quadrant_batch = bool(cached_four_quadrant_batch)
        self.global_context_long_side = int(global_context_long_side)
        self.global_context_modalities = _normalise_global_modalities(global_context_modalities)
        self.dtype = _resolve_torch_dtype(read_dtype if read_dtype is not None else dtype)
        self.split = _normalise_split(split)
        self.validation_shard_indices = {int(v) for v in (validation_shard_indices or (33,))}
        self._include_clips = set(include_clips) if include_clips else None
        self._exclude_clips = set(exclude_clips) if exclude_clips else set()
        self._decode_pool: Optional[ThreadPoolExecutor] = None

        root = Path(root_dir)
        source_key = str(dataset_source).strip().lower()
        if source_key in {"web", "hf", "huggingface", "remote"}:
            root = _download_hf_snapshot(
                repo_id=str(webdataset_repo_id or DEFAULT_WEB_DATASET_REPO),
                split=self.split,
                validation_shard_indices=tuple(self.validation_shard_indices),
                shard_glob=self.shard_glob,
                manifest_filename=self.manifest_filename,
                cache_dir=web_shard_cache_dir,
            )
        if not root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {root}")
        self.root_dir = root

        self.clips: List[WebClipIndex] = []
        self.samples: List[Tuple[int, int]] = []
        self._build_index(preindex_shards=bool(preindex_shards))

    def _selected_shard(self, shard_index: int) -> bool:
        is_val = shard_index in self.validation_shard_indices
        if self.split == "all":
            return True
        if self.split == "validation":
            return is_val
        return not is_val

    def _get_decode_pool(self) -> Optional[ThreadPoolExecutor]:
        if self.exr_decode_threads <= 1:
            return None
        if self._decode_pool is None:
            self._decode_pool = ThreadPoolExecutor(max_workers=self.exr_decode_threads)
        return self._decode_pool

    def _build_index(self, *, preindex_shards: bool) -> None:
        manifest_path = _find_manifest(self.root_dir, self.manifest_filename)
        manifest = _read_json_or_jsonl(manifest_path)
        min_required = (self.sequence_length - 1) * self.frame_stride + 1
        if isinstance(manifest, dict):
            self._build_index_from_json_manifest(manifest, min_required=min_required)
        elif isinstance(manifest, list):
            self._build_index_from_jsonl_records(manifest, min_required=min_required)
        else:
            raise RuntimeError(f"Unsupported manifest payload in {manifest_path}: {type(manifest)!r}")
        if not self.samples:
            raise RuntimeError(
                "No valid sequence windows found in selected CorridorKey shards. "
                f"split={self.split!r} validation_shards={sorted(self.validation_shard_indices)}"
            )
        if preindex_shards:
            for shard_path in sorted({clip.shard_path for clip in self.clips}):
                _get_tar_member_map(shard_path)

    def _build_index_from_json_manifest(self, manifest: Dict[str, Any], *, min_required: int) -> None:
        raw_samples = manifest.get("samples")
        raw_shards = manifest.get("shards")
        if not isinstance(raw_samples, list) or not isinstance(raw_shards, list):
            raise RuntimeError("manifest.json must contain list fields 'samples' and 'shards'")
        sample_pos = {
            str(sample.get("key", "")): i
            for i, sample in enumerate(raw_samples)
            if isinstance(sample, dict) and sample.get("key")
        }
        frames_per_clip = int(manifest.get("frames_per_clip", 48))
        cursor = 0
        for shard_record in raw_shards:
            if not isinstance(shard_record, dict):
                continue
            try:
                shard_index = int(shard_record.get("shard_index"))
            except (TypeError, ValueError):
                continue
            if not self._selected_shard(shard_index):
                continue
            shard_path = _shard_path_from_manifest(self.root_dir, shard_record, self.shard_glob)
            if shard_path is None:
                continue
            first_key = str(shard_record.get("first_key", ""))
            last_key = str(shard_record.get("last_key", ""))
            if first_key in sample_pos and last_key in sample_pos:
                start, end = sample_pos[first_key], sample_pos[last_key]
            else:
                count = int(shard_record.get("sample_count", 0))
                start, end = cursor, cursor + count - 1
            cursor = max(cursor, end + 1)
            for sample in raw_samples[start : end + 1]:
                if not isinstance(sample, dict):
                    continue
                clip = self._clip_from_record(
                    sample,
                    shard_path=shard_path,
                    shard_index=shard_index,
                    default_frame_count=frames_per_clip,
                )
                self._append_clip_samples(clip, min_required=min_required)

    def _build_index_from_jsonl_records(self, records: Sequence[Any], *, min_required: int) -> None:
        shard_map = {path.name: path for path in sorted(self.root_dir.glob(self.shard_glob))}
        for record in records:
            if not isinstance(record, dict):
                continue
            shard_name = str(record.get("shard", "")).strip()
            shard_path = shard_map.get(shard_name) or shard_map.get(Path(shard_name).name)
            if shard_path is None:
                continue
            shard_index = _extract_shard_index(shard_path)
            if shard_index is None or not self._selected_shard(shard_index):
                continue
            clip = self._clip_from_record(
                record,
                shard_path=shard_path,
                shard_index=shard_index,
                default_frame_count=48,
            )
            self._append_clip_samples(clip, min_required=min_required)

    def _clip_from_record(
        self,
        record: Dict[str, Any],
        *,
        shard_path: Path,
        shard_index: int,
        default_frame_count: int,
    ) -> Optional[WebClipIndex]:
        key = str(record.get("key", "")).strip()
        if not key:
            return None
        clip_name = str(record.get("clip_name", key))
        if not _clip_is_selected(clip_name, key, self._include_clips, self._exclude_clips):
            return None
        sample_modalities = tuple(
            _normalise_modality_name(str(m))
            for m in record.get("modalities", record.get("source_counts", {}).keys())
        )
        if sample_modalities:
            available = set(sample_modalities)
            if any(field not in available for field in self.modality_fields):
                return None
        frame_numbers = _frame_numbers_from_record(
            record,
            requested_fields=self.modality_fields,
            default_count=default_frame_count,
        )
        if not frame_numbers:
            return None
        return WebClipIndex(
            key=key,
            name=clip_name,
            shard_path=shard_path,
            shard_index=shard_index,
            frame_numbers=frame_numbers,
            modalities=sample_modalities,
        )

    def _append_clip_samples(self, clip: Optional[WebClipIndex], *, min_required: int) -> None:
        if clip is None or len(clip.frame_numbers) < min_required:
            return
        clip_idx = len(self.clips)
        self.clips.append(clip)
        max_start = len(clip.frame_numbers) - min_required
        for start in range(0, max_start + 1, self.sequence_stride):
            self.samples.append((clip_idx, start))

    def __len__(self) -> int:
        return len(self.samples)

    def _member_candidates(self, clip: WebClipIndex, modality: str, frame_number: int) -> Tuple[str, ...]:
        field = _normalise_modality_name(modality)
        return (
            f"{clip.key}.{field}.{frame_number:02d}.exr",
            f"{clip.key}.{field}.{frame_number:05d}.exr",
            f"{clip.key}.{field}.{frame_number}.exr",
        )

    def _resolve_member(self, clip: WebClipIndex, modality: str, frame_number: int) -> TarMemberRef:
        member_map = _get_tar_member_map(clip.shard_path)
        for member_name in self._member_candidates(clip, modality, frame_number):
            member = member_map.get(member_name)
            if member is not None:
                return member
        raise KeyError(
            f"Missing EXR member for clip={clip.name!r} modality={modality!r} "
            f"frame={frame_number} shard={clip.shard_path}"
        )

    def _resolve_global_member(self, clip: WebClipIndex, modality: str, frame_number: int) -> Tuple[TarMemberRef, bool]:
        member_map = _get_tar_member_map(clip.shard_path)
        field = _normalise_modality_name(modality)
        side = str(self.global_context_long_side) if self.global_context_long_side > 0 else ""
        nums = (f"{frame_number:02d}", f"{frame_number:05d}", str(frame_number))
        prefixes = []
        if side:
            prefixes.extend(
                [
                    f"{clip.key}.global_{side}.{field}",
                    f"{clip.key}.global{side}.{field}",
                    f"{clip.key}.{field}_{side}",
                    f"{clip.key}.{field}.global_{side}",
                ]
            )
        prefixes.extend([f"{clip.key}.global.{field}", f"{clip.key}.global_{field}"])
        for prefix in prefixes:
            for num in nums:
                member = member_map.get(f"{prefix}.{num}.exr")
                if member is not None:
                    return member, False
        return self._resolve_member(clip, modality, frame_number), True

    def _global_frame_fallback_tile_grid(self) -> Tuple[int, int, int, int]:
        return _full_tile_grid(self.source_hw, self.exr_tile_size)

    def _fast_global_context_from_frame_member(
        self,
        clip: WebClipIndex,
        member: TarMemberRef,
        modality: str,
        tile_grid: Optional[Tuple[int, int, int, int]],
    ) -> Optional[Tensor]:
        if tile_grid is None or self.global_context_long_side <= 0:
            return None
        source_h, source_w = self.source_hw
        source_long = max(source_h, source_w)
        target_long = int(self.global_context_long_side)
        if source_long != target_long * 2:
            return None
        target_h = max(1, int(round(source_h * (target_long / float(source_long)))))
        target_w = max(1, int(round(source_w * (target_long / float(source_long)))))
        context_h, context_w = target_h * 2, target_w * 2
        native_tile = max(1, int(self.exr_tile_size))
        tx0, _, ty0, _ = tile_grid
        context_y0 = min(max(0, ty0 * native_tile), max(0, source_h - context_h))
        context_x0 = min(max(0, tx0 * native_tile), max(0, source_w - context_w))
        half_h, half_w = context_h // 2, context_w // 2
        if half_h <= 0 or half_w <= 0:
            return None

        def grid_for(y0: int, x0: int, y1: int, x1: int) -> Tuple[int, int, int, int]:
            return (
                x0 // native_tile,
                max(x0 // native_tile, (x1 - 1) // native_tile),
                y0 // native_tile,
                max(y0 // native_tile, (y1 - 1) // native_tile),
            )

        path = _write_payload_to_decode_slot(_read_member_payload(clip.shard_path, member))
        specs: List[Tuple[int, int, int, int, int, int, int]] = []
        grids: List[Tuple[int, int, int, int]] = []
        for y_off in (0, half_h):
            for x_off in (0, half_w):
                y0, x0 = context_y0 + y_off, context_x0 + x_off
                y1, x1 = min(source_h, y0 + half_h), min(source_w, x0 + half_w)
                grid = grid_for(y0, x0, y1, x1)
                specs.append((y_off, x_off, y1 - y0, x1 - x0, y0 - grid[2] * native_tile, x0 - grid[0] * native_tile, len(grids)))
                grids.append(grid)
        tensors = [
            _to_frame_tensor(
                _read_exr_tile_array(path, modality=modality, tile_grid=grid, source_hw=self.source_hw, tile_size=self.exr_tile_size),
                self.dtype,
            )
            for grid in grids
        ]
        canvas = torch.empty((int(tensors[0].shape[0]), context_h, context_w), dtype=self.dtype)
        for y_off, x_off, crop_h, crop_w, local_y0, local_x0, tensor_idx in specs:
            tensor = tensors[tensor_idx]
            piece = tensor[:, local_y0 : local_y0 + crop_h, local_x0 : local_x0 + crop_w]
            canvas[:, y_off : y_off + crop_h, x_off : x_off + crop_w] = piece
        return _downscale_2x_box_chw(canvas)

    def _load_global_modalities(
        self,
        clip: WebClipIndex,
        frame_numbers: Sequence[int],
        tile_grid: Optional[Tuple[int, int, int, int]],
    ) -> Dict[str, Tensor]:
        if not self.decode_global_context:
            return {}
        pool = self._get_decode_pool()
        out: Dict[str, Tensor] = {}
        for modality in self.global_context_modalities:
            tasks = [self._resolve_global_member(clip, modality, frame_number) for frame_number in frame_numbers]

            def decode(task: Tuple[TarMemberRef, bool]) -> Tensor:
                member, fallback_from_frame = task
                if fallback_from_frame:
                    fast = self._fast_global_context_from_frame_member(clip, member, modality, tile_grid)
                    if fast is not None:
                        return fast
                read_grid = self._global_frame_fallback_tile_grid() if fallback_from_frame else None
                return _load_member_tensor(
                    clip.shard_path,
                    member,
                    modality=modality,
                    dtype=self.dtype,
                    tile_grid=read_grid,
                    source_hw=self.source_hw,
                    tile_size=self.exr_tile_size,
                )

            frames = [decode(task) for task in tasks] if pool is None else list(pool.map(decode, tasks))
            out[_global_key(modality)] = torch.stack(frames, dim=0)
        return out

    def _load_all_modalities(
        self,
        clip: WebClipIndex,
        frame_numbers: Sequence[int],
        tile_grid: Optional[Tuple[int, int, int, int]],
    ) -> Dict[str, Tensor]:
        tasks: List[Tuple[str, str, int, TarMemberRef]] = []
        for out_name, field in zip(self.modalities, self.modality_fields):
            for i, frame_number in enumerate(frame_numbers):
                tasks.append((out_name, field, i, self._resolve_member(clip, field, frame_number)))

        def decode(task: Tuple[str, str, int, TarMemberRef]) -> Tensor:
            _, field, _, member = task
            return _load_member_tensor(
                clip.shard_path,
                member,
                modality=field,
                dtype=self.dtype,
                tile_grid=tile_grid,
                source_hw=self.source_hw,
                tile_size=self.exr_tile_size,
            )

        pool = self._get_decode_pool()
        decoded = [decode(task) for task in tasks] if pool is None else list(pool.map(decode, tasks))
        grouped: Dict[str, List[Optional[Tensor]]] = {out_name: [None] * len(frame_numbers) for out_name in self.modalities}
        for (out_name, _, i, _), tensor in zip(tasks, decoded):
            grouped[out_name][i] = tensor
        out: Dict[str, Tensor] = {}
        for out_name in self.modalities:
            frames = grouped[out_name]
            if any(frame is None for frame in frames):
                raise RuntimeError(f"Missing decoded frames for modality={out_name!r} clip={clip.name!r}")
            out[out_name] = torch.stack([frame for frame in frames if frame is not None], dim=0)
        return out

    def _sample_frame_numbers_for_index(self, index: int) -> Tuple[WebClipIndex, List[int]]:
        clip_idx, start = self.samples[index]
        clip = self.clips[clip_idx]
        all_positions = [start + (i * self.frame_stride) for i in range(self.sequence_length)]
        if self.clip_len_range is not None:
            lo, hi = self.clip_len_range
            lo, hi = min(int(lo), int(hi)), max(int(lo), int(hi))
            n = len(all_positions)
            clip_len = random.randint(max(1, min(lo, n)), max(1, min(hi, n)))
            sub_start = random.randint(0, max(0, n - clip_len))
            positions = all_positions[sub_start : sub_start + clip_len]
        else:
            positions = all_positions
        return clip, [clip.frame_numbers[pos] for pos in positions]

    def _four_quadrant_tile_grids(self) -> Tuple[Tuple[int, int, int, int], ...]:
        source_h, source_w = self.source_hw
        tile = max(1, int(self.exr_tile_size))
        span = max(1, int(self.local_tile_span))
        if self.decode_full_frame or source_h != source_w or source_h != tile * span * 2:
            raise ValueError(
                "cached_four_quadrant_batch expects a square 2x2 quadrant layout with "
                "source_hw == [2 * exr_tile_size * local_tile_span] on both axes. "
                f"Got source_hw={self.source_hw}, exr_tile_size={tile}, local_tile_span={span}."
            )
        return (
            (0, span - 1, 0, span - 1),
            (span, (span * 2) - 1, 0, span - 1),
            (0, span - 1, span, (span * 2) - 1),
            (span, (span * 2) - 1, span, (span * 2) - 1),
        )

    def _load_cached_quadrant_tensor(
        self,
        clip: WebClipIndex,
        field: str,
        frame_number: int,
        tile_grid: Tuple[int, int, int, int],
    ) -> Tensor:
        return _load_member_tensor(
            clip.shard_path,
            self._resolve_member(clip, field, frame_number),
            modality=field,
            dtype=self.dtype,
            tile_grid=tile_grid,
            source_hw=self.source_hw,
            tile_size=self.exr_tile_size,
        )

    def _assemble_cached_full_frame(
        self,
        quadrant_tensors: Sequence[Tensor],
        grids: Sequence[Tuple[int, int, int, int]],
    ) -> Tensor:
        source_h, source_w = self.source_hw
        tile = max(1, int(self.exr_tile_size))
        canvas = torch.empty((int(quadrant_tensors[0].shape[0]), source_h, source_w), dtype=self.dtype)
        for tensor, grid in zip(quadrant_tensors, grids):
            tx0, tx1, ty0, ty1 = grid
            del tx1, ty1
            y0 = ty0 * tile
            x0 = tx0 * tile
            y1 = min(source_h, y0 + int(tensor.shape[-2]))
            x1 = min(source_w, x0 + int(tensor.shape[-1]))
            canvas[:, y0:y1, x0:x1] = tensor[:, : y1 - y0, : x1 - x0]
        return canvas

    def _downscale_cached_global_frame(self, frame_chw: Tensor) -> Tensor:
        source_long = max(self.source_hw)
        target_long = int(self.global_context_long_side)
        if target_long > 0 and source_long == target_long * 2:
            return _downscale_2x_box_chw(frame_chw)
        raise ValueError(
            "cached_four_quadrant_batch currently expects global_context_long_side to be "
            f"half the source long side. Got source_hw={self.source_hw}, "
            f"global_context_long_side={self.global_context_long_side}."
        )

    def _getitem_cached_four_quadrants(self, index: int) -> Dict[str, object]:
        clip, frame_numbers = self._sample_frame_numbers_for_index(index)
        grids = self._four_quadrant_tile_grids()
        local_fields_by_name = {
            out_name: field
            for out_name, field in zip(self.modalities, self.modality_fields)
        }
        global_fields = tuple(_normalise_modality_name(m) for m in self.global_context_modalities)
        needed_fields = tuple(dict.fromkeys([*self.modality_fields, *global_fields]))

        tasks: List[Tuple[str, int, int, Tuple[int, int, int, int]]] = []
        for frame_number in frame_numbers:
            for grid_idx, grid in enumerate(grids):
                for field in needed_fields:
                    tasks.append((field, int(frame_number), grid_idx, grid))

        def decode(task: Tuple[str, int, int, Tuple[int, int, int, int]]) -> Tuple[Tuple[str, int, int], Tensor]:
            field, frame_number, grid_idx, grid = task
            return (field, frame_number, grid_idx), self._load_cached_quadrant_tensor(clip, field, frame_number, grid)

        pool = self._get_decode_pool()
        decoded = [decode(task) for task in tasks] if pool is None else list(pool.map(decode, tasks))
        cache: Dict[Tuple[str, int, int], Tensor] = dict(decoded)

        global_tensors: Dict[str, Tensor] = {}
        if self.decode_global_context:
            for modality in self.global_context_modalities:
                field = _normalise_modality_name(modality)
                frames = []
                for frame_number in frame_numbers:
                    quadrants = [cache[(field, int(frame_number), grid_idx)] for grid_idx in range(len(grids))]
                    full = self._assemble_cached_full_frame(quadrants, grids)
                    frames.append(self._downscale_cached_global_frame(full))
                global_tensors[_global_key(modality)] = torch.stack(frames, dim=0)

        source_h, source_w = self.source_hw
        tile = max(1, int(self.exr_tile_size))
        prebatched: List[Dict[str, object]] = []
        for grid_idx, grid in enumerate(grids):
            tx0, tx1, ty0, ty1 = grid
            y0 = ty0 * tile
            x0 = tx0 * tile
            y1 = min(source_h, (ty1 + 1) * tile)
            x1 = min(source_w, (tx1 + 1) * tile)
            sample: Dict[str, object] = {
                "clip_name": clip.name,
                "frame_numbers": torch.tensor(frame_numbers, dtype=torch.long),
            }
            if self.emit_tile_metadata:
                sample["tile_coords"] = torch.tensor([float(y0), float(y1), float(x0), float(x1)], dtype=torch.float32)
                sample["source_hw"] = torch.tensor([float(source_h), float(source_w)], dtype=torch.float32)
                sample["tile_grid"] = torch.tensor([tx0, tx1, ty0, ty1], dtype=torch.long)
            for out_name, field in local_fields_by_name.items():
                sample[out_name] = torch.stack(
                    [cache[(field, int(frame_number), grid_idx)] for frame_number in frame_numbers],
                    dim=0,
                )
            sample.update(global_tensors)
            if self.transform is not None:
                sample = self.transform(sample)
            prebatched.append(sample)
        return {"_prebatched_samples": prebatched}

    def __getitem__(self, index: int) -> Dict[str, object]:
        if self.cached_four_quadrant_batch:
            return self._getitem_cached_four_quadrants(index)
        clip, frame_numbers = self._sample_frame_numbers_for_index(index)
        tile_grid, tile_coords, source_hw, tile_grid_t = _sample_tile_selection(
            decode_full_frame=self.decode_full_frame,
            source_hw=self.source_hw,
            tile_size=self.exr_tile_size,
            local_tile_span=self.local_tile_span,
        )
        sample: Dict[str, object] = {"clip_name": clip.name, "frame_numbers": torch.tensor(frame_numbers, dtype=torch.long)}
        if self.emit_tile_metadata:
            sample["tile_coords"] = tile_coords
            sample["source_hw"] = source_hw
            sample["tile_grid"] = tile_grid_t
        sample.update(self._load_all_modalities(clip, frame_numbers, tile_grid))
        sample.update(self._load_global_modalities(clip, frame_numbers, tile_grid))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


class CorridorKeySequenceDataset(CorridorKeyWebSequenceDataset):
    """Compatibility alias: this project now trains directly from sharded tars."""


def _make_worker_init(num_torch_threads: int, exr_internal_threads: int = 0) -> Callable[[int], None]:
    per_worker = max(1, int(num_torch_threads))
    exr_n = max(0, int(exr_internal_threads))

    def _init(worker_id: int) -> None:
        import signal as _signal

        for sig in (_signal.SIGINT, _signal.SIGTERM):
            try:
                _signal.signal(sig, _signal.SIG_DFL)
            except (OSError, ValueError):
                pass
        seed = torch.initial_seed() % (2**32)
        random.seed(seed)
        np.random.seed(seed)
        torch.set_num_threads(per_worker)
        if exr_n > 0:
            apply_openexr_internal_threads(exr_n)

    return _init


seed_worker = _make_worker_init(1)


def create_single_gpu_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    seed: int = 1337,
    collate_fn: Optional[Callable] = None,
    num_torch_threads: int = 1,
    exr_internal_threads: int = 0,
    **_unused: object,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    loader_kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": bool(drop_last),
        "worker_init_fn": _make_worker_init(num_torch_threads, exr_internal_threads=exr_internal_threads),
        "generator": generator,
        "collate_fn": collate_fn,
    }
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = bool(persistent_workers)
        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))
    return DataLoader(**loader_kwargs)


def set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:
    del dataloader, epoch
