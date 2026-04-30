from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import fnmatch
import io
import json
import math
import os
import random
import re
import tarfile
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple

import OpenEXR  # type: ignore[import-not-found]
import Imath
import numpy as np
import torch
import torch.distributed as dist
from safetensors import safe_open
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


DEFAULT_MODALITIES: Tuple[str, ...] = ("Input", "FG", "BG", "Alpha")
FRAME_INDEX_RE = re.compile(r"(\d+)(?!.*\d)")
SHARD_INDEX_RE = re.compile(r"(\d+)$")
SHARD_FILENAME = "_cache.safetensors"


@dataclass(frozen=True, slots=True)
class TarMemberRef:
    offset_data: int
    size: int


# Per-worker cache of mmap'd shard handles (populated lazily after fork).
_shard_handles: Dict[str, Any] = {}
_tar_archives: Dict[str, tarfile.TarFile] = {}
_tar_member_maps: "OrderedDict[str, Dict[str, TarMemberRef]]" = OrderedDict()
# Per-thread raw file handles keyed by (shard_path, thread_id). Used to read
# raw bytes at known member offsets without going through the (thread-unsafe)
# ``tarfile.TarFile`` object, so multiple decode threads can read from the
# same shard concurrently.
_raw_tar_fds: Dict[Tuple[str, int], BinaryIO] = {}

# ``OpenEXR.File`` (the modern OpenEXR >= 3.3 API) releases the GIL during
# reads, unlike the legacy ``OpenEXR.InputFile``. That matters a lot for us:
# the legacy path pegged a single Python thread at ~635 ms per 2048x2048
# DWAB-compressed EXR regardless of ``exr_decode_threads`` because the GIL
# was held for the entire decode. The new API does not have a "read from
# bytes" constructor, though, so to decode a frame that lives inside a tar
# shard we spill the payload to a small tempfile on tmpfs and hand that
# filename to ``OpenEXR.File``.
_USE_NEW_EXR_API: bool = hasattr(OpenEXR, "File")
_TMPFS_DIR: Optional[str] = "/dev/shm" if os.path.isdir("/dev/shm") else None

# Whether the (patched) OpenEXR Python binding exposes the
# ``Imf::setGlobalThreadCount`` knob. When True, each dataloader worker can
# size the IlmThread global pool used internally by ``OpenEXR.File`` to
# parallelise a *single* decode across multiple cores. When False (stock
# binding), this is a no-op and per-decode parallelism stays at 1.
_HAS_EXR_THREAD_API: bool = hasattr(OpenEXR, "setGlobalThreadCount") and hasattr(
    OpenEXR, "globalThreadCount"
)

# Force the parent process's IlmThread global pool to size 0 at import time.
#
# Why: the patched binding spawns its globalThreadCount() worker threads
# eagerly inside ``setGlobalThreadCount`` (and the patched default is >0).
# DataLoader workers are forked from the parent on Linux, and ``fork()``
# only carries the calling thread into the child. Any IlmThread workers
# alive in the parent vanish in the child but leave their pthread mutexes
# / condvars in a half-locked state -- the first decode dispatched in the
# child then blocks forever on a broken mutex. Symptom: workers spawn,
# never produce a batch, no CPU, no RAM growth, training pegged at 0%.
#
# Setting the pool to 0 here destroys those threads in the parent BEFORE
# fork. Each dataloader worker re-allocates its own pool from scratch in
# ``worker_init_fn`` (``apply_openexr_internal_threads``), which runs
# post-fork and is therefore safe.
if _HAS_EXR_THREAD_API:
    OpenEXR.setGlobalThreadCount(0)


def apply_openexr_internal_threads(n_threads: int) -> int:
    """Set the OpenEXR global thread pool size for *this* process.

    Returns the value actually applied (may be 0 if the binding does not
    expose ``setGlobalThreadCount``). Safe to call multiple times.

    ``Imf::setGlobalThreadCount(n)`` allocates ``n`` persistent worker
    threads in the IlmThread global pool. Those workers parallelise the
    per-block decompression inside one ``OpenEXR.File`` call. ``n=0``
    disables the pool entirely; ``n=1`` allocates a single worker (no
    speedup vs ``n=0`` for a single decode); ``n>=2`` is where the
    per-decode wall-clock actually drops.
    """
    n = max(0, int(n_threads))
    if not _HAS_EXR_THREAD_API:
        return 0
    OpenEXR.setGlobalThreadCount(n)
    return int(OpenEXR.globalThreadCount())


def _get_shard(shard_path: Path) -> Any:
    key = str(shard_path)
    handle = _shard_handles.get(key)
    if handle is None:
        handle = safe_open(key, framework="pt", device="cpu")
        _shard_handles[key] = handle
    return handle


def _get_tar_archive(shard_path: Path) -> tarfile.TarFile:
    key = str(shard_path)
    archive = _tar_archives.get(key)
    if archive is None:
        archive = tarfile.open(key, mode="r:")
        _tar_archives[key] = archive
    return archive


def _tar_member_map_cache_max() -> int:
    raw = os.environ.get("CORRIDORKEY_TAR_MEMBER_MAP_CACHE_MAX_SHARDS", "0").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _close_tar_resources_for_shard(key: str) -> None:
    archive = _tar_archives.pop(key, None)
    if archive is not None:
        try:
            archive.close()
        except Exception:
            pass

    stale_fd_keys = [fd_key for fd_key in _raw_tar_fds if fd_key[0] == key]
    for fd_key in stale_fd_keys:
        fh = _raw_tar_fds.pop(fd_key, None)
        if fh is not None:
            try:
                fh.close()
            except Exception:
                pass


def _enforce_tar_member_map_cache_limit() -> None:
    max_maps = _tar_member_map_cache_max()
    if max_maps <= 0:
        return

    while len(_tar_member_maps) > max_maps:
        old_key, _ = _tar_member_maps.popitem(last=False)
        _close_tar_resources_for_shard(old_key)


def _get_tar_member_map(shard_path: Path) -> Dict[str, TarMemberRef]:
    key = str(shard_path)
    member_map = _tar_member_maps.get(key)
    if member_map is not None:
        _tar_member_maps.move_to_end(key)
        return member_map

    # Do not retain ``TarInfo`` objects or the ``TarFile.members`` list here:
    # large shards can contain millions of members, and every DataLoader
    # worker would otherwise build its own full Python-object copy as it
    # touches shards. We only need the byte range for random reads.
    with tarfile.open(key, mode="r:") as archive:
        member_map = {
            member.name: TarMemberRef(
                offset_data=int(member.offset_data),
                size=int(member.size),
            )
            for member in archive
            if member.isfile()
        }
    _tar_member_maps[key] = member_map
    _enforce_tar_member_map_cache_limit()
    return member_map


def _get_raw_tar_fd(shard_path: Path) -> BinaryIO:
    """Return a raw file handle pointing at *shard_path*, unique per thread.

    ``tarfile.TarFile`` is not thread-safe, and we want to read member bytes
    from several decode threads in parallel without any locking. Since tar
    archives are just concatenated headers + payloads, we can skip the
    ``TarFile`` machinery entirely once we know ``member.offset_data`` and
    ``member.size``.
    """
    key = (str(shard_path), threading.get_ident())
    fh = _raw_tar_fds.get(key)
    if fh is None:
        fh = open(shard_path, "rb")
        _raw_tar_fds[key] = fh
    return fh


def _read_member_payload(shard_path: Path, member: TarMemberRef) -> bytes:
    fh = _get_raw_tar_fd(shard_path)
    fh.seek(member.offset_data)
    return fh.read(member.size)


@dataclass
class ClipIndex:
    name: str
    root: Path
    frame_numbers: List[int]
    frame_paths: Dict[str, List[Path]]
    shard_path: Optional[Path] = None


@dataclass
class WebClipIndex:
    key: str
    name: str
    shard_path: Path
    frame_numbers: List[int]
    modalities: Tuple[str, ...] = field(default_factory=tuple)


def _extract_frame_index(path: Path) -> Optional[int]:
    match = FRAME_INDEX_RE.search(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _extract_shard_index(path: Path) -> Optional[int]:
    match = SHARD_INDEX_RE.search(path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _ensure_chw_tensor(array: np.ndarray) -> Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(array))

    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim == 3:
        tensor = tensor.permute(2, 0, 1)
    else:
        raise ValueError(f"Expected image with 2D/3D shape, got {tuple(tensor.shape)}")

    return tensor


def _normalize_image_tensor(tensor: Tensor) -> Tensor:
    if torch.is_floating_point(tensor):
        return tensor.to(dtype=torch.float32)
    if tensor.dtype == torch.uint8:
        return tensor.to(dtype=torch.float32) / 255.0
    if tensor.dtype == torch.uint16:
        return tensor.to(dtype=torch.float32) / 65535.0
    return tensor.to(dtype=torch.float32)


def _load_exr_array(exr: OpenEXR.InputFile) -> np.ndarray:
    """Legacy decode path using ``OpenEXR.InputFile`` (holds the GIL).

    Kept as a fallback for older ``OpenEXR`` builds that lack the ``File`` API.
    """
    header = exr.header()
    data_window = header.get("dataWindow")
    if data_window is None:
        raise RuntimeError("EXR file missing dataWindow header")

    width = int(data_window.max.x - data_window.min.x + 1)
    height = int(data_window.max.y - data_window.min.y + 1)
    if width < 1 or height < 1:
        raise RuntimeError(f"Invalid EXR dimensions: {width}x{height}")

    channel_map = header.get("channels", {})
    channel_names = set(channel_map.keys()) if isinstance(channel_map, dict) else set()

    pixel_type = Imath.PixelType(Imath.PixelType.FLOAT)

    def read_channel(name: str) -> np.ndarray:
        raw = exr.channel(name, pixel_type)
        values = np.frombuffer(raw, dtype=np.float32)
        expected = width * height
        if values.size != expected:
            raise RuntimeError(
                f"Unexpected EXR channel size for {name}: got {values.size}, expected {expected}"
            )
        return values.reshape((height, width))

    return _compose_array_from_channels(
        {name: read_channel(name) for name in channel_names}
    )


def _compose_array_from_channels(channels: Dict[str, np.ndarray]) -> np.ndarray:
    """Assemble a single HxW[xC] array from a dict of per-channel 2D arrays.

    Implements the same precedence the legacy path used:
      1. R,G,B (+ optional A) -> HxWx3 or HxWx4
      2. single Y channel     -> HxW (grayscale)
      3. single A channel     -> HxW (alpha-only mattes)
      4. fallback: sorted channel names, stacked on axis=2
    """
    if not channels:
        raise RuntimeError("EXR file has no readable channels")

    channel_names = set(channels)
    if {"R", "G", "B"}.issubset(channel_names):
        stack = [channels["R"], channels["G"], channels["B"]]
        if "A" in channel_names:
            stack.append(channels["A"])
        return np.stack(stack, axis=2)

    if "Y" in channel_names:
        return channels["Y"]

    if "A" in channel_names and len(channel_names) == 1:
        return channels["A"]

    fallback = [channels[name] for name in sorted(channel_names)]
    if len(fallback) == 1:
        return fallback[0]
    return np.stack(fallback, axis=2)


def _load_exr_from_path_newapi(path: str, tile_grid: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Decode an EXR using the modern ``OpenEXR.File`` API.

    ``OpenEXR.File`` drops the GIL during channel reads, so multiple decode
    threads in a DataLoader worker can actually run in parallel.
    ``separate_channels=True`` gives us a dict ``{channel_name: 2D ndarray}``
    which we hand to :func:`_compose_array_from_channels` for the same
    R/G/B/A/Y precedence as the legacy decoder.
    """
    if tile_grid:
        # header_only=True keeps the MultiPartInputFile in native tiled mode
        # so TiledInputPart can be constructed inside readTiles().  The default
        # constructor eagerly reads all pixels via the scanline path, which
        # makes a subsequent readTiles() fail with "bad any_cast".
        xmin, xmax, ymin, ymax = tile_grid
        with OpenEXR.File(path, header_only=True) as exr:
            channels = {name: ch.pixels for name, ch in exr.readTiles(xmin, xmax, ymin, ymax, separate_channels=True).items()}
    else:
        with OpenEXR.File(path, separate_channels=True) as exr:
            part = exr.parts[0]
            channels = {name: ch.pixels for name, ch in part.channels.items()}
    return _compose_array_from_channels(channels)


def _load_exr_tiles_from_path_newapi(
    path: str,
    tile_grids: Sequence[Tuple[int, int, int, int]],
) -> List[np.ndarray]:
    arrays: List[np.ndarray] = []
    with OpenEXR.File(path, header_only=True) as exr:
        for xmin, xmax, ymin, ymax in tile_grids:
            channels = {
                name: ch.pixels
                for name, ch in exr.readTiles(xmin, xmax, ymin, ymax, separate_channels=True).items()
            }
            arrays.append(_compose_array_from_channels(channels))
    return arrays


def _to_frame_tensor(array: np.ndarray, convert_to_float: bool) -> Tensor:
    tensor = _ensure_chw_tensor(array)
    return _normalize_image_tensor(tensor) if convert_to_float else tensor


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


def _coerce_source_hw(source_hw: Optional[Sequence[int]]) -> Tuple[int, int]:
    if source_hw is None:
        return (2048, 2048)
    if len(source_hw) != 2:
        raise ValueError(f"source_hw must be [H, W], got {source_hw!r}")
    h, w = int(source_hw[0]), int(source_hw[1])
    if h < 1 or w < 1:
        raise ValueError(f"source_hw values must be positive, got {(h, w)!r}")
    return h, w


def _sample_tile_selection(
    *,
    decode_full_frame: bool,
    source_hw: Tuple[int, int],
    tile_size: int,
    local_tile_span: int,
) -> Tuple[Optional[Tuple[int, int, int, int]], Tensor, Tensor, Tensor]:
    """Choose an EXR tile-grid ROI and return coordinate metadata.

    tile_grid uses the patched OpenEXR.readTiles convention:
    (tile_x_min, tile_x_max, tile_y_min, tile_y_max) with inclusive maxima.

    tile_coords uses the V3 model convention: [y0, y1, x0, x1] in
    source-frame pixels.
    """
    source_h, source_w = source_hw
    source_hw_t = torch.tensor([float(source_h), float(source_w)], dtype=torch.float32)

    if decode_full_frame:
        tile_coords = torch.tensor([0.0, float(source_h), 0.0, float(source_w)], dtype=torch.float32)
        tile_grid_t = torch.tensor([-1, -1, -1, -1], dtype=torch.long)
        return None, tile_coords, source_hw_t, tile_grid_t

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

    tile_grid = (tx0, tx1, ty0, ty1)
    tile_coords = torch.tensor([float(y0), float(y1), float(x0), float(x1)], dtype=torch.float32)
    tile_grid_t = torch.tensor([tx0, tx1, ty0, ty1], dtype=torch.long)
    return tile_grid, tile_coords, source_hw_t, tile_grid_t


def _normalise_global_modalities(modalities: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for modality in modalities:
        m = str(modality).strip()
        if m and m not in out:
            out.append(m)
    return tuple(out)


def _global_key(modality: str) -> str:
    return f"global_{modality}"


def load_frame(path: Path, convert_to_float: bool = True, tile_grid: Optional[Tuple[int, int, int, int]] = None) -> Tensor:
    if path.suffix.lower() != ".exr":
        raise ValueError(f"Expected .exr frame, got: {path.name}")

    if _USE_NEW_EXR_API:
        array = _load_exr_from_path_newapi(str(path), tile_grid=tile_grid)
    else:
        exr = OpenEXR.InputFile(str(path))
        try:
            array = _load_exr_array(exr)
        finally:
            exr.close()

    return _to_frame_tensor(array, convert_to_float)


def _load_payload_via_tmpfile(payload: bytes, tile_grid: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Decode *payload* using the new EXR API by spilling it to a tmpfile.

    ``OpenEXR.File`` has no bytes/stream constructor. A tmpfs-backed tmpfile
    (``/dev/shm`` on Linux) keeps the round trip in RAM and costs well under
    a millisecond per frame compared to the ~600 ms EXR decode itself.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".exr", dir=_TMPFS_DIR)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        return _load_exr_from_path_newapi(tmp_path, tile_grid=tile_grid)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _load_payload_tiles_via_tmpfile(
    payload: bytes,
    tile_grids: Sequence[Tuple[int, int, int, int]],
) -> List[np.ndarray]:
    fd, tmp_path = tempfile.mkstemp(suffix=".exr", dir=_TMPFS_DIR)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
        return _load_exr_tiles_from_path_newapi(tmp_path, tile_grids=tile_grids)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def load_frame_bytes(payload: bytes, convert_to_float: bool = True, tile_grid: Optional[Tuple[int, int, int, int]] = None) -> Tensor:
    if not payload:
        raise RuntimeError("Cannot decode empty frame payload")

    if _USE_NEW_EXR_API:
        array = _load_payload_via_tmpfile(payload, tile_grid=tile_grid)
    else:
        exr_stream = io.BytesIO(payload)
        exr = OpenEXR.InputFile(exr_stream)
        try:
            array = _load_exr_array(exr)
        finally:
            exr.close()

    return _to_frame_tensor(array, convert_to_float)


def _load_member_tensor(
    shard_path: Path,
    member: TarMemberRef,
    convert_to_float: bool,
    tile_grid: Optional[Tuple[int, int, int, int]] = None,
) -> Tensor:
    """Read *member* from *shard_path* and decode to a CHW tensor.

    Combines the raw-bytes read (via a thread-local file handle) and the EXR
    decode so a single thread-pool task can do both steps without jumping
    back to the main Python thread.
    """
    payload = _read_member_payload(shard_path, member)
    if _USE_NEW_EXR_API:
        array = _load_payload_via_tmpfile(payload, tile_grid=tile_grid)
    else:
        exr_stream = io.BytesIO(payload)
        exr = OpenEXR.InputFile(exr_stream)
        try:
            array = _load_exr_array(exr)
        finally:
            exr.close()
    return _to_frame_tensor(array, convert_to_float)


class CorridorKeySequenceDataset(Dataset):
    """
    Sequence-preserving frame loader for CorridorKeyDataset.

    Each item is a temporal window from a single clip and contains:
    - one tensor per modality with shape [T, C, H, W]
    - frame_numbers with shape [T]
    - clip_name as a string

    Temporal order is preserved by sorted frame indices.
    """

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
        exr_decode_threads: int = 1,
        decode_full_frame: bool = False,
        local_tile_span: int = 4,
        exr_tile_size: int = 256,
        source_hw: Optional[Sequence[int]] = None,
        emit_tile_metadata: bool = True,
        decode_global_context: bool = False,
        global_context_root_dir: Path | str | None = None,
        global_context_long_side: int = 0,
        global_context_modalities: Sequence[str] = ("FG", "Alpha"),
        **_compat_kwargs: object,
    ) -> None:
        super().__init__()

        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if sequence_stride < 1:
            raise ValueError("sequence_stride must be >= 1")

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self.sequence_stride = sequence_stride
        self.modalities = tuple(modalities)
        if not self.modalities:
            raise ValueError("modalities must not be empty")
        self.transform = transform
        self.convert_to_float = convert_to_float
        self.clip_len_range = clip_len_range
        self.exr_decode_threads = max(1, int(exr_decode_threads))
        self.decode_full_frame = bool(decode_full_frame)
        self.local_tile_span = max(1, int(local_tile_span))
        self.exr_tile_size = max(1, int(exr_tile_size))
        self.source_hw = _coerce_source_hw(source_hw)
        self.emit_tile_metadata = bool(emit_tile_metadata)
        self.decode_global_context = bool(decode_global_context)
        self.global_context_root_dir = Path(global_context_root_dir) if global_context_root_dir else None
        self.global_context_long_side = int(global_context_long_side)
        self.global_context_modalities = _normalise_global_modalities(global_context_modalities)
        self._decode_pool: Optional[ThreadPoolExecutor] = None

        self._include_clips = set(include_clips) if include_clips else None
        self._exclude_clips = set(exclude_clips) if exclude_clips else set()

        self.clips: List[ClipIndex] = []
        self.samples: List[Tuple[int, int]] = []
        self._build_index()

    def _build_index(self) -> None:
        min_required = (self.sequence_length - 1) * self.frame_stride + 1

        for clip_dir in self._iter_clip_dirs():
            clip = self._index_clip(clip_dir)
            if clip is None:
                continue

            if len(clip.frame_numbers) < min_required:
                continue

            clip_idx = len(self.clips)
            self.clips.append(clip)

            max_start = len(clip.frame_numbers) - min_required
            for start in range(0, max_start + 1, self.sequence_stride):
                self.samples.append((clip_idx, start))

        if not self.samples:
            raise RuntimeError(
                "No valid sequence windows found. Check root path/modalities/extensions."
            )

    def _iter_clip_dirs(self):
        for entry in sorted(self.root_dir.iterdir()):
            if not entry.is_dir():
                continue
            if entry.name.startswith("."):
                continue
            if self._include_clips is not None and entry.name not in self._include_clips:
                continue
            if entry.name in self._exclude_clips:
                continue
            yield entry

    def _index_clip(self, clip_dir: Path) -> Optional[ClipIndex]:
        modality_maps: Dict[str, Dict[int, Path]] = {}

        for modality in self.modalities:
            modality_dir = clip_dir / modality
            if not modality_dir.is_dir():
                return None

            frame_map = self._index_modality_dir(modality_dir)
            if not frame_map:
                return None
            modality_maps[modality] = frame_map

        common_indices = set(modality_maps[self.modalities[0]].keys())
        for modality in self.modalities[1:]:
            common_indices &= set(modality_maps[modality].keys())

        if not common_indices:
            return None

        frame_numbers = sorted(common_indices)
        frame_paths = {
            modality: [modality_maps[modality][idx] for idx in frame_numbers]
            for modality in self.modalities
        }

        shard = clip_dir / SHARD_FILENAME
        return ClipIndex(
            name=clip_dir.name,
            root=clip_dir,
            frame_numbers=frame_numbers,
            frame_paths=frame_paths,
            shard_path=shard if shard.exists() else None,
        )

    def _index_modality_dir(self, modality_dir: Path) -> Dict[int, Path]:
        frame_map: Dict[int, Path] = {}

        for frame_path in sorted(modality_dir.iterdir()):
            if not frame_path.is_file():
                continue
            if frame_path.suffix.lower() != ".exr":
                continue

            frame_index = _extract_frame_index(frame_path)
            if frame_index is None:
                continue

            frame_map[frame_index] = frame_path

        return frame_map

    def __len__(self) -> int:
        return len(self.samples)

    def _get_decode_pool(self) -> Optional[ThreadPoolExecutor]:
        if self.exr_decode_threads <= 1:
            return None
        if self._decode_pool is None:
            self._decode_pool = ThreadPoolExecutor(max_workers=self.exr_decode_threads)
        return self._decode_pool

    def _load_local_frames(self, paths: List[Path], positions: List[int]) -> List[Tensor]:
        selected_paths = [paths[pos] for pos in positions]
        pool = self._get_decode_pool()
        if pool is None:
            return [
                load_frame(path, convert_to_float=self.convert_to_float)
                for path in selected_paths
            ]

        futures = [pool.submit(load_frame, path, self.convert_to_float) for path in selected_paths]
        return [future.result() for future in futures]

    def _load_all_modalities_local(
        self,
        clip: "ClipIndex",
        positions: List[int],
        tile_grid: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, List[Tensor]]:
        """Decode every (modality, frame) in one thread-pool flight.

        Doing all modalities at once gives the pool ``len(modalities) * T``
        independent work items, so with ``exr_decode_threads >= 8`` we fully
        saturate the decode threads instead of leaving them idle between
        modality loops.
        """
        tasks: List[Tuple[str, int, Path]] = []
        for modality in self.modalities:
            paths = clip.frame_paths[modality]
            for i, pos in enumerate(positions):
                tasks.append((modality, i, paths[pos]))

        pool = self._get_decode_pool()
        if pool is None:
            decoded = [
                load_frame(path, convert_to_float=self.convert_to_float, tile_grid=tile_grid)
                for _, _, path in tasks
            ]
        else:
            futures = [
                pool.submit(load_frame, path, self.convert_to_float, tile_grid)
                for _, _, path in tasks
            ]
            decoded = [f.result() for f in futures]

        by_mod: Dict[str, List[Tensor]] = {m: [None] * len(positions) for m in self.modalities}  # type: ignore[assignment]
        for (modality, i, _), tensor in zip(tasks, decoded):
            by_mod[modality][i] = tensor
        return by_mod

    def _global_frame_candidates(self, clip: "ClipIndex", modality: str, pos: int) -> List[Path]:
        paths = clip.frame_paths.get(modality)
        if not paths:
            return []
        original = paths[pos]
        side = str(int(self.global_context_long_side)) if self.global_context_long_side > 0 else ""
        candidates: List[Path] = []
        if self.global_context_root_dir is not None:
            root = self.global_context_root_dir
            candidates.extend([
                root / clip.name / modality / original.name,
                root / clip.name / f"{modality}_{side}" / original.name if side else root / clip.name / modality / original.name,
                root / clip.name / f"global_{side}" / modality / original.name if side else root / clip.name / "global" / modality / original.name,
                root / clip.name / "Global" / modality / original.name,
            ])
        candidates.extend([
            original.parent.parent / f"{modality}_{side}" / original.name if side else original.parent / original.name,
            original.parent.parent / f"global_{side}" / modality / original.name if side else original.parent / original.name,
            original.parent.parent / "Global" / modality / original.name,
        ])
        # Keep order but drop exact duplicates.
        out: List[Path] = []
        seen = set()
        for path in candidates:
            key = str(path)
            if key not in seen:
                out.append(path)
                seen.add(key)
        return out

    def _resolve_global_frame_path(self, clip: "ClipIndex", modality: str, pos: int) -> Path:
        for candidate in self._global_frame_candidates(clip, modality, pos):
            if candidate.is_file():
                return candidate
        tried = ", ".join(str(p) for p in self._global_frame_candidates(clip, modality, pos)[:6])
        raise FileNotFoundError(
            f"Missing global-context sidecar for clip={clip.name!r} modality={modality!r} "
            f"frame_pos={pos}. Tried: {tried}"
        )

    def _load_global_modalities_local(self, clip: "ClipIndex", positions: List[int]) -> Dict[str, Tensor]:
        if not self.decode_global_context:
            return {}
        by_mod: Dict[str, Tensor] = {}
        pool = self._get_decode_pool()
        for modality in self.global_context_modalities:
            paths = [self._resolve_global_frame_path(clip, modality, pos) for pos in positions]
            if pool is None:
                frames = [load_frame(path, convert_to_float=self.convert_to_float) for path in paths]
            else:
                frames = list(pool.map(lambda path: load_frame(path, self.convert_to_float), paths))
            by_mod[_global_key(modality)] = torch.stack(frames, dim=0)
        return by_mod

    def __getitem__(self, index: int) -> Dict[str, object]:
        clip_idx, start = self.samples[index]
        clip = self.clips[clip_idx]

        all_positions = [start + (i * self.frame_stride) for i in range(self.sequence_length)]

        if self.clip_len_range is not None:
            lo, hi = self.clip_len_range
            n = len(all_positions)
            clip_len = random.randint(min(lo, n), min(hi, n))
            sub_start = random.randint(0, max(0, n - clip_len))
            positions = all_positions[sub_start : sub_start + clip_len]
        else:
            positions = all_positions

        frame_numbers = [clip.frame_numbers[pos] for pos in positions]
        tile_grid, tile_coords, source_hw, tile_grid_t = _sample_tile_selection(
            decode_full_frame=self.decode_full_frame,
            source_hw=self.source_hw,
            tile_size=self.exr_tile_size,
            local_tile_span=self.local_tile_span,
        )

        sample: Dict[str, object] = {
            "clip_name": clip.name,
            "frame_numbers": torch.tensor(frame_numbers, dtype=torch.long),
        }
        if self.emit_tile_metadata:
            sample["tile_coords"] = tile_coords
            sample["source_hw"] = source_hw
            sample["tile_grid"] = tile_grid_t

        if clip.shard_path is not None and self.decode_full_frame:
            # Legacy safetensors cache stores full tensors, so it cannot use
            # OpenEXR.readTiles(). Only use it for full-frame decode mode;
            # tiled V3 training must fall through to the EXR readTiles path.
            shard = _get_shard(clip.shard_path)
            for modality in self.modalities:
                paths = clip.frame_paths[modality]
                frames = []
                for pos in positions:
                    key = f"{modality}/{paths[pos].stem}"
                    tensor = shard.get_tensor(key)
                    if self.convert_to_float:
                        tensor = tensor.to(dtype=torch.float32)
                    frames.append(tensor)
                sample[modality] = torch.stack(frames, dim=0)
        else:
            decoded = self._load_all_modalities_local(clip=clip, positions=positions, tile_grid=tile_grid)
            for modality in self.modalities:
                sample[modality] = torch.stack(decoded[modality], dim=0)
            sample.update(self._load_global_modalities_local(clip=clip, positions=positions))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class CorridorKeyWebSequenceDataset(Dataset):
    """
    Sequence-preserving loader for clip-level WebDataset shards.

    Expected sample layout inside each tar shard:
    - <key>.json with metadata fields: key, clip_name, frame_numbers, modalities
    - <key>.<Modality>.<frame_idx>.exr for each aligned frame

    Split behavior:
    - split="all": load every matching shard.
    - split="train": load all shards except validation_shard_indices.
    - split="validation" / "val" / "dev" / "eval": load only validation_shard_indices.
    """

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
        manifest_filename: str = "clips_manifest.jsonl",
        exr_decode_threads: int = 1,
        split: str = "all",
        validation_shard_indices: Optional[Sequence[int]] = None,
        decode_full_frame: bool = False,
        local_tile_span: int = 4,
        exr_tile_size: int = 256,
        source_hw: Optional[Sequence[int]] = None,
        emit_tile_metadata: bool = True,
        decode_global_context: bool = False,
        global_context_root_dir: Path | str | None = None,
        global_context_long_side: int = 0,
        global_context_modalities: Sequence[str] = ("FG", "Alpha"),
        **_compat_kwargs: object,
    ) -> None:
        super().__init__()

        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if frame_stride < 1:
            raise ValueError("frame_stride must be >= 1")
        if sequence_stride < 1:
            raise ValueError("sequence_stride must be >= 1")

        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self.sequence_stride = sequence_stride
        self.modalities = tuple(modalities)
        if not self.modalities:
            raise ValueError("modalities must not be empty")
        self.transform = transform
        self.convert_to_float = convert_to_float
        self.clip_len_range = clip_len_range
        self.shard_glob = shard_glob
        self.manifest_filename = manifest_filename
        self.exr_decode_threads = max(1, int(exr_decode_threads))
        self.decode_full_frame = bool(decode_full_frame)
        self.local_tile_span = max(1, int(local_tile_span))
        self.exr_tile_size = max(1, int(exr_tile_size))
        self.source_hw = _coerce_source_hw(source_hw)
        self.emit_tile_metadata = bool(emit_tile_metadata)
        self.decode_global_context = bool(decode_global_context)
        self.global_context_root_dir = Path(global_context_root_dir) if global_context_root_dir else None
        self.global_context_long_side = int(global_context_long_side)
        self.global_context_modalities = _normalise_global_modalities(global_context_modalities)
        self._decode_pool: Optional[ThreadPoolExecutor] = None
        split_key = str(split).strip().lower()
        if split_key in {"val", "dev", "eval"}:
            split_key = "validation"
        if split_key not in {"all", "train", "validation"}:
            raise ValueError(
                f"Unsupported split={split!r}. Expected one of: all, train, validation."
            )
        self.split = split_key
        if validation_shard_indices is None:
            validation_shard_indices = ()
        self.validation_shard_indices = {int(v) for v in validation_shard_indices}

        self._include_clips = set(include_clips) if include_clips else None
        self._exclude_clips = set(exclude_clips) if exclude_clips else set()

        self.clips: List[WebClipIndex] = []
        self.samples: List[Tuple[int, int]] = []
        self._build_index()

    def _select_split_shards(self, shard_paths: Sequence[Path]) -> List[Path]:
        if self.split == "all":
            return list(shard_paths)

        selected: List[Path] = []
        for shard_path in shard_paths:
            shard_idx = _extract_shard_index(shard_path)
            is_validation_shard = (
                shard_idx is not None and shard_idx in self.validation_shard_indices
            )
            if self.split == "validation":
                if is_validation_shard:
                    selected.append(shard_path)
            else:  # self.split == "train"
                if not is_validation_shard:
                    selected.append(shard_path)
        return selected

    def _get_decode_pool(self) -> Optional[ThreadPoolExecutor]:
        if self.exr_decode_threads <= 1:
            return None
        if self._decode_pool is None:
            self._decode_pool = ThreadPoolExecutor(max_workers=self.exr_decode_threads)
        return self._decode_pool

    def _decode_payloads(self, payloads: Sequence[bytes]) -> List[Tensor]:
        """Legacy decode path that takes raw payloads (kept for compatibility)."""
        pool = self._get_decode_pool()
        if pool is None:
            return [
                load_frame_bytes(payload, convert_to_float=self.convert_to_float)
                for payload in payloads
            ]

        futures = [pool.submit(load_frame_bytes, payload, self.convert_to_float) for payload in payloads]
        return [future.result() for future in futures]

    def _resolve_member(
        self,
        clip: WebClipIndex,
        modality: str,
        frame_number: int,
    ) -> TarMemberRef:
        member_map = _get_tar_member_map(clip.shard_path)
        for member_name in (
            f"{clip.key}.{modality.lower()}.{frame_number:05d}.exr",
            f"{clip.key}.{modality.lower()}.{frame_number:02d}.exr",
            f"{clip.key}.{modality.lower()}.{frame_number}.exr",
        ):
            member = member_map.get(member_name)
            if member is not None:
                return member
        raise KeyError(
            f"Missing frame member for clip='{clip.name}', modality='{modality}', "
            f"frame={frame_number} in shard {clip.shard_path}"
        )

    def _resolve_global_member(
        self,
        clip: WebClipIndex,
        modality: str,
        frame_number: int,
    ) -> TarMemberRef:
        member_map = _get_tar_member_map(clip.shard_path)
        side = str(int(self.global_context_long_side)) if self.global_context_long_side > 0 else ""
        nums = (f"{frame_number:05d}", f"{frame_number:02d}", f"{frame_number}")
        prefixes = []
        if side:
            prefixes.extend([
                f"{clip.key}.global_{side}.{modality.lower()}",
                f"{clip.key}.global{side}.{modality.lower()}",
                f"{clip.key}.{modality.lower()}_{side}",
                f"{clip.key}.{modality.lower()}.global_{side}",
            ])
        prefixes.extend([
            f"{clip.key}.global.{modality.lower()}",
            f"{clip.key}.global_{modality.lower()}",
        ])
        for prefix in prefixes:
            for num in nums:
                member = member_map.get(f"{prefix}.{num}.exr")
                if member is not None:
                    return member
        raise KeyError(
            f"Missing global-context sidecar for clip='{clip.name}', modality='{modality}', "
            f"frame={frame_number} in shard {clip.shard_path}"
        )

    def _resolve_global_or_frame_member(
        self,
        clip: WebClipIndex,
        modality: str,
        frame_number: int,
    ) -> Tuple[TarMemberRef, bool]:
        try:
            return self._resolve_global_member(
                clip=clip,
                modality=modality,
                frame_number=frame_number,
            ), False
        except KeyError:
            return self._resolve_member(
                clip=clip,
                modality=modality,
                frame_number=frame_number,
            ), True

    def _resize_global_tensor_if_needed(self, tensor: Tensor, fallback_from_frame: bool) -> Tensor:
        if not fallback_from_frame or self.global_context_long_side <= 0:
            return tensor
        h, w = tensor.shape[-2:]
        long_side = max(h, w)
        target_long_side = int(self.global_context_long_side)
        if long_side == target_long_side:
            return tensor
        scale = target_long_side / float(long_side)
        target_h = max(1, int(round(h * scale)))
        target_w = max(1, int(round(w * scale)))
        resized = torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
        return resized.squeeze(0)

    def _global_frame_fallback_tile_grid(self) -> Tuple[int, int, int, int]:
        source_h, source_w = self.source_hw
        tile = max(1, int(self.exr_tile_size))
        tiles_y = max(1, int(math.ceil(source_h / tile)))
        tiles_x = max(1, int(math.ceil(source_w / tile)))
        return (0, tiles_x - 1, 0, tiles_y - 1)

    def _fast_global_context_from_frame_member(
        self,
        shard_path: Path,
        member: TarMemberRef,
        tile_grid: Optional[Tuple[int, int, int, int]],
        convert_to_float: bool,
    ) -> Optional[Tensor]:
        if not _USE_NEW_EXR_API or tile_grid is None or self.global_context_long_side <= 0:
            return None

        source_h, source_w = self.source_hw
        source_long = max(source_h, source_w)
        target_long = int(self.global_context_long_side)
        if source_long != target_long * 2:
            return None

        target_h = max(1, int(round(source_h * (target_long / float(source_long)))))
        target_w = max(1, int(round(source_w * (target_long / float(source_long)))))
        context_h = target_h * 2
        context_w = target_w * 2
        if context_h > source_h or context_w > source_w:
            return None

        native_tile = max(1, int(self.exr_tile_size))
        tx0, _, ty0, _ = tile_grid
        tile_y0 = ty0 * native_tile
        tile_x0 = tx0 * native_tile
        context_y0 = min(max(0, tile_y0), max(0, source_h - context_h))
        context_x0 = min(max(0, tile_x0), max(0, source_w - context_w))
        half_h = context_h // 2
        half_w = context_w // 2
        if half_h <= 0 or half_w <= 0:
            return None

        def _grid_for(y0: int, x0: int, y1: int, x1: int) -> Tuple[int, int, int, int]:
            return (
                x0 // native_tile,
                max(x0 // native_tile, (x1 - 1) // native_tile),
                y0 // native_tile,
                max(y0 // native_tile, (y1 - 1) // native_tile),
            )

        specs: List[Tuple[int, int, int, int, int, int, int]] = []
        grids: List[Tuple[int, int, int, int]] = []
        for y_off in (0, half_h):
            for x_off in (0, half_w):
                y0 = context_y0 + y_off
                x0 = context_x0 + x_off
                y1 = min(source_h, y0 + half_h)
                x1 = min(source_w, x0 + half_w)
                grid = _grid_for(y0, x0, y1, x1)
                local_y0 = y0 - (y0 // native_tile) * native_tile
                local_x0 = x0 - (x0 // native_tile) * native_tile
                crop_h = y1 - y0
                crop_w = x1 - x0
                specs.append((y_off, x_off, crop_h, crop_w, local_y0, local_x0, len(grids)))
                grids.append(grid)

        if _USE_NEW_EXR_API:
            payload = _read_member_payload(shard_path, member)
            tensors = [
                _to_frame_tensor(array, convert_to_float)
                for array in _load_payload_tiles_via_tmpfile(payload, grids)
            ]
        else:
            tensors = [
                _load_member_tensor(shard_path, member, convert_to_float, tile_grid=grid)
                for grid in grids
            ]

        c = int(tensors[0].shape[0])
        canvas = torch.empty((c, context_h, context_w), dtype=tensors[0].dtype)
        for y_off, x_off, crop_h, crop_w, local_y0, local_x0, tensor_idx in specs:
            tensor = tensors[tensor_idx]
            piece = tensor[:, local_y0 : local_y0 + crop_h, local_x0 : local_x0 + crop_w]
            canvas[:, y_off : y_off + crop_h, x_off : x_off + crop_w] = piece
        return _downscale_2x_box_chw(canvas, accumulate_dtype=torch.float16)

    def _load_global_modalities_webdataset(
        self,
        clip: WebClipIndex,
        frame_numbers: Sequence[int],
        tile_grid: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Tensor]:
        if not self.decode_global_context:
            return {}
        shard_path = clip.shard_path
        convert = self.convert_to_float
        pool = self._get_decode_pool()
        by_mod: Dict[str, Tensor] = {}

        def _do(item: Tuple[TarMemberRef, bool]) -> Tensor:
            member, fallback_from_frame = item
            if fallback_from_frame:
                fast_tensor = self._fast_global_context_from_frame_member(
                    shard_path,
                    member,
                    tile_grid,
                    convert,
                )
                if fast_tensor is not None:
                    return fast_tensor
            read_tile_grid = self._global_frame_fallback_tile_grid() if fallback_from_frame else None
            tensor = _load_member_tensor(shard_path, member, convert, tile_grid=read_tile_grid)
            return self._resize_global_tensor_if_needed(tensor, fallback_from_frame)

        for modality in self.global_context_modalities:
            members = [
                self._resolve_global_or_frame_member(clip=clip, modality=modality, frame_number=fn)
                for fn in frame_numbers
            ]
            if pool is None:
                frame_tensors: List[Tensor] = [_do(m) for m in members]
            else:
                frame_tensors = list(pool.map(_do, members))
            by_mod[_global_key(modality)] = torch.stack(frame_tensors, dim=0)
            del frame_tensors
        return by_mod

    def _load_all_modalities_webdataset(
        self,
        clip: WebClipIndex,
        frame_numbers: Sequence[int],
        tile_grid: Optional[Tuple[int, int, int, int]] = None,
    ) -> Dict[str, Tensor]:
        """Extract, decode and stack every (modality, frame) for a clip.

        Memory profile vs the previous all-36-tasks-at-once implementation:
        that version held every individual frame tensor alive in a Python
        list until the caller stacked them per-modality, peaking at
        ``T * len(modalities) * frame_bytes`` (~1.7 GB for a 12-frame fp32
        2048x2048 3-channel sample). With 4 workers x 4 ranks that was
        ~27 GB of worker scratch on top of the IPC-prefetch queue and was
        enough to push the host over the edge at iter ~115.

        Here we process one modality at a time. We still submit all T
        decodes of a modality to the thread pool so the GIL-free EXR
        decoder stays saturated, but as soon as the modality is stacked
        into a single contiguous tensor we drop the list of individuals
        before starting the next modality. Peak scratch per worker drops
        to ~1 modality's-worth of individual tensors + the already-stacked
        prior modalities (~600 MB for a 12-frame fp32 clip), a ~2x win.

        Throughput is unchanged: the decode pool is CPU-bound on
        ``exr_decode_threads`` and that's true whether we feed it 12 tasks
        or 36.
        """
        modalities = self.modalities
        shard_path = clip.shard_path
        convert = self.convert_to_float
        pool = self._get_decode_pool()

        def _do(member: TarMemberRef) -> Tensor:
            return _load_member_tensor(shard_path, member, convert, tile_grid=tile_grid)

        by_mod: Dict[str, Tensor] = {}
        for modality in modalities:
            members = [
                self._resolve_member(clip=clip, modality=modality, frame_number=fn)
                for fn in frame_numbers
            ]
            if pool is None:
                frame_tensors: List[Tensor] = [_do(m) for m in members]
            else:
                frame_tensors = list(pool.map(_do, members))
            by_mod[modality] = torch.stack(frame_tensors, dim=0)
            del frame_tensors
        return by_mod

    def _load_all_modalities_webdataset_fast_tile(
        self,
        clip: WebClipIndex,
        frame_numbers: Sequence[int],
        tile_grid: Tuple[int, int, int, int],
    ) -> Dict[str, Tensor]:
        """Fast WebDataset path for OpenEXR tiled ROI reads.

        The full-frame path above intentionally serialises by modality to keep
        worker scratch memory bounded. For tiled ROI training, each decoded
        tensor is much smaller, so submit all (modality, frame) decodes in one
        pool flight. This removes the Input -> FG -> BG -> Alpha barriers that
        otherwise leave the GPU waiting between batches.
        """
        shard_path = clip.shard_path
        convert = self.convert_to_float
        pool = self._get_decode_pool()

        tasks: List[Tuple[str, int, TarMemberRef]] = []
        for modality in self.modalities:
            for i, frame_number in enumerate(frame_numbers):
                tasks.append((
                    modality,
                    i,
                    self._resolve_member(
                        clip=clip,
                        modality=modality,
                        frame_number=frame_number,
                    ),
                ))

        def _decode_task(task: Tuple[str, int, TarMemberRef]) -> Tensor:
            _, _, member = task
            return _load_member_tensor(
                shard_path,
                member,
                convert,
                tile_grid=tile_grid,
            )

        if pool is None:
            decoded = [_decode_task(task) for task in tasks]
        else:
            decoded = list(pool.map(_decode_task, tasks))

        grouped: Dict[str, List[Optional[Tensor]]] = {
            modality: [None] * len(frame_numbers) for modality in self.modalities
        }
        for (modality, i, _), tensor in zip(tasks, decoded):
            grouped[modality][i] = tensor

        by_mod: Dict[str, Tensor] = {}
        for modality in self.modalities:
            frames: List[Tensor] = []
            for i, tensor in enumerate(grouped[modality]):
                if tensor is None:
                    raise RuntimeError(
                        f"Missing decoded tensor for modality={modality!r}, "
                        f"frame_index={i}, clip={clip.name!r}"
                    )
                frames.append(tensor)
            by_mod[modality] = torch.stack(frames, dim=0)
        return by_mod

    def _build_index(self) -> None:
        shard_paths = sorted(self.root_dir.glob(self.shard_glob))
        if not shard_paths:
            raise RuntimeError(
                f"No WebDataset shards matching '{self.shard_glob}' found in: {self.root_dir}"
            )
        shard_paths = self._select_split_shards(shard_paths)
        if not shard_paths:
            raise RuntimeError(
                f"No shards selected for split='{self.split}' with "
                f"validation_shard_indices={sorted(self.validation_shard_indices)}."
            )

        shard_map = {path.name: path for path in shard_paths}

        min_required = (self.sequence_length - 1) * self.frame_stride + 1

        loaded_from_manifest = self._build_index_from_manifest(
            shard_map=shard_map,
            min_required=min_required,
        )

        if not loaded_from_manifest:
            for shard_path in shard_paths:
                for meta in self._iter_shard_metadata(shard_path):
                    clip = self._clip_from_metadata(shard_path=shard_path, meta=meta)
                    self._append_clip_samples(clip=clip, min_required=min_required)

        if not self.samples:
            raise RuntimeError(
                "No valid sequence windows found in WebDataset shards. "
                "Check root path, shard pattern, and modalities."
            )

    def _append_clip_samples(self, clip: Optional[WebClipIndex], min_required: int) -> None:
        if clip is None:
            return
        if len(clip.frame_numbers) < min_required:
            return

        clip_idx = len(self.clips)
        self.clips.append(clip)

        max_start = len(clip.frame_numbers) - min_required
        for start in range(0, max_start + 1, self.sequence_stride):
            self.samples.append((clip_idx, start))

    def _build_index_from_manifest(self, shard_map: Dict[str, Path], min_required: int) -> bool:
        manifest_path = self.root_dir / self.manifest_filename
        if not manifest_path.is_file():
            return False

        parsed_records = 0
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                payload = line.strip()
                if not payload:
                    continue

                parsed_records += 1
                try:
                    record = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(
                        f"Invalid JSON in manifest {manifest_path} at line {line_no}"
                    ) from exc

                if not isinstance(record, dict):
                    continue

                clip = self._clip_from_manifest_record(record=record, shard_map=shard_map)
                self._append_clip_samples(clip=clip, min_required=min_required)

        return parsed_records > 0

    def _clip_from_manifest_record(
        self,
        record: Dict[str, Any],
        shard_map: Dict[str, Path],
    ) -> Optional[WebClipIndex]:
        key = str(record.get("key", "")).strip()
        if not key:
            return None

        clip_name = str(record.get("clip_name", key))
        if not self._clip_is_selected(clip_name=clip_name, key=key):
            return None

        shard_name = str(record.get("shard", "")).strip()
        if not shard_name:
            return None

        shard_path = shard_map.get(shard_name)
        if shard_path is None:
            shard_path = shard_map.get(Path(shard_name).name)
        if shard_path is None:
            return None
        if not fnmatch.fnmatch(shard_path.name, self.shard_glob):
            return None

        frame_numbers = self._frame_numbers_from_record(record)
        if frame_numbers is None:
            frame_numbers = self._frame_numbers_from_shard_metadata(shard_path=shard_path, key=key)
        if not frame_numbers:
            return None

        sample_modalities = tuple(str(m) for m in record.get("modalities", []))
        if sample_modalities:
            sample_mod_lower = {m.lower() for m in sample_modalities}
            if any(modality.lower() not in sample_mod_lower for modality in self.modalities):
                return None

        return WebClipIndex(
            key=key,
            name=clip_name,
            shard_path=shard_path,
            frame_numbers=frame_numbers,
            modalities=sample_modalities,
        )

    def _clip_is_selected(self, clip_name: str, key: str) -> bool:
        if self._include_clips is not None:
            if clip_name not in self._include_clips and key not in self._include_clips:
                return False

        if clip_name in self._exclude_clips or key in self._exclude_clips:
            return False

        return True

    def _frame_numbers_from_record(self, record: Dict[str, Any]) -> Optional[List[int]]:
        frame_values = record.get("frame_numbers")
        if isinstance(frame_values, list) and frame_values:
            try:
                frame_numbers = sorted({int(v) for v in frame_values})
            except (TypeError, ValueError):
                frame_numbers = []
            if frame_numbers:
                return frame_numbers

        num_frames = record.get("num_frames")
        frame_start = record.get("frame_start")
        frame_end = record.get("frame_end")

        try:
            n = int(num_frames) if num_frames is not None else None
            start = int(frame_start) if frame_start is not None else None
            end = int(frame_end) if frame_end is not None else None
        except (TypeError, ValueError):
            return None

        if start is not None and end is not None and end >= start:
            contiguous = list(range(start, end + 1))
            if n is not None and n > 0 and n != len(contiguous):
                return list(range(start, start + n))
            return contiguous

        if start is not None and n is not None and n > 0:
            return list(range(start, start + n))

        return None

    def _frame_numbers_from_shard_metadata(self, shard_path: Path, key: str) -> Optional[List[int]]:
        member_map = _get_tar_member_map(shard_path)
        member = member_map.get(f"{key}.json")
        if member is None:
            return None

        payload = _read_member_payload(shard_path, member)

        try:
            meta = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

        if not isinstance(meta, dict):
            return None

        frame_numbers = None
        frame_values = meta.get("frame_numbers")
        if isinstance(frame_values, list) and frame_values:
            try:
                frame_numbers = sorted({int(v) for v in frame_values})
            except (TypeError, ValueError):
                pass
        
        if not frame_numbers:
            fpm = meta.get("frames_per_modality")
            if isinstance(fpm, int) and fpm > 0:
                frame_numbers = list(range(1, fpm + 1))
        
        return frame_numbers if frame_numbers else None

    def _iter_shard_metadata(self, shard_path: Path):
        with tarfile.open(shard_path, mode="r:") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue

                extracted = archive.extractfile(member)
                if extracted is None:
                    continue

                with extracted:
                    payload = extracted.read()

                try:
                    meta = json.loads(payload.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise RuntimeError(f"Invalid metadata in {shard_path}:{member.name}") from exc

                if isinstance(meta, dict):
                    yield meta

    def _clip_from_metadata(self, shard_path: Path, meta: Dict[str, Any]) -> Optional[WebClipIndex]:
        key = str(meta.get("key", "")).strip()
        if not key:
            return None

        clip_name = str(meta.get("clip_name", key))
        if not self._clip_is_selected(clip_name=clip_name, key=key):
            return None

        frame_numbers = None
        frame_values = meta.get("frame_numbers")
        if isinstance(frame_values, list) and frame_values:
            try:
                frame_numbers = sorted({int(v) for v in frame_values})
            except (TypeError, ValueError):
                pass
        
        if not frame_numbers:
            fpm = meta.get("frames_per_modality")
            if isinstance(fpm, int) and fpm > 0:
                frame_numbers = list(range(1, fpm + 1))
        
        if not frame_numbers:
            return None

        sample_modalities = tuple(str(m) for m in meta.get("modalities", []))
        if sample_modalities:
            sample_mod_lower = {m.lower() for m in sample_modalities}
            if any(modality.lower() not in sample_mod_lower for modality in self.modalities):
                return None

        return WebClipIndex(
            key=key,
            name=clip_name,
            shard_path=shard_path,
            frame_numbers=frame_numbers,
            modalities=sample_modalities,
        )

    def _load_member_payload(self, clip: WebClipIndex, modality: str, frame_number: int) -> bytes:
        member_map = _get_tar_member_map(clip.shard_path)
        candidate_names = (
            f"{clip.key}.{modality.lower()}.{frame_number:05d}.exr",
            f"{clip.key}.{modality.lower()}.{frame_number:02d}.exr",
            f"{clip.key}.{modality.lower()}.{frame_number}.exr",
        )

        payload: Optional[bytes] = None
        for member_name in candidate_names:
            member = member_map.get(member_name)
            if member is None:
                continue

            payload = _read_member_payload(clip.shard_path, member)
            break

        if payload is None:
            raise KeyError(
                f"Missing frame member for clip='{clip.name}', modality='{modality}', "
                f"frame={frame_number} in shard {clip.shard_path}"
            )

        return payload

    def _load_member_frame(self, clip: WebClipIndex, modality: str, frame_number: int) -> Tensor:
        payload = self._load_member_payload(clip=clip, modality=modality, frame_number=frame_number)
        return load_frame_bytes(payload, convert_to_float=self.convert_to_float)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        clip_idx, start = self.samples[index]
        clip = self.clips[clip_idx]

        all_positions = [start + (i * self.frame_stride) for i in range(self.sequence_length)]

        if self.clip_len_range is not None:
            lo, hi = self.clip_len_range
            lo, hi = min(lo, hi), max(lo, hi)
            n = len(all_positions)
            clip_len = random.randint(max(1, min(lo, n)), max(1, min(hi, n)))
            sub_start = random.randint(0, max(0, n - clip_len))
            positions = all_positions[sub_start : sub_start + clip_len]
        else:
            positions = all_positions

        frame_numbers = [clip.frame_numbers[pos] for pos in positions]
        tile_grid, tile_coords, source_hw, tile_grid_t = _sample_tile_selection(
            decode_full_frame=self.decode_full_frame,
            source_hw=self.source_hw,
            tile_size=self.exr_tile_size,
            local_tile_span=self.local_tile_span,
        )

        sample: Dict[str, object] = {
            "clip_name": clip.name,
            "frame_numbers": torch.tensor(frame_numbers, dtype=torch.long),
        }
        if self.emit_tile_metadata:
            sample["tile_coords"] = tile_coords
            sample["source_hw"] = source_hw
            sample["tile_grid"] = tile_grid_t

        if tile_grid is not None:
            decoded = self._load_all_modalities_webdataset_fast_tile(
                clip=clip,
                frame_numbers=frame_numbers,
                tile_grid=tile_grid,
            )
        else:
            decoded = self._load_all_modalities_webdataset(
                clip=clip,
                frame_numbers=frame_numbers,
                tile_grid=tile_grid,
            )
        for modality in self.modalities:
            sample[modality] = decoded[modality]
        sample.update(
            self._load_global_modalities_webdataset(
                clip=clip,
                frame_numbers=frame_numbers,
                tile_grid=tile_grid,
            )
        )

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


def resolve_ddp_rank_world_size(
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> Tuple[int, int]:
    ddp_ready = dist.is_available() and dist.is_initialized()

    if world_size is None:
        world_size = dist.get_world_size() if ddp_ready else 1
    if rank is None:
        rank = dist.get_rank() if ddp_ready else 0

    if world_size < 1:
        raise ValueError("world_size must be >= 1")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size - 1}], got rank={rank}")

    return rank, world_size


def _make_worker_init(
    num_torch_threads: int,
    exr_internal_threads: int = 0,
) -> Callable[[int], None]:
    """Build a ``worker_init_fn`` that seeds RNGs and sizes thread pools.

    ``torch.set_num_threads(1)`` is the traditional DataLoader-worker setting
    because many-worker setups would otherwise over-subscribe the CPU. But on
    a 48-core host with only 4 workers per rank, single-threaded CPU ops like
    ``F.interpolate`` (used for 2048->1280 resizes) become the dominant
    per-sample cost (~2 s/sample measured). Letting each worker use a handful
    of torch threads brings that back down; we still cap the total across
    workers so EXR decode threads (GIL-free) don't get starved.

    ``exr_internal_threads`` controls the per-decode IlmThread pool inside a
    *single* ``OpenEXR.File`` call (parallel block decompression). Layered
    with ``exr_decode_threads`` on the dataset (frame-level Python pool), the
    total per-worker decode parallelism is roughly
    ``exr_decode_threads * exr_internal_threads`` cores. Requires the patched
    OpenEXR binding that exposes ``setGlobalThreadCount``; on a stock
    binding this argument is silently ignored.
    """
    per_worker = max(1, int(num_torch_threads))
    exr_n = max(0, int(exr_internal_threads))

    def _init(worker_id: int) -> None:
        # The trainer installs a flag-setting SIGINT handler in the main
        # process *before* DataLoader workers are forked. Without restoring
        # the default handler in each worker, our flag-setter is inherited
        # and Ctrl-C no longer raises KeyboardInterrupt in the worker -
        # which is exactly the mechanism torch.utils.data uses to shut
        # workers down. Restore default SIGINT here so a Ctrl-C in the
        # foreground process group kills the workers cleanly instead of
        # leaving them alive and blocking the parent from exiting.
        import signal as _signal
        for _sig in (_signal.SIGINT, _signal.SIGTERM):
            try:
                _signal.signal(_sig, _signal.SIG_DFL)
            except (ValueError, OSError):
                pass

        worker_seed = torch.initial_seed() % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.set_num_threads(per_worker)
        if exr_n > 0:
            apply_openexr_internal_threads(exr_n)

    return _init


# Keep the legacy ``seed_worker`` name so existing callers that reference it
# directly keep working with the previous (single-threaded) default.
seed_worker = _make_worker_init(1)


def create_ddp_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    drop_last: bool = False,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    prefetch_factor: int = 2,
    seed: int = 1337,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    collate_fn: Optional[Callable] = None,
    num_torch_threads: int = 1,
    exr_internal_threads: int = 0,
) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    rank, world_size = resolve_ddp_rank_world_size(rank=rank, world_size=world_size)

    sampler = None
    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
        )

    generator = torch.Generator()
    generator.manual_seed(seed + rank)

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": bool(shuffle and sampler is None),
        "sampler": sampler,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "worker_init_fn": _make_worker_init(
            num_torch_threads,
            exr_internal_threads=exr_internal_threads,
        ),
        "generator": generator,
        "collate_fn": collate_fn,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        loader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(**loader_kwargs)


def set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:
    sampler = getattr(dataloader, "sampler", None)
    if isinstance(sampler, DistributedSampler):
        sampler.set_epoch(epoch)
