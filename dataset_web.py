from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import fnmatch
import io
import json
import math
import os
import random
import re
import time
import tarfile
import tempfile
import threading
import urllib.parse
import urllib.request
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


def _to_frame_tensor(array: np.ndarray, convert_to_float: bool) -> Tensor:
    tensor = _ensure_chw_tensor(array)
    return _normalize_image_tensor(tensor) if convert_to_float else tensor


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



import webdataset as wds
import huggingface_hub
from webdataset import tariterators


DEFAULT_WEB_DATASET_REPO = "Vano04/CorridorKeyDataset_Custom"
DEFAULT_WEB_SHARD_CACHE_DIR = Path(tempfile.gettempdir()) / "corridorkey_wds_cache"

def clip_to_windows(clip_sample, sequence_length, frame_stride, sequence_stride, modalities, convert_to_float, clip_len_range, decode_full_frame, exr_tile_size, local_tile_span, source_hw, emit_tile_metadata, transform, global_context_long_side, global_context_modalities, decode_global_context):
    key = clip_sample.get("__key__", "unknown")
    
    try:
        meta_bytes = clip_sample.get("json", b"{}")
        if not isinstance(meta_bytes, bytes):
            meta_bytes = meta_bytes.encode('utf-8')
        meta = json.loads(meta_bytes.decode("utf-8"))
    except:
        meta = {}
        
    frame_numbers = None
    frame_values = meta.get("frame_numbers")
    if isinstance(frame_values, list) and frame_values:
        frame_numbers = sorted({int(v) for v in frame_values})
    
    if not frame_numbers:
        fpm = meta.get("frames_per_modality")
        if isinstance(fpm, int) and fpm > 0:
            frame_numbers = list(range(1, fpm + 1))
            
    if not frame_numbers:
        frames = set()
        for k in clip_sample.keys():
            m = re.search(r'\.([0-9]+)\.exr$', k)
            if m:
                frames.add(int(m.group(1)))
        frame_numbers = sorted(frames)

    if not frame_numbers:
        return

    min_required = (sequence_length - 1) * frame_stride + 1
    if len(frame_numbers) < min_required:
        return

    max_start = len(frame_numbers) - min_required
    for start in range(0, max_start + 1, sequence_stride):
        all_positions = [start + (i * frame_stride) for i in range(sequence_length)]
        
        if clip_len_range is not None:
            lo, hi = clip_len_range
            n = len(all_positions)
            clip_len = random.randint(max(1, min(lo, n)), max(1, min(hi, n)))
            sub_start = random.randint(0, max(0, n - clip_len))
            positions = all_positions[sub_start : sub_start + clip_len]
        else:
            positions = all_positions
            
        selected_frames = [frame_numbers[pos] for pos in positions]
        
        tile_grid, tile_coords, source_hw_t, tile_grid_t = _sample_tile_selection(
            decode_full_frame=decode_full_frame,
            source_hw=_coerce_source_hw(source_hw),
            tile_size=exr_tile_size,
            local_tile_span=local_tile_span,
        )
        
        out_sample = {
            "clip_name": meta.get("clip_name", key),
            "frame_numbers": torch.tensor(selected_frames, dtype=torch.long),
        }
        if emit_tile_metadata:
            out_sample["tile_coords"] = tile_coords
            out_sample["source_hw"] = source_hw_t
            out_sample["tile_grid"] = tile_grid_t

        by_mod = {}
        for modality in modalities:
            frame_tensors = []
            for fn in selected_frames:
                candidate_keys = [
                    f"{modality.lower()}.{fn:05d}.exr",
                    f"{modality.lower()}.{fn:02d}.exr",
                    f"{modality.lower()}.{fn}.exr",
                ]
                payload = None
                for ck in candidate_keys:
                    if ck in clip_sample:
                        payload = clip_sample[ck]
                        break
                if payload is None:
                    print(f"WARNING: Missing frame {modality} {fn} in {key}")
                    return # skip this window entirely if missing frames
                
                tensor = load_frame_bytes(payload, convert_to_float=convert_to_float, tile_grid=tile_grid)
                frame_tensors.append(tensor)
            by_mod[modality] = torch.stack(frame_tensors, dim=0)
            
        for modality in modalities:
            out_sample[modality] = by_mod[modality]

        if decode_global_context:
            for modality in global_context_modalities:
                side = str(int(global_context_long_side)) if global_context_long_side > 0 else ""
                frame_tensors = []
                for fn in selected_frames:
                    candidate_keys = []
                    nums = (f"{fn:05d}", f"{fn:02d}", f"{fn}")
                    if side:
                        candidate_keys.extend([f"global_{side}.{modality.lower()}.{num}.exr" for num in nums])
                        candidate_keys.extend([f"global{side}.{modality.lower()}.{num}.exr" for num in nums])
                        candidate_keys.extend([f"{modality.lower()}_{side}.{num}.exr" for num in nums])
                        candidate_keys.extend([f"{modality.lower()}.global_{side}.{num}.exr" for num in nums])
                    candidate_keys.extend([f"global.{modality.lower()}.{num}.exr" for num in nums])
                    candidate_keys.extend([f"global_{modality.lower()}.{num}.exr" for num in nums])
                    
                    payload = None
                    for ck in candidate_keys:
                        if ck in clip_sample:
                            payload = clip_sample[ck]
                            break
                    if payload is None:
                        print(f"WARNING: Missing global {modality} {fn} in {key}")
                        return
                    tensor = load_frame_bytes(payload, convert_to_float=convert_to_float, tile_grid=None)
                    frame_tensors.append(tensor)
                out_sample[_global_key(modality)] = torch.stack(frame_tensors, dim=0)
                
        if transform is not None:
            out_sample = transform(out_sample)
            
        yield out_sample


def _resolve_web_shard_cache_dir(cache_dir: Optional[str | os.PathLike[str]]) -> Path:
    if cache_dir is None:
        return DEFAULT_WEB_SHARD_CACHE_DIR
    raw = str(cache_dir).strip()
    if not raw:
        return DEFAULT_WEB_SHARD_CACHE_DIR
    return Path(raw).expanduser()


def _cache_lock_path(cache_dir: Path) -> Path:
    return cache_dir / ".cache.lock"


def _shard_lock_path(local_path: Path) -> Path:
    return local_path.with_name(f"{local_path.name}.lock")


def _lease_glob(local_path: Path) -> str:
    return f"{local_path.name}.lease.*"


def _local_shard_name(remote_url: str) -> str:
    name = Path(urllib.parse.urlparse(remote_url).path).name
    if not name.endswith(".tar"):
        raise RuntimeError(f"Expected remote shard URL ending in .tar, got: {remote_url}")
    return name


def _local_shard_path(cache_dir: Path, remote_url: str) -> Path:
    return cache_dir / _local_shard_name(remote_url)


def _touch_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b"):
        os.utime(path, None)


def _download_shard_to_path(
    *,
    remote_url: str,
    local_path: Path,
    token: str,
    download_retries: int,
    download_timeout_s: int,
) -> None:
    attempts = max(1, int(download_retries))
    timeout = max(30, int(download_timeout_s))
    local_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, attempts + 1):
        tmp_path = local_path.with_name(
            f"{local_path.name}.part.{os.getpid()}.{threading.get_ident()}"
        )
        try:
            req = urllib.request.Request(
                remote_url,
                headers={"Authorization": f"Bearer {token}"},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response, open(tmp_path, "wb") as handle:
                while True:
                    chunk = response.read(16 * 1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)

            if tmp_path.stat().st_size <= 0:
                raise RuntimeError(f"Downloaded empty shard from {remote_url}")

            os.replace(tmp_path, local_path)
            os.utime(local_path, None)
            return
        except Exception:
            try:
                tmp_path.unlink()
            except OSError:
                pass
            if attempt >= attempts:
                raise
            time.sleep(min(10.0, float(attempt)))


def _ensure_local_shard(
    *,
    remote_url: str,
    token: str,
    cache_dir: Path,
    download_retries: int,
    download_timeout_s: int,
) -> Path:
    local_path = _local_shard_path(cache_dir, remote_url)
    lock_path = _shard_lock_path(local_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    with open(lock_path, "a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if local_path.exists() and local_path.stat().st_size > 0:
                os.utime(local_path, None)
                return local_path

            _download_shard_to_path(
                remote_url=remote_url,
                local_path=local_path,
                token=token,
                download_retries=download_retries,
                download_timeout_s=download_timeout_s,
            )
            return local_path
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _create_shard_lease(local_path: Path) -> Path:
    lease_name = (
        f"{local_path.name}.lease.{os.getpid()}."
        f"{threading.get_ident()}.{time.time_ns()}"
    )
    lease_path = local_path.with_name(lease_name)
    _touch_file(lease_path)
    return lease_path


def _release_shard_lease(lease_path: Path) -> None:
    try:
        lease_path.unlink()
    except OSError:
        pass


def _shard_has_leases(local_path: Path) -> bool:
    return any(local_path.parent.glob(_lease_glob(local_path)))


def _delete_local_shard_if_idle(local_path: Path) -> None:
    cache_dir = local_path.parent
    with open(_cache_lock_path(cache_dir), "a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            if local_path.exists() and not _shard_has_leases(local_path):
                try:
                    local_path.unlink()
                except OSError:
                    pass
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _evict_local_shard_cache(
    *,
    cache_dir: Path,
    max_shards: int,
    exclude_paths: Sequence[Path] = (),
) -> None:
    if max_shards < 1:
        return

    excluded = {path.resolve() for path in exclude_paths}
    with open(_cache_lock_path(cache_dir), "a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            shard_paths = sorted(
                (path for path in cache_dir.glob("*.tar") if path.is_file()),
                key=lambda path: path.stat().st_mtime,
            )
            active_count = len(shard_paths)
            for shard_path in shard_paths:
                if active_count <= max_shards:
                    break
                try:
                    resolved = shard_path.resolve()
                except OSError:
                    resolved = shard_path
                if resolved in excluded:
                    continue
                if _shard_has_leases(shard_path):
                    continue
                try:
                    shard_path.unlink()
                    active_count -= 1
                except OSError:
                    continue
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def _local_shard_opener(
    src: Sequence[Dict[str, Any]] | Any,
    *,
    token: str,
    cache_dir: Path,
    max_local_shards: int,
    delete_after_use: bool,
    download_retries: int,
    download_timeout_s: int,
    handler: Callable[[Exception], bool],
):
    for sample in src:
        assert isinstance(sample, dict), sample
        assert "url" in sample
        remote_url = str(sample["url"])
        try:
            local_path = _ensure_local_shard(
                remote_url=remote_url,
                token=token,
                cache_dir=cache_dir,
                download_retries=download_retries,
                download_timeout_s=download_timeout_s,
            )
            lease_path = _create_shard_lease(local_path)
            try:
                with open(local_path, "rb") as stream:
                    localised = dict(sample)
                    localised["stream"] = stream
                    localised["local_path"] = str(local_path)
                    yield localised
            finally:
                _release_shard_lease(lease_path)
                if delete_after_use:
                    _delete_local_shard_if_idle(local_path)
                else:
                    try:
                        os.utime(local_path, None)
                    except OSError:
                        pass
                    _evict_local_shard_cache(
                        cache_dir=cache_dir,
                        max_shards=max_local_shards,
                        exclude_paths=(local_path,),
                    )
        except Exception as exn:
            exn.args = exn.args + (remote_url,)
            if handler(exn):
                continue
            raise


def _local_tarfile_to_samples(
    src,
    *,
    token: str,
    cache_dir: Path,
    max_local_shards: int,
    delete_after_use: bool,
    download_retries: int,
    download_timeout_s: int,
    handler: Callable[[Exception], bool],
):
    streams = _local_shard_opener(
        src,
        token=token,
        cache_dir=cache_dir,
        max_local_shards=max_local_shards,
        delete_after_use=delete_after_use,
        download_retries=download_retries,
        download_timeout_s=download_timeout_s,
        handler=handler,
    )
    files = tariterators.tar_file_expander(streams, handler=handler)
    return tariterators.group_by_keys(files, handler=handler)

def create_dataset_pipeline(
    sequence_length: int = 1,
    frame_stride: int = 1,
    sequence_stride: int = 1,
    modalities: Sequence[str] = DEFAULT_MODALITIES,
    transform: Optional[Callable[[Dict[str, object]], Dict[str, object]]] = None,
    convert_to_float: bool = True,
    clip_len_range: Optional[Tuple[int, int]] = None,
    decode_full_frame: bool = False,
    local_tile_span: int = 4,
    exr_tile_size: int = 256,
    source_hw: Optional[Sequence[int]] = None,
    emit_tile_metadata: bool = True,
    decode_global_context: bool = False,
    global_context_long_side: int = 0,
    global_context_modalities: Sequence[str] = ("FG", "Alpha"),
    split: str = "all",
    validation_shard_indices: Optional[Sequence[int]] = None,
    webdataset_repo_id: str = DEFAULT_WEB_DATASET_REPO,
    web_shard_cache_dir: Optional[str] = None,
    web_shard_cache_max_shards: int = 3,
    web_shard_delete_after_use: bool = False,
    web_shard_download_retries: int = 6,
    web_shard_download_timeout_s: int = 120,
    **kwargs # ignore other kwargs for compatibility
):
    token = huggingface_hub.get_token()
    if not token:
        raise RuntimeError("No huggingface token found. Please run huggingface-cli login.")
    
    api = huggingface_hub.HfApi()
    repo_id = str(webdataset_repo_id or DEFAULT_WEB_DATASET_REPO).strip()
    info = api.dataset_info(repo_id)
    cache_dir = _resolve_web_shard_cache_dir(web_shard_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    val_indices = set(validation_shard_indices) if validation_shard_indices else set()
    split_key = str(split).strip().lower()
    if split_key in {"val", "dev", "eval"}:
        split_key = "validation"
    
    urls = []
    for f in info.siblings:
        if f.rfilename.endswith(".tar"):
            match = SHARD_INDEX_RE.search(f.rfilename.replace(".tar", ""))
            shard_idx = int(match.group(1)) if match else None
            
            is_val = shard_idx is not None and shard_idx in val_indices
            if split_key == "validation" and not is_val:
                continue
            if split_key == "train" and is_val:
                continue
                
            urls.append(
                huggingface_hub.hf_hub_url(
                    repo_id=repo_id,
                    filename=f.rfilename,
                    repo_type="dataset",
                )
            )
            
    if not urls:
        raise RuntimeError(f"No URLs found for split {split_key}")
        
    if split_key == "train":
        # Infinite random sampling of shards prevents DDP deadlock and early epoch truncation
        pipeline = [wds.ResampledShards(urls)]
    else:
        pipeline = [
            wds.SimpleShardList(urls),
            wds.split_by_node,
            wds.split_by_worker
        ]
    
    # Materialise remote shards locally before WebDataset expands them.
    pipeline.append(
        lambda src: _local_tarfile_to_samples(
            src,
            token=token,
            cache_dir=cache_dir,
            max_local_shards=max(1, int(web_shard_cache_max_shards)),
            delete_after_use=bool(web_shard_delete_after_use),
            download_retries=max(1, int(web_shard_download_retries)),
            download_timeout_s=max(30, int(web_shard_download_timeout_s)),
            handler=wds.warn_and_continue,
        )
    )
    
    if split_key == "train":
        # Buffer a small number of full clips to mix them up without OOMing.
        # A single clip can be hundreds of MBs in compressed EXR bytes.
        pipeline.append(wds.shuffle(5))
        
    def window_generator(src):
        for clip in src:
            yield from clip_to_windows(
                clip,
                sequence_length=sequence_length,
                frame_stride=frame_stride,
                sequence_stride=sequence_stride,
                modalities=modalities,
                convert_to_float=convert_to_float,
                clip_len_range=clip_len_range,
                decode_full_frame=decode_full_frame,
                exr_tile_size=exr_tile_size,
                local_tile_span=local_tile_span,
                source_hw=source_hw,
                emit_tile_metadata=emit_tile_metadata,
                transform=transform,
                global_context_long_side=global_context_long_side,
                global_context_modalities=global_context_modalities,
                decode_global_context=decode_global_context
            )
            
    pipeline.append(window_generator)
    
    if split_key == "train":
        # Buffer a small number of decoded windows. Decoded windows are HUGE 
        # (potentially 1GB+ each for 16 frames of 1024x1024 across 4 modalities + global).
        # We rely on PyTorch DataLoader round-robining across the 24 workers to mix clips.
        pipeline.append(wds.shuffle(4))
        
    return wds.DataPipeline(*pipeline).with_epoch(10000)


# Provide compatibility classes/functions for training.py
class CorridorKeyWebSequenceDataset:
    def __init__(self, **kwargs):
        self.dataset = create_dataset_pipeline(**kwargs)
    def __iter__(self):
        return iter(self.dataset)

class CorridorKeySequenceDataset(CorridorKeyWebSequenceDataset):
    pass

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
    per_worker = max(1, int(num_torch_threads))
    exr_n = max(0, int(exr_internal_threads))

    def _init(worker_id: int) -> None:
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

class WebDatasetWrapper(torch.utils.data.IterableDataset):
    def __init__(self, dataset, length):
        super().__init__()
        self.dataset = dataset
        self.length = length
    def __iter__(self):
        return iter(self.dataset)
    def __len__(self):
        return self.length

def create_ddp_dataloader(
    dataset: Any,
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

    generator = torch.Generator()
    rank, _ = resolve_ddp_rank_world_size(rank=rank, world_size=world_size)
    generator.manual_seed(seed + rank)

    actual_dataset = getattr(dataset, "dataset", dataset)
    actual_dataset = WebDatasetWrapper(actual_dataset, length=10000)

    loader_kwargs = {
        "dataset": actual_dataset,
        "batch_size": batch_size,
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
    # WebDataset handles epoch seeding via wds.split_by_worker etc., or we 
    # could set epoch explicitly if needed. For now, doing nothing.
    pass
