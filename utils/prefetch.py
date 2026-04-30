from __future__ import annotations
import queue
import threading
from typing import Any, Callable, Dict, Iterator, Optional
import torch
from torch.utils.data import DataLoader

_SENTINEL = object()

def move_batch_to_device(batch: Dict[str, Any], device: torch.device, non_blocking: bool = True) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    seen: Dict[int, torch.Tensor] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            key = id(v)
            cached = seen.get(key)
            if cached is None:
                cached = v.to(device=device, non_blocking=non_blocking)
                seen[key] = cached
            out[k] = cached
        else:
            out[k] = v
    return out

def _record_batch_stream(batch: Any, stream: torch.cuda.Stream, seen: Optional[set[int]] = None) -> None:
    if seen is None: seen = set()
    if torch.is_tensor(batch):
        ident = id(batch)
        if ident not in seen:
            batch.record_stream(stream)
            seen.add(ident)
    elif isinstance(batch, dict):
        for value in batch.values(): _record_batch_stream(value, stream, seen)
    elif isinstance(batch, (list, tuple)):
        for value in batch: _record_batch_stream(value, stream, seen)

class AsyncDevicePrefetcher:
    def __init__(self, dataloader: DataLoader, device: torch.device, queue_size: int = 2, max_batches: Optional[int] = None, move_fn: Optional[Callable] = None, post_move_fn: Optional[Callable] = None) -> None:
        self._dataloader = dataloader
        self._device = device
        self._queue_size = queue_size
        self._max_batches = max_batches
        self._move_fn = move_fn or move_batch_to_device
        self._post_move_fn = post_move_fn
        self._use_cuda = device.type == "cuda"
        self._stream = torch.cuda.Stream(device=device) if self._use_cuda else None
        self._queue = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = None

    def _put_until_stopped(self, item: Any) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            count = 0
            for host_batch in self._dataloader:
                if self._stop.is_set(): return
                if self._max_batches is not None and count >= self._max_batches: break
                
                if self._use_cuda and self._stream is not None:
                    with torch.cuda.stream(self._stream):
                        with torch.no_grad():
                            gpu_batch = self._move_fn(host_batch, self._device, True)
                            if self._post_move_fn is not None: gpu_batch = self._post_move_fn(gpu_batch)
                    event = torch.cuda.Event()
                    event.record(self._stream)
                    payload = (gpu_batch, event)
                else:
                    with torch.no_grad():
                        gpu_batch = self._move_fn(host_batch, self._device, False)
                        if self._post_move_fn is not None: gpu_batch = self._post_move_fn(gpu_batch)
                    payload = (gpu_batch, None)
                
                while not self._stop.is_set():
                    try:
                        self._queue.put(payload, timeout=0.1)
                        break
                    except queue.Full: continue
                count += 1
        except Exception as exc:
            self._put_until_stopped(exc)
        finally:
            self._put_until_stopped(_SENTINEL)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self._stop.clear()
        self._thread = threading.Thread(target=self._produce, daemon=True)
        self._thread.start()
        try:
            while True:
                try:
                    item = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._thread is not None and not self._thread.is_alive():
                        break
                    continue
                if item is _SENTINEL: break
                if isinstance(item, Exception): raise item
                gpu_batch, event = item
                if event is not None:
                    current_stream = torch.cuda.current_stream(self._device)
                    event.wait(current_stream)
                    _record_batch_stream(gpu_batch, current_stream)
                yield gpu_batch
        finally:
            self.close()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            try:
                while True: self._queue.get_nowait()
            except queue.Empty: pass
            self._thread.join(timeout=0.1)
        self._thread = None
