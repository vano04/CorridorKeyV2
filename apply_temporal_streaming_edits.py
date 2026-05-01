from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path.cwd()


def read(rel: str) -> str:
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(f"Missing {rel}; run this from repo root")
    return p.read_text()


def write(rel: str, text: str) -> None:
    (ROOT / rel).write_text(text)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f"{label}: expected 1 match, found {n}")
    return text.replace(old, new, 1)


def insert_before(text: str, marker: str, block: str, label: str) -> str:
    if block.strip() in text:
        return text
    return replace_once(text, marker, block + marker, label)


def patch_dataset() -> None:
    rel = "CorridorKeyDataset/dataset.py"
    s = read(rel)
    if "class TemporalChunkSpec" in s and "class TemporalChunkBatchSampler" in s:
        print(f"{rel}: streaming classes already present, skipping most dataset edits")
        return

    s = replace_once(
        s,
        "from typing import Any, BinaryIO, Callable, Dict, List, Optional, Sequence, Tuple\n",
        "from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Sequence, Tuple\n",
        "dataset typing import",
    )
    s = replace_once(
        s,
        "from torch.utils.data import DataLoader, Dataset\n",
        "from torch.utils.data import DataLoader, Dataset, Sampler\n",
        "dataset torch dataloader import",
    )

    temporal_spec = '''\n@dataclass(frozen=True, slots=True)\nclass TemporalChunkSpec:\n    """Index payload used by TemporalChunkBatchSampler."""\n\n    sample_index: int\n    frame_numbers: Tuple[int, ...]\n    chunk_start: int\n    chunk_end: int\n    chunk_index: int\n    num_chunks: int\n    logical_batch_index: int\n    tile_grid: Optional[Tuple[int, int, int, int]]\n    tile_coords: Tuple[float, float, float, float]\n    source_hw: Tuple[float, float]\n    tile_grid_tensor: Tuple[int, int, int, int]\n\n\n'''
    s = insert_before(s, "ClipIndex = WebClipIndex\n", temporal_spec, "insert TemporalChunkSpec")

    s = replace_once(
        s,
        "    local_tile_span: int,\n) -> Tuple[Optional[Tuple[int, int, int, int]], Tensor, Tensor, Tensor]:\n    source_h, source_w = source_hw\n",
        "    local_tile_span: int,\n    rng: Optional[random.Random] = None,\n) -> Tuple[Optional[Tuple[int, int, int, int]], Tensor, Tensor, Tensor]:\n    r = rng if rng is not None else random\n    source_h, source_w = source_hw\n",
        "dataset _sample_tile_selection signature",
    )
    s = replace_once(s, "    tx0 = random.randint(0, max(0, tiles_x - span_x))\n", "    tx0 = r.randint(0, max(0, tiles_x - span_x))\n", "dataset tx0 rng")
    s = replace_once(s, "    ty0 = random.randint(0, max(0, tiles_y - span_y))\n", "    ty0 = r.randint(0, max(0, tiles_y - span_y))\n", "dataset ty0 rng")

    s = replace_once(
        s,
        "    def _sample_frame_numbers_for_index(self, index: int) -> Tuple[WebClipIndex, List[int]]:\n        clip_idx, start = self.samples[index]\n",
        "    def _sample_frame_numbers_for_index(\n        self,\n        index: int,\n        rng: Optional[random.Random] = None,\n    ) -> Tuple[WebClipIndex, List[int]]:\n        r = rng if rng is not None else random\n        clip_idx, start = self.samples[index]\n",
        "dataset frame sampler signature",
    )
    s = replace_once(s, "            clip_len = random.randint(max(1, min(lo, n)), max(1, min(hi, n)))\n", "            clip_len = r.randint(max(1, min(lo, n)), max(1, min(hi, n)))\n", "dataset clip_len rng")
    s = replace_once(s, "            sub_start = random.randint(0, max(0, n - clip_len))\n", "            sub_start = r.randint(0, max(0, n - clip_len))\n", "dataset sub_start rng")

    s = replace_once(
        s,
        "    def _getitem_cached_four_quadrants(self, index: int) -> Dict[str, object]:\n        clip, frame_numbers = self._sample_frame_numbers_for_index(index)\n",
        "    def _getitem_cached_four_quadrants(\n        self,\n        index: int,\n        frame_numbers: Optional[Sequence[int]] = None,\n    ) -> Dict[str, object]:\n        clip, sampled_frame_numbers = self._sample_frame_numbers_for_index(index)\n        if frame_numbers is None:\n            frame_numbers = sampled_frame_numbers\n",
        "dataset cached quadrants signature",
    )

    stream_methods = '''    def _temporal_stream_metadata(self, spec: TemporalChunkSpec) -> Dict[str, object]:\n        return {\n            "temporal_stream_chunk_index": int(spec.chunk_index),\n            "temporal_stream_num_chunks": int(spec.num_chunks),\n            "temporal_stream_full_frames": int(len(spec.frame_numbers)),\n            "temporal_stream_chunk_start": int(spec.chunk_start),\n            "temporal_stream_chunk_end": int(spec.chunk_end),\n            "temporal_stream_logical_batch": int(spec.logical_batch_index),\n        }\n\n    def _attach_temporal_stream_metadata(\n        self,\n        sample: Dict[str, object],\n        spec: TemporalChunkSpec,\n    ) -> Dict[str, object]:\n        metadata = self._temporal_stream_metadata(spec)\n        prebatched = sample.get("_prebatched_samples")\n        if isinstance(prebatched, list):\n            for item in prebatched:\n                if isinstance(item, dict):\n                    item.update(metadata)\n            return sample\n        sample.update(metadata)\n        return sample\n\n    def _getitem_temporal_chunk(self, spec: TemporalChunkSpec) -> Dict[str, object]:\n        clip_idx, _ = self.samples[int(spec.sample_index)]\n        clip = self.clips[clip_idx]\n        frame_numbers = list(spec.frame_numbers[spec.chunk_start : spec.chunk_end])\n        if not frame_numbers:\n            raise IndexError(f"Empty temporal chunk spec: {spec!r}")\n\n        if self.cached_four_quadrant_batch:\n            sample = self._getitem_cached_four_quadrants(int(spec.sample_index), frame_numbers=frame_numbers)\n            return self._attach_temporal_stream_metadata(sample, spec)\n\n        tile_grid = spec.tile_grid\n        tile_coords = torch.tensor(spec.tile_coords, dtype=torch.float32)\n        source_hw = torch.tensor(spec.source_hw, dtype=torch.float32)\n        tile_grid_t = torch.tensor(spec.tile_grid_tensor, dtype=torch.long)\n        sample: Dict[str, object] = {\n            "clip_name": clip.name,\n            "frame_numbers": torch.tensor(frame_numbers, dtype=torch.long),\n        }\n        if self.emit_tile_metadata:\n            sample["tile_coords"] = tile_coords\n            sample["source_hw"] = source_hw\n            sample["tile_grid"] = tile_grid_t\n        sample.update(self._load_all_modalities(clip, frame_numbers, tile_grid))\n        sample.update(self._load_global_modalities(clip, frame_numbers, tile_grid))\n        if self.transform is not None:\n            sample = self.transform(sample)\n        return self._attach_temporal_stream_metadata(sample, spec)\n\n'''
    s = insert_before(s, "    def __getitem__(self, index: int) -> Dict[str, object]:\n", stream_methods, "insert dataset stream methods")
    s = replace_once(
        s,
        "    def __getitem__(self, index: int) -> Dict[str, object]:\n        if self.cached_four_quadrant_batch:\n",
        "    def __getitem__(self, index: int | TemporalChunkSpec) -> Dict[str, object]:\n        if isinstance(index, TemporalChunkSpec):\n            return self._getitem_temporal_chunk(index)\n        if self.cached_four_quadrant_batch:\n",
        "dataset __getitem__ stream dispatch",
    )

    sampler_class = '''\nclass TemporalChunkBatchSampler(Sampler[List[TemporalChunkSpec]]):\n    """Yield consecutive temporal chunk batches for each logical batch."""\n\n    def __init__(\n        self,\n        dataset: CorridorKeyWebSequenceDataset,\n        batch_size: int,\n        chunk_size: int,\n        *,\n        shuffle: bool = True,\n        drop_last: bool = False,\n        seed: int = 1337,\n    ) -> None:\n        if batch_size < 1:\n            raise ValueError("batch_size must be >= 1")\n        if chunk_size < 1:\n            raise ValueError("temporal stream chunk_size must be >= 1")\n        self.dataset = dataset\n        self.batch_size = int(batch_size)\n        self.chunk_size = int(chunk_size)\n        self.shuffle = bool(shuffle)\n        self.drop_last = bool(drop_last)\n        self.seed = int(seed)\n        self.epoch = 0\n\n    @property\n    def logical_len(self) -> int:\n        n = len(self.dataset)\n        if self.drop_last:\n            return n // self.batch_size\n        return int(math.ceil(n / self.batch_size))\n\n    def __len__(self) -> int:\n        if self.dataset.clip_len_range is not None:\n            _, max_len = self.dataset.clip_len_range\n            full_len = min(int(max_len), int(self.dataset.sequence_length))\n        else:\n            full_len = int(self.dataset.sequence_length)\n        chunks = max(1, int(math.ceil(max(1, full_len) / self.chunk_size)))\n        return self.logical_len * chunks\n\n    def set_epoch(self, epoch: int) -> None:\n        self.epoch = int(epoch)\n\n    def __iter__(self) -> Iterator[List[TemporalChunkSpec]]:\n        rng = random.Random(self.seed + self.epoch)\n        indices = list(range(len(self.dataset)))\n        if self.shuffle:\n            rng.shuffle(indices)\n\n        for logical_batch_idx in range(self.logical_len):\n            start = logical_batch_idx * self.batch_size\n            sample_indices = indices[start : start + self.batch_size]\n            if self.drop_last and len(sample_indices) < self.batch_size:\n                continue\n\n            planned: List[Tuple[int, Tuple[int, ...], Optional[Tuple[int, int, int, int]], Tuple[float, float, float, float], Tuple[float, float], Tuple[int, int, int, int]]] = []\n            max_frames = 0\n            for sample_index in sample_indices:\n                _, frame_numbers_l = self.dataset._sample_frame_numbers_for_index(sample_index, rng=rng)\n                frame_numbers = tuple(int(v) for v in frame_numbers_l)\n                tile_grid, tile_coords, source_hw, tile_grid_t = _sample_tile_selection(\n                    decode_full_frame=self.dataset.decode_full_frame,\n                    source_hw=self.dataset.source_hw,\n                    tile_size=self.dataset.exr_tile_size,\n                    local_tile_span=self.dataset.local_tile_span,\n                    rng=rng,\n                )\n                planned.append((\n                    int(sample_index),\n                    frame_numbers,\n                    tile_grid,\n                    tuple(float(v) for v in tile_coords.tolist()),\n                    tuple(float(v) for v in source_hw.tolist()),\n                    tuple(int(v) for v in tile_grid_t.tolist()),\n                ))\n                max_frames = max(max_frames, len(frame_numbers))\n\n            num_chunks = max(1, int(math.ceil(max_frames / self.chunk_size)))\n            for chunk_idx in range(num_chunks):\n                chunk_start = chunk_idx * self.chunk_size\n                chunk_end = min(max_frames, chunk_start + self.chunk_size)\n                chunk_specs: List[TemporalChunkSpec] = []\n                for sample_index, frame_numbers, tile_grid, tile_coords, source_hw, tile_grid_t in planned:\n                    if chunk_start >= len(frame_numbers):\n                        continue\n                    chunk_specs.append(\n                        TemporalChunkSpec(\n                            sample_index=sample_index,\n                            frame_numbers=frame_numbers,\n                            chunk_start=chunk_start,\n                            chunk_end=min(len(frame_numbers), chunk_end),\n                            chunk_index=chunk_idx,\n                            num_chunks=num_chunks,\n                            logical_batch_index=logical_batch_idx,\n                            tile_grid=tile_grid,\n                            tile_coords=tile_coords,\n                            source_hw=source_hw,\n                            tile_grid_tensor=tile_grid_t,\n                        )\n                    )\n                if chunk_specs:\n                    yield chunk_specs\n\n\n'''
    s = insert_before(s, "seed_worker = _make_worker_init(1)\n", sampler_class, "insert TemporalChunkBatchSampler")

    old_loader = '''def create_single_gpu_dataloader(\n    dataset: Dataset,\n    batch_size: int,\n    num_workers: int = 4,\n    shuffle: bool = True,\n    drop_last: bool = False,\n    pin_memory: bool = True,\n    persistent_workers: bool = True,\n    prefetch_factor: int = 2,\n    seed: int = 1337,\n    collate_fn: Optional[Callable] = None,\n    num_torch_threads: int = 1,\n    exr_internal_threads: int = 0,\n    **_unused: object,\n) -> DataLoader:\n    if batch_size < 1:\n        raise ValueError("batch_size must be >= 1")\n    generator = torch.Generator()\n    generator.manual_seed(int(seed))\n    loader_kwargs: Dict[str, Any] = {\n        "dataset": dataset,\n        "batch_size": int(batch_size),\n        "shuffle": bool(shuffle),\n        "num_workers": int(num_workers),\n        "pin_memory": bool(pin_memory),\n        "drop_last": bool(drop_last),\n        "worker_init_fn": _make_worker_init(num_torch_threads, exr_internal_threads=exr_internal_threads),\n        "generator": generator,\n        "collate_fn": collate_fn,\n    }\n    if int(num_workers) > 0:\n        loader_kwargs["persistent_workers"] = bool(persistent_workers)\n        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))\n    return DataLoader(**loader_kwargs)\n\n\ndef set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:\n    del dataloader, epoch\n'''
    new_loader = '''def create_single_gpu_dataloader(\n    dataset: Dataset,\n    batch_size: int,\n    num_workers: int = 4,\n    shuffle: bool = True,\n    drop_last: bool = False,\n    pin_memory: bool = True,\n    persistent_workers: bool = True,\n    prefetch_factor: int = 2,\n    seed: int = 1337,\n    collate_fn: Optional[Callable] = None,\n    num_torch_threads: int = 1,\n    exr_internal_threads: int = 0,\n    temporal_stream_chunk_size: int = 0,\n    **_unused: object,\n) -> DataLoader:\n    if batch_size < 1:\n        raise ValueError("batch_size must be >= 1")\n    stream_chunk_size = max(0, int(temporal_stream_chunk_size))\n    loader_kwargs: Dict[str, Any] = {\n        "dataset": dataset,\n        "num_workers": int(num_workers),\n        "pin_memory": bool(pin_memory),\n        "worker_init_fn": _make_worker_init(num_torch_threads, exr_internal_threads=exr_internal_threads),\n        "collate_fn": collate_fn,\n    }\n    if stream_chunk_size > 0:\n        if not isinstance(dataset, CorridorKeyWebSequenceDataset):\n            raise TypeError("temporal streaming requires CorridorKeyWebSequenceDataset")\n        batch_sampler = TemporalChunkBatchSampler(\n            dataset=dataset,\n            batch_size=int(batch_size),\n            chunk_size=stream_chunk_size,\n            shuffle=bool(shuffle),\n            drop_last=bool(drop_last),\n            seed=int(seed),\n        )\n        loader_kwargs["batch_sampler"] = batch_sampler\n    else:\n        generator = torch.Generator()\n        generator.manual_seed(int(seed))\n        loader_kwargs.update({\n            "batch_size": int(batch_size),\n            "shuffle": bool(shuffle),\n            "drop_last": bool(drop_last),\n            "generator": generator,\n        })\n    if int(num_workers) > 0:\n        loader_kwargs["persistent_workers"] = bool(persistent_workers)\n        loader_kwargs["prefetch_factor"] = max(1, int(prefetch_factor))\n    loader = DataLoader(**loader_kwargs)\n    if stream_chunk_size > 0:\n        setattr(loader, "_corridorkey_temporal_stream", True)\n        setattr(loader, "_corridorkey_logical_len", batch_sampler.logical_len)\n    return loader\n\n\ndef set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:\n    batch_sampler = getattr(dataloader, "batch_sampler", None)\n    if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):\n        batch_sampler.set_epoch(epoch)\n'''
    s = replace_once(s, old_loader, new_loader, "dataset dataloader factory")
    write(rel, s)
    print(f"{rel}: patched")


def patch_utils() -> None:
    rel = "utils/data.py"
    s = read(rel)
    if "disable_temporal_augment" in s and "temporal_stream_chunk_index" in s:
        print(f"{rel}: already patched")
        return
    s = replace_once(
        s,
        "    temporal_jitter_p: float = 0.1\n    skip_temporal_sample: bool = False\n",
        "    temporal_jitter_p: float = 0.1\n    skip_temporal_sample: bool = False\n    disable_temporal_augment: bool = False\n",
        "utils add disable_temporal_augment field",
    )
    s = replace_once(
        s,
        "        if self.skip_temporal_sample:\n            temporal_tensors = self._temporal_augment(temporal_tensors)\n        else:\n            temporal_tensors = self._temporal_sample(temporal_tensors)\n",
        "        if self.disable_temporal_augment:\n            temporal_tensors = list(temporal_tensors)\n        elif self.skip_temporal_sample:\n            temporal_tensors = self._temporal_augment(temporal_tensors)\n        else:\n            temporal_tensors = self._temporal_sample(temporal_tensors)\n",
        "utils temporal augment branch",
    )
    s = replace_once(
        s,
        "        for meta_key in (\"tile_coords\", \"source_hw\", \"tile_grid\"):\n            meta = item.get(meta_key)\n            if isinstance(meta, Tensor):\n                out[meta_key] = meta.unsqueeze(0)\n        return out\n",
        "        for meta_key in (\"tile_coords\", \"source_hw\", \"tile_grid\"):\n            meta = item.get(meta_key)\n            if isinstance(meta, Tensor):\n                out[meta_key] = meta.unsqueeze(0)\n        for meta_key in (\"temporal_stream_chunk_index\", \"temporal_stream_num_chunks\", \"temporal_stream_full_frames\", \"temporal_stream_chunk_start\", \"temporal_stream_chunk_end\", \"temporal_stream_logical_batch\"):\n            meta = item.get(meta_key)\n            if meta is not None:\n                out[meta_key] = torch.tensor([int(meta)], dtype=torch.long)\n        return out\n",
        "utils single-item temporal metadata",
    )
    s = replace_once(
        s,
        "    for meta_key in (\"tile_coords\", \"source_hw\", \"tile_grid\"):\n        metas = [item.get(meta_key) for item in batch]\n        if all(isinstance(m, Tensor) for m in metas):\n            out[meta_key] = torch.stack([m for m in metas if isinstance(m, Tensor)], dim=0)\n    return out\n",
        "    for meta_key in (\"tile_coords\", \"source_hw\", \"tile_grid\"):\n        metas = [item.get(meta_key) for item in batch]\n        if all(isinstance(m, Tensor) for m in metas):\n            out[meta_key] = torch.stack([m for m in metas if isinstance(m, Tensor)], dim=0)\n\n    for meta_key in (\"temporal_stream_chunk_index\", \"temporal_stream_num_chunks\", \"temporal_stream_full_frames\", \"temporal_stream_chunk_start\", \"temporal_stream_chunk_end\", \"temporal_stream_logical_batch\"):\n        metas = [item.get(meta_key) for item in batch]\n        if all(m is not None for m in metas):\n            out[meta_key] = torch.tensor([int(m) for m in metas], dtype=torch.long, device=device)\n    return out\n",
        "utils batch temporal metadata",
    )
    write(rel, s)
    print(f"{rel}: patched")


def patch_training() -> None:
    rel = "training.py"
    s = read(rel)
    if "stream_temporal_batches" in s and "stream_coarse_seed" in s:
        print(f"{rel}: already patched")
        return

    s = replace_once(
        s,
        "    clip_len_min = int(data_cfg.get(\"clip_len_min\", 4))\n    clip_len_max = int(data_cfg.get(\"clip_len_max\", 12))\n",
        "    clip_len_min = int(data_cfg.get(\"clip_len_min\", 4))\n    clip_len_max = int(data_cfg.get(\"clip_len_max\", 12))\n    stream_temporal_batches = bool(data_cfg.get(\"stream_temporal_batches\", False))\n    temporal_stream_chunk_size = int(\n        data_cfg.get(\n            \"temporal_stream_chunk_size\",\n            config.get(\"train\", {}).get(\"temporal_batch_size\", clip_len_max),\n        )\n    )\n",
        "training add stream config",
    )
    s = replace_once(
        s,
        "    device_offload = bool(data_cfg.get(\"device_augment\", False))\n\n    # Optional dtype downcast",
        "    device_offload = bool(data_cfg.get(\"device_augment\", False))\n    if stream_temporal_batches:\n        if temporal_stream_chunk_size < 1:\n            raise ValueError(\"data.temporal_stream_chunk_size must be >= 1 when streaming is enabled\")\n        if not device_offload:\n            raise ValueError(\"data.stream_temporal_batches requires data.device_augment=true\")\n        if clip_len_min != clip_len_max:\n            raise ValueError(\"data.stream_temporal_batches currently requires clip_len_min == clip_len_max\")\n        if float(data_cfg.get(\"temporal_jitter_p\", 0.1)) != 0.0:\n            raise ValueError(\"data.stream_temporal_batches currently requires temporal_jitter_p: 0.0\")\n\n    # Optional dtype downcast",
        "training stream config validation",
    )
    s = replace_once(
        s,
        "        temporal_jitter_p=float(data_cfg.get(\"temporal_jitter_p\", 0.1)),\n        skip_temporal_sample=True,\n        device_offload=device_offload,\n",
        "        temporal_jitter_p=float(data_cfg.get(\"temporal_jitter_p\", 0.1)),\n        skip_temporal_sample=True,\n        disable_temporal_augment=stream_temporal_batches,\n        device_offload=device_offload,\n",
        "training pass disable_temporal_augment",
    )
    s = replace_once(
        s,
        "        num_torch_threads=int(data_cfg.get(\"num_torch_threads\", 1)),\n        exr_internal_threads=int(data_cfg.get(\"exr_internal_threads\", 0)),\n    )\n\n    setattr(loader, \"_corridorkey_set_dataloader_epoch\", dataset_runtime.set_dataloader_epoch)\n",
        "        num_torch_threads=int(data_cfg.get(\"num_torch_threads\", 1)),\n        exr_internal_threads=int(data_cfg.get(\"exr_internal_threads\", 0)),\n        temporal_stream_chunk_size=temporal_stream_chunk_size if stream_temporal_batches else 0,\n    )\n\n    setattr(loader, \"_corridorkey_set_dataloader_epoch\", dataset_runtime.set_dataloader_epoch)\n    if stream_temporal_batches:\n        setattr(loader, \"_corridorkey_temporal_stream\", True)\n        logical_len = getattr(getattr(loader, \"batch_sampler\", None), \"logical_len\", len(loader))\n        setattr(loader, \"_corridorkey_logical_len\", int(logical_len))\n",
        "training dataloader stream args",
    )
    s = replace_once(
        s,
        "    epoch_steps = math.ceil(len(dataloader) / max(1, grad_accum_steps))\n    epochs_total_steps = int(train_cfg.get(\"epochs\", 40)) * epoch_steps\n",
        "    dataloader_iter_len = len(dataloader)\n    dataloader_logical_len = int(getattr(dataloader, \"_corridorkey_logical_len\", dataloader_iter_len))\n    epoch_steps = math.ceil(dataloader_logical_len / max(1, grad_accum_steps))\n    epochs_total_steps = int(train_cfg.get(\"epochs\", 40)) * epoch_steps\n",
        "training step accounting vars",
    )
    s = replace_once(
        s,
        "        f\"Step accounting: dataloader_iters_per_epoch={len(dataloader)}, \"\n        f\"grad_accum_steps={grad_accum_steps}, optimizer_steps_per_epoch={epoch_steps}\",\n",
        "        f\"Step accounting: dataloader_iters_per_epoch={dataloader_iter_len}, \"\n        f\"logical_batches_per_epoch={dataloader_logical_len}, \"\n        f\"grad_accum_steps={grad_accum_steps}, optimizer_steps_per_epoch={epoch_steps}\",\n",
        "training step accounting print",
    )
    s = replace_once(
        s,
        "            pbar = tqdm(\n                total=len(dataloader),\n",
        "            pbar = tqdm(\n                total=dataloader_iter_len,\n",
        "training pbar total",
    )
    s = replace_once(
        s,
        "            prefetch_device_transform = device_transform if cuda_prefetch else None\n\n            for it, batch in enumerate(\n",
        "            prefetch_device_transform = device_transform if cuda_prefetch else None\n            stream_coarse_seed: torch.Tensor | None = None\n            stream_current_logical_batch: int | None = None\n            stream_total_loss: torch.Tensor | None = None\n            stream_loss_items: Dict[str, torch.Tensor] = {}\n\n            for it, batch in enumerate(\n",
        "training stream state vars",
    )

    old_temporal = '''                temporal_chunks = build_temporal_chunks(\n                    total_frames=int(batch["video_rgb"].shape[1]),\n                    chunk_size=temporal_batch_size,\n                )\n                real_n_chunks = len(temporal_chunks)\n                while len(temporal_chunks) < fixed_n_chunks:\n                    temporal_chunks.append(temporal_chunks[-1])\n\n                valid_mask = batch.get("valid_mask")\n                if valid_mask is not None:\n                    total_weight_denom = valid_mask.sum().clamp_min(1.0)\n                else:\n                    total_weight_denom = batch["video_rgb"].new_tensor(float(batch["video_rgb"].shape[1]))\n\n                coarse_seed = batch["coarse_alpha_init"]\n                did_backward = False\n                batch_total_loss = batch["video_rgb"].new_tensor(0.0)\n                batch_loss_items: Dict[str, torch.Tensor] = {}\n                do_step = ((it + 1) % grad_accum_steps == 0) or (it + 1 == len(dataloader))\n'''
    new_temporal = '''                is_stream_batch = "temporal_stream_chunk_index" in batch\n                logical_it = it\n                stream_is_final_chunk = True\n                if is_stream_batch:\n                    logical_it = int(batch["temporal_stream_logical_batch"][0].detach().to(device="cpu"))\n                    stream_chunk_idx = int(batch["temporal_stream_chunk_index"][0].detach().to(device="cpu"))\n                    stream_num_chunks = int(batch["temporal_stream_num_chunks"][0].detach().to(device="cpu"))\n                    stream_is_final_chunk = stream_chunk_idx + 1 >= stream_num_chunks\n                    if stream_current_logical_batch != logical_it or stream_chunk_idx == 0:\n                        stream_current_logical_batch = logical_it\n                        stream_coarse_seed = batch["coarse_alpha_init"]\n                        stream_total_loss = batch["video_rgb"].new_tensor(0.0)\n                        stream_loss_items = {}\n                    elif stream_coarse_seed is not None:\n                        batch["coarse_alpha_init"] = stream_coarse_seed\n                    temporal_chunks = [(0, int(batch["video_rgb"].shape[1]))]\n                    real_n_chunks = 1\n                else:\n                    temporal_chunks = build_temporal_chunks(\n                        total_frames=int(batch["video_rgb"].shape[1]),\n                        chunk_size=temporal_batch_size,\n                    )\n                    real_n_chunks = len(temporal_chunks)\n                    while len(temporal_chunks) < fixed_n_chunks:\n                        temporal_chunks.append(temporal_chunks[-1])\n\n                valid_mask = batch.get("valid_mask")\n                if valid_mask is not None:\n                    if is_stream_batch:\n                        total_weight_denom = batch["temporal_stream_full_frames"].sum().clamp_min(1).to(\n                            dtype=batch["video_rgb"].dtype\n                        )\n                    else:\n                        total_weight_denom = valid_mask.sum().clamp_min(1.0)\n                elif is_stream_batch:\n                    total_weight_denom = batch["temporal_stream_full_frames"].sum().clamp_min(1).to(\n                        dtype=batch["video_rgb"].dtype\n                    )\n                else:\n                    total_weight_denom = batch["video_rgb"].new_tensor(float(batch["video_rgb"].shape[1]))\n\n                coarse_seed = batch["coarse_alpha_init"]\n                did_backward = False\n                batch_total_loss = stream_total_loss if is_stream_batch and stream_total_loss is not None else batch["video_rgb"].new_tensor(0.0)\n                batch_loss_items = stream_loss_items if is_stream_batch else {}\n                do_step = stream_is_final_chunk and (\n                    ((logical_it + 1) % grad_accum_steps == 0)\n                    or (logical_it + 1 == dataloader_logical_len)\n                )\n'''
    s = replace_once(s, old_temporal, new_temporal, "training temporal setup block")

    s = replace_once(
        s,
        "                        else:\n                            chunk_weight = batch[\"video_rgb\"].new_tensor(\n                                (end_t - start_t) / max(1, int(batch[\"video_rgb\"].shape[1]))\n                            )\n",
        "                        else:\n                            if is_stream_batch:\n                                chunk_weight = (\n                                    batch[\"video_rgb\"].new_tensor(float(batch[\"video_rgb\"].shape[0] * (end_t - start_t)))\n                                    / total_weight_denom\n                                ).to(dtype=batch[\"video_rgb\"].dtype)\n                            else:\n                                chunk_weight = batch[\"video_rgb\"].new_tensor(\n                                    (end_t - start_t) / max(1, int(batch[\"video_rgb\"].shape[1]))\n                                )\n",
        "training stream chunk weight",
    )
    s = replace_once(
        s,
        "                        coarse_seed = pred[\"alpha_pred\"][:, -1].detach()\n\n                if do_step and did_backward:\n",
        "                        coarse_seed = pred[\"alpha_pred\"][:, -1].detach()\n\n                if is_stream_batch:\n                    stream_coarse_seed = coarse_seed\n                    stream_total_loss = batch_total_loss.detach()\n                    stream_loss_items = batch_loss_items\n\n                if do_step and did_backward:\n",
        "training update stream state after chunk",
    )
    s = replace_once(
        s,
        "                should_log = (it + 1) % log_interval == 0\n                should_wandb_log = wandb_logger.enabled and (global_step % wandb_log_interval == 0)\n",
        "                should_log = stream_is_final_chunk and ((logical_it + 1) % log_interval == 0)\n                should_wandb_log = (\n                    stream_is_final_chunk\n                    and wandb_logger.enabled\n                    and (global_step % wandb_log_interval == 0)\n                )\n",
        "training log gates",
    )
    s = replace_once(
        s,
        "                running_loss = running_loss + batch_total_loss.detach().to(dtype=torch.float32)\n",
        "                if stream_is_final_chunk:\n                    running_loss = running_loss + batch_total_loss.detach().to(dtype=torch.float32)\n",
        "training running loss final chunks only",
    )
    s = replace_once(s, "f\"epoch={epoch:03d} iter={it + 1:05d}/{len(dataloader):05d} \"", "f\"epoch={epoch:03d} iter={logical_it + 1:05d}/{dataloader_logical_len:05d} \"", "training tqdm log iter")
    s = replace_once(s, "\"train/iter\": int(it + 1),", "\"train/iter\": int(logical_it + 1),", "training wandb iter")
    s = replace_once(
        s,
        "                if max_epoch_batches > 0 and it + 1 >= max_epoch_batches:\n                    tqdm.write(\n                        f\"Reached max_epoch_batches={max_epoch_batches} at epoch iter={it + 1}; \"\n",
        "                if max_epoch_batches > 0 and stream_is_final_chunk and logical_it + 1 >= max_epoch_batches:\n                    tqdm.write(\n                        f\"Reached max_epoch_batches={max_epoch_batches} at epoch iter={logical_it + 1}; \"\n",
        "training max_epoch_batches logical",
    )
    s = replace_once(
        s,
        "            epoch_loss = running_loss / max(1, len(dataloader))\n",
        "            epoch_loss = running_loss / max(1, dataloader_logical_len)\n",
        "training epoch loss logical len",
    )
    write(rel, s)
    print(f"{rel}: patched")


def main() -> None:
    try:
        patch_dataset()
        patch_utils()
        patch_training()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    print("Done. Now run: uv run python -S -m py_compile training.py utils/data.py CorridorKeyDataset/dataset.py")


if __name__ == "__main__":
    main()
