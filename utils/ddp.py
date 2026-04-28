from __future__ import annotations
import os
import torch
import torch.distributed as dist
from typing import Any, Optional
from dataclasses import dataclass

@dataclass
class DDPInfo:
    rank: int
    world_size: int
    device: torch.device
    is_distributed: bool
    timeout: Optional[Any] = None

def is_rank0() -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0

def get_rank() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()

def init_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    timeout_minutes: Optional[int] = 30,
) -> DDPInfo:
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if rank is None:
        rank = int(os.environ.get("RANK", "0"))
    
    device_idx = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    device = torch.device(f"cuda:{device_idx}" if torch.cuda.is_available() else "cpu")

    from datetime import timedelta
    timeout_obj = timedelta(minutes=timeout_minutes) if timeout_minutes is not None else None
    if world_size > 1:
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend, 
                init_method=init_method, 
                world_size=world_size, 
                rank=rank,
                timeout=timeout_obj
            )
            if backend == "nccl":
                torch.cuda.set_device(device_idx)
    
    return DDPInfo(rank=get_rank(), world_size=get_world_size(), device=device, is_distributed=(get_world_size() > 1), timeout=timeout_obj)

def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def reduce_scalar(val: Any, average: bool = True) -> torch.Tensor:
    if not torch.is_tensor(val):
        val = torch.tensor(val)
    t = val.detach().clone().to(device="cuda" if torch.cuda.is_available() else "cpu")
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return t
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    if average:
        t = t / dist.get_world_size()
    return t
