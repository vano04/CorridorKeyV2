"""Transformer Engine FP8 wrappers for V3.

Provides config-controlled wrappers that use NVIDIA Transformer Engine modules
when available and enabled, falling back to standard PyTorch modules otherwise.

FP8 is intended for the single-card RTX 5090 path.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def transformer_engine_available() -> bool:
    """Check if NVIDIA Transformer Engine is installed and importable."""
    try:
        import transformer_engine.pytorch as te  # noqa: F401
        return True
    except Exception:
        return False


def _get_te_module():
    """Import and return transformer_engine.pytorch, or None."""
    try:
        import transformer_engine.pytorch as te
        return te
    except Exception:
        return None


def _autocast_target_dtype(x: Tensor) -> Optional[torch.dtype]:
    """Return active autocast dtype for x's device, if any."""
    if not x.is_floating_point():
        return None
    device_type = x.device.type
    try:
        enabled = torch.is_autocast_enabled(device_type)
        dtype = torch.get_autocast_dtype(device_type)
    except TypeError:
        enabled = torch.is_autocast_enabled()
        if x.is_cuda:
            dtype = torch.get_autocast_gpu_dtype()
        elif device_type == "cpu":
            dtype = torch.get_autocast_cpu_dtype()
        else:
            dtype = None
    if not enabled or dtype not in (torch.float16, torch.bfloat16):
        return None
    return dtype


class MaybeTELinear(nn.Module):
    """Linear layer that uses TE Linear when FP8 is enabled and available."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        use_te: bool = False,
    ) -> None:
        super().__init__()
        self.use_te = False
        te = _get_te_module()
        if use_te and te is not None:
            self.linear = te.Linear(in_features, out_features, bias=bias)
            self.use_te = True
        else:
            if use_te and te is None:
                import warnings
                warnings.warn(
                    "Transformer Engine requested but not available. "
                    "Falling back to standard nn.Linear.",
                    stacklevel=2,
                )
            self.linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        if not self.use_te:
            return self.linear(x)

        target_dtype = _autocast_target_dtype(x)
        if target_dtype is not None and x.dtype == torch.float32:
            x = x.to(target_dtype)

        # FP8 alignment check for Transformer Engine. TE FP8 Linear expects
        # aligned matrix dimensions, so flatten all leading dims and pad rows
        # to a multiple of 16 before slicing the padding away.
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])
        n = x_flat.shape[0]

        align = 16
        if n % align != 0:
            pad = align - (n % align)
            x_padded = F.pad(x_flat, (0, 0, 0, pad))
            out_padded = self.linear(x_padded)
            out = out_padded[:n]
        else:
            out = self.linear(x_flat)

        if len(orig_shape) > 2:
            return out.reshape(*orig_shape[:-1], -1)
        return out



class MaybeTELayerNorm(nn.Module):
    """LayerNorm that uses TE LayerNorm when FP8 is enabled and available."""

    def __init__(
        self,
        normalized_shape: int,
        *,
        use_te: bool = False,
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.use_te = False
        te = _get_te_module()
        if use_te and te is not None:
            self.norm = te.LayerNorm(normalized_shape, eps=eps)
            self.use_te = True
        else:
            self.norm = nn.LayerNorm(normalized_shape, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x)


def resolve_fp8_config(model_cfg: dict) -> dict:
    """Parse and validate the model.fp8 config block."""
    fp8_cfg = model_cfg.get("fp8", {})
    if not isinstance(fp8_cfg, dict):
        fp8_cfg = {}

    enabled = bool(fp8_cfg.get("enabled", False))
    backend = str(fp8_cfg.get("backend", "none")).strip().lower()

    if enabled and backend == "transformer_engine" and not transformer_engine_available():
        raise RuntimeError(
            "model.fp8.enabled=true with backend=transformer_engine, "
            "but Transformer Engine is not installed. "
            "Install it or set model.fp8.enabled=false."
        )

    return {
        "enabled": enabled,
        "backend": backend,
        "use_te_linear": bool(fp8_cfg.get("use_te_linear", True)) and enabled,
        "use_te_attention": bool(fp8_cfg.get("use_te_attention", True)) and enabled,
        "exclude_heads": bool(fp8_cfg.get("exclude_heads", True)),
        "exclude_native_refiner": bool(fp8_cfg.get("exclude_native_refiner", True)),
        "fp8_recipe": str(fp8_cfg.get("fp8_recipe", "delayed_scaling")),
    }
