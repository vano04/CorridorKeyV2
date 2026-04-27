"""Green / chroma prior map generation for V3.

Deterministic maps computed from RGB that help the model distinguish
greenscreen spill from foreground green content.

Must produce identical results on GPU (training) and CPU (inference).
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor


def green_excess(rgb: Tensor) -> Tensor:
    """Green excess: G - max(R, B), clamped to [0, 1].

    Args:
        rgb: [..., 3, H, W] with channel dim at -3
    Returns:
        [..., 1, H, W]
    """
    g = rgb[..., 1:2, :, :]
    r = rgb[..., 0:1, :, :]
    b = rgb[..., 2:3, :, :]
    return (g - torch.maximum(r, b)).clamp(0.0, 1.0)


def normalized_green_excess(rgb: Tensor, eps: float = 1e-6) -> Tensor:
    """Normalised green excess: (G - max(R,B)) / (R + G + B + eps).

    Args:
        rgb: [..., 3, H, W]
    Returns:
        [..., 1, H, W] in approximately [0, 1]
    """
    g = rgb[..., 1:2, :, :]
    r = rgb[..., 0:1, :, :]
    b = rgb[..., 2:3, :, :]
    total = r + g + b + eps
    return ((g - torch.maximum(r, b)) / total).clamp(0.0, 1.0)


def chroma_distance_to_key(
    rgb: Tensor,
    key_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> Tensor:
    """Per-pixel chroma distance to a reference key colour.

    Removes luminance before computing distance so pure brightness
    differences don't register as chroma distance.

    Args:
        rgb: [..., 3, H, W]
        key_color: (R, G, B) reference
    Returns:
        [..., 1, H, W] in [0, ~1.7] (clamped to [0, 1] for model input)
    """
    luma = 0.299 * rgb[..., 0:1, :, :] + 0.587 * rgb[..., 1:2, :, :] + 0.114 * rgb[..., 2:3, :, :]
    chroma = rgb - luma
    kr, kg, kb = key_color
    key_luma = 0.299 * kr + 0.587 * kg + 0.114 * kb
    key_chroma = torch.tensor([kr - key_luma, kg - key_luma, kb - key_luma],
                              device=rgb.device, dtype=rgb.dtype)
    key_chroma = key_chroma.reshape(*([1] * (rgb.ndim - 3)), 3, 1, 1)
    diff = chroma - key_chroma
    dist = (diff ** 2).sum(dim=-3, keepdim=True).sqrt()
    return dist.clamp(0.0, 1.0)


def screen_likelihood(rgb: Tensor, threshold: float = 0.05) -> Tensor:
    """Binary-ish screen likelihood: where green excess > threshold.

    Args:
        rgb: [..., 3, H, W]
    Returns:
        [..., 1, H, W] soft mask in [0, 1]
    """
    ge = green_excess(rgb)
    return torch.sigmoid((ge - threshold) * 20.0)


def spill_prior(rgb: Tensor, alpha: Tensor, band_kernel: int = 5) -> Tensor:
    """Estimate green spill prior from boundary band + green excess.

    Spill is most likely where green excess is high AND alpha is transitional.

    Args:
        rgb:   [..., 3, H, W]
        alpha: [..., 1, H, W]
    Returns:
        [..., 1, H, W] in [0, 1]
    """
    ge = green_excess(rgb)
    # Soft boundary band via morphological ops
    shape = alpha.shape
    flat = alpha.reshape(-1, 1, shape[-2], shape[-1])
    pad = band_kernel // 2
    dil = torch.nn.functional.max_pool2d(flat, band_kernel, stride=1, padding=pad)
    ero = -torch.nn.functional.max_pool2d(-flat, band_kernel, stride=1, padding=pad)
    band = (dil - ero).clamp(0.0, 1.0).reshape(shape)
    return (ge * band).clamp(0.0, 1.0)


def unknown_band_from_hint(hint: Tensor, lo: float = 0.02, hi: float = 0.98) -> Tensor:
    """Mark uncertain regions from a coarse alpha hint.

    Args:
        hint: [..., 1, H, W]
    Returns:
        [..., 1, H, W] binary-ish mask of transitional region
    """
    return ((hint > lo) & (hint < hi)).to(hint.dtype)


def compute_all_priors(
    rgb: Tensor,
    hint: Tensor,
    key_color: Tuple[float, float, float] = (0.0, 1.0, 0.0),
) -> Tensor:
    """Compute all green/chroma prior channels.

    Args:
        rgb:  [..., 3, H, W]
        hint: [..., 1, H, W] coarse alpha hint
        key_color: reference greenscreen colour

    Returns:
        [..., 4, H, W] with channels:
            0: green_excess
            1: chroma_distance_to_key
            2: spill_prior
            3: unknown_band
    """
    ge = green_excess(rgb)
    cd = chroma_distance_to_key(rgb, key_color)
    sp = spill_prior(rgb, hint)
    ub = unknown_band_from_hint(hint)
    return torch.cat([ge, cd, sp, ub], dim=-3)
