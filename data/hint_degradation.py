"""Hint degradation for V3.

Augments coarse_alpha_init to train the model to correct noisy/imperfect hints.
Applied inside DeviceMattingTransform after spatial/temporal alignment and clean
coarse_alpha_init creation, before the batch is returned.

Severity distribution (configurable):
  clean:  25% — no degradation
  mild:   50% — small geometric + blur perturbations
  severe: 25% — large distortions, holes, temporal lag
"""
from __future__ import annotations

import random
from typing import Dict, Any

import torch
from torch import Tensor
import torch.nn.functional as F


def _morphological(x: Tensor, op: str, kernel: int) -> Tensor:
    pad = kernel // 2
    if op == "dilate":
        return F.max_pool2d(x, kernel_size=kernel, stride=1, padding=pad)
    if op == "erode":
        return -F.max_pool2d(-x, kernel_size=kernel, stride=1, padding=pad)
    raise ValueError(f"Unknown op: {op}")


def _gaussian_blur(x: Tensor, kernel_size: int, sigma: float) -> Tensor:
    if kernel_size % 2 == 0:
        kernel_size += 1
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma + 1e-6))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    c = x.shape[1]
    kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(c, 1, kernel_size, kernel_size)
    return F.conv2d(x, kernel, padding=radius, groups=c)


def _random_holes(alpha: Tensor, max_holes: int = 5, max_radius: int = 30) -> Tensor:
    """Punch random circular holes into the hint mask."""
    b, c, h, w = alpha.shape
    mask = torch.ones_like(alpha)
    n_holes = random.randint(1, max_holes)
    for _ in range(n_holes):
        cy = random.randint(0, h - 1)
        cx = random.randint(0, w - 1)
        r = random.randint(5, max_radius)
        yy = torch.arange(max(0, cy - r), min(h, cy + r + 1), device=alpha.device)
        xx = torch.arange(max(0, cx - r), min(w, cx + r + 1), device=alpha.device)
        yy, xx = torch.meshgrid(yy, xx, indexing="ij")
        dist2 = (yy - cy).float() ** 2 + (xx - cx).float() ** 2
        circle = (dist2 <= r * r).unsqueeze(0).unsqueeze(0).expand(b, c, -1, -1)
        y0, x0 = max(0, cy - r), max(0, cx - r)
        mask[:, :, y0:y0 + circle.shape[2], x0:x0 + circle.shape[3]] *= (~circle).float()
    return alpha * mask


def _threshold_wobble(alpha: Tensor, strength: float = 0.15) -> Tensor:
    """Add random per-pixel threshold wobble."""
    noise = torch.randn_like(alpha) * strength
    return (alpha + noise).clamp(0.0, 1.0)


def degrade_hint(
    coarse_alpha_init: Tensor,
    severity: str = "auto",
    clean_prob: float = 0.25,
    mild_prob: float = 0.50,
) -> Tensor:
    """Degrade a coarse alpha hint with configurable severity.

    Args:
        coarse_alpha_init: [B, 1, H, W] clean coarse alpha hint
        severity: "clean", "mild", "severe", or "auto" (random choice)
        clean_prob: probability of clean (no degradation) when auto
        mild_prob: probability of mild degradation when auto

    Returns:
        [B, 1, H, W] degraded hint
    """
    if severity == "auto":
        r = random.random()
        if r < clean_prob:
            severity = "clean"
        elif r < clean_prob + mild_prob:
            severity = "mild"
        else:
            severity = "severe"

    if severity == "clean":
        return coarse_alpha_init

    alpha = coarse_alpha_init.clone()

    if severity == "mild":
        # Mild: small morph + blur + noise
        if random.random() < 0.7:
            op = random.choice(["erode", "dilate"])
            k = random.choice([3, 5])
            alpha = _morphological(alpha, op=op, kernel=k)

        if random.random() < 0.6:
            k = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 1.5)
            alpha = _gaussian_blur(alpha, kernel_size=k, sigma=sigma)

        if random.random() < 0.5:
            alpha = _threshold_wobble(alpha, strength=random.uniform(0.02, 0.08))

        # Small shift
        if random.random() < 0.3:
            _, _, h, w = alpha.shape
            sx = random.uniform(-3.0, 3.0)
            sy = random.uniform(-3.0, 3.0)
            theta = torch.tensor(
                [[1.0, 0.0, 2.0 * sx / max(1.0, w)],
                 [0.0, 1.0, 2.0 * sy / max(1.0, h)]],
                device=alpha.device, dtype=alpha.dtype,
            ).unsqueeze(0).expand(alpha.shape[0], -1, -1)
            grid = F.affine_grid(theta, alpha.shape, align_corners=False)
            alpha = F.grid_sample(alpha, grid, mode="bilinear", padding_mode="border", align_corners=False)

    elif severity == "severe":
        # Severe: large morph + big blur + holes + threshold wobble + downsample
        if random.random() < 0.8:
            op = random.choice(["erode", "dilate"])
            k = random.choice([5, 7, 9, 11])
            alpha = _morphological(alpha, op=op, kernel=k)

        if random.random() < 0.8:
            k = random.choice([7, 9, 11, 15])
            sigma = random.uniform(1.5, 4.0)
            alpha = _gaussian_blur(alpha, kernel_size=k, sigma=sigma)

        if random.random() < 0.5:
            alpha = _random_holes(alpha, max_holes=random.randint(2, 8), max_radius=random.randint(15, 50))

        if random.random() < 0.7:
            alpha = _threshold_wobble(alpha, strength=random.uniform(0.05, 0.20))

        # Aggressive downsample-upsample
        if random.random() < 0.6:
            _, _, h, w = alpha.shape
            factor = random.choice([4, 6, 8, 10])
            h2, w2 = max(4, h // factor), max(4, w // factor)
            alpha = F.interpolate(alpha, size=(h2, w2), mode="bilinear", align_corners=False)
            alpha = F.interpolate(alpha, size=(h, w), mode="bilinear", align_corners=False)

        # Large shift
        if random.random() < 0.5:
            _, _, h, w = alpha.shape
            sx = random.uniform(-8.0, 8.0)
            sy = random.uniform(-8.0, 8.0)
            theta = torch.tensor(
                [[1.0, 0.0, 2.0 * sx / max(1.0, w)],
                 [0.0, 1.0, 2.0 * sy / max(1.0, h)]],
                device=alpha.device, dtype=alpha.dtype,
            ).unsqueeze(0).expand(alpha.shape[0], -1, -1)
            grid = F.affine_grid(theta, alpha.shape, align_corners=False)
            alpha = F.grid_sample(alpha, grid, mode="bilinear", padding_mode="border", align_corners=False)

        # Partial missing: zero out a random quadrant
        if random.random() < 0.3:
            _, _, h, w = alpha.shape
            quadrant = random.randint(0, 3)
            h2, w2 = h // 2, w // 2
            if quadrant == 0:
                alpha[:, :, :h2, :w2] = 0.0
            elif quadrant == 1:
                alpha[:, :, :h2, w2:] = 0.0
            elif quadrant == 2:
                alpha[:, :, h2:, :w2] = 0.0
            else:
                alpha[:, :, h2:, w2:] = 0.0

    return alpha.clamp(0.0, 1.0)
