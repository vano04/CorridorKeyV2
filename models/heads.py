"""Prediction heads for V3.

Quality eval head (train-time only) and any auxiliary heads.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class QualityEvalHead(nn.Module):
    """Train-time error prediction head.

    Learns to predict per-pixel error magnitude from decoder features.
    Used for uncertainty calibration and bad-region mining weights.
    Disabled during inference.
    """

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, in_dim // 4, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_dim // 4, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: [N, C, H, W] → [N, 1, H, W] predicted error magnitude in [0, 1]."""
        return torch.sigmoid(self.net(x))


class AlphaHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.net(x))


class ForegroundHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, 3, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.net(x))


class UncertaintyHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.net(x))


class SpillMaskHead(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, in_dim // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(in_dim // 2, 1, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.net(x))
