"""Native detail refiner for V3.

Rewritten from V2's NativeResidualRefineHead. Key V3 differences:
- Accepts global context projection as additional input
- Always executed (no conditional skip) — uses refine_mask as soft gate only
- DDP-safe: every parameter receives gradient every iteration
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _norm_groups(channels: int, min_groups: int = 4) -> int:
    for g in (32, 16, 8, min_groups):
        if channels % g == 0 and channels // g >= 1:
            return g
    return 1


class NativeResidualBlock(nn.Module):
    """Lightweight residual block for native-resolution detail processing."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        g = _norm_groups(channels)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.GELU()
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.act(self.norm1(x))
        x = self.conv1(x)
        x = self.act(self.norm2(x))
        x = self.conv2(x)
        return x + residual


class NativeDetailRefiner(nn.Module):
    """Native-resolution detail refiner for V3.

    Takes the coarse decoder output (upsampled to native 1024x1024) along with
    full-res RGB and produces bounded residual deltas for alpha, FG, and spill.
    The refine_mask is a differentiable soft gate that weights the residuals.

    This module always runs during training (no conditional branching) so that
    DDP static_graph sees a deterministic parameter-usage pattern.
    """

    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 64,
        num_blocks: int = 3,
        max_alpha_delta: float = 0.25,
        max_fg_delta: float = 0.15,
        max_spill_delta: float = 0.10,
        predict_spill: bool = True,
        global_context_dim: int = 0,
    ) -> None:
        super().__init__()
        self.max_alpha_delta = max_alpha_delta
        self.max_fg_delta = max_fg_delta
        self.max_spill_delta = max_spill_delta
        self.predict_spill = predict_spill
        self.global_context_dim = global_context_dim

        # Input: RGB(3) + coarse_alpha(1) + coarse_fg(3) + uncertainty(1) + spill(1) = 9
        # Optionally + global context projection
        actual_in = in_channels
        if global_context_dim > 0:
            self.global_proj = nn.Sequential(
                nn.Linear(global_context_dim, hidden_channels),
                nn.GELU(),
            )
            actual_in += hidden_channels
        else:
            self.global_proj = None

        self.stem = nn.Sequential(
            nn.Conv2d(actual_in, hidden_channels, 3, padding=1),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(*[
            NativeResidualBlock(hidden_channels) for _ in range(num_blocks)
        ])

        # Residual heads
        self.alpha_delta = nn.Conv2d(hidden_channels, 1, 1)
        self.fg_delta = nn.Conv2d(hidden_channels, 3, 1)
        if predict_spill:
            self.spill_delta = nn.Conv2d(hidden_channels, 1, 1)
        else:
            self.spill_delta = None

        # Soft gate head — predicts per-pixel confidence for applying residuals
        self.gate_head = nn.Sequential(
            nn.Conv2d(hidden_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        rgb: Tensor,
        coarse_alpha: Tensor,
        coarse_fg: Tensor,
        uncertainty: Optional[Tensor],
        coarse_spill: Optional[Tensor],
        refine_mask: Optional[Tensor] = None,
        alpha_refine_mask: Optional[Tensor] = None,
        fg_refine_mask: Optional[Tensor] = None,
        spill_refine_mask: Optional[Tensor] = None,
        global_tokens: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        All inputs: [B*T, C, H, W] (temporal axis already folded).
        refine_mask: [B*T, 1, H, W] legacy soft gate used for all residuals.
        *_refine_mask: optional per-head gates. Alpha/spill usually want an
            edge/uncertainty mask, while FG needs foreground-wide access to
            restore native RGB detail instead of only touching the matte edge.
        global_tokens: [B, M, C] compact tokens (broadcast across T).

        Returns dict with:
            alpha_refined: [B*T, 1, H, W]
            fg_refined:    [B*T, 3, H, W]
            spill_refined: [B*T, 1, H, W] (if predict_spill)
            native_alpha_delta_pred: [B*T, 1, H, W]
            native_fg_delta_pred:    [B*T, 3, H, W]
            native_spill_delta_pred: [B*T, 1, H, W] (if predict_spill)
            refine_gate:   [B*T, 1, H, W] learned soft gate
        """
        bt = rgb.shape[0]
        h, w = rgb.shape[-2:]

        # Build input tensor
        parts = [rgb, coarse_alpha, coarse_fg]
        if uncertainty is not None:
            parts.append(uncertainty)
        else:
            parts.append(torch.zeros(bt, 1, h, w, device=rgb.device, dtype=rgb.dtype))
        if coarse_spill is not None:
            parts.append(coarse_spill)
        else:
            parts.append(torch.zeros(bt, 1, h, w, device=rgb.device, dtype=rgb.dtype))

        x_in = torch.cat(parts, dim=1)

        # Project and spatially broadcast global context if available
        if self.global_proj is not None and global_tokens is not None:
            # global_tokens: [B, M, C] → pool → [B, C_proj] → broadcast to [B*T, C_proj, H, W]
            g_pooled = global_tokens.mean(dim=1)  # [B, C_global]
            g_proj = self.global_proj(g_pooled)    # [B, C_hidden]
            # Infer T from bt/B
            b = global_tokens.shape[0]
            t = bt // b
            g_spatial = g_proj[:, :, None, None].expand(b, -1, h, w)
            g_spatial = g_spatial.unsqueeze(1).expand(b, t, -1, h, w).reshape(bt, -1, h, w)
            x_in = torch.cat([x_in, g_spatial], dim=1)

        x = self.stem(x_in)
        x = self.blocks(x)

        # Bounded residuals
        gate = self.gate_head(x)
        alpha_gate = gate
        fg_gate = gate
        spill_gate = gate
        if refine_mask is not None:
            alpha_gate = alpha_gate * refine_mask
            fg_gate = fg_gate * refine_mask
            spill_gate = spill_gate * refine_mask
        if alpha_refine_mask is not None:
            alpha_gate = gate * alpha_refine_mask
        if fg_refine_mask is not None:
            fg_gate = gate * fg_refine_mask
        if spill_refine_mask is not None:
            spill_gate = gate * spill_refine_mask

        alpha_delta_raw = torch.tanh(self.alpha_delta(x)) * self.max_alpha_delta
        fg_delta_raw = torch.tanh(self.fg_delta(x)) * self.max_fg_delta

        alpha_delta = alpha_delta_raw * alpha_gate
        fg_delta = fg_delta_raw * fg_gate

        alpha_refined = (coarse_alpha + alpha_delta).clamp(0.0, 1.0)
        fg_refined = (coarse_fg + fg_delta).clamp(0.0, 1.0)

        out: Dict[str, Tensor] = {
            "alpha_refined": alpha_refined,
            "fg_refined": fg_refined,
            "native_alpha_delta_pred": alpha_delta_raw,
            "native_fg_delta_pred": fg_delta_raw,
            "refine_gate": torch.maximum(alpha_gate, fg_gate),
        }

        if self.spill_delta is not None:
            spill_delta_raw = torch.tanh(self.spill_delta(x)) * self.max_spill_delta
            spill_delta = spill_delta_raw * spill_gate
            coarse_s = coarse_spill if coarse_spill is not None else torch.zeros_like(coarse_alpha)
            spill_refined = (coarse_s + spill_delta).clamp(0.0, 1.0)
            out["spill_refined"] = spill_refined
            out["native_spill_delta_pred"] = spill_delta_raw

        return out
