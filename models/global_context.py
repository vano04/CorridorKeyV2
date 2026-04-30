"""Global context branch for V3.

Runs once per temporal window on downscaled full-frame inputs (default 512px
long side). Produces compact global tokens and an optional spatial feature
pyramid that are reused across all 1024 tiles in the window.

This branch answers "what is foreground globally?" — it does not produce
final detail. It is transformer-heavy and is a good FP8/TE target.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from .transformer_engine_utils import MaybeTELinear, MaybeTELayerNorm


EPS = 1e-6


def _compute_padding(h: int, w: int, multiple: int) -> Tuple[int, int]:
    return (multiple - h % multiple) % multiple, (multiple - w % multiple) % multiple


def _safe_reflect_pad(x: Tensor, pad_w: int, pad_h: int) -> Tensor:
    h, w = x.shape[-2], x.shape[-1]
    if pad_h >= h or pad_w >= w:
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


class GlobalPatchEmbed(nn.Module):
    """Patch embedding for the global context branch."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 8) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, C, H, W]
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.proj(x)
        h1, w1 = x.shape[-2:]
        return x.permute(0, 2, 3, 1).reshape(b, t, h1, w1, -1)


class GlobalSelfAttention(nn.Module):
    """Multi-head self-attention for the global context branch.

    Uses full spatial attention (no windowing) since the global branch operates
    on small downscaled inputs (~64x64 patches at 512px / patch_size=8).
    """

    def __init__(self, dim: int, num_heads: int, fp8_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = MaybeTELinear(dim, dim * 3, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.proj = MaybeTELinear(dim, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)

    def forward(self, x: Tensor) -> Tensor:
        # x: [N, L, C] where N = B*T, L = H*W
        n, l, c = x.shape
        qkv = self.qkv(x).reshape(n, l, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(n, l, c))


class GlobalTemporalAttention(nn.Module):
    """Temporal attention across frames at each spatial location."""

    def __init__(self, dim: int, num_heads: int, fp8_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = MaybeTELinear(dim, dim * 3, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.proj = MaybeTELinear(dim, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, H, W, C]
        b, t, h, w, c = x.shape
        s = h * w
        x_seq = x.permute(0, 2, 3, 1, 4).reshape(b * s, t, c)
        qkv = self.qkv(x_seq).reshape(b * s, t, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # Some SDPA CUDA backends reject very large effective batch counts.
        # At 1024px global context with patch_size=8, B=4 gives
        # B*H*W == 65,536 temporal sequences; chunking keeps the same math
        # while staying below backend launch limits.
        max_sdpa_batch = 32768
        if q.shape[0] > max_sdpa_batch and q.is_cuda:
            out = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        q[start : start + max_sdpa_batch],
                        k[start : start + max_sdpa_batch],
                        v[start : start + max_sdpa_batch],
                    )
                    for start in range(0, q.shape[0], max_sdpa_batch)
                ],
                dim=0,
            )
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b * s, t, c)
        out = self.proj(out)
        return out.reshape(b, h, w, t, c).permute(0, 3, 1, 2, 4)


class GlobalTransformerBlock(nn.Module):
    """Single transformer block: spatial self-attn → temporal attn → MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, use_temporal: bool = True, fp8_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        self.norm1 = MaybeTELayerNorm(dim)
        self.spatial_attn = GlobalSelfAttention(dim, num_heads, fp8_cfg=fp8_cfg)
        self.norm2 = MaybeTELayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            MaybeTELinear(dim, hidden, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False),
            nn.GELU(),
            MaybeTELinear(hidden, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False),
        )
        if use_temporal:
            self.norm_t = MaybeTELayerNorm(dim)
            self.temporal_attn = GlobalTemporalAttention(dim, num_heads, fp8_cfg=fp8_cfg)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, T, H, W, C]
        b, t, h, w, c = x.shape
        # Spatial attention
        flat = x.reshape(b * t, h * w, c)
        flat = flat + self.spatial_attn(self.norm1(flat))
        x = flat.reshape(b, t, h, w, c)
        # Temporal attention
        if self.use_temporal and t > 1:
            x = x + self.temporal_attn(self.norm_t(x))
        # MLP
        flat = x.reshape(b * t * h * w, c)
        flat = flat + self.mlp(self.norm2(flat))
        return flat.reshape(b, t, h, w, c)


class GlobalContextSummary(nn.Module):
    """Cross-attention from learnable summary tokens to global spatial features.

    Produces a compact set of global_tokens that tiles cross-attend to.
    """

    def __init__(self, dim: int, num_heads: int, num_tokens: int = 64, fp8_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.num_tokens = num_tokens

        self.summary_tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.norm_q = MaybeTELayerNorm(dim)
        self.norm_kv = MaybeTELayerNorm(dim)
        self.q_proj = MaybeTELinear(dim, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.kv_proj = MaybeTELinear(dim, dim * 2, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.out_proj = MaybeTELinear(dim, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: [B, T*H*W, C] flattened spatial-temporal features.
        Returns:
            [B, num_tokens, C] compact global tokens.
        """
        b = features.shape[0]
        queries = self.summary_tokens.unsqueeze(0).expand(b, -1, -1)  # [B, M, C]
        q = self.q_proj(self.norm_q(queries))
        kv = self.kv_proj(self.norm_kv(features))

        q = q.reshape(b, self.num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        kv = kv.reshape(b, -1, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, self.num_tokens, -1)
        return self.out_proj(out)


class GlobalContextBranch(nn.Module):
    """Global context encoder for V3.

    Operates on downscaled full-frame inputs, producing global spatial tokens
    and temporal memory tokens that are reused across all 1024 tiles.
    """

    def __init__(
        self,
        in_channels: int = 4,
        embed_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        patch_size: int = 8,
        memory_tokens: int = 64,
        mlp_ratio: float = 4.0,
        gradient_checkpointing: bool = False,
        fp8_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.gradient_checkpointing = gradient_checkpointing
        self.fp8_cfg = fp8_cfg

        self.patch_embed = GlobalPatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )

        self.blocks = nn.ModuleList([
            GlobalTransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                use_temporal=(i >= num_layers // 2),
                fp8_cfg=fp8_cfg,
            )
            for i in range(num_layers)
        ])

        self.summary = GlobalContextSummary(
            dim=embed_dim,
            num_heads=num_heads,
            num_tokens=memory_tokens,
            fp8_cfg=fp8_cfg,
        )

        self.norm_out = MaybeTELayerNorm(embed_dim)

    def forward(
        self,
        video_rgb_global: Tensor,
        coarse_alpha_global: Tensor,
        green_priors_global: Optional[Tensor] = None,
        fg_guidance_global: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            video_rgb_global:    [B, T, 3, Hg, Wg]
            coarse_alpha_global: [B, T, 1, Hg, Wg]
            green_priors_global: [B, T, C_priors, Hg, Wg] optional
            fg_guidance_global:  [B, T, 3, Hg, Wg] optional dedicated FG context layer

        Returns:
            global_tokens:  [B, M, C]  compact global context tokens
            spatial_features: [B, T, Hp, Wp, C]  spatial feature map (for optional pyramid)
        """
        # Build input
        inputs = [video_rgb_global, coarse_alpha_global]
        if green_priors_global is not None:
            inputs.append(green_priors_global)
        if fg_guidance_global is not None:
            inputs.append(fg_guidance_global)
        x = torch.cat(inputs, dim=2)  # [B, T, C_in, Hg, Wg]

        b, t, c, h, w = x.shape
        # Pad to patch_size multiple
        pad_h, pad_w = _compute_padding(h, w, self.patch_size)
        if pad_h or pad_w:
            x_flat = x.reshape(b * t, c, h, w)
            x_flat = _safe_reflect_pad(x_flat, pad_w, pad_h)
            x = x_flat.reshape(b, t, c, h + pad_h, w + pad_w)

        # Patch embed
        x = self.patch_embed(x)  # [B, T, Hp, Wp, C]

        # Transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = _checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Summarise into compact tokens
        _, _, hp, wp, c_dim = x.shape
        flat = self.norm_out(x.reshape(b, t * hp * wp, c_dim))
        global_tokens = self.summary(flat)

        return global_tokens, x
