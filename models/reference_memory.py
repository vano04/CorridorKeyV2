"""Reference memory bank for V3.

MatAnyone2-style long-range reference context. Maintains reference tokens from
first frame, random far frame, and/or previous window frame to anchor temporal
consistency without requiring huge temporal windows.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


EPS = 1e-6


class ReferenceCrossAttention(nn.Module):
    """Cross-attention from spatial features to reference tokens."""

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor, ref_tokens: Tensor) -> Tensor:
        """
        Args:
            x:          [B, L, C] spatial feature tokens
            ref_tokens: [B, R, C] reference tokens
        Returns:
            [B, L, C] updated spatial features
        """
        b, l, c = x.shape
        q = self.q_proj(self.norm_q(x)).reshape(b, l, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(self.norm_kv(ref_tokens)).reshape(b, -1, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, l, c)
        return x + torch.sigmoid(self.gate) * self.out_proj(out)


class ReferenceEncoder(nn.Module):
    """Encode a single reference frame into compact tokens."""

    def __init__(self, dim: int, num_tokens: int = 32) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.query_tokens = nn.Parameter(torch.randn(num_tokens, dim) * 0.02)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)

    def forward(self, frame_features: Tensor) -> Tensor:
        """
        Args:
            frame_features: [B, H*W, C] features from one reference frame
        Returns:
            [B, num_tokens, C] reference tokens
        """
        b = frame_features.shape[0]
        queries = self.query_tokens.unsqueeze(0).expand(b, -1, -1)
        q = self.norm_q(queries)
        kv = self.norm_kv(frame_features)
        out, _ = self.cross_attn(q, kv, kv)
        return out


class ReferenceMemoryBank(nn.Module):
    """Long-range reference memory for V3.

    During training, samples reference frames and encodes them into tokens.
    During inference, caches tokens across windows and refreshes on scene changes.
    """

    def __init__(
        self,
        dim: int,
        num_reference_frames: int = 2,
        tokens_per_reference: int = 32,
        use_first_frame: bool = True,
        use_random_far_frame: bool = True,
        use_previous_window_frame: bool = True,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_reference_frames = num_reference_frames
        self.tokens_per_reference = tokens_per_reference
        self.use_first_frame = use_first_frame
        self.use_random_far_frame = use_random_far_frame
        self.use_previous_window_frame = use_previous_window_frame
        self.dropout = dropout

        self.ref_encoder = ReferenceEncoder(dim=dim, num_tokens=tokens_per_reference)
        self.cross_attn = ReferenceCrossAttention(dim=dim, num_heads=8)
        self.null_token = nn.Parameter(torch.randn(1, tokens_per_reference, dim) * 0.02)

    def _select_reference_indices(self, t: int) -> list[int]:
        """Select a fixed-size deterministic reference set for compile/DDP stability."""
        if t <= 0 or self.num_reference_frames <= 0:
            return []

        candidates: list[int] = []
        if self.use_first_frame:
            candidates.append(0)
        if self.use_random_far_frame and t > 1:
            candidates.append(t - 1)
        if self.use_previous_window_frame and t > 2:
            candidates.append(max(0, t // 2))
        if not candidates:
            candidates.append(0)

        deduped: list[int] = []
        for idx in candidates:
            idx = max(0, min(int(idx), t - 1))
            if idx not in deduped:
                deduped.append(idx)
        while len(deduped) < self.num_reference_frames:
            deduped.append(deduped[-1])
        return deduped[:self.num_reference_frames]

    def encode_references(
        self,
        features: Tensor,
        reference_indices: Optional[list[int]] = None,
    ) -> Tensor:
        """Encode selected reference frames into tokens.

        Args:
            features: [B, T, H, W, C] encoder features
            reference_indices: which temporal indices to use as references

        Returns:
            [B, R*tokens_per_reference, C] reference tokens
        """
        b, t, h, w, c = features.shape

        if reference_indices is None:
            reference_indices = self._select_reference_indices(t)

        if not reference_indices:
            return self.null_token.expand(b, -1, -1)

        all_tokens = []
        null = self.null_token.expand(b, -1, -1)
        for idx in reference_indices:
            frame_feat = features[:, idx].reshape(b, h * w, c)
            tokens = self.ref_encoder(frame_feat)
            if self.training and self.dropout > 0.0:
                drop = (torch.rand(b, 1, 1, device=features.device) < float(self.dropout)).to(tokens.dtype)
                tokens = tokens * (1.0 - drop) + null.to(tokens.dtype) * drop
            all_tokens.append(tokens)

        result = torch.cat(all_tokens, dim=1)
        # Always add null_token as a tiny differentiable bias so the parameter
        # is in the computation graph every iteration (DDP static_graph).
        result = result + self.null_token.expand(b, -1, -1).mean() * 0.0
        return result

    def forward(
        self,
        features: Tensor,
        ref_tokens: Optional[Tensor] = None,
        reference_indices: Optional[list[int]] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            features:          [B, T, H, W, C] encoder features
            ref_tokens:        [B, R, C] precomputed reference tokens (inference cache)
            reference_indices: which frames to encode as references

        Returns:
            updated_features: [B, T, H, W, C]
            ref_tokens:       [B, R, C] for caching
        """
        b, t, h, w, c = features.shape

        if ref_tokens is None:
            ref_tokens = self.encode_references(features, reference_indices)

        # Apply reference cross-attention to each frame
        flat = features.reshape(b * t, h * w, c)
        ref_expanded = ref_tokens.unsqueeze(1).expand(b, t, -1, c).reshape(b * t, -1, c)
        updated = self.cross_attn(flat, ref_expanded)
        updated = updated.reshape(b, t, h, w, c)

        return updated, ref_tokens
