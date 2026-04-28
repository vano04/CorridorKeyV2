"""Local tile encoder for V3.

The main backbone transformer that processes 1024x1024 tiles at native
resolution. Rewritten from V1's encoder path with V3-specific additions:
- Extended input channels (green priors, coordinate maps)
- Cross-attention fusion from local tile tokens to global context tokens
- Integration with reference memory at the deepest stage
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint

from .transformer_engine_utils import MaybeTELinear, MaybeTELayerNorm


EPS = 1e-6


# ---------------------------------------------------------------------------
# Building blocks (rewritten from V1 for self-containment)
# ---------------------------------------------------------------------------


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.register_buffer("drop_prob", torch.tensor(float(drop_prob), dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.bernoulli(torch.full(shape, keep_prob, device=x.device, dtype=x.dtype))
        return x / keep_prob * random_tensor


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, dropout: float = 0.0, fp8_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = MaybeTELinear(dim, hidden, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.act = nn.GELU()
        self.fc2 = MaybeTELinear(hidden, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.drop(self.act(self.fc1(x)))
        return self.drop(self.fc2(x))


class RelativePositionBias2D(nn.Module):
    def __init__(self, window_size: int, num_heads: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        size = (2 * window_size - 1) * (2 * window_size - 1)
        self.table = nn.Parameter(torch.zeros(size, num_heads))
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_index = relative_coords.sum(-1)
        self.register_buffer("relative_index", relative_index, persistent=False)
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self) -> Tensor:
        n = self.window_size * self.window_size
        bias = self.table[self.relative_index.reshape(-1)].reshape(n, n, self.num_heads)
        return bias.permute(2, 0, 1)


def _compute_padding(h: int, w: int, multiple: int) -> Tuple[int, int]:
    return (multiple - h % multiple) % multiple, (multiple - w % multiple) % multiple


def _safe_reflect_pad(x: Tensor, pad_w: int, pad_h: int) -> Tensor:
    h, w = x.shape[-2], x.shape[-1]
    if pad_h >= h or pad_w >= w:
        return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")


class SpatialWindowAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int, fp8_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.qkv = MaybeTELinear(dim, dim * 3, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.proj = MaybeTELinear(dim, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.rel_bias = RelativePositionBias2D(window_size, num_heads)

    def forward(self, x: Tensor) -> Tensor:
        b, t, h, w, c = x.shape
        ws = self.window_size
        pad_h, pad_w = _compute_padding(h, w, ws)
        if pad_h or pad_w:
            x_nchw = x.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
            x_nchw = _safe_reflect_pad(x_nchw, pad_w, pad_h)
            h_pad, w_pad = h + pad_h, w + pad_w
            x = x_nchw.reshape(b, t, c, h_pad, w_pad).permute(0, 1, 3, 4, 2)
        else:
            h_pad, w_pad = h, w

        windows = x.reshape(b * t, h_pad // ws, ws, w_pad // ws, ws, c)
        windows = windows.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)

        n_win, n_tok, _ = windows.shape
        qkv = self.qkv(windows).reshape(n_win, n_tok, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        bias = self.rel_bias().unsqueeze(0).to(dtype=q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        out = out.transpose(1, 2).reshape(n_win, n_tok, c)
        out = self.proj(out)

        out = out.reshape(b * t, h_pad // ws, w_pad // ws, ws, ws, c)
        out = out.permute(0, 1, 3, 2, 4, 5).reshape(b, t, h_pad, w_pad, c)
        return out[:, :, :h, :w, :]


class TemporalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, temporal_window: int = 5, fp8_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.temporal_window = temporal_window
        self.qkv = MaybeTELinear(dim, dim * 3, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.proj = MaybeTELinear(dim, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        # Cache: (t, device_str, dtype_str) -> [b=1, 1, t, t] base mask without valid_mask applied.
        # The base window mask is static for fixed t and temporal_window; it is
        # broadcast over batch at call time so B-dependent valid_mask is handled
        # separately without invalidating the cache.
        self._window_mask_cache: Dict[Tuple[int, str, str], Tensor] = {}

    def _get_window_mask(self, t: int, device: torch.device, dtype: torch.dtype) -> Optional[Tensor]:
        """Return the [1, 1, t, t] window bias mask, building it once per (t, device, dtype)."""
        if not (0 < self.temporal_window < t):
            return None
        key = (t, str(device), str(dtype))
        if key not in self._window_mask_cache:
            radius = max(0, int(self.temporal_window) // 2)
            idx = torch.arange(t, device=device)
            window_allowed = (idx[:, None] - idx[None, :]).abs() <= radius  # [t, t]
            eye = torch.eye(t, device=device, dtype=torch.bool)
            allowed = window_allowed | eye  # self-attend always
            mask = torch.zeros(1, 1, t, t, device=device, dtype=dtype)
            mask.masked_fill_(~allowed.unsqueeze(0).unsqueeze(0), torch.finfo(dtype).min)
            self._window_mask_cache[key] = mask
        return self._window_mask_cache[key]

    def forward(self, x: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        b, t, h, w, c = x.shape
        s = h * w
        x_seq = x.permute(0, 2, 3, 1, 4).reshape(b * s, t, c)
        qkv = self.qkv(x_seq).reshape(b * s, t, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_mask: Optional[Tensor] = None
        if valid_mask is not None or (0 < self.temporal_window < t):
            # Start from the cached window mask (0 launches on cache hit)
            window_base = self._get_window_mask(t, x.device, q.dtype)

            if valid_mask is not None:
                # Build per-batch allowed mask incorporating valid_mask
                valid = valid_mask.to(device=x.device, dtype=torch.bool)  # [B, T]
                # allowed[b, i, j] = valid[b, j] (query i can attend to valid key j)
                allowed_v = valid[:, None, :]  # [B, 1, T]
                if window_base is not None:
                    # window_base: [1,1,T,T] → [B,1,T,T]; combine with valid_mask
                    allowed_w = (window_base > torch.finfo(q.dtype).min / 2)  # back to bool
                    allowed = allowed_w.squeeze(0) & allowed_v  # [B, 1, T] broadcast → [B, T, T]
                    # Always allow self-attention
                    eye = torch.eye(t, device=x.device, dtype=torch.bool).unsqueeze(0)
                    allowed = (allowed | eye)  # [B, T, T]
                    attn_mask = torch.zeros(b, 1, t, t, device=x.device, dtype=q.dtype)
                    attn_mask.masked_fill_(~allowed.unsqueeze(1), torch.finfo(q.dtype).min)
                else:
                    eye = torch.eye(t, device=x.device, dtype=torch.bool).unsqueeze(0)
                    allowed = (allowed_v.expand(b, t, t) | eye)
                    attn_mask = torch.zeros(b, 1, t, t, device=x.device, dtype=q.dtype)
                    attn_mask.masked_fill_(~allowed.unsqueeze(1), torch.finfo(q.dtype).min)
                # Expand over spatial: [B, 1, T, T] → [B*S, 1, T, T]
                attn_mask = attn_mask.unsqueeze(1).expand(b, s, 1, t, t).reshape(b * s, 1, t, t)
            else:
                # No valid_mask — just use the cached window mask (0 extra launches)
                attn_mask = window_base  # [1, 1, t, t] broadcast over b*s

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.permute(0, 2, 1, 3).reshape(b * s, t, c)
        out = self.proj(out)

        if valid_mask is not None:
            out = out.reshape(b, s, t, c)
            out = out * valid_mask.to(device=x.device, dtype=out.dtype)[:, None, :, None]
            return out.reshape(b, h, w, t, c).permute(0, 3, 1, 2, 4)

        return out.reshape(b, h, w, t, c).permute(0, 3, 1, 2, 4)


class GlobalFusionCrossAttention(nn.Module):
    """Cross-attention from local tile features to global context tokens."""

    def __init__(self, dim: int, global_dim: int, num_heads: int, fp8_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = MaybeTELayerNorm(dim)
        self.norm_kv = MaybeTELayerNorm(global_dim)
        self.q_proj = MaybeTELinear(dim, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.kv_proj = MaybeTELinear(global_dim, dim * 2, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.out_proj = MaybeTELinear(dim, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: Tensor, global_tokens: Tensor) -> Tensor:
        """
        x:             [B, T, H, W, C] local features
        global_tokens: [B, M, C_global]
        """
        b, t, h, w, c = x.shape
        flat = x.reshape(b, t * h * w, c)
        q = self.q_proj(self.norm_q(flat))
        kv = self.kv_proj(self.norm_kv(global_tokens))

        q = q.reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = kv.reshape(b, -1, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)

        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, t, h, w, c)
        return x + torch.sigmoid(self.gate) * self.out_proj(out.reshape(b, t * h * w, c)).reshape(b, t, h, w, c)


class MemoryCrossAttention(nn.Module):
    """Cross-attention from local features to subject memory tokens."""

    def __init__(self, dim: int, num_heads: int, fp8_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm_q = MaybeTELayerNorm(dim)
        self.norm_kv = MaybeTELayerNorm(dim)
        self.q_proj = MaybeTELinear(dim, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.kv_proj = MaybeTELinear(dim, dim * 2, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.out_proj = MaybeTELinear(dim, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)

    def forward(self, x: Tensor, memory: Tensor) -> Tensor:
        b, t, h, w, c = x.shape
        q = self.norm_q(x.reshape(b, t * h * w, c))
        kv = self.norm_kv(memory)
        q = self.q_proj(q).reshape(b, -1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(kv).reshape(b, -1, 2, self.num_heads, self.head_dim)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, t, h, w, c)
        return self.out_proj(out)


class MemoryUpdateGate(nn.Module):
    def __init__(self, dim: int, fp8_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.gate = MaybeTELinear(dim * 2, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)

    def forward(self, memory: Tensor, candidate: Tensor, confidence: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate(torch.cat([memory, candidate], dim=-1)))
        g = g * confidence
        return memory + g * (candidate - memory)


class SubjectMemoryBank(nn.Module):
    """Subject-aware memory bank for temporal reasoning."""

    def __init__(self, dim: int, memory_tokens: int, fp8_cfg: Optional[Dict[str, Any]] = None) -> None:
        super().__init__()
        self.dim = dim
        self.memory_tokens = memory_tokens
        self.base_tokens = nn.Parameter(torch.randn(memory_tokens, dim) * 0.02)
        self.update_proj = MaybeTELinear(dim, dim, use_te=fp8_cfg["use_te_linear"] if fp8_cfg else False)
        self.update_gate = MemoryUpdateGate(dim=dim, fp8_cfg=fp8_cfg)

    def forward(self, features: Tensor, coarse_alpha_init: Tensor, video: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        b, t, h, w, c = features.shape
        alpha0 = F.interpolate(coarse_alpha_init, size=(h, w), mode="bilinear", align_corners=False)

        # Build init tokens
        feat0 = features[:, 0]  # [B, H, W, C]
        fg_mask = (alpha0.permute(0, 2, 3, 1) > 0.7).to(feat0.dtype)
        pooled = (feat0 * fg_mask).sum(dim=(1, 2)) / fg_mask.sum(dim=(1, 2)).clamp_min(EPS)
        base = self.base_tokens.unsqueeze(0).expand(b, -1, -1)
        memory = base.clone()
        memory[:, 0] = pooled

        # Vectorized: project all frame summaries in one batched linear call
        # then run the GRU-style scan without Python-loop kernel launches for the linear.
        frame_summary = features.mean(dim=(2, 3))           # [B, T, C]
        all_summaries = self.update_proj(
            frame_summary.reshape(b * t, c)
        ).reshape(b, t, c)                                   # [B, T, C] — one linear call

        # Confidence weights: frame 0 = 1.0, others = 0.25
        # [B, T] built as a tensor to avoid per-frame torch.full calls
        confs = torch.full((b, t), 0.25, dtype=memory.dtype, device=memory.device)
        confs[:, 0] = 1.0
        if valid_mask is not None:
            confs = confs * valid_mask.to(dtype=memory.dtype, device=memory.device)

        # Batched gate inputs: compute all [memory, summary] cats for all frames at once.
        # memory needs to be expanded per-frame inside the scan since it updates,
        # but we pre-compute the gate linear for *all* frames simultaneously,
        # then apply them sequentially (scan cannot be parallelised, but at least
        # the expensive linear is batched, cutting gate launches from 3t → t+2).
        # summary_exp: [B, T, M, C]
        summary_exp = all_summaries.unsqueeze(2).expand(-1, -1, self.memory_tokens, -1)

        for i in range(t):
            s_i = summary_exp[:, i]                          # [B, M, C]
            conf_i = confs[:, i].view(b, 1, 1)              # [B, 1, 1]
            g = torch.sigmoid(self.update_gate.gate(torch.cat([memory, s_i], dim=-1)))
            g = g * conf_i
            memory = memory + g * (s_i - memory)
        return memory


class LocalTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        temporal_window: int,
        drop_path: float = 0.0,
        mlp_ratio: float = 4.0,
        use_temporal: bool = True,
        use_memory: bool = True,
        use_global_fusion: bool = False,
        global_dim: int = 256,
        fp8_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.use_temporal = use_temporal
        self.use_memory = use_memory
        self.use_global_fusion = use_global_fusion

        self.norm1 = MaybeTELayerNorm(dim)
        self.spatial_attn = SpatialWindowAttention(dim, num_heads, window_size, fp8_cfg=fp8_cfg)
        self.drop_path1 = DropPath(drop_path)

        if use_temporal:
            self.norm2 = MaybeTELayerNorm(dim)
            self.temporal_attn = TemporalAttention(dim, num_heads, temporal_window, fp8_cfg=fp8_cfg)
            self.drop_path2 = DropPath(drop_path)

        if use_memory:
            self.norm3 = MaybeTELayerNorm(dim)
            self.memory_attn = MemoryCrossAttention(dim, num_heads, fp8_cfg=fp8_cfg)
            self.drop_path3 = DropPath(drop_path)

        if use_global_fusion:
            self.global_fusion = GlobalFusionCrossAttention(dim, global_dim, num_heads, fp8_cfg=fp8_cfg)
            self.drop_path_g = DropPath(drop_path)

        self.norm4 = MaybeTELayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, fp8_cfg=fp8_cfg)
        self.drop_path4 = DropPath(drop_path)

    def forward(self, x: Tensor, memory: Optional[Tensor] = None, valid_mask: Optional[Tensor] = None, global_tokens: Optional[Tensor] = None) -> Tensor:
        x = x + self.drop_path1(self.spatial_attn(self.norm1(x)))
        if self.use_temporal:
            x = x + self.drop_path2(self.temporal_attn(self.norm2(x), valid_mask=valid_mask))
        if self.use_memory and memory is not None:
            x = x + self.drop_path3(self.memory_attn(self.norm3(x), memory=memory))
        if self.use_global_fusion and global_tokens is not None:
            x = x + self.drop_path_g(self.global_fusion(x, global_tokens))
        x = x + self.drop_path4(self.mlp(self.norm4(x)))
        return x


class LocalTransformerStage(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        temporal_window: int,
        drop_path_rates: Sequence[float],
        use_temporal: bool,
        use_memory: bool,
        use_global_fusion: bool = False,
        global_dim: int = 256,
        gradient_checkpointing: bool = False,
        fp8_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.blocks = nn.ModuleList([
            LocalTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                temporal_window=temporal_window,
                drop_path=drop_path_rates[i],
                use_temporal=use_temporal,
                use_memory=use_memory,
                use_global_fusion=use_global_fusion,
                global_dim=global_dim,
                fp8_cfg=fp8_cfg,
            )
            for i in range(depth)
        ])

    def forward(self, x: Tensor, memory: Optional[Tensor], valid_mask: Optional[Tensor], global_tokens: Optional[Tensor] = None) -> Tensor:
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = _checkpoint(block, x, memory, valid_mask, global_tokens, use_reentrant=False)
            else:
                x = block(x, memory=memory, valid_mask=valid_mask, global_tokens=global_tokens)
        return x


class PatchDownsample(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)

    def forward(self, x: Tensor) -> Tensor:
        b, t, h, w, c = x.shape
        x = x.reshape(b * t, h, w, c).permute(0, 3, 1, 2)
        x = self.proj(x)
        h2, w2 = x.shape[-2:]
        return x.permute(0, 2, 3, 1).reshape(b, t, h2, w2, -1)


class FramePatchEmbed(nn.Module):
    def __init__(self, in_channels: int, embed_dim: int, patch_size: int = 8) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: Tensor) -> Tensor:
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.proj(x)
        h1, w1 = x.shape[-2:]
        return x.permute(0, 2, 3, 1).reshape(b, t, h1, w1, -1)


class DecoderFPN(nn.Module):
    def __init__(self, in_dims: Sequence[int], out_dim: int = 256) -> None:
        super().__init__()
        if len(in_dims) != 4:
            raise ValueError("DecoderFPN expects 4 scales")
        self.lateral = nn.ModuleList([nn.Conv2d(dim, out_dim, 1) for dim in in_dims])
        self.smooth = nn.ModuleList([
            nn.Sequential(nn.Conv2d(out_dim, out_dim, 3, padding=1), nn.GELU(), nn.Conv2d(out_dim, out_dim, 3, padding=1))
            for _ in in_dims
        ])

    def forward(self, feats: Sequence[Tensor]) -> Tensor:
        x = [f.reshape(f.shape[0] * f.shape[1], f.shape[2], f.shape[3], f.shape[-1]).permute(0, 3, 1, 2) for f in feats]
        p = [lat(v) for lat, v in zip(self.lateral, x)]
        p4 = self.smooth[3](p[3])
        p3 = self.smooth[2](p[2] + F.interpolate(p4, size=p[2].shape[-2:], mode="bilinear", align_corners=False))
        p2 = self.smooth[1](p[1] + F.interpolate(p3, size=p[1].shape[-2:], mode="bilinear", align_corners=False))
        p1 = self.smooth[0](p[0] + F.interpolate(p2, size=p[0].shape[-2:], mode="bilinear", align_corners=False))
        return p1


class LocalTileEncoder(nn.Module):
    """Complete local tile encoder with global fusion and memory.

    This wraps the 4-stage transformer encoder, subject memory bank,
    decoder FPN, and all prediction heads into a single module.
    """

    def __init__(
        self,
        in_channels: int = 14,
        patch_size: int = 8,
        embed_dims: Sequence[int] = (128, 256, 384, 512),
        depths: Sequence[int] = (2, 2, 6, 2),
        num_heads: Sequence[int] = (4, 8, 12, 16),
        window_sizes: Sequence[int] = (8, 8, 4, 4),
        memory_tokens: int = 64,
        temporal_window: int = 5,
        drop_path_rate: float = 0.1,
        gradient_checkpointing: bool = False,
        global_context_dim: int = 256,
        decoder_out_dim: int = 256,
        fp8_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.fp8_cfg = fp8_cfg

        self.patch_embed = FramePatchEmbed(in_channels, embed_dims[0], patch_size)

        total_blocks = sum(depths)
        dpr = torch.linspace(0.0, drop_path_rate, total_blocks).tolist()

        dp_cursor = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            depth = depths[i]
            stage = LocalTransformerStage(
                dim=embed_dims[i],
                depth=depth,
                num_heads=num_heads[i],
                window_size=window_sizes[i],
                temporal_window=temporal_window,
                drop_path_rates=dpr[dp_cursor:dp_cursor + depth],
                use_temporal=(i >= 2),
                use_memory=(i == 3),
                use_global_fusion=(i >= 2),
                global_dim=global_context_dim,
                gradient_checkpointing=gradient_checkpointing,
                fp8_cfg=fp8_cfg,
            )
            self.stages.append(stage)
            dp_cursor += depth

        self.downsamples = nn.ModuleList([
            PatchDownsample(embed_dims[i], embed_dims[i + 1]) for i in range(3)
        ])

        self.memory_bank = SubjectMemoryBank(dim=embed_dims[-1], memory_tokens=memory_tokens, fp8_cfg=fp8_cfg)
        self.decoder = DecoderFPN(in_dims=embed_dims, out_dim=decoder_out_dim)

    def encode(
        self,
        x: Tensor,
        coarse_alpha_init: Tensor,
        video_rgb: Tensor,
        valid_mask: Optional[Tensor] = None,
        global_tokens: Optional[Tensor] = None,
    ) -> List[Tensor]:
        """Run the encoder stages only, returning multi-scale features.

        Use this instead of ``forward()`` when you need to modify stage
        features (e.g. with reference memory) before running the decoder.

        Returns:
            stage_features: list of [B, T, H_i, W_i, C_i] for each stage
        """
        x = self.patch_embed(x)

        if valid_mask is not None:
            x = x * valid_mask[:, :, None, None, None].to(x.dtype)

        feats: List[Tensor] = []
        memory: Optional[Tensor] = None

        for i, stage in enumerate(self.stages):
            if i == 3:
                memory = self.memory_bank(x, coarse_alpha_init, video_rgb, valid_mask)

            x = stage(x, memory=memory, valid_mask=valid_mask, global_tokens=global_tokens)
            feats.append(x)

            if i < 3:
                x = self.downsamples[i](x)

        return feats

    def forward(
        self,
        x: Tensor,
        coarse_alpha_init: Tensor,
        video_rgb: Tensor,
        valid_mask: Optional[Tensor] = None,
        global_tokens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """
        Args:
            x: [B, T, C_in, H, W] prepared input (already includes all channels)
            coarse_alpha_init: [B, 1, H, W]
            video_rgb: [B, T, 3, H, W]
            valid_mask: [B, T] optional
            global_tokens: [B, M, C_global] from global context branch

        Returns:
            decoded: [B*T, decoder_out_dim, Hp, Wp] FPN output at 1/patch_size resolution
            stage_features: list of [B, T, H_i, W_i, C_i] for skip connections
        """
        feats = self.encode(x, coarse_alpha_init, video_rgb, valid_mask, global_tokens)
        decoded = self.decoder(feats)
        return decoded, feats
