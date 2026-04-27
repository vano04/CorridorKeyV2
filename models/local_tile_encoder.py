"""Local tile encoder for V3.

The main backbone transformer that processes 1024x1024 tiles at native
resolution. Rewritten from V1's encoder path with V3-specific additions:
- Extended input channels (green priors, coordinate maps)
- Cross-attention fusion from local tile tokens to global context tokens
- Integration with reference memory at the deepest stage
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

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

    def forward(self, x: Tensor, valid_mask: Optional[Tensor] = None) -> Tensor:
        b, t, h, w, c = x.shape
        s = h * w
        x_seq = x.permute(0, 2, 3, 1, 4).reshape(b * s, t, c)
        qkv = self.qkv(x_seq).reshape(b * s, t, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 2, 1, 3).reshape(b * s, t, c)
        out = self.proj(out)
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
        rgb0 = F.interpolate(video[:, 0], size=(h, w), mode="bilinear", align_corners=False)

        # Build init tokens
        feat0 = features[:, 0]  # [B, H, W, C]
        fg_mask = (alpha0.permute(0, 2, 3, 1) > 0.7).to(feat0.dtype)
        pooled = (feat0 * fg_mask).sum(dim=(1, 2)) / fg_mask.sum(dim=(1, 2)).clamp_min(EPS)
        base = self.base_tokens.unsqueeze(0).expand(b, -1, -1)
        memory = base.clone()
        memory[:, 0] = pooled

        frame_summary = features.mean(dim=(2, 3))
        all_summaries = self.update_proj(frame_summary.reshape(b * t, c)).reshape(b, t, c)
        for i in range(t):
            summary = all_summaries[:, i].unsqueeze(1).expand(-1, self.memory_tokens, -1)
            confidence = torch.full((b, 1, 1), 1.0 if i == 0 else 0.25, dtype=memory.dtype, device=memory.device)
            if valid_mask is not None:
                confidence = confidence * valid_mask[:, i:i+1].to(memory.dtype).view(b, 1, 1)
            memory = self.update_gate(memory, summary, confidence)
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

        decoded = self.decoder(feats)
        return decoded, feats
