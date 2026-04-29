"""Native windowed transformer detail refiner for V3.

The head runs at input resolution, uses per-pixel linear projection plus
shifted-window attention blocks, and predicts bounded residual corrections.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _as_list(value: Optional[Sequence[int]], depth: int, default: int) -> Tuple[int, ...]:
    if value is None:
        return tuple(default for _ in range(depth))
    vals = tuple(int(v) for v in value)
    if len(vals) == depth:
        return vals
    if len(vals) == 1:
        return tuple(vals[0] for _ in range(depth))
    raise ValueError(f"Expected {depth} window sizes, got {len(vals)}")


def _as_type_list(value: Optional[Sequence[str]], depth: int, default: str) -> Tuple[str, ...]:
    if value is None:
        return tuple(default for _ in range(depth))
    vals = tuple(str(v).strip().lower() for v in value)
    if len(vals) == depth:
        return vals
    if len(vals) == 1:
        return tuple(vals[0] for _ in range(depth))
    raise ValueError(f"Expected {depth} attention types, got {len(vals)}")


def _window_partition(x: Tensor, window_size: int) -> Tensor:
    b, h, w, c = x.shape
    x = x.reshape(b, h // window_size, window_size, w // window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, c)


def _window_unpartition(windows: Tensor, window_size: int, b: int, h: int, w: int, c: int) -> Tensor:
    x = windows.reshape(b, h // window_size, w // window_size, window_size, window_size, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)


def _pad_nhwc_to_window(x: Tensor, window_size: int) -> Tuple[Tensor, Tuple[int, int]]:
    h, w = x.shape[1], x.shape[2]
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return F.pad(x, (0, 0, 0, pad_w, 0, pad_h)), (pad_h, pad_w)


def _feature_map(x: Tensor, name: str, eps: float) -> Tensor:
    if name == "elu_plus_one":
        return F.elu(x) + 1.0
    if name == "softplus":
        return F.softplus(x) + eps
    if name == "relu_squared":
        y = F.relu(x)
        return y * y + eps
    raise ValueError(f"Unsupported linear feature map: {name}")


def _apply_rotary(part: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    even = part[..., 0::2]
    odd = part[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return rotated.flatten(-2)


def _apply_2d_rope(q: Tensor, k: Tensor, window_size: int) -> Tuple[Tensor, Tensor]:
    head_dim = q.shape[-1]
    axis_dim = (head_dim // 4) * 2
    if axis_dim < 2:
        return q, k

    device = q.device
    dtype = q.dtype
    coords = torch.arange(window_size, device=device, dtype=torch.float32)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    yy = yy.reshape(-1)
    xx = xx.reshape(-1)

    inv_freq = 1.0 / (10000 ** (torch.arange(0, axis_dim, 2, device=device, dtype=torch.float32) / axis_dim))
    y_freq = yy[:, None] * inv_freq[None, :]
    x_freq = xx[:, None] * inv_freq[None, :]
    y_sin = y_freq.sin()[None, None, :, :].to(dtype=dtype)
    y_cos = y_freq.cos()[None, None, :, :].to(dtype=dtype)
    x_sin = x_freq.sin()[None, None, :, :].to(dtype=dtype)
    x_cos = x_freq.cos()[None, None, :, :].to(dtype=dtype)

    q_y = _apply_rotary(q[..., :axis_dim], y_sin, y_cos)
    q_x = _apply_rotary(q[..., axis_dim:2 * axis_dim], x_sin, x_cos)
    k_y = _apply_rotary(k[..., :axis_dim], y_sin, y_cos)
    k_x = _apply_rotary(k[..., axis_dim:2 * axis_dim], x_sin, x_cos)
    if 2 * axis_dim == head_dim:
        return torch.cat((q_y, q_x), dim=-1), torch.cat((k_y, k_x), dim=-1)
    return (
        torch.cat((q_y, q_x, q[..., 2 * axis_dim:]), dim=-1),
        torch.cat((k_y, k_x, k[..., 2 * axis_dim:]), dim=-1),
    )


class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float) -> None:
        super().__init__()
        hidden = max(dim, int(dim * mlp_ratio))
        self.fc1 = nn.Linear(dim, hidden * 2)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(F.silu(gate) * value)


class NativeWindowAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        window_size: int,
        attention_backend: str = "linear",
        linear_feature_map: str = "elu_plus_one",
        linear_attention_eps: float = 1.0e-6,
        use_2d_rope: bool = True,
        performer_features: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.inner_dim = self.num_heads * self.head_dim
        self.window_size = int(window_size)
        self.attention_backend = str(attention_backend).strip().lower()
        self.linear_feature_map = str(linear_feature_map).strip().lower()
        self.linear_attention_eps = float(linear_attention_eps)
        self.use_2d_rope = bool(use_2d_rope)
        self.performer_features = int(performer_features or self.head_dim)

        self.qkv = nn.Linear(self.dim, self.inner_dim * 3)
        self.proj = nn.Linear(self.inner_dim, self.dim)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        proj = torch.randn(self.num_heads, self.head_dim, self.performer_features, generator=gen)
        proj = F.normalize(proj, dim=1)
        self.register_buffer("performer_projection", proj, persistent=False)

        self.last_denominator_min: Optional[Tensor] = None
        self.last_denominator_mean: Optional[Tensor] = None

    def _linear_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        q_phi = _feature_map(q, self.linear_feature_map, self.linear_attention_eps)
        k_phi = _feature_map(k, self.linear_feature_map, self.linear_attention_eps)
        return self._positive_feature_attention(q_phi, k_phi, v)

    def _performer_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        proj = self.performer_projection.to(device=q.device, dtype=torch.float32)
        qf = q.float()
        kf = k.float()
        q_proj = torch.einsum("bhnd,hdm->bhnm", qf, proj)
        k_proj = torch.einsum("bhnd,hdm->bhnm", kf, proj)
        q_norm = 0.5 * qf.square().sum(dim=-1, keepdim=True)
        k_norm = 0.5 * kf.square().sum(dim=-1, keepdim=True)
        q_phi = torch.exp((q_proj - q_norm).clamp(min=-30.0, max=30.0))
        k_phi = torch.exp((k_proj - k_norm).clamp(min=-30.0, max=30.0))
        scale = self.performer_features ** -0.5
        return self._positive_feature_attention(q_phi * scale, k_phi * scale, v)

    def _positive_feature_attention(self, q_phi: Tensor, k_phi: Tensor, v: Tensor) -> Tensor:
        qf = q_phi.float()
        kf = k_phi.float()
        vf = v.float()
        kv = torch.einsum("bhnd,bhne->bhde", kf, vf)
        k_sum = kf.sum(dim=-2)
        denom = torch.einsum("bhnd,bhd->bhn", qf, k_sum).clamp_min(self.linear_attention_eps)
        self.last_denominator_min = denom.detach().amin()
        self.last_denominator_mean = denom.detach().mean()
        out = torch.einsum("bhnd,bhde->bhne", qf, kv) / denom.unsqueeze(-1)
        return out.to(dtype=v.dtype)

    def _softmax_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        if self.attention_backend == "softmax_flash":
            try:
                from flash_attn import flash_attn_func  # type: ignore

                q_f = q.transpose(1, 2).contiguous()
                k_f = k.transpose(1, 2).contiguous()
                v_f = v.transpose(1, 2).contiguous()
                return flash_attn_func(q_f, k_f, v_f).transpose(1, 2)
            except Exception:
                pass
        return F.scaled_dot_product_attention(q, k, v)

    def forward(self, windows: Tensor) -> Tensor:
        n_win, n_tok, _ = windows.shape
        qkv = self.qkv(windows).reshape(n_win, n_tok, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.use_2d_rope and n_tok == self.window_size * self.window_size:
            q, k = _apply_2d_rope(q, k, self.window_size)

        if self.attention_backend in {"linear", "auto"}:
            out = self._linear_attention(q, k, v)
        elif self.attention_backend == "performer":
            out = self._performer_attention(q, k, v)
        elif self.attention_backend in {"softmax", "softmax_flash", "softmax_sdpa"}:
            out = self._softmax_attention(q, k, v)
        else:
            raise ValueError(f"Unsupported attention backend: {self.attention_backend}")

        out = out.transpose(1, 2).reshape(n_win, n_tok, self.inner_dim)
        return self.proj(out)


class NativeWindowLinearBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        head_dim: int,
        window_size: int,
        shift_size: int,
        mlp_ratio: float,
        attention_backend: str,
        linear_feature_map: str,
        linear_attention_eps: float,
        use_2d_rope: bool,
    ) -> None:
        super().__init__()
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = NativeWindowAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            window_size=window_size,
            attention_backend=attention_backend,
            linear_feature_map=linear_feature_map,
            linear_attention_eps=linear_attention_eps,
            use_2d_rope=use_2d_rope,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLUMLP(dim, mlp_ratio)

    def forward(self, x: Tensor) -> Tensor:
        b, h, w, c = x.shape
        shortcut = x
        y = self.norm1(x)
        y, pad_hw = _pad_nhwc_to_window(y, self.window_size)
        pad_h, pad_w = pad_hw
        hp, wp = y.shape[1], y.shape[2]

        shifted = self.shift_size > 0
        if shifted:
            y = torch.roll(y, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))

        windows = _window_partition(y, self.window_size)
        windows = self.attn(windows)
        y = _window_unpartition(windows, self.window_size, b, hp, wp, c)

        if shifted:
            y = torch.roll(y, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        if pad_h or pad_w:
            y = y[:, :h, :w, :]

        x = shortcut + y
        return x + self.mlp(self.norm2(x))


class NativeWindowLinearRefineHead(nn.Module):
    """Native-resolution windowed transformer refine head."""

    def __init__(
        self,
        in_channels: int = 9,
        hidden_channels: int = 64,
        num_blocks: int = 8,
        max_alpha_delta: float = 0.03,
        max_fg_delta: float = 0.02,
        max_spill_delta: float = 0.15,
        predict_spill: bool = True,
        global_context_dim: int = 0,
        num_heads: int = 4,
        head_dim: int = 16,
        mlp_ratio: float = 2.0,
        window_sizes: Optional[Sequence[int]] = None,
        shift_every_other_block: bool = True,
        attention_backend: str = "linear",
        attention_types: Optional[Sequence[str]] = None,
        linear_feature_map: str = "elu_plus_one",
        linear_attention_eps: float = 1.0e-6,
        use_2d_rope: bool = True,
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.max_alpha_delta = float(max_alpha_delta)
        self.max_fg_delta = float(max_fg_delta)
        self.max_spill_delta = float(max_spill_delta)
        self.predict_spill = bool(predict_spill)
        self.global_context_dim = int(global_context_dim)
        self.embed_dim = int(hidden_channels)
        self.depth = int(num_blocks)
        self.use_checkpointing = bool(use_checkpointing)
        self.base_in_channels = int(in_channels)

        actual_in = self.base_in_channels
        if self.global_context_dim > 0:
            self.global_proj = nn.Sequential(
                nn.Linear(self.global_context_dim, self.embed_dim),
                nn.GELU(),
            )
            actual_in += self.embed_dim
        else:
            self.global_proj = None

        self.input_proj = nn.Linear(actual_in, self.embed_dim)

        ws = _as_list(window_sizes, self.depth, 16)
        if window_sizes is None and self.depth == 8:
            ws = (16, 16, 24, 24, 24, 24, 16, 16)

        attn_types = _as_type_list(attention_types, self.depth, attention_backend)
        blocks = []
        for i in range(self.depth):
            block_backend = attn_types[i]
            if block_backend == "softmax":
                block_backend = "softmax_flash" if attention_backend in {"auto", "softmax_flash"} else "softmax_sdpa"
            shift = ws[i] // 2 if shift_every_other_block and i % 2 == 1 else 0
            blocks.append(
                NativeWindowLinearBlock(
                    dim=self.embed_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    window_size=ws[i],
                    shift_size=shift,
                    mlp_ratio=mlp_ratio,
                    attention_backend=block_backend,
                    linear_feature_map=linear_feature_map,
                    linear_attention_eps=linear_attention_eps,
                    use_2d_rope=use_2d_rope,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.LayerNorm(self.embed_dim)
        self.alpha_delta = nn.Linear(self.embed_dim, 1)
        self.fg_delta = nn.Linear(self.embed_dim, 3)
        self.uncertainty_delta = nn.Linear(self.embed_dim, 1)
        self.spill_delta = nn.Linear(self.embed_dim, 1) if self.predict_spill else None
        self.gate_head = nn.Sequential(nn.Linear(self.embed_dim, 1), nn.Sigmoid())

    def _global_spatial(self, global_tokens: Tensor, bt: int, h: int, w: int, dtype: torch.dtype) -> Tensor:
        b = global_tokens.shape[0]
        t = max(1, bt // b)
        g_pooled = global_tokens.mean(dim=1)
        g_proj = self.global_proj(g_pooled).to(dtype=dtype)  # type: ignore[operator]
        g_spatial = g_proj[:, None, None, :].expand(b, h, w, -1)
        return g_spatial.unsqueeze(1).expand(b, t, h, w, -1).reshape(bt, h, w, -1)

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
        native_features: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        bt = rgb.shape[0]
        h, w = rgb.shape[-2:]

        parts = [rgb, coarse_alpha, coarse_fg]
        if uncertainty is not None:
            parts.append(uncertainty)
        else:
            parts.append(torch.zeros(bt, 1, h, w, device=rgb.device, dtype=rgb.dtype))
        if self.base_in_channels >= 9:
            if coarse_spill is not None:
                parts.append(coarse_spill)
            else:
                parts.append(torch.zeros(bt, 1, h, w, device=rgb.device, dtype=rgb.dtype))
        if native_features is not None:
            parts.append(native_features)

        x_in = torch.cat(parts, dim=1).permute(0, 2, 3, 1)
        if self.global_proj is not None:
            if global_tokens is not None:
                global_part = self._global_spatial(global_tokens, bt, h, w, x_in.dtype)
            else:
                global_part = torch.zeros(bt, h, w, self.embed_dim, device=rgb.device, dtype=x_in.dtype)
            x_in = torch.cat([x_in, global_part], dim=-1)

        x = self.input_proj(x_in)
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.norm(x)

        gate = self.gate_head(x).permute(0, 3, 1, 2)
        alpha_gate = gate
        fg_gate = gate
        spill_gate = gate
        uncertainty_gate = gate
        if refine_mask is not None:
            alpha_gate = alpha_gate * refine_mask
            fg_gate = fg_gate * refine_mask
            spill_gate = spill_gate * refine_mask
            uncertainty_gate = uncertainty_gate * refine_mask
        if alpha_refine_mask is not None:
            alpha_gate = gate * alpha_refine_mask
            uncertainty_gate = gate * alpha_refine_mask
        if fg_refine_mask is not None:
            fg_gate = gate * fg_refine_mask
        if spill_refine_mask is not None:
            spill_gate = gate * spill_refine_mask

        alpha_delta_raw = torch.tanh(self.alpha_delta(x)).permute(0, 3, 1, 2) * self.max_alpha_delta
        fg_delta_raw = torch.tanh(self.fg_delta(x)).permute(0, 3, 1, 2) * self.max_fg_delta
        uncertainty_delta_raw = torch.tanh(self.uncertainty_delta(x)).permute(0, 3, 1, 2) * self.max_alpha_delta

        alpha_delta = alpha_delta_raw * alpha_gate
        fg_delta = fg_delta_raw * fg_gate
        uncertainty_delta = uncertainty_delta_raw * uncertainty_gate

        alpha_refined = (coarse_alpha + alpha_delta).clamp(0.0, 1.0)
        fg_refined = (coarse_fg + fg_delta).clamp(0.0, 1.0)

        out: Dict[str, Tensor] = {
            "alpha_refined": alpha_refined,
            "fg_refined": fg_refined,
            "native_alpha_delta_pred": alpha_delta_raw,
            "native_fg_delta_pred": fg_delta_raw,
            "native_uncertainty_delta_pred": uncertainty_delta_raw,
            "refine_gate": torch.maximum(alpha_gate, fg_gate),
        }

        if uncertainty is not None:
            out["uncertainty_refined"] = (uncertainty + uncertainty_delta).clamp(0.0, 1.0)

        if self.spill_delta is not None:
            spill_delta_raw = torch.tanh(self.spill_delta(x)).permute(0, 3, 1, 2) * self.max_spill_delta
            spill_delta = spill_delta_raw * spill_gate
            coarse_s = coarse_spill if coarse_spill is not None else torch.zeros_like(coarse_alpha)
            out["spill_refined"] = coarse_s + spill_delta
            out["native_spill_delta_pred"] = spill_delta_raw

        assert alpha_refined.shape[-2:] == coarse_alpha.shape[-2:]
        assert fg_refined.shape[-2:] == coarse_fg.shape[-2:]
        return out


class NativeDetailRefiner(NativeWindowLinearRefineHead):
    """Backward-compatible name for the native windowed linear-attention head."""
