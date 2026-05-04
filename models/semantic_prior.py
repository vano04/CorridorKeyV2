"""DINO-style semantic foreground prior for V3.

The module is intentionally DINO-style rather than a hard dependency on a
downloaded DINO checkpoint: patchify image features, run global self-attention over
patches, summarise semantic tokens, and supervise a dense fg/bg logit map from
alpha. It gives global context semantic subject awareness without feeding it a
decaying alpha seed at inference time.
"""
from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _checkpoint


class DinoStyleSemanticPrior(nn.Module):
    """Small ViT-style semantic prior trained from fg/bg alpha labels."""

    def __init__(
        self,
        *,
        dim: int,
        global_dim: int,
        patch_size: int,
        depth: int,
        num_heads: int,
        num_tokens: int,
        dropout: float = 0.0,
        pos_grid_size: int = 32,
        grayscale_input: bool = False,
        grayscale_prob: float = 0.0,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.global_dim = int(global_dim)
        self.patch_size = int(patch_size)
        self.num_tokens = int(num_tokens)
        self.grayscale_input = bool(grayscale_input)
        self.grayscale_prob = float(max(0.0, min(1.0, grayscale_prob)))
        self.gradient_checkpointing = bool(gradient_checkpointing)

        self.patch_embed = nn.Conv2d(3, self.dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, self.dim, pos_grid_size, pos_grid_size) * 0.02)
        self.blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=self.dim,
                    nhead=int(num_heads),
                    dim_feedforward=self.dim * 4,
                    dropout=float(dropout),
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(int(depth))
            ]
        )
        self.norm = nn.LayerNorm(self.dim)
        self.logit_head = nn.Conv2d(self.dim, 1, kernel_size=1)

        self.token_proj = nn.Linear(self.dim, self.global_dim) if self.dim != self.global_dim else nn.Identity()
        self.summary_tokens = nn.Parameter(torch.randn(self.num_tokens, self.global_dim) * 0.02)
        self.summary_attn = nn.MultiheadAttention(
            embed_dim=self.global_dim,
            num_heads=int(num_heads) if self.global_dim % int(num_heads) == 0 else 1,
            dropout=float(dropout),
            batch_first=True,
        )
        self.summary_norm = nn.LayerNorm(self.global_dim)

    def forward(self, video_rgb: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            video_rgb: [B, T, 3, H, W]

        Returns:
            semantic_tokens: [B, M, global_dim]
            fg_logits:       [B, T, 1, H, W]
        """
        b, t, c, h, w = video_rgb.shape
        x_bt = video_rgb
        if self.grayscale_input or (self.training and self.grayscale_prob > 0.0):
            gray = (
                x_bt[:, :, 0:1] * 0.299
                + x_bt[:, :, 1:2] * 0.587
                + x_bt[:, :, 2:3] * 0.114
            )
            gray = gray.expand(-1, -1, 3, -1, -1)
            if self.grayscale_input:
                x_bt = gray
            else:
                use_gray = (
                    torch.rand((b, 1, 1, 1, 1), device=video_rgb.device)
                    < self.grayscale_prob
                )
                x_bt = torch.where(use_gray, gray, x_bt)
        x = x_bt.reshape(b * t, c, h, w)
        patches = self.patch_embed(x)
        hp, wp = patches.shape[-2:]
        pos = F.interpolate(self.pos_embed, size=(hp, wp), mode="bicubic", align_corners=False)
        patches = patches + pos.to(dtype=patches.dtype)

        tokens = patches.flatten(2).transpose(1, 2)
        for block in self.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                tokens = _checkpoint(block, tokens, use_reentrant=False)
            else:
                tokens = block(tokens)
        tokens = self.norm(tokens)

        patch_features = tokens.transpose(1, 2).reshape(b * t, self.dim, hp, wp)
        fg_logits = self.logit_head(patch_features)
        fg_logits = F.interpolate(fg_logits, size=(h, w), mode="bilinear", align_corners=False)
        fg_logits = fg_logits.reshape(b, t, 1, h, w)

        semantic_features = self.token_proj(tokens).reshape(b, t * hp * wp, self.global_dim)
        queries = self.summary_tokens.unsqueeze(0).expand(b, -1, -1)
        summary, _ = self.summary_attn(
            query=queries,
            key=semantic_features,
            value=semantic_features,
            need_weights=False,
        )
        return self.summary_norm(summary + queries), fg_logits
