"""Inference context cache for V3.

Caches global context tokens and reference memory tokens across temporal
windows during inference, avoiding redundant computation.
"""
from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


class ContextCache:
    """Manages cached global and reference tokens during inference.

    Usage:
        cache = ContextCache()

        for window in temporal_windows:
            if cache.should_refresh_global(window):
                global_tokens = model.global_context(window_video)
                cache.update_global(global_tokens)

            ref_tokens = cache.get_ref_tokens()
            output = model.forward(..., global_tokens=cache.global_tokens, ref_tokens=ref_tokens)
            cache.update_ref_tokens(output.get('ref_tokens'))
    """

    def __init__(self) -> None:
        self.global_tokens: Optional[Tensor] = None
        self.ref_tokens: Optional[Tensor] = None
        self._window_count: int = 0
        self._global_refresh_interval: int = 1

    def should_refresh_global(self, window_idx: int) -> bool:
        """Whether to recompute global context for this window."""
        if self.global_tokens is None:
            return True
        return window_idx % self._global_refresh_interval == 0

    def update_global(self, tokens: Tensor) -> None:
        self.global_tokens = tokens.detach()

    def get_ref_tokens(self) -> Optional[Tensor]:
        return self.ref_tokens

    def update_ref_tokens(self, tokens: Optional[Tensor]) -> None:
        if tokens is not None:
            self.ref_tokens = tokens.detach()

    def reset(self) -> None:
        self.global_tokens = None
        self.ref_tokens = None
        self._window_count = 0
