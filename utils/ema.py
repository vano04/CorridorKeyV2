"""Exponential moving average helpers for model training."""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


class ModelEma:
    """Track a shadow EMA copy of a module state_dict."""

    def __init__(self, model: nn.Module, *, decay: float = 0.9999, device: torch.device | str | None = None) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be >= 0.0 and < 1.0")
        self.decay = float(decay)
        self.num_updates = 0
        self.device = torch.device(device) if device is not None else None
        self.shadow: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.reset(model)

    def reset(self, model: nn.Module) -> None:
        self.shadow.clear()
        for key, value in unwrap_model(model).state_dict().items():
            tensor = value.detach()
            if self.device is not None:
                tensor = tensor.to(device=self.device)
            self.shadow[key] = tensor.clone()
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        model_state = unwrap_model(model).state_dict()
        for key, value in model_state.items():
            value_detached = value.detach()
            if key not in self.shadow:
                tensor = value_detached
                if self.device is not None:
                    tensor = tensor.to(device=self.device)
                self.shadow[key] = tensor.clone()
                continue

            shadow = self.shadow[key]
            if self.device is not None and shadow.device != self.device:
                shadow = shadow.to(device=self.device)
                self.shadow[key] = shadow
            value_for_ema = value_detached.to(device=shadow.device)
            if torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(value_for_ema, alpha=1.0 - self.decay)
            else:
                shadow.copy_(value_for_ema)

    def state_dict(self) -> Dict[str, object]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": OrderedDict((k, v.detach().clone()) for k, v in self.shadow.items()),
        }

    def load_state_dict(self, state_dict: Dict[str, object]) -> None:
        self.decay = float(state_dict.get("decay", self.decay))
        self.num_updates = int(state_dict.get("num_updates", 0))
        shadow = state_dict.get("shadow", state_dict)
        if not isinstance(shadow, dict):
            raise ValueError("EMA checkpoint state must contain a tensor mapping")
        self.shadow = OrderedDict()
        for key, value in shadow.items():
            if not torch.is_tensor(value):
                continue
            tensor = value.detach()
            if self.device is not None:
                tensor = tensor.to(device=self.device)
            self.shadow[str(key)] = tensor.clone()

    def model_state_dict(self) -> "OrderedDict[str, torch.Tensor]":
        return OrderedDict((k, v.detach().clone()) for k, v in self.shadow.items())
