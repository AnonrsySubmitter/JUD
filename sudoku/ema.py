# Adapted from S-FLM (https://github.com/jdeschena/s-flm), Apache-2.0.
"""EMA with early-update warmup."""

from __future__ import annotations

import torch
from torch import nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.updates = 0
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - decay)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.copy_(self.shadow[name])

    def state_dict(self) -> dict[str, object]:
        return {"decay": self.decay, "updates": self.updates, "shadow": self.shadow}

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.decay = float(state["decay"])
        self.updates = int(state["updates"])
        shadow = state["shadow"]
        if not isinstance(shadow, dict):
            raise TypeError("invalid EMA shadow state")
        self.shadow = shadow


def load_ema_model(model: nn.Module, checkpoint: dict[str, object]) -> None:
    model_state = checkpoint.get("model")
    if not isinstance(model_state, dict):
        raise TypeError("checkpoint has no model state")
    model.load_state_dict(model_state)
    ema_state = checkpoint.get("ema")
    if isinstance(ema_state, dict):
        ema = EMA(model)
        ema.load_state_dict(ema_state)
        ema.copy_to(model)
