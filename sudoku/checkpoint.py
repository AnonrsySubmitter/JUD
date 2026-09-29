"""Checkpoint utilities."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from .ema import EMA


def checkpoint_path(output: Path, step: int) -> Path:
    return output / f"checkpoint-{step:06d}.pt"


def latest_checkpoint(output: Path) -> Path | None:
    candidates = sorted(output.glob("checkpoint-*.pt"))
    return candidates[-1] if candidates else None


def save_checkpoint(
    output: Path,
    step: int,
    epoch: int,
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    config: dict[str, object],
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_path(output, step)
    temporary = destination.with_suffix(".tmp")
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "config": config,
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        temporary,
    )
    os.replace(temporary, destination)
    return destination


def load_checkpoint(path: Path, device: torch.device | str = "cpu") -> dict[str, object]:
    return torch.load(path, map_location=device, weights_only=False)
