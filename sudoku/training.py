"""Single-device training for N/L MSE and direct posterior CE."""

import json
import math
import signal as signal_module
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler

from .checkpoint import latest_checkpoint, load_checkpoint, save_checkpoint
from .data import SudokuDataset
from .ema import EMA
from .layout import PROMPT_LENGTH, extract_solution_cells
from .model import JUDDiT, parameter_count
from .process import ProcessConfig, corrupt, mse_loss
from .runtime import initialize

LR_SCHEDULES = ("flat", "cosine")


@dataclass(frozen=True)
class TrainConfig:
    objective: str
    data: Path
    output: Path
    steps: int = 10_000
    train_examples: int = 48_000
    global_batch: int = 256
    learning_rate: float = 3e-4
    warmup_steps: int = 2_500
    lr_schedule: str = "cosine"
    min_learning_rate: float = 3e-5
    lr_horizon: int | None = 15_000
    weight_decay: float = 0.05
    digit_permutation: bool = False
    ema_decay: float = 0.9999
    gradient_clip: float = 1.0
    seed: int = 101
    workers: int = 4
    checkpoint_every: int = 5_000
    log_every: int = 100
    resume: bool = True


def _lr_horizon(config: TrainConfig) -> int | None:
    if config.lr_schedule == "cosine":
        return config.steps if config.lr_horizon is None else config.lr_horizon
    return config.lr_horizon


def _validate_lr_schedule(config: TrainConfig) -> None:
    if config.lr_schedule not in LR_SCHEDULES:
        raise ValueError(f"unknown learning-rate schedule: {config.lr_schedule!r}")
    if config.learning_rate < 0.0:
        raise ValueError("learning rate must be nonnegative")
    if config.warmup_steps < 1:
        raise ValueError("warmup needs at least one step")
    if config.lr_schedule == "flat":
        if config.min_learning_rate != 0.0 or config.lr_horizon is not None:
            raise ValueError("minimum learning rate and horizon require the cosine schedule")
        return
    if not 0.0 <= config.min_learning_rate <= config.learning_rate:
        raise ValueError("minimum learning rate must be between zero and the maximum")
    horizon = _lr_horizon(config)
    if horizon is None or horizon <= config.warmup_steps:
        raise ValueError("cosine horizon must be greater than warmup steps")


def _learning_rate(config: TrainConfig, step: int) -> float:
    """Return the learning rate for a one-based optimizer step."""
    if step < 1:
        raise ValueError("optimizer step must be positive")
    if step <= config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    if config.lr_schedule == "flat":
        return config.learning_rate
    horizon = _lr_horizon(config)
    if horizon is None:
        raise ValueError("cosine schedule has no horizon")
    progress = min((step - config.warmup_steps) / (horizon - config.warmup_steps), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + (config.learning_rate - config.min_learning_rate) * cosine


def _optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.AdamW:
    options = {
        "lr": config.learning_rate,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    if config.weight_decay == 0.0:
        return torch.optim.AdamW(model.parameters(), weight_decay=0.0, **options)
    decay = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim >= 2
    ]
    no_decay = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.ndim < 2
    ]
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, weight_decay=0.0, **options)


def _permute_digits(
    sequences: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    permutations = (
        torch.multinomial(
            torch.ones((len(sequences), 9), device=sequences.device),
            9,
            replacement=False,
            generator=generator,
        )
        .add_(1)
        .to(sequences.dtype)
    )
    lookup = torch.cat(
        [
            torch.zeros((len(sequences), 1), dtype=sequences.dtype, device=sequences.device),
            permutations,
        ],
        dim=1,
    )
    mapped = lookup.gather(1, sequences.clamp(0, 9).long())
    is_cell = (sequences >= 0) & (sequences <= 9)
    return torch.where(is_cell, mapped, sequences)


def _maybe_permute_digits(
    sequences: torch.Tensor,
    enabled: bool,
    step_seed: int,
) -> torch.Tensor:
    if not enabled:
        return sequences
    generator = torch.Generator(device=sequences.device).manual_seed(step_seed + 1_000_000_007)
    return _permute_digits(sequences, generator)


def _model_config(config: TrainConfig, process: ProcessConfig) -> dict:
    training = asdict(config)
    training.update(data=str(config.data), output=str(config.output))
    return {
        "objective": config.objective,
        "denoiser_parameterization": "noise-lost",
        "process": asdict(process),
        "architecture": {
            "dimension": 512,
            "condition_dimension": 128,
            "blocks": 8,
            "heads": 8,
            "dropout": 0.1,
        },
        "training": training,
    }


def _load_resume(config, model, ema, optimizer, expected, device):
    path = latest_checkpoint(config.output) if config.resume else None
    if path is None:
        return 0
    checkpoint = load_checkpoint(path, device)
    saved = checkpoint["config"]
    for key in ("objective", "denoiser_parameterization", "process", "architecture"):
        if saved[key] != expected[key]:
            raise ValueError(f"resume mismatch for {key}")
    mutable = {"output", "steps", "workers", "checkpoint_every", "log_every", "resume"}
    for key, value in expected["training"].items():
        if key not in mutable and saved["training"][key] != value:
            raise ValueError(f"resume mismatch for training.{key}")
    model.load_state_dict(checkpoint["model"])
    ema.load_state_dict(checkpoint["ema"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def train(config: TrainConfig) -> None:
    _validate_lr_schedule(config)
    if min(config.steps, config.global_batch, config.checkpoint_every, config.log_every) < 1:
        raise ValueError("steps, batch size, and logging/checkpoint intervals must be positive")
    device = initialize(config.seed)
    process = ProcessConfig()
    model = JUDDiT(config.objective, process).to(device)
    ema = EMA(model, config.ema_decay)
    serialized_config = _model_config(config, process)
    optimizer = _optimizer(model, config)
    step = _load_resume(config, model, ema, optimizer, serialized_config, device)

    dataset = SudokuDataset(config.data / "train.npy", stop=config.train_examples)
    if len(dataset) < config.global_batch:
        raise ValueError("training data must contain at least one full batch")
    # With one replica this preserves the epoch shuffle of the original experiment.
    sampler = DistributedSampler(
        dataset, num_replicas=1, rank=0, shuffle=True, seed=config.seed, drop_last=True
    )
    loader = DataLoader(
        dataset,
        batch_size=config.global_batch,
        sampler=sampler,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.workers > 0,
        drop_last=True,
    )
    epoch = step // len(loader)
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    for _ in range(step % len(loader)):
        next(iterator)
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    signal_module.signal(signal_module.SIGTERM, request_stop)
    signal_module.signal(signal_module.SIGINT, request_stop)
    config.output.mkdir(parents=True, exist_ok=True)
    (config.output / "config.json").write_text(json.dumps(serialized_config, indent=2) + "\n")
    print(
        json.dumps(
            {
                "event": "start",
                "objective": config.objective,
                "parameters": parameter_count(model),
                "step": step,
            }
        ),
        flush=True,
    )
    started = time.monotonic()

    while step < config.steps:
        try:
            sequences = next(iterator)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)
            iterator = iter(loader)
            sequences = next(iterator)
        sequences = sequences.to(device, non_blocking=True)
        step_seed = config.seed * 1_000_003 + step
        torch.manual_seed(step_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed(step_seed)
        sequences = _maybe_permute_digits(sequences, config.digit_permutation, step_seed)
        prompt = sequences[:, :PROMPT_LENGTH]
        clean = extract_solution_cells(sequences).float()
        # Stratified uniform proposal: each batch fills the interval [eps, 1-eps].
        uniform = (
            torch.arange(config.global_batch, device=device)
            + torch.rand(config.global_batch, device=device)
        ) / config.global_batch
        signal = (process.signal_epsilon + (1 - 2 * process.signal_epsilon) * uniform).clamp(
            process.signal_epsilon, 1 - process.signal_epsilon
        )
        noisy, added, missing = corrupt(clean, signal, process.poisson_rate)
        learning_rate = _learning_rate(config, step + 1)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            output = model(prompt, noisy, signal)
        if config.objective == "posterior_ce":
            loss = F.cross_entropy(output.float().reshape(-1, 9), (clean.long() - 1).reshape(-1))
        else:
            loss, _ = mse_loss(output.float(), noisy, signal, added, missing, process)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        ema.update(model)
        step += 1

        if step % config.log_every == 0 or step == 1:
            record = {
                "step": step,
                "epoch": epoch,
                "loss": loss.item(),
                "gradient_norm": gradient_norm.item(),
                "learning_rate": learning_rate,
                "elapsed_seconds": time.monotonic() - started,
            }
            with (config.output / "training.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % config.checkpoint_every == 0 or step == config.steps or stop_requested:
            path = save_checkpoint(
                config.output, step, epoch, model, ema, optimizer, serialized_config
            )
            print(json.dumps({"event": "checkpoint", "path": str(path), "step": step}), flush=True)
        if stop_requested:
            break
    print(json.dumps({"event": "complete", "step": step}), flush=True)
