"""Exact-board accuracy on the fixed validation split."""

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import load_checkpoint
from .data import SudokuDataset
from .ema import load_ema_model
from .layout import PROMPT_LENGTH, extract_solution_cells
from .model import JUDDiT
from .process import ProcessConfig
from .runtime import initialize
from .sampling import sample


def model_from_checkpoint(path: Path, device: torch.device):
    checkpoint = load_checkpoint(path, device)
    config = checkpoint["config"]
    if config.get("denoiser_parameterization") != "noise-lost":
        raise ValueError("expected a checkpoint trained with the noise/lost-count parameterization")
    model = JUDDiT(
        config["objective"], ProcessConfig(**config["process"]), **config["architecture"]
    ).to(device)
    load_ema_model(model, checkpoint)
    model.eval().requires_grad_(False)
    return model, checkpoint


@torch.no_grad()
def evaluate(
    checkpoint_path: Path,
    data: Path,
    output: Path,
    steps: int = 180,
    batch_size: int = 64,
    limit: int | None = None,
    seed: int = 2001,
) -> dict:
    if batch_size < 1 or (limit is not None and limit < 1):
        raise ValueError("batch size and optional example limit must be positive")
    device = initialize(seed)
    model, checkpoint = model_from_checkpoint(checkpoint_path, device)
    dataset = SudokuDataset(data / "validation.npy", stop=limit)
    loader = DataLoader(dataset, batch_size=batch_size, pin_memory=device.type == "cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    exact_boards = correct_cells = total = 0
    for sequences in loader:
        sequences = sequences.to(device, non_blocking=True)
        truth = extract_solution_cells(sequences)
        generated = sample(model, sequences[:, :PROMPT_LENGTH], steps, generator)
        correct = generated.digits.eq(truth)
        exact_boards += int(correct.all(dim=1).sum())
        correct_cells += int(correct.sum())
        total += len(sequences)
    if total == 0:
        raise ValueError("validation dataset is empty")
    result = {
        "objective": model.objective,
        "checkpoint_step": int(checkpoint["step"]),
        "digit_permutation": checkpoint["config"]["training"]["digit_permutation"],
        "training_seed": checkpoint["config"]["training"]["seed"],
        "sampling_seed": seed,
        "sampling_nfe": steps,
        "sampling_batch_size": batch_size,
        "examples": total,
        "exact_boards": exact_boards,
        "exact_board_accuracy": exact_boards / total,
        "cell_accuracy": correct_cells / (81 * total),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / f"validation-{steps}nfe.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return result
