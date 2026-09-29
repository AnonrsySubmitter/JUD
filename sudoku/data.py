# Benchmark layout: S-FLM (https://github.com/jdeschena/s-flm), Apache-2.0.
"""Data preparation and loading for the pinned hard-Sudoku benchmark."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import layout
from .sudoku_generator import generate_sudoku_dataset

SFLM_REPOSITORY = "https://github.com/jdeschena/s-flm"
SFLM_COMMIT = "30def79976ca3b1cdcb5fcbfa0165524e44001de"


def array_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def validate_sequences(sequences: np.ndarray) -> dict[str, object]:
    if sequences.ndim != 2 or sequences.shape[1] != layout.TOTAL_SEQUENCE_LENGTH:
        raise ValueError(f"expected [N, 180] sequences, got {sequences.shape}")
    if sequences.dtype != np.uint8:
        raise ValueError(f"expected uint8 data, got {sequences.dtype}")
    if not np.all(sequences[:, 0] == layout.BOS) or not np.all(
        sequences[:, layout.PROMPT_LENGTH - 1] == layout.BOS
    ):
        raise ValueError("invalid BOS layout")

    separator_positions = np.concatenate(
        [
            1 + np.arange(9, 89, 10),
            layout.PROMPT_LENGTH + np.arange(9, 89, 10),
        ]
    )
    if not np.all(sequences[:, separator_positions] == layout.ROW_SEPARATOR):
        raise ValueError("invalid row separators")

    cell_offsets = layout.grid_cell_offsets().numpy()
    puzzles = sequences[:, 1 + cell_offsets]
    solutions = sequences[:, layout.PROMPT_LENGTH + cell_offsets]
    if not np.all((puzzles >= 0) & (puzzles <= 9)):
        raise ValueError("puzzle cells must be in 0..9")
    if not np.all((solutions >= 1) & (solutions <= 9)):
        raise ValueError("solution cells must be in 1..9")
    clue_counts = np.count_nonzero(puzzles, axis=1)
    if not np.all((puzzles == 0) | (puzzles == solutions)):
        raise ValueError("a puzzle clue disagrees with its solution")

    return {
        "count": len(sequences),
        "sha256": array_sha256(sequences),
        "clues_min": int(clue_counts.min()),
        "clues_max": int(clue_counts.max()),
        "clues_mean": float(clue_counts.mean()),
        "unique_solutions": len(np.unique(solutions, axis=0)),
    }


def prepare_data(
    output: Path,
    workers: int = 1,
    force: bool = False,
    train_examples: int = 48_000,
    validation_examples: int = 2_000,
) -> dict[str, object]:
    if train_examples < 1 or validation_examples < 1 or workers < 1:
        raise ValueError("dataset sizes and workers must be positive")
    output = output.resolve()
    manifest_path = output / "manifest.json"
    train_path = output / "train.npy"
    validation_path = output / "validation.npy"
    if not force and all(path.exists() for path in (manifest_path, train_path, validation_path)):
        manifest = json.loads(manifest_path.read_text())
        if (manifest["train"]["count"], manifest["validation"]["count"]) != (
            train_examples,
            validation_examples,
        ):
            raise ValueError("existing dataset has different sizes; use a new output directory")
        return manifest

    output.mkdir(parents=True, exist_ok=True)
    generated = generate_sudoku_dataset(
        num_train=train_examples, num_valid=validation_examples, num_workers=workers
    )
    train = np.asarray(generated["train"], dtype=np.uint8)
    validation = np.asarray(generated["validation"], dtype=np.uint8)
    train_report = validate_sequences(train)
    validation_report = validate_sequences(validation)
    if train_report["unique_solutions"] != len(train):
        raise ValueError("training solutions are not unique")
    if validation_report["unique_solutions"] != len(validation):
        raise ValueError("validation solutions are not unique")

    np.save(train_path, train, allow_pickle=False)
    np.save(validation_path, validation, allow_pickle=False)
    manifest = {
        "source_repository": SFLM_REPOSITORY,
        "source_commit": SFLM_COMMIT,
        "difficulty": "hard",
        "data_seed": 42,
        "train": train_report,
        "validation": validation_report,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class SudokuDataset(Dataset[torch.Tensor]):
    def __init__(self, path: Path, start: int = 0, stop: int | None = None):
        sequences = np.load(path, mmap_mode="r", allow_pickle=False)
        self.sequences = sequences[start:stop]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.sequences[index]).copy()).long()
