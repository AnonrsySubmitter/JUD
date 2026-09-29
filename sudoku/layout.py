# Benchmark layout: S-FLM (https://github.com/jdeschena/s-flm), Apache-2.0.
"""The official S-FLM Sudoku token layout."""

from __future__ import annotations

import torch

EMPTY = 0
DIGITS = tuple(range(1, 10))
ROW_SEPARATOR = 10
BOS = 11
VOCAB_SIZE = 12

GRID_SIZE = 9
NUM_CELLS = 81
GRID_SEQUENCE_LENGTH = 89
PROMPT_LENGTH = 91
TOTAL_SEQUENCE_LENGTH = 180


def grid_cell_offsets() -> torch.Tensor:
    """Offsets of the 81 cells in an 89-token grid."""
    return torch.tensor([row * 10 + col for row in range(9) for col in range(9)])


def solution_cell_positions() -> torch.Tensor:
    return PROMPT_LENGTH + grid_cell_offsets()


def solution_separator_positions() -> torch.Tensor:
    return PROMPT_LENGTH + torch.tensor([row * 10 + 9 for row in range(8)])


def extract_solution_cells(sequences: torch.Tensor) -> torch.Tensor:
    return sequences.index_select(-1, solution_cell_positions().to(sequences.device))
