# Sudoku

Run from the repository root after `uv sync`.
Training and evaluation use one CUDA GPU per run.

## Data

Prepare 48,000 training boards and 2,000 validation boards, each with 30 clues:

```sh
uv run python -m sudoku prepare-data --output data/sudoku --workers 8
```

## Train JUD

Train noise/lost-count MSE and posterior CE models with original or shuffled digits:

```sh
uv run python -m sudoku train --objective mse --encoding original \
  --data data/sudoku --output runs/sudoku/mse-original
uv run python -m sudoku train --objective posterior_ce --encoding original \
  --data data/sudoku --output runs/sudoku/ce-original
uv run python -m sudoku train --objective mse --encoding shuffled \
  --data data/sudoku --output runs/sudoku/mse-shuffled
uv run python -m sudoku train --objective posterior_ce --encoding shuffled \
  --data data/sudoku --output runs/sudoku/ce-shuffled
```

## Sample and Evaluate

Evaluate a checkpoint's EMA weights on all 2,000 validation boards using tau-leaping with 180 network evaluations:

```sh
uv run python -m sudoku evaluate \
  --checkpoint runs/sudoku/mse-original/checkpoint-010000.pt \
  --data data/sudoku --output runs/sudoku/mse-original/evaluation
```

Use the corresponding checkpoint and output directory for each configuration:

| Run directory under `runs/sudoku/` | Final checkpoint       |
| ---------------------------------- | ---------------------- |
| `mse-original`, `ce-original`      | `checkpoint-010000.pt` |
| `mse-shuffled`, `ce-shuffled`      | `checkpoint-020000.pt` |

Evaluation writes exact-board and cell accuracy to `validation-180nfe.json` in the output directory.
