# Jumping Up and Down

Synthetic count-data, CIFAR-10 and hard-Sudoku experiments for _Jumping Up and Down_.

## Setup

Install [uv](https://docs.astral.sh/uv/) and the pinned dependencies:

```sh
uv sync --locked --python 3.14
```

The CIFAR-10 and Sudoku training requires a CUDA-capable GPU, and their commands run from the repository root. The synthetic experiments run on CPU and are run from `synthetic/` (see its README).

## Experiments

| Experiment  | Instructions                           | Models                                                              |
| ----------- | -------------------------------------- | ------------------------------------------------------------------- |
| Synthetic   | [synthetic/README.md](synthetic/README.md) | JUD (up-and-down N/Z) vs. binomial, Poisson-only, Count Bridges, Count-FM and categorical (SEDD) baselines on count distributions |
| CIFAR-10    | [cifar10/README.md](cifar10/README.md) | JUD and the EDM baseline                                            |
| Hard Sudoku | [sudoku/README.md](sudoku/README.md)   | JUD with MSE regression / posterior CE × original / shuffled digits |

The instructions cover data preparation, training, sampling, and evaluation.
