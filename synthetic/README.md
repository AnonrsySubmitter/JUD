# Synthetic count-data experiments

Trains discrete generative models (binomial, Poisson-only, up-and-down N/Z,
Count Bridges, Count-FM, categorical/SEDD) on synthetic count distributions.

```bash
# from the repository root: uv sync --locked --python 3.14
cd synthetic
uv run python train_paths.py --distribution PoissonMixture      # train + sample, saves runs/<run>/paths_*.pt
uv run python plot_paths.py --run runs/<run>/paths_PoissonMixture.pt
uv run python compare_noise_types.py                             # TV / W1 across distributions and methods
```

Hyperparameters live in `config.yaml`. Runs on CPU; dependencies are in the root `pyproject.toml`.
