# CIFAR-10

Run from the repository root after `uv sync`.
Multiple GPUs are supported (use the `--nproc_per_node` argument), as well as gradient accumulation (use the `--batch-gpu` argument).

## Data

Convert the official archive's 50,000 training images to the EDM ZIP format, without labels, for unconditional generation.

```sh
mkdir -p data
curl -L https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz \
  -o data/cifar-10-python.tar.gz
uv run python cifar10/prepare_data.py \
  --source=data/cifar-10-python.tar.gz --dest=data/cifar10-32x32.zip
```

## Train JUD

Train a 5-million-image pilot with uniform perturbation times (adjust for the number of available GPUs, here eight):

```sh
uv run torchrun --standalone --nproc_per_node=8 cifar10/train.py \
  --data=data/cifar10-32x32.zip --outdir=runs/cifar10-pilot \
  --precond=jud --duration=5 --batch=512 --batch-gpu=64
```

Estimate a fixed time proposal from 4,096 images at 81 logit-spaced times: 10% uniform mass and 90% proportional to positive improvement over the affine predictor.

```sh
uv run python cifar10/jud_time.py \
  --network=runs/cifar10-pilot/network-snapshot-005000.pkl \
  --data=data/cifar10-32x32.zip \
  --out=runs/cifar10-pilot/time-cdf-005000.npz \
  --signals=81 --samples=4096 --batch=128 --uniform-floor=.1
```

Resume the model and optimizer to 200 million images total, including the pilot:

```sh
uv run torchrun --standalone --nproc_per_node=8 cifar10/train.py \
  --data=data/cifar10-32x32.zip --outdir=runs/cifar10-jud \
  --precond=jud --duration=200 --batch=512 --batch-gpu=64 \
  --jud-time-sampling=cdf \
  --jud-time-cdf=runs/cifar10-pilot/time-cdf-005000.npz \
  --resume=runs/cifar10-pilot/training-state-005000.pt
```

## Sample and Evaluate

Generate 50,000 images with 2,048-step tau-leaping, compute FID, and display the first 64 images.
FID downloads NVIDIA's Inception model and reference statistics.

```sh
uv run torchrun --standalone --nproc_per_node=8 cifar10/generate.py \
  --network=runs/cifar10-jud/network-snapshot-200000.pkl \
  --outdir=results/cifar10/tau-2048/images --subdirs \
  --batch=64 --sampler=real-tau --tau-steps=2048
uv run torchrun --standalone --nproc_per_node=8 cifar10/fid.py calc \
  --images=results/cifar10/tau-2048/images --num=50000 --batch=512 \
  --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
uv run python cifar10/grid.py \
  --images=results/cifar10/tau-2048/images \
  --out=results/cifar10/tau-2048/grid-first-64.png
```

To reproduce the sampling / NFE table, evaluate these 12 configurations using the same checkpoint:

| Condition                               | Generation options                            | `--tau-steps` (NFE)  |
| --------------------------------------- | --------------------------------------------- | -------------------- |
| Euler                                   | `--sampler=euler --jud-corrector=none`        | 256, 512, 1024, 2048 |
| Tau-leaping                             | `--sampler=real-tau --jud-corrector=none`     | 256, 512, 1024, 2048 |
| Tau-leaping + marginal-preserving rates | `--sampler=real-tau --jud-corrector=campbell` | 256, 512, 1024, 2048 |

## EDM baseline

Train EDM in FP32, then sample with deterministic 18-step Heun (35 network evaluations):

```sh
uv run torchrun --standalone --nproc_per_node=8 cifar10/train.py \
  --data=data/cifar10-32x32.zip --outdir=runs/cifar10-edm \
  --precond=edm --bf16=0 --duration=200 --batch=512 --batch-gpu=64
uv run torchrun --standalone --nproc_per_node=8 cifar10/generate.py \
  --network=runs/cifar10-edm/network-snapshot-200000.pkl \
  --outdir=results/cifar10/edm/images --subdirs \
  --batch=64 --sampler=edm
uv run torchrun --standalone --nproc_per_node=8 cifar10/fid.py calc \
  --images=results/cifar10/edm/images --num=50000 --batch=512 \
  --ref=https://nvlabs-fi-cdn.nvidia.com/edm/fid-refs/cifar10-32x32.npz
```
