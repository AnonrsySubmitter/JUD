"""Measure JUD learning over signal time and build a post-hoc proposal CDF."""

import os
import pickle

import click
import dnnlib
import numpy as np
import torch
import torch.distributions as torch_dist
import tqdm
from scipy.integrate import trapezoid
from training.dataset import ImageFolderDataset
from training.jump_up_down import head_preconditioning, moments, sigma_from_signal


@click.command()
@click.option("--network", "network_pkl", type=str, required=True, help="Network snapshot PKL")
@click.option(
    "--data", "data_path", type=click.Path(exists=True), required=True, help="Training dataset"
)
@click.option(
    "--out", "out_path", type=click.Path(dir_okay=False), required=True, help="Output time-CDF NPZ"
)
@click.option("--signals", "num_signals", type=click.IntRange(min=3), default=81, show_default=True)
@click.option(
    "--samples", "num_samples", type=click.IntRange(min=1), default=4096, show_default=True
)
@click.option("--batch", "batch_size", type=click.IntRange(min=1), default=128, show_default=True)
@click.option(
    "--uniform-floor", type=click.FloatRange(min=0, max=1), default=0.1, show_default=True
)
@click.option("--seed", type=int, default=123, show_default=True)
@click.option("--device", type=str, default="cuda", show_default=True)
def main(
    network_pkl,
    data_path,
    out_path,
    num_signals,
    num_samples,
    batch_size,
    uniform_floor,
    seed,
    device,
):
    """Compare the learned predictor with its analytic affine baseline."""
    device = torch.device(device)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=True) as file:
        snapshot = pickle.load(file)
    net = snapshot["ema"].to(device).eval().requires_grad_(False)
    if not hasattr(net, "jud_poisson_rate"):
        raise click.ClickException("Snapshot is not a jump-up/down network")
    loss_fn = snapshot.get("loss_fn")

    dataset = ImageFolderDataset(path=data_path, use_labels=net.label_dim > 0, cache=False)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=num_samples, replace=num_samples > len(dataset))
    examples = [dataset[int(index)] for index in indices]
    images = torch.from_numpy(np.stack([image for image, _ in examples]))
    labels = torch.from_numpy(np.stack([label for _, label in examples])).to(torch.float32)

    eps = float(net.log_noise_eps)
    logit_bound = np.log((1.0 - eps) / eps)
    signal_grid = 1.0 / (1.0 + np.exp(-np.linspace(-logit_bound, logit_bound, num_signals)))
    result = {
        key: [] for key in ["model_added", "model_removed", "baseline_added", "baseline_removed"]
    }

    for signal_value in tqdm.tqdm(signal_grid, unit="signal"):
        totals = dict.fromkeys(result, 0.0)
        elements = 0
        for start in range(0, num_samples, batch_size):
            x_data = images[start : start + batch_size].to(device=device, dtype=torch.float32)
            class_labels = labels[start : start + batch_size].to(device)
            signal = torch.full([len(x_data), 1, 1, 1], float(signal_value), device=device)
            stats = moments(
                signal,
                net.mean_data,
                net.var_data,
                net.jud_poisson_rate,
                net.jud_retention_power,
                net.jud_poisson_power,
            )
            survived = torch_dist.Binomial(total_count=x_data, probs=stats["retention"]).sample()
            added = torch.poisson(stats["rate"].expand_as(x_data))
            removed = x_data - survived
            noisy = survived + added

            with torch.no_grad():
                pred_added, pred_removed = net(
                    noisy, sigma_from_signal(signal, eps), class_labels
                ).chunk(2, dim=1)

            n_skip, n_out, n_offset, n_mean = head_preconditioning(
                stats["mean_n"],
                stats["var_n"],
                stats["cov_n_y"],
                stats["mean_y"],
                stats["var_y"],
                affine=net.jud_affine,
                eps=net.precond_eps,
            )
            l_skip, l_out, l_offset, l_mean = head_preconditioning(
                stats["mean_l"],
                stats["var_l"],
                stats["cov_l_y"],
                stats["mean_y"],
                stats["var_y"],
                affine=net.jud_affine,
                eps=net.precond_eps,
            )
            loss_eps = float(getattr(loss_fn, "loss_eps", 1e-8))
            n_bias_sq = torch.zeros_like(n_mean) if net.jud_affine else n_mean.square()
            l_bias_sq = torch.zeros_like(l_mean) if net.jud_affine else l_mean.square()
            n_weight = 1.0 / (n_out.square() + n_bias_sq + loss_eps)
            l_weight = 1.0 / (l_out.square() + l_bias_sq + loss_eps)
            base_added = n_offset + n_skip * noisy
            base_removed = l_offset + l_skip * noisy

            losses = {
                "model_added": n_weight * (pred_added - added).square(),
                "model_removed": l_weight * (pred_removed - removed).square(),
                "baseline_added": n_weight * (base_added - added).square(),
                "baseline_removed": l_weight * (base_removed - removed).square(),
            }
            elements += added.numel()
            for key, value in losses.items():
                totals[key] += value.sum().item()

        for key, values in result.items():
            values.append(totals[key] / elements)

    result = {key: np.asarray(value, dtype=np.float64) for key, value in result.items()}
    baseline = 0.5 * (result["baseline_added"] + result["baseline_removed"])
    learned = 0.5 * (result["model_added"] + result["model_removed"])
    gain = np.maximum(baseline - learned, 0.0)
    span = signal_grid[-1] - signal_grid[0]
    gain_area = trapezoid(gain, signal_grid)
    gain_density = gain / gain_area if gain_area > 0 else np.full_like(gain, 1.0 / span)
    density = uniform_floor / span + (1.0 - uniform_floor) * gain_density
    increments = 0.5 * (density[1:] + density[:-1]) * np.diff(signal_grid)
    cdf = np.concatenate([[0.0], np.cumsum(increments)])
    cdf /= cdf[-1]

    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    np.savez(
        out_path,
        cdf=cdf.astype(np.float32),
        signal=signal_grid.astype(np.float32),
        density=density.astype(np.float32),
        gain=gain.astype(np.float32),
        **{key: value.astype(np.float32) for key, value in result.items()},
    )
    quantiles = np.interp([0.1, 0.5, 0.9], cdf, signal_grid)
    print(f"Saved {out_path}")
    print(
        f"baseline={baseline.mean():.4f}, learned={learned.mean():.4f}, positive gain area={gain_area:.4g}"
    )
    print(
        f"proposal signal quantiles: 10%={quantiles[0]:.4g}, 50%={quantiles[1]:.4g}, 90%={quantiles[2]:.4g}"
    )


if __name__ == "__main__":
    main()
