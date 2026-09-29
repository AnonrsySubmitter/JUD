# Preconditioning follows NVIDIA EDM (https://github.com/NVlabs/edm), CC BY-NC-SA 4.0.
"""Binomial–Poisson corruption, N/L denoisers, and reverse rates."""

from __future__ import annotations

from dataclasses import dataclass

import torch

DATA_MEAN = 5.0
DATA_VARIANCE = 20.0 / 3.0


@dataclass(frozen=True)
class ProcessConfig:
    poisson_rate: float = DATA_MEAN
    signal_epsilon: float = 1e-3
    input_epsilon: float = 1e-8
    preconditioning_epsilon: float = 1e-12
    loss_epsilon: float = 1e-6


DEFAULT_PROCESS = ProcessConfig()


def sigma_from_signal(signal: torch.Tensor, epsilon: float) -> torch.Tensor:
    return -torch.log(signal + epsilon)


def moments(signal: torch.Tensor, poisson_rate: float = DATA_MEAN) -> dict[str, torch.Tensor]:
    """Moments for Y=B+N, N, and L=X-B under linear schedules."""
    retention = signal
    rate = poisson_rate * (1.0 - signal)
    missing_fraction = 1.0 - retention
    mean_y = retention * DATA_MEAN + rate
    var_y = retention * missing_fraction * DATA_MEAN + retention.square() * DATA_VARIANCE + rate
    return {
        "retention": retention,
        "rate": rate,
        "mean_y": mean_y,
        "var_y": var_y,
        "mean_n": rate,
        "var_n": rate,
        "cov_n_y": rate,
        "mean_l": missing_fraction * DATA_MEAN,
        "var_l": (
            retention * missing_fraction * DATA_MEAN + missing_fraction.square() * DATA_VARIANCE
        ),
        "cov_l_y": retention * missing_fraction * (DATA_VARIANCE - DATA_MEAN),
    }


def normalize_input(
    noisy: torch.Tensor,
    signal: torch.Tensor,
    epsilon: float = 1e-8,
    poisson_rate: float = DATA_MEAN,
) -> torch.Tensor:
    stats = moments(signal[:, None], poisson_rate)
    return (noisy.float() - stats["mean_y"]) / torch.sqrt(stats["var_y"] + epsilon)


def corrupt(
    clean: torch.Tensor,
    signal: torch.Tensor,
    poisson_rate: float = DATA_MEAN,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return noisy state Y, added particles N, and missing particles L."""
    clean = clean.float()
    retention = signal[:, None].expand_as(clean)
    rate = (poisson_rate * (1.0 - signal))[:, None].expand_as(clean)
    survived = torch.binomial(clean, retention, generator=generator)
    added = torch.poisson(rate, generator=generator)
    missing = clean - survived
    return survived + added, added, missing


def _head_preconditioning(
    target_mean: torch.Tensor,
    target_variance: torch.Tensor,
    target_input_covariance: torch.Tensor,
    mean_y: torch.Tensor,
    var_y: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_var_y = var_y.clamp_min(epsilon)
    skip = target_input_covariance / safe_var_y
    out_squared = (target_variance - target_input_covariance.square() / safe_var_y).clamp_min(0.0)
    offset = target_mean - skip * mean_y
    return skip, out_squared.sqrt(), offset


def mse_predictions(
    residuals: torch.Tensor,
    noisy: torch.Tensor,
    signal: torch.Tensor,
    config: ProcessConfig = DEFAULT_PROCESS,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Apply moment-matched EDM-style output preconditioning to both MSE heads."""
    stats = moments(signal[:, None], config.poisson_rate)
    n_skip, n_out, n_offset = _head_preconditioning(
        stats["mean_n"],
        stats["var_n"],
        stats["cov_n_y"],
        stats["mean_y"],
        stats["var_y"],
        config.preconditioning_epsilon,
    )
    l_skip, l_out, l_offset = _head_preconditioning(
        stats["mean_l"],
        stats["var_l"],
        stats["cov_l_y"],
        stats["mean_y"],
        stats["var_y"],
        config.preconditioning_epsilon,
    )
    expected_n = n_offset + n_skip * noisy + n_out * residuals[..., 0]
    expected_l = l_offset + l_skip * noisy + l_out * residuals[..., 1]
    coefficients = {
        "n_out": n_out,
        "l_out": l_out,
        "n_baseline": n_offset + n_skip * noisy,
        "l_baseline": l_offset + l_skip * noisy,
    }
    return expected_n, expected_l, coefficients


def mse_loss(
    residuals: torch.Tensor,
    noisy: torch.Tensor,
    signal: torch.Tensor,
    added: torch.Tensor,
    missing: torch.Tensor,
    config: ProcessConfig = DEFAULT_PROCESS,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    expected_n, expected_l, coefficients = mse_predictions(residuals, noisy, signal, config)
    weight_n = 1.0 / (coefficients["n_out"].square() + config.loss_epsilon)
    weight_l = 1.0 / (coefficients["l_out"].square() + config.loss_epsilon)
    loss_n = weight_n * (expected_n - added).square()
    loss_l = weight_l * (expected_l - missing).square()
    baseline_n = weight_n * (coefficients["n_baseline"] - added).square()
    baseline_l = weight_l * (coefficients["l_baseline"] - missing).square()
    metrics = {
        "loss_n": loss_n.mean(),
        "loss_l": loss_l.mean(),
        "baseline": 0.5 * (baseline_n.mean() + baseline_l.mean()),
    }
    return 0.5 * (loss_n.mean() + loss_l.mean()), metrics


def _survivor_log_weights(
    noisy: torch.Tensor,
    signal: torch.Tensor,
    poisson_rate: float = DATA_MEAN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = noisy.dtype if noisy.is_floating_point() else torch.float32
    y = noisy.to(dtype)[..., None, None]
    s = signal.to(dtype).reshape(-1, *([1] * (noisy.ndim - 1)), 1, 1)
    k = torch.arange(1, 10, device=noisy.device, dtype=dtype).reshape(*([1] * noisy.ndim), 9, 1)
    b = torch.arange(0, 10, device=noisy.device, dtype=dtype).reshape(*([1] * noisy.ndim), 1, 10)

    valid = (b <= k) & (b <= y)
    safe_s = s.clamp(torch.finfo(dtype).eps, 1.0 - torch.finfo(dtype).eps)
    rate = (poisson_rate * (1.0 - s)).clamp_min(torch.finfo(dtype).tiny)
    noise = y - b
    log_combination = torch.lgamma(k + 1.0) - torch.lgamma(b + 1.0) - torch.lgamma(k - b + 1.0)
    log_weight = (
        log_combination
        + b * torch.log(safe_s)
        + (k - b) * torch.log1p(-safe_s)
        + noise * torch.log(rate)
        - torch.lgamma(noise + 1.0)
    )
    log_weight = log_weight.masked_fill(~valid, -torch.inf)
    return log_weight, b, s


def conditional_survivor_mean(
    noisy: torch.Tensor,
    signal: torch.Tensor,
    poisson_rate: float = DATA_MEAN,
) -> torch.Tensor:
    """Compute E[B | Y=y, X=k] for every k=1,...,9."""
    log_weight, b, expanded_signal = _survivor_log_weights(noisy, signal, poisson_rate)
    probabilities = torch.softmax(log_weight, dim=-1)
    result = (probabilities * b).sum(dim=-1)
    return torch.where(
        expanded_signal.squeeze(-1) == 0.0,
        torch.zeros_like(result),
        result,
    )


def posterior_expectations(
    logits: torch.Tensor,
    noisy: torch.Tensor,
    signal: torch.Tensor,
    poisson_rate: float = DATA_MEAN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map a clean-category posterior to E[X|Y], E[N|Y], and E[L|Y]."""
    probabilities = torch.softmax(logits.float(), dim=-1)
    digits = torch.arange(1, 10, device=logits.device, dtype=torch.float32)
    expected_x = (probabilities * digits).sum(dim=-1)
    survivor_given_x = conditional_survivor_mean(noisy.float(), signal, poisson_rate)
    expected_b = (probabilities * survivor_given_x).sum(dim=-1)
    expected_n = noisy.float() - expected_b
    expected_l = expected_x - expected_b
    return expected_x, expected_n, expected_l


def sanitize_rates(
    noisy: torch.Tensor,
    expected_n: torch.Tensor,
    expected_l: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    expected_n = expected_n.float().clamp_min(0.0)
    expected_n = torch.minimum(expected_n, noisy.float())
    expected_l = expected_l.float().clamp(0.0, 9.0)
    return expected_n, expected_l


def step_rate(signal: torch.Tensor, next_signal: torch.Tensor) -> torch.Tensor:
    """Euler tau-leap coefficient: delta_signal / (1 - signal)."""
    return (next_signal - signal) / (1.0 - signal).clamp_min(torch.finfo(signal.dtype).tiny)
