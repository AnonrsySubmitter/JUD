# Marginal-preserving correction: Campbell et al. (2022), https://arxiv.org/abs/2205.14987.
"""Shared Poisson--Binomial schedules and moment formulas."""

import torch

FORMULA_VERSION = 1


def signal_from_sigma(sigma, eps):
    """Map the external PF-style noise coordinate to signal time in [0, 1]."""
    return (torch.exp(-sigma) - eps).clamp(0.0, 1.0)


def sigma_from_signal(signal, eps):
    return -torch.log(signal + eps)


def schedules(signal, poisson_rate, retention_power=1.0, poisson_power=1.0):
    """Return retention probability and added-Poisson rate at signal time."""
    retention = signal.pow(retention_power)
    rate = poisson_rate * (1.0 - signal).pow(poisson_power)
    return retention, rate


def moments(signal, mean_data, var_data, poisson_rate, retention_power=1.0, poisson_power=1.0):
    """Scalar moments for Y=B+N and the local rate targets N and L=X-B."""
    retention, rate = schedules(signal, poisson_rate, retention_power, poisson_power)
    one_minus_retention = 1.0 - retention

    mean_b = retention * mean_data
    var_b = retention * one_minus_retention * mean_data + retention.square() * var_data
    mean_y = mean_b + rate
    var_y = var_b + rate

    mean_n = rate
    var_n = rate
    cov_n_y = rate

    mean_l = one_minus_retention * mean_data
    var_l = retention * one_minus_retention * mean_data + one_minus_retention.square() * var_data
    cov_l_y = retention * one_minus_retention * (var_data - mean_data)

    return {
        "retention": retention,
        "rate": rate,
        "mean_b": mean_b,
        "var_b": var_b,
        "mean_y": mean_y,
        "var_y": var_y,
        "mean_n": mean_n,
        "var_n": var_n,
        "cov_n_y": cov_n_y,
        "mean_l": mean_l,
        "var_l": var_l,
        "cov_l_y": cov_l_y,
    }


def head_preconditioning(
    target_mean, target_var, target_input_cov, mean_y, var_y, affine, eps=1e-12
):
    """Optimal scalar linear preconditioning for a target regressed on Y."""
    safe_var_y = var_y.clamp_min(eps)
    c_skip = target_input_cov / safe_var_y
    c_out_sq = (target_var - target_input_cov.square() / safe_var_y).clamp_min(0.0)
    residual_mean = target_mean - c_skip * mean_y
    offset = residual_mean if affine else torch.zeros_like(residual_mean)
    return c_skip, c_out_sq.sqrt(), offset, residual_mean


def rate_coefficients(signal, retention_power=1.0, poisson_power=1.0, eps=1e-8):
    """Coefficients multiplying E[L|Y] (up) and E[N|Y] (down)."""
    retention = signal.pow(retention_power)
    retention_derivative = retention_power * signal.clamp_min(eps).pow(retention_power - 1.0)
    up = retention_derivative / (1.0 - retention).clamp_min(eps)
    down = poisson_power / (1.0 - signal).clamp_min(eps)
    return up, down


def campbell_corrector_rates(
    signal,
    state,
    expected_noise,
    expected_missing,
    poisson_rate,
    mean_data,
    retention_power=1.0,
    poisson_power=1.0,
    eps=1e-8,
):
    """Marginal-preserving Q_forward + Q_reverse rates."""
    reverse_up, reverse_down = rate_coefficients(
        signal, retention_power=retention_power, poisson_power=poisson_power, eps=eps
    )
    forward_up = poisson_rate * poisson_power * (1.0 - signal).pow(poisson_power - 1.0)
    expected_survivors = (state - expected_noise).clamp_min(0.0)
    forward_down = retention_power * expected_survivors / signal.clamp_min(eps)
    endpoint_down = (
        (mean_data / poisson_rate) * state if retention_power == 1.0 else torch.zeros_like(state)
    )
    forward_down = torch.where(signal > 0.0, forward_down, endpoint_down)
    return reverse_up * expected_missing + forward_up, reverse_down * expected_noise + forward_down


def step_rate_coefficients(signal_cur, signal_next, retention_power=1.0, poisson_power=1.0):
    """Schedule increments after cancelling each vanishing conditional target."""
    retention_cur = signal_cur.pow(retention_power)
    retention_next = signal_next.pow(retention_power)
    remaining_noise_cur = (1.0 - signal_cur).pow(poisson_power)
    remaining_noise_next = (1.0 - signal_next).pow(poisson_power)
    tiny = torch.finfo(signal_cur.dtype).tiny
    up = (retention_next - retention_cur) / (1.0 - retention_cur).clamp_min(tiny)
    down = (remaining_noise_cur - remaining_noise_next) / remaining_noise_cur.clamp_min(tiny)
    return up, down
