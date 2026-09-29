# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from NVIDIA EDM (https://github.com/NVlabs/edm), CC BY-NC-SA 4.0; see cifar10/LICENSE-NVIDIA.txt.

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import numpy as np
import torch
import torch.distributions as dist
from torch_utils import persistence, training_stats

from training.jump_up_down import FORMULA_VERSION as JUD_FORMULA_VERSION
from training.jump_up_down import head_preconditioning, moments, sigma_from_signal


@persistence.persistent_class
class JumpUpDownLoss:
    def __init__(
        self,
        log_noise_eps=1e-5,
        time_sampling="uniform",
        time_mu=0.0,
        time_std=1.5,
        time_cdf_path=None,
        mean_data=120.7,
        var_data=4115.2,
        poisson_rate=120.7,
        retention_power=1.0,
        poisson_power=1.0,
        precond_eps=1e-12,
        loss_eps=1e-8,
        affine=True,
    ):
        if time_sampling not in ["uniform", "logit-normal", "cdf"]:
            raise ValueError(f"Invalid time_sampling: {time_sampling!r}")
        if not 0 < log_noise_eps < 0.5:
            raise ValueError("log_noise_eps must be in (0, 0.5)")
        if poisson_rate <= 0 or retention_power <= 0 or poisson_power <= 0:
            raise ValueError("Poisson rate and schedule powers must be positive")
        if time_std <= 0:
            raise ValueError("time_std must be positive")
        self.log_noise_eps = float(log_noise_eps)
        self.time_sampling = time_sampling
        self.time_mu = float(time_mu)
        self.time_std = float(time_std)
        self.time_cdf_path = time_cdf_path
        self.mean_data = float(mean_data)
        self.var_data = float(var_data)
        self.poisson_rate = float(poisson_rate)
        self.retention_power = float(retention_power)
        self.poisson_power = float(poisson_power)
        self.precond_eps = float(precond_eps)
        self.loss_eps = float(loss_eps)
        self.affine = bool(affine)
        self.formula_version = JUD_FORMULA_VERSION
        self.time_cdf = None
        self.time_signal = None
        if self.time_sampling == "cdf":
            if self.time_cdf_path is None:
                raise ValueError("time_cdf_path is required for CDF time sampling")
            with np.load(self.time_cdf_path) as data:
                self.time_cdf = torch.as_tensor(np.asarray(data["cdf"], dtype=np.float32).copy())
                key = "signal" if "signal" in data else "t_sorted"
                self.time_signal = torch.as_tensor(np.asarray(data[key], dtype=np.float32).copy())
            if (
                self.time_cdf.ndim != 1
                or self.time_cdf.shape != self.time_signal.shape
                or len(self.time_cdf) < 2
            ):
                raise ValueError("Time CDF arrays must be matching one-dimensional arrays")
            if (
                not torch.isfinite(self.time_cdf).all()
                or not torch.isfinite(self.time_signal).all()
            ):
                raise ValueError("Time CDF arrays must be finite")
            if not torch.all(self.time_cdf[1:] >= self.time_cdf[:-1]):
                raise ValueError("Time CDF must be nondecreasing")
            if not torch.all(self.time_signal[1:] >= self.time_signal[:-1]):
                raise ValueError("Time signal grid must be nondecreasing")
            span = self.time_cdf[-1] - self.time_cdf[0]
            if not span > 0:
                raise ValueError("Time CDF must have positive mass")
            self.time_cdf = (self.time_cdf - self.time_cdf[0]) / span
            self.time_signal = self.time_signal.clamp(self.log_noise_eps, 1.0 - self.log_noise_eps)

    def sample_signal(self, batch_size, device):
        eps = self.log_noise_eps
        if self.time_sampling == "uniform":
            signal = eps + (1.0 - 2.0 * eps) * torch.rand([batch_size, 1, 1, 1], device=device)
        elif self.time_sampling == "logit-normal":
            z = torch.randn([batch_size, 1, 1, 1], device=device) * self.time_std + self.time_mu
            signal = torch.sigmoid(z)
        else:
            cdf = self.time_cdf.to(device)
            signal_grid = self.time_signal.to(device)
            uniform = torch.rand([batch_size], device=device)
            hi = torch.searchsorted(cdf, uniform, right=True).clamp(1, len(cdf) - 1)
            lo = hi - 1
            fraction = (uniform - cdf[lo]) / (cdf[hi] - cdf[lo]).clamp_min(1e-12)
            signal = (signal_grid[lo] + fraction * (signal_grid[hi] - signal_grid[lo])).reshape(
                -1, 1, 1, 1
            )
        return signal.clamp(eps, 1.0 - eps)

    def __call__(self, net, images, labels=None, augment_pipe=None):
        if self.formula_version != JUD_FORMULA_VERSION:
            raise RuntimeError("Jump-up/down formula version mismatch")
        signal = self.sample_signal(images.shape[0], images.device)
        sigma = sigma_from_signal(signal, self.log_noise_eps)
        stats = moments(
            signal,
            self.mean_data,
            self.var_data,
            self.poisson_rate,
            self.retention_power,
            self.poisson_power,
        )

        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        x_data = torch.clamp((y + 1) * 127.5, 0, 255).round().to(torch.float32)
        survived = dist.Binomial(total_count=x_data, probs=stats["retention"]).sample()
        added = torch.poisson(stats["rate"].expand_as(x_data))
        removed = x_data - survived
        noisy = survived + added

        pred_n, pred_l = net(noisy, sigma, labels, augment_labels=augment_labels).chunk(2, dim=1)
        _, n_out, _, n_residual_mean = head_preconditioning(
            stats["mean_n"],
            stats["var_n"],
            stats["cov_n_y"],
            stats["mean_y"],
            stats["var_y"],
            affine=self.affine,
            eps=self.precond_eps,
        )
        _, l_out, _, l_residual_mean = head_preconditioning(
            stats["mean_l"],
            stats["var_l"],
            stats["cov_l_y"],
            stats["mean_y"],
            stats["var_y"],
            affine=self.affine,
            eps=self.precond_eps,
        )
        n_bias_sq = torch.zeros_like(n_residual_mean) if self.affine else n_residual_mean.square()
        l_bias_sq = torch.zeros_like(l_residual_mean) if self.affine else l_residual_mean.square()
        n_weight = 1.0 / (n_out.square() + n_bias_sq + self.loss_eps)
        l_weight = 1.0 / (l_out.square() + l_bias_sq + self.loss_eps)
        loss_n = n_weight * (pred_n - added).square()
        loss_l = l_weight * (pred_l - removed).square()
        loss = torch.cat([loss_n, loss_l], dim=1)

        training_stats.report("Loss/loss_base", loss)
        training_stats.report("Loss/jud_added", loss_n)
        training_stats.report("Loss/jud_removed", loss_l)
        training_stats.report("JUD/signal", signal)
        return loss


# Standard EDM denoising loss from https://github.com/NVlabs/edm.
@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0], 1, 1, 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        noise = torch.randn_like(y)
        weight = (sigma.square() + self.sigma_data**2) / (sigma * self.sigma_data).square()
        prediction = net(y + sigma * noise, sigma, labels, augment_labels=augment_labels)
        loss = weight * (prediction - y).square()
        training_stats.report("Loss/loss_base", loss)
        return loss
