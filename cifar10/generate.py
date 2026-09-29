# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Adapted from NVIDIA EDM (https://github.com/NVlabs/edm), CC BY-NC-SA 4.0; see cifar10/LICENSE-NVIDIA.txt.

import hashlib
import os
import pickle
import re
import struct

import click
import PIL.Image
import torch
import tqdm
from torch_utils import distributed as dist
from training.jump_up_down import (
    campbell_corrector_rates,
    rate_coefficients,
    sigma_from_signal,
    step_rate_coefficients,
)


def tau_leap_jump_up_down_sampler(
    net,
    batch_size,
    class_labels=None,
    num_steps=1024,
    log_noise_eps=None,
    base_seed=None,
    euler=False,
    jud_corrector="none",
):
    if jud_corrector not in ["none", "campbell"]:
        raise ValueError(f"Unknown JUD corrector {jud_corrector!r}")
    device = next(net.parameters()).device
    eps = float(log_noise_eps if log_noise_eps is not None else getattr(net, "log_noise_eps", 1e-5))
    poisson_rate = float(net.jud_poisson_rate)
    retention_power = float(net.jud_retention_power)
    poisson_power = float(net.jud_poisson_power)
    if not (eps > 0):
        raise ValueError(f"log_noise_eps must be > 0, got {eps}")
    if not (poisson_rate > 0):
        raise ValueError(f"jud_poisson_rate must be > 0, got {poisson_rate}")
    if not (retention_power > 0 and poisson_power > 0):
        raise ValueError("Jump-up/down schedule powers must be > 0")
    if jud_corrector == "campbell" and retention_power < 1.0:
        raise ValueError("Campbell correction requires jud_retention_power >= 1")

    B = int(batch_size)
    x_shape = [B, net.img_channels, net.img_resolution, net.img_resolution]

    if base_seed is not None:
        base_seed = int(base_seed) % (1 << 32)
        torch.manual_seed(base_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(base_seed)

    x = torch.poisson(torch.full(x_shape, poisson_rate, device=device, dtype=torch.float32))
    signal_steps = torch.linspace(
        0.0, 1.0 - eps, int(num_steps) + 1, device=device, dtype=torch.float32
    )

    for signal_cur, signal_next in zip(signal_steps[:-1], signal_steps[1:]):
        sigma_cur = sigma_from_signal(signal_cur, eps)
        with torch.no_grad():
            pred = net(x, sigma_cur.expand([B]), class_labels).to(torch.float32)
        if pred.shape[1] != 2 * net.img_channels:
            raise ValueError(
                f"Jump-up/down network must output {2 * net.img_channels} channels, "
                f"got {pred.shape[1]}"
            )
        expected_noise, expected_missing = pred.chunk(2, dim=1)
        expected_noise = expected_noise.clamp(min=0.0)
        expected_noise = torch.minimum(expected_noise, x)
        expected_missing = expected_missing.clamp(0.0, 255.0)

        dt = signal_next - signal_cur
        if not euler:
            up_rate, down_rate = rate_coefficients(
                signal_cur,
                retention_power=retention_power,
                poisson_power=poisson_power,
            )
            up_mean = dt * up_rate * expected_missing
            down_mean = dt * down_rate * expected_noise
        else:
            up_step, down_step = step_rate_coefficients(
                signal_cur,
                signal_next,
                retention_power=retention_power,
                poisson_power=poisson_power,
            )
            up_mean = up_step * expected_missing
            down_mean = down_step * expected_noise

        if jud_corrector == "campbell":
            corrector_up, corrector_down = campbell_corrector_rates(
                signal_cur,
                x,
                expected_noise,
                expected_missing,
                poisson_rate=poisson_rate,
                mean_data=float(getattr(net, "mean_data", poisson_rate)),
                retention_power=retention_power,
                poisson_power=poisson_power,
            )
            up_mean = up_mean + dt * corrector_up
            down_mean = down_mean + dt * corrector_down

        if euler:
            p_up = up_mean.clamp(0.0, 1.0)
            p_down = down_mean.clamp(0.0, 1.0)
            p_down = torch.where(x > 0, p_down, torch.zeros_like(p_down))
            scale = (p_up + p_down).clamp_min(1.0)
            p_up = p_up / scale
            p_down = p_down / scale
            uniform = torch.rand_like(x)
            up = (uniform < p_up).to(x.dtype)
            down = ((uniform >= p_up) & (uniform < p_up + p_down)).to(x.dtype)
        else:
            up = torch.poisson(up_mean.clamp(min=0.0))
            down = torch.poisson(down_mean.clamp(min=0.0))
            down = torch.minimum(down, x)
        x = (x + up - down).clamp_min(0.0)

    return x.clamp(0.0, 255.0)


class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [
            torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds
        ]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack(
            [torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators]
        )


def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


def edm_sampler(net, latents, num_steps=18, sigma_min=0.002, sigma_max=80, rho=7):
    """NVIDIA EDM's deterministic Heun sampler (18 steps, 35 NFE)."""
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)
    indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    times = (
        sigma_max ** (1 / rho)
        + indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    times = torch.cat([net.round_sigma(times), torch.zeros_like(times[:1])])
    x_next = latents.to(torch.float64) * times[0]
    for index, (t_cur, t_next) in enumerate(zip(times[:-1], times[1:])):
        x_cur = x_next
        denoised = net(x_cur, t_cur, None).to(torch.float64)
        derivative = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * derivative
        if index < num_steps - 1:
            denoised = net(x_next, t_next, None).to(torch.float64)
            derivative_next = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * derivative + 0.5 * derivative_next)
    return x_next


@click.command()
@click.option(
    "--network", "network_pkl", required=True, type=click.Path(exists=True, dir_okay=False)
)
@click.option("--outdir", required=True, type=click.Path(file_okay=False))
@click.option("--seeds", type=parse_int_list, default="0-49999", show_default=True)
@click.option(
    "--batch", "max_batch_size", type=click.IntRange(min=1), default=64, show_default=True
)
@click.option("--subdirs", is_flag=True, help="Group every 1,000 sample IDs into a subdirectory.")
@click.option(
    "--sampler",
    type=click.Choice(["real-tau", "euler", "edm"]),
    default="real-tau",
    show_default=True,
)
@click.option("--tau-steps", type=click.IntRange(min=1), default=2048, show_default=True)
@click.option(
    "--jud-corrector", type=click.Choice(["none", "campbell"]), default="none", show_default=True
)
def main(network_pkl, outdir, seeds, max_batch_size, subdirs, sampler, tau_steps, jud_corrector):
    """Generate unconditional CIFAR-10 PNGs from a locally trained EMA snapshot."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise click.ClickException("Sample seeds must be nonempty and unique")
    if sampler == "edm" and jud_corrector != "none":
        raise click.ClickException("The marginal-preserving corrector applies to JUD")
    if sampler == "euler" and jud_corrector != "none":
        raise click.ClickException("The paper applies the corrector to tau-leaping")
    dist.init()
    device = torch.device("cuda")
    num_batches = (
        (len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1
    ) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    if dist.get_rank() != 0:
        torch.distributed.barrier()
    with open(network_pkl, "rb") as file:
        net = pickle.load(file)["ema"].to(device).eval().requires_grad_(False)
    is_jud = hasattr(net, "jud_poisson_rate")
    if (sampler == "edm") == is_jud or net.label_dim != 0:
        raise click.ClickException("Sampler and unconditional model type do not match")
    if dist.get_rank() == 0:
        torch.distributed.barrier()
    dist.print0(f"Generating {len(seeds)} samples to {outdir}")
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=dist.get_rank() != 0):
        torch.distributed.barrier()
        if not len(batch_seeds):
            continue
        if is_jud:
            # The original discrete sampler uses one RNG stream per batch/rank.
            # Record world size and batch size when comparing seeded results.
            digest = hashlib.blake2b(digest_size=4)
            digest.update(struct.pack("<I", dist.get_rank() % (1 << 32)))
            for seed in batch_seeds.tolist():
                digest.update(struct.pack("<I", int(seed) % (1 << 32)))
            images = tau_leap_jump_up_down_sampler(
                net,
                len(batch_seeds),
                num_steps=tau_steps,
                base_seed=int.from_bytes(digest.digest(), "little"),
                euler=sampler == "euler",
                jud_corrector=jud_corrector,
            )
            pixels = images.round().clamp(0, 255).to(torch.uint8)
        else:
            random = StackedRandomGenerator(device, batch_seeds)
            latents = random.randn([len(batch_seeds), 3, 32, 32], device=device)
            images = edm_sampler(net, latents)
            pixels = (images * 127.5 + 128).clip(0, 255).to(torch.uint8)
        for seed, image in zip(batch_seeds.tolist(), pixels.permute(0, 2, 3, 1).cpu().numpy()):
            directory = os.path.join(outdir, f"{seed - seed % 1000:06d}") if subdirs else outdir
            os.makedirs(directory, exist_ok=True)
            PIL.Image.fromarray(image).save(os.path.join(directory, f"{seed:06d}.png"))
    torch.distributed.barrier()
    dist.print0("Done.")


if __name__ == "__main__":
    main()
