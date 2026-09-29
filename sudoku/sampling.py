"""Tau-leaping with 179 transitions and one final denoise at 180 NFE."""

from dataclasses import dataclass
from itertools import pairwise

import torch

from .model import JUDDiT
from .process import mse_predictions, posterior_expectations, sanitize_rates, step_rate


@dataclass(frozen=True)
class Samples:
    digits: torch.Tensor
    raw_state: torch.Tensor
    nfe: int


def _expectations(model, output, state, signal):
    if model.objective == "posterior_ce":
        return posterior_expectations(output, state, signal, model.process.poisson_rate)
    expected_n, expected_l, _ = mse_predictions(output.float(), state, signal, model.process)
    return state - expected_n + expected_l, expected_n, expected_l


@torch.no_grad()
def sample(
    model: JUDDiT,
    prompt: torch.Tensor,
    steps: int = 180,
    generator: torch.Generator | None = None,
) -> Samples:
    if steps < 2:
        raise ValueError("sampling needs at least two network evaluations")
    batch = len(prompt)
    state = torch.poisson(
        torch.full((batch, 81), model.process.poisson_rate, device=prompt.device),
        generator=generator,
    )
    # Signal s = 1 - t. The grid is linear; the network embeds -log(s + eps).
    grid = torch.linspace(0.0, 1.0 - model.process.signal_epsilon, steps, device=prompt.device)

    def predict(signal):
        with torch.autocast(
            device_type=state.device.type,
            dtype=torch.bfloat16,
            enabled=state.device.type == "cuda",
        ):
            return model(prompt, state, signal)

    for current, following in pairwise(grid):
        signal = current.expand(batch)
        output = predict(signal)
        _, expected_n, expected_l = _expectations(model, output, state, signal)
        expected_n, expected_l = sanitize_rates(state, expected_n, expected_l)
        coefficient = step_rate(current, following)
        up = torch.poisson((coefficient * expected_l).clamp_min(0.0), generator=generator)
        down = torch.poisson((coefficient * expected_n).clamp_min(0.0), generator=generator)
        state = (state + up - torch.minimum(down, state)).clamp_min(0.0)

    signal = grid[-1].expand(batch)
    output = predict(signal)
    if model.objective == "posterior_ce":
        digits = output.float().argmax(dim=-1) + 1
    else:
        expected_x, _, _ = _expectations(model, output, state, signal)
        digits = expected_x.round().clamp(1.0, 9.0).long()
    return Samples(digits, state, steps)
