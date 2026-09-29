# Adapted from S-FLM models/dit.py (https://github.com/jdeschena/s-flm), Apache-2.0.
# Timestep embedding: OpenAI GLIDE (https://github.com/openai/glide-text2im), MIT; see LICENSE-GLIDE.txt.
# Copyright (c) 2021 OpenAI (timestep embedding).
"""DiT with scalar noisy inputs, N/L or posterior heads, and PyTorch attention."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from . import layout
from .process import (
    DEFAULT_PROCESS,
    ProcessConfig,
    normalize_input,
    sigma_from_signal,
)


class LayerNorm(nn.Module):
    """The weight-only LayerNorm used by the official S-FLM DiT."""

    def __init__(self, dimension: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.dimension = dimension

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        normalized = F.layer_norm(inputs.float(), [self.dimension])
        return normalized * self.weight


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_size: int = 256):
        super().__init__()
        self.frequency_size = frequency_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half = self.frequency_size // 2
        frequencies = torch.exp(
            -math.log(10_000) * torch.arange(half, dtype=torch.float32, device=time.device) / half
        )
        arguments = time.float()[:, None] * frequencies[None]
        embedding = torch.cat([torch.cos(arguments), torch.sin(arguments)], dim=-1)
        return self.mlp(embedding)


def _modulate(inputs: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return inputs * (1.0 + scale[:, None]) + shift[:, None]


def _apply_rope(inputs: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    first, second = inputs.chunk(2, dim=-1)
    cosine = cosine[None, None].to(inputs.dtype)
    sine = sine[None, None].to(inputs.dtype)
    return torch.cat([first * cosine - second * sine, second * cosine + first * sine], dim=-1)


class DiTBlock(nn.Module):
    def __init__(
        self,
        dimension: int = 512,
        heads: int = 8,
        condition_dimension: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.heads = heads
        self.head_dimension = dimension // heads
        self.dropout = dropout
        self.norm1 = LayerNorm(dimension)
        self.qkv = nn.Linear(dimension, 3 * dimension, bias=False)
        self.attention_output = nn.Linear(dimension, dimension, bias=False)
        self.norm2 = LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, 4 * dimension),
            nn.GELU(approximate="tanh"),
            nn.Linear(4 * dimension, dimension),
        )
        self.modulation = nn.Linear(condition_dimension, 6 * dimension)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(
        self,
        inputs: torch.Tensor,
        condition: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> torch.Tensor:
        shift_attention, scale_attention, gate_attention, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation(condition).chunk(6, dim=-1)
        )
        normalized = _modulate(self.norm1(inputs), shift_attention, scale_attention)
        batch, sequence, _ = normalized.shape
        qkv = self.qkv(normalized).reshape(batch, sequence, 3, self.heads, self.head_dimension)
        query = _apply_rope(qkv[:, :, 0].transpose(1, 2), cosine, sine)
        key = _apply_rope(qkv[:, :, 1].transpose(1, 2), cosine, sine)
        value = qkv[:, :, 2].transpose(1, 2)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(1, 2).reshape(batch, sequence, -1)
        inputs = inputs + gate_attention[:, None] * F.dropout(
            self.attention_output(attended), p=self.dropout, training=self.training
        )
        hidden = self.mlp(_modulate(self.norm2(inputs), shift_mlp, scale_mlp))
        return inputs + gate_mlp[:, None] * F.dropout(
            hidden, p=self.dropout, training=self.training
        )


class JUDDiT(nn.Module):
    def __init__(
        self,
        objective: str,
        process: ProcessConfig = DEFAULT_PROCESS,
        dimension: int = 512,
        condition_dimension: int = 128,
        blocks: int = 8,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        if objective not in {"mse", "posterior_ce"}:
            raise ValueError(f"unknown objective: {objective!r}")
        self.objective = objective
        self.process = process
        self.token_embedding = nn.Embedding(layout.VOCAB_SIZE, dimension)
        nn.init.kaiming_uniform_(self.token_embedding.weight, a=math.sqrt(5))
        self.noisy_projection = nn.Linear(1, dimension)
        self.time_embedding = TimestepEmbedder(condition_dimension)
        self.blocks = nn.ModuleList(
            [DiTBlock(dimension, heads, condition_dimension, dropout) for _ in range(blocks)]
        )
        self.final_norm = LayerNorm(dimension)
        self.final_modulation = nn.Linear(condition_dimension, 2 * dimension)
        output_size = 2 if objective == "mse" else 9
        self.output = nn.Linear(dimension, output_size)
        nn.init.zeros_(self.final_modulation.weight)
        nn.init.zeros_(self.final_modulation.bias)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        suffix = torch.zeros(layout.GRID_SEQUENCE_LENGTH, dtype=torch.long)
        suffix[torch.arange(9, 89, 10)] = layout.ROW_SEPARATOR
        self.register_buffer("suffix_template", suffix, persistent=False)
        self.register_buffer(
            "solution_positions", layout.solution_cell_positions(), persistent=False
        )
        head_dimension = dimension // heads
        inverse_frequency = 1.0 / (
            10_000 ** (torch.arange(0, head_dimension, 2, dtype=torch.float32) / head_dimension)
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    def _rotary(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(layout.TOTAL_SEQUENCE_LENGTH, device=device, dtype=torch.float32)
        angles = torch.outer(positions, self.inverse_frequency)
        return angles.cos(), angles.sin()

    def forward(
        self,
        prompt: torch.Tensor,
        noisy: torch.Tensor,
        signal: torch.Tensor,
    ) -> torch.Tensor:
        batch = prompt.shape[0]
        suffix = self.suffix_template.expand(batch, -1)
        tokens = torch.cat([prompt, suffix], dim=1)
        hidden = self.token_embedding(tokens)
        normalized_noisy = normalize_input(
            noisy,
            signal,
            self.process.input_epsilon,
            self.process.poisson_rate,
        )
        projected = self.noisy_projection(normalized_noisy[..., None]).to(hidden.dtype)
        hidden[:, self.solution_positions] = projected

        sigma = sigma_from_signal(signal.float(), self.process.signal_epsilon)
        condition = F.silu(self.time_embedding(sigma))
        cosine, sine = self._rotary(hidden.device)
        for block in self.blocks:
            hidden = block(hidden, condition, cosine, sine)
        hidden = hidden.index_select(1, self.solution_positions)
        shift, scale = self.final_modulation(condition).chunk(2, dim=-1)
        hidden = _modulate(self.final_norm(hidden), shift, scale)
        return self.output(hidden)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
