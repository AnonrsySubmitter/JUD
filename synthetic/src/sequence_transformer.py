# Conditional-sequence Transformer backbone.
#
# ConditionalSequenceTransformer predicts x0_pred (raw, unpostprocessed) from
# a discrete conditioning context (ctx_ids) and a continuous noisy sequence
# (xt_ids) via cross-attention; PFConditionalPrecond wraps it with EDM-style
# input/output preconditioning derived specifically for the PoissonFolmer
# (pure binomial-thinning) forward process -- NOT a generic wrapper, so it
# is not directly applicable to UpDownNdNoise/UpDownZdNoise's own denoising
# targets without deriving the analogous cin/cskip/cout formulas for those
# processes (see model.py's edm_coefficients/_ResidualBackbone._edm_predict
# for how that's done there).
import math
import torch
import torch.nn as nn


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    timesteps: [B] float or long
    returns:   [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model, dtype=torch.float32)
        pe[0, :, 0::2] = torch.sin(position.float() * div_term)
        pe[0, :, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, L, d_model]
        return self.dropout(x + self.pe[:, : x.size(1), :])


class AdaLN(nn.Module):
    """
    Adaptive LayerNorm conditioned on timestep embedding, adaLN-Zero style
    (Peebles & Xie 2023, DiT): emits (scale, shift, gate) instead of just
    (scale, shift), and to_scale_shift_gate is zero-initialized so every
    block starts as an exact identity/residual pass (gate=0 => the sublayer
    contributes nothing at init, scale=0/shift=0 => norm is a no-op on the
    path that does eventually contribute once gate moves off zero). This
    stabilizes training as depth grows, unlike the un-gated, default-init
    version this replaces -- which let every block perturb the residual
    stream non-trivially from step zero, a likely cause of deeper
    (num_layers>1) configs not translating faster loss descent into better
    samples.
    """
    def __init__(self, d_model, temb_dim):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.to_scale_shift_gate = nn.Linear(temb_dim, 3 * d_model)
        nn.init.zeros_(self.to_scale_shift_gate.weight)
        nn.init.zeros_(self.to_scale_shift_gate.bias)

    def forward(self, x, temb):
        # x: [B, L, d_model], temb: [B, temb_dim]
        # returns: (modulated_x, gate), gate: [B, 1, d_model] in (-1, 1) at
        # init (tanh of an all-zero pre-activation = 0), so the caller's
        # gate * sublayer_output residual term starts at exactly zero.
        ssg = self.to_scale_shift_gate(temb)[:, None, :]  # [B, 1, 3*d_model]
        scale, shift, gate = ssg.chunk(3, dim=-1)
        x = self.norm(x)
        return x * (1.0 + scale) + shift, torch.tanh(gate)


class FFResidual(nn.Module):
    """
    Feed-forward residual block with adaLN-Zero timestep conditioning.
    Used as a lightweight per-token refinement layer after the main decoder stack.
    """
    def __init__(self, d_model, dim_feedforward, temb_dim, dropout=0.0):
        super().__init__()
        self.adaln = AdaLN(d_model, temb_dim)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, temb):
        # x: [B, L, d_model]  temb: [B, temb_dim]
        h, gate = self.adaln(x, temb)
        return x + gate * self.ff(h)


class CrossAttentionBlock(nn.Module):
    """
    Decoder-style block:
      1) self-attn on xt
      2) cross-attn: xt queries ctx
      3) FFN
    All sublayers use timestep-conditioned adaLN-Zero (scale/shift/gate).
    """
    def __init__(self, d_model, num_heads, dim_feedforward, dropout, temb_dim):
        super().__init__()
        self.adaln_self = AdaLN(d_model, temb_dim)
        self.adaln_cross = AdaLN(d_model, temb_dim)
        self.adaln_ff = AdaLN(d_model, temb_dim)

        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, xt, ctx, temb):
        # Self-attention on xt
        h, gate = self.adaln_self(xt, temb)
        xt = xt + gate * self.self_attn(h, h, h, need_weights=False)[0]

        # Cross-attention: xt (Q) attends to ctx (K, V)
        h, gate = self.adaln_cross(xt, temb)
        xt = xt + gate * self.cross_attn(h, ctx, ctx, need_weights=False)[0]

        # Feed-forward
        h, gate = self.adaln_ff(xt, temb)
        xt = xt + gate * self.ff(h)
        return xt


class ConditionalSequenceTransformer(nn.Module):
    """
    Conditional denoiser.
      input:  ctx [B, condition_dim], xt [B, output_dim], times [B]
      output: x0_pred [B, output_dim] in normalized space
    """
    def __init__(self, cfg):
        super().__init__()
        self.condition_dim = int(cfg.data.condition_dim)
        self.total_dim = int(cfg.data.dim)
        self.output_dim = self.total_dim - self.condition_dim
        self.S = int(cfg.data.S)

        d_model = int(cfg.model.d_model)
        n_layers = int(cfg.model.num_layers)
        n_heads = int(cfg.model.num_heads)
        d_ff = int(cfg.model.dim_feedforward)
        dropout = float(cfg.model.dropout)
        # Recommended: temb_dim = 4 * d_model
        temb_dim = int(cfg.model.temb_dim)
        n_output_resid = int(getattr(cfg.model, "num_output_FFresiduals", 0))
        self.time_scale_factor = float(cfg.model.time_scale_factor)
        # If True, forward() expects sigma = -log(t) instead of t in [0,1].
        # The sinusoidal embedding is then computed directly on sigma, which
        # spreads the low-t (high-noise) regime more evenly across frequencies.
        self.sigma_input = bool(getattr(cfg.model, "sigma_input", False))
        self.log_noise_eps = float(getattr(cfg.model, "log_noise_eps", 1e-5))

        # Discrete context tokens: values in [0, S]
        self.ctx_value_embed = nn.Embedding(self.S + 1, d_model)
        # Continuous preconditioned xt: project scalar per token
        self.xt_cont_proj = nn.Linear(1, d_model)

        # Segment embeddings (0 = ctx, 1 = xt)
        self.segment_embed = nn.Embedding(2, d_model)

        self.ctx_pos = PositionalEncoding(d_model, dropout=dropout, max_len=self.condition_dim)
        self.xt_pos = PositionalEncoding(d_model, dropout=dropout, max_len=self.output_dim)

        self.temb_net = nn.Sequential(
            nn.Linear(temb_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, temb_dim),
        )

        self.blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_model=d_model,
                    num_heads=n_heads,
                    dim_feedforward=d_ff,
                    dropout=dropout,
                    temb_dim=temb_dim,
                )
                for _ in range(n_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model, eps=1e-5)

        # Optional per-token refinement after the cross-attention stack.
        # Set cfg.model.num_output_FFresiduals = 0 to disable.
        self.output_resid = nn.ModuleList([
            FFResidual(d_model, d_ff, temb_dim, dropout)
            for _ in range(n_output_resid)
        ])

        self.output_linear = nn.Linear(d_model, 1)

    def _to_embed_idx(self, x):
        """Clamp to valid embedding range. Raises during training to catch data bugs."""
        if self.training:
            assert x.min() >= 0 and x.max() <= self.S, (
                f"ctx/xt index out of range [0, {self.S}]: "
                f"min={x.min().item()}, max={x.max().item()}"
            )
        return x.clamp(min=0, max=self.S).long()

    def forward_features(self, ctx_ids, xt_ids, t):
        """
        Same inputs as forward() (see there), but returns the shared
        per-token trunk features [B, output_dim, d_model] just before the
        final output_linear projection -- lets a caller attach its own
        (possibly multiple) output heads on top of this trunk instead of
        the single scalar-per-token head below, without duplicating any of
        the embedding/cross-attention/output_resid parameters.
        """
        B = ctx_ids.size(0)
        device = ctx_ids.device

        assert ctx_ids.size(1) == self.condition_dim, (
            f"Expected ctx dim {self.condition_dim}, got {ctx_ids.size(1)}"
        )
        assert xt_ids.size(1) == self.output_dim, (
            f"Expected xt dim {self.output_dim}, got {xt_ids.size(1)}"
        )

        ctx_tokens = self.ctx_value_embed(self._to_embed_idx(ctx_ids))   # [B, ctx_len, d_model]
        xt_tokens = self.xt_cont_proj(xt_ids.float().unsqueeze(-1))       # [B, xt_len, d_model]

        ctx_seg = torch.zeros(B, self.condition_dim, dtype=torch.long, device=device)
        xt_seg = torch.ones(B, self.output_dim, dtype=torch.long, device=device)
        ctx_tokens = ctx_tokens + self.segment_embed(ctx_seg)
        xt_tokens = xt_tokens + self.segment_embed(xt_seg)

        ctx_tokens = self.ctx_pos(ctx_tokens)
        xt_tokens = self.xt_pos(xt_tokens)

        # Convert to the sinusoidal embedding argument.
        # sigma mode: times is already sigma = -log(t + eps) in [0, inf),
        #             embed directly — sigma spreads the high-noise regime
        #             more evenly across sinusoidal frequencies than raw t.
        # t mode:     scale t into a friendlier numerical range first.
        times = t.float()
        if self.sigma_input:
            emb_arg = times                              # sigma in [0, inf)
        else:
            emb_arg = times * self.time_scale_factor     # scaled t

        temb_in = timestep_embedding(emb_arg, self.temb_net[0].in_features)
        temb = self.temb_net(temb_in)

        for block in self.blocks:
            xt_tokens = block(xt_tokens, ctx_tokens, temb)

        xt_tokens = self.final_norm(xt_tokens)

        for layer in self.output_resid:
            xt_tokens = layer(xt_tokens, temb)

        return xt_tokens  # [B, output_dim, d_model]

    def forward(self, ctx_ids, xt_ids, t):
        """
        ctx:   [B, condition_dim] discrete ints (clean context)
        xt:    [B, output_dim]    continuous preconditioned noisy signal
        times: [B]                noise level — interpretation set by sigma_input:
                                    sigma_input=False  ->  t in [0, 1]
                                    sigma_input=True   ->  sigma = -log(t + eps) in [0, inf)
        returns: [B, output_dim]  raw F_x prediction (not yet postprocessed)
        """
        xt_tokens = self.forward_features(ctx_ids, xt_ids, t)
        return self.output_linear(xt_tokens).squeeze(-1)


class PFConditionalPrecond(nn.Module):
    """
    PF preconditioning wrapper for the conditional sequence denoiser.

    Takes raw noisy counts Xt and diffusion time t in [0, 1], applies
    input/output preconditioning following the EDM framework, and returns
    a denoised prediction D_x in the original (count) space.
    """
    def __init__(self, core_model, mean_data, var_data, log_noise_eps=1e-5, cin_eps=0.01):
        super().__init__()
        self.core = core_model
        self.log_noise_eps = float(log_noise_eps)
        self.mean_data = float(mean_data)
        self.var_data = float(var_data)
        self.cin_eps = float(cin_eps)

    def forward(self, ctx_ids, xt_ids, t):
        """
        ctx_ids: [B, condition_dim]  discrete context tokens (clean)
        xt_ids:  [B, output_dim]     noisy counts Xt (raw, integer-valued)
        t:       [B]                 diffusion time in [0, 1] (always t here;
                                     conversion to sigma is handled internally
                                     if core.sigma_input is True)
        returns: [B, output_dim]     denoised prediction D_x in count space
        """
        x = xt_ids.float()
        t = t.float().view(-1, 1)                         # [B, 1]

        s_in = -self.mean_data / math.sqrt(self.var_data)

        denom_in = (
            self.mean_data * t * (1.0 - t)
            + self.var_data * (t ** 2)
            + self.cin_eps
        )
        c_in = 1.0 / torch.sqrt(denom_in)                # [B, 1]
        x_in = c_in * x + s_in                           # [B, output_dim]

        # Convert t -> sigma before passing to core if it expects sigma input.
        times_for_core = t.squeeze(1)
        if self.core.sigma_input:
            times_for_core = -torch.log(times_for_core + self.core.log_noise_eps)

        F_x = self.core(
            ctx_ids=ctx_ids,
            xt_ids=x_in,
            t=times_for_core,
        )

        denom_out = (
            self.mean_data * (1.0 - t)
            + self.var_data * t
            + self.cin_eps                                # same guard as denom_in
        )
        c_skip = self.var_data / denom_out
        c_out = torch.sqrt(self.var_data * self.mean_data * (1.0 - t) / denom_out)

        D_x = c_skip * x + c_out * F_x
        return D_x
