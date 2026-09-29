import torch
import math
import torch.nn as nn
from torch.nn import functional as F

from sequence_transformer import ConditionalSequenceTransformer

def timestep_embedding(timesteps, embedding_dim, max_positions=10000):
    assert len(timesteps.shape) == 1  # and timesteps.dtype == tf.int32
    half_dim = embedding_dim // 2
    # set the embedding frequencies to form a geometric progression from 1 to max_positions^(half_dim-1)/half_dim
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode='constant')
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb

# ============================================================
# EDM (Karras et al. 2022) preconditioning: given the target
# Z_0's own moments (mdata0, vdata0), its covariance with the noised
# observation Z_sigma (cdata), and Z_sigma's moments (mdata_sigma,
# vdata_sigma), returns the coefficients of
#
#   m_theta(x, sigma) = cskip(sigma)*x + cout(sigma)*net_theta(cin(sigma)*x
#                        + sin(sigma), sigma)
#
# cin/sin/cskip/cout/w are all functions of sigma
# (here, sigma == the diffusion time t, evaluated per-example): sin(sigma)
# = -mdata_sigma(sigma)/sqrt(vdata_sigma(sigma)) centers the network's
# input Z_sigma = P_t at its OWN per-t mean, not a single fixed reference
# -- unlike cin/cskip/
# cout, which only depend on the denoiser's Z_0 target, sin depends only on
# Z_sigma's own moments, so it is identical for every denoiser row sharing
# the same forward process (den/dend/etc.), but still varies with t.
#
# cout^2 = vdata0 - cdata^2/vdata_sigma is guaranteed >= 0 by
# Cauchy-Schwarz (cdata = Cov(Z_0, Z_sigma), so cdata^2 <=
# Var(Z_0)*Var(Z_sigma) = vdata0*vdata_sigma) as long as (mdata0, vdata0,
# cdata, vdata_sigma) all come from the actual joint law of (Z_0, Z_sigma)
# at the same t -- which is how forward.*.edm_target_moments computes them.
# ============================================================
def edm_input_rescale(x: torch.Tensor, t: torch.Tensor, edm_moments_fn, eps: float = 1e-6) -> torch.Tensor:
    """
    EDM's INPUT-only rescaling z = cin(t)*x + sin(t), for
    models whose OUTPUT needs no EDM preconditioning (a softmax posterior
    head already returns a properly normalized distribution on the right
    support, unlike a regression target -- see CrossEntropyMLP/
    TwoHeadCategoricalNdDenoiser/SequenceTwoHeadCategoricalDenoiser). cin/
    sin only depend on Z_sigma = P_t's own moments (mdata_sigma,
    vdata_sigma), NOT on cskip/cout/w (which need the Z_0 target's own
    moments/covariance with Z_sigma, mdata0/vdata0/cdata) -- so
    edm_moments_fn(t) only needs to return (., ., ., mdata_sigma,
    vdata_sigma), unlike the full edm_coefficients below. This still
    recenters/rescales the network's input at its OWN per-t mean/std,
    instead of the legacy fixed x/S*2-1 rescaling used elsewhere in this
    file's non-EDM backbones.
    """
    _, _, _, mdata_sigma, vdata_sigma = edm_moments_fn(t)
    cin = vdata_sigma.clamp(min=eps).rsqrt()[:, None]
    sin = -mdata_sigma[:, None] * cin
    return cin * x + sin


def edm_coefficients(mdata0, vdata0, cdata, mdata_sigma, vdata_sigma, eps: float = 1e-6):
    """Returns (cin, sin, cskip, cout, w), each broadcastable to (B, 1)."""
    # evaluate c_in, s_in, c_skip, c_out
    vdata_sigma = vdata_sigma.clamp(min=eps)
    cin   = vdata_sigma.rsqrt()
    sin   = -mdata_sigma * cin
    cskip = cdata / vdata_sigma
    resid_var = vdata0 - cdata.pow(2) / vdata_sigma
    cout = resid_var.clamp(min=eps).sqrt()
    # evaluate weight
    resid_mean = mdata0 - cskip * mdata_sigma
    wlambda_denom = (resid_var + resid_mean.pow(2)).clamp(min=eps)
    wlambda = wlambda_denom.rsqrt()
    return cin, sin, cskip, cout, wlambda

class _ResidualBackbone(nn.Module):
    def __init__(self, cfg):
        """ FiLM-conditioned residual MLP shared model types. """
        super().__init__()
        # save model parameters
        self.S           = cfg.data.S
        self.D           = cfg.data.dim
        self.TIME_SCALE  = cfg.model.time_scale_factor
        self.D_MODEL     = cfg.model.d_model
        self.HIDDEN_DIM  = cfg.model.hidden_dim
        self.NUM_LAYERS  = cfg.model.num_layers
        self.TEMB_DIM    = cfg.model.temb_dim
        self.edm_scaling = bool(getattr(cfg.model, 'edm_scaling', False))
        # define all middle layers
        self.input_proj  = nn.Linear(self.D, self.D_MODEL)
        self.layers1     = nn.ModuleList([nn.Linear(self.D_MODEL, self.HIDDEN_DIM)    for _ in range(self.NUM_LAYERS)])
        self.layers2     = nn.ModuleList([nn.Linear(self.HIDDEN_DIM, self.D_MODEL)    for _ in range(self.NUM_LAYERS)])
        self.norms       = nn.ModuleList([nn.LayerNorm(self.D_MODEL)                  for _ in range(self.NUM_LAYERS)])
        self.film        = nn.ModuleList([nn.Linear(4*self.TEMB_DIM, 2*self.D_MODEL)  for _ in range(self.NUM_LAYERS)])
        self.temb_net    = nn.Sequential(
            nn.Linear(self.TEMB_DIM, self.HIDDEN_DIM), nn.ReLU(),
            nn.Linear(self.HIDDEN_DIM, 4*self.TEMB_DIM)
        )
        self.act = nn.ReLU()

    def normalize_input(self, x):
        x = x/self.S # (0, 1)
        x = x*2 - 1 # (-1, 1)
        return x

    def unnormalize_input(self, x):
        return (x + 1.0) / 2.0 * self.S

    def _backbone_core(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # z: (B, D) already-normalized network input, t: (B,) float in [0, 1]
        te = self.temb_net(timestep_embedding(t * self.TIME_SCALE, self.TEMB_DIM))
        h  = self.input_proj(z)
        for i in range(self.NUM_LAYERS):
            h        = self.norms[i](h + self.layers2[i](self.act(self.layers1[i](h))))
            gam, bet = self.film[i](te).chunk(2, dim=-1)
            h        = gam * h + bet
        return h  # (B, d_model)

    def _backbone(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: (B, 1) float in [0, S],  t: (B,) float in [0, 1]. Legacy
        # (non-EDM) input normalization: fixed /S rescaling to [-1, 1].
        return self._backbone_core(x / float(self.S) * 2.0 - 1.0, t)

    def _edm_predict(self, x: torch.Tensor, t: torch.Tensor, edm_moments_fn) -> torch.Tensor:
        """
        Shared EDM forward path for BinomialDenoiser/ZDenoiser: computes
        m_theta(x, t) = cskip(t)*x + cout(t)*net_theta(cin(t)*x + sin(t), t)
        edm_moments_fn(t) -> (mdata0, vdata0, cdata,
        mdata_sigma, vdata_sigma), all (B,) tensors.
        Caches cout, w on self for edm_lambda() (used by the loss weight).
        """
        mdata0, vdata0, cdata, mdata_sigma, vdata_sigma = edm_moments_fn(t)
        cin, sin, cskip, cout, w = edm_coefficients(mdata0, vdata0, cdata, mdata_sigma, vdata_sigma)
        cin, sin, cskip, cout = (c[:, None] for c in (cin, sin, cskip, cout))
        z = cin * x + sin
        net_out = self.out(self._backbone_core(z, t))
        self._edm_last_cout, self._edm_last_w = cout, w
        return cskip * x + cout * net_out

    def edm_lambda(self) -> torch.Tensor:
        """
        Loss weight lambda(t) = w(t)/cout(t)^2, from the most
        recent forward() call: since the training loss here is computed in
        PREDICTION space (bregman_divergence(m_theta(x,t), z0)), and
        m_theta - z0 = cout*(net_theta - net_target), multiplying the raw
        prediction-space loss by w/cout^2 recovers the intended
        w(t)*|net_theta - net_target|^2 term whose
        expectation is constant across t at initialization (net_theta=0).
        """
        assert hasattr(self, '_edm_last_cout'), "call forward() before edm_lambda()"
        return self._edm_last_w / self._edm_last_cout.squeeze(-1).pow(2)

    def _edm_input_only(self, x: torch.Tensor, t: torch.Tensor, edm_moments_fn) -> torch.Tensor:
        """ See module-level edm_input_rescale (same rescaling, factored out so
        non-_ResidualBackbone models -- e.g. SequenceTwoHeadCategoricalDenoiser --
        can reuse it too). """
        return edm_input_rescale(x, t, edm_moments_fn)


# ============================================================
# Binomial denoiser: outputs E[Bin_t | P_t] for the up_down_Nd
# noising process (see forward.UpDownNdNoise)
# ============================================================
class BinomialDenoiser(_ResidualBackbone):
    """
    Predicts den(x, t) = E[Bin_t | P_t = x] from the discrete Tweedie
    formula. The raw output is passed through a sigmoid gate and rescaled
    by x itself, so that 0 <= den(x, t) <= x holds by construction --
    matching the fact that the Binomial component can never exceed the
    observed (noised) count x.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.out = nn.Linear(self.D_MODEL, self.D)
        self._edm_forward_process = None

    def configure_edm(self, forward_process, mdata: float, vdata: float):
        """
        Required before forward()/training when cfg.model.edm_scaling is
        True. forward_process must be a forward.UpDownNdNoise instance
        (exposes edm_target_moments_bin(t, mdata, vdata)). mdata, vdata: population mean/
        variance of the data X_0 (assumed i.i.d. coordinates).
        """
        self._edm_forward_process = forward_process
        self._edm_mdata, self._edm_vdata = float(mdata), float(vdata)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.edm_scaling:
            assert self._edm_forward_process is not None, \
                "cfg.model.edm_scaling=True requires configure_edm(...) before forward()"

            def moments(t):
                return self._edm_forward_process.edm_target_moments_bin(t, self._edm_mdata, self._edm_vdata)
            # EDM parametrization is unconstrained (the EDM parametrization has no
            # positivity/gating term), unlike the legacy 0 <= den <= x
            # sigmoid-gate path below.
            return self._edm_predict(x, t, moments)
        # returns: (B, D) predicted E[Bin_t | P_t = x], in [0, x]
        h = self._backbone(x, t)
        gate = torch.sigmoid(self.out(h))  # (B, D) in (0, 1)
        return gate * x


# ============================================================
# Sequence-conditioned "two denoiser" model for UpDownNdNoise: den(ctx,x,t)
# = E[Bin_t | P_t = x] and dend(ctx,x,t) = E[X_0 | P_t = x] SHARE a single
# ConditionalSequenceTransformer trunk (cross-attention over a discrete
# conditioning context, for sequence/conditional data) and only branch into two separate linear
# output heads at the very end. This replaces the earlier design where den
# and dend were two entirely separate SequenceBinomialDenoiser/
# SequenceDataDenoiser networks (each with its own ConditionalSequenceTransformer,
# trained/optimized/checkpointed independently): here there is one set of
# trunk parameters, jointly trained from both losses, so the two targets
# only differ in their EDM preconditioning (edm_target_moments vs
# edm_target_moments_data) and
# final Linear(d_model, 1) head -- not in any earlier layer. The trunk
# still has to be evaluated once per head (den's z != dend's z, since their
# EDM cin/cskip/cout differ -- though sin is the same for both, since it
# only depends on the shared Z_sigma), so this does not save forward-pass
# compute, only
# parameters/training signal sharing. The den+dend pairing
# replaces ReverseRates' fragile in-network-output division in up_rate
# with the division-free ReverseRatesTwo formula (only division left is by the deterministic
# schedule quantity 1-Rd(t)).
# ============================================================
class SequenceTwoHeadDenoiser(nn.Module):
    """
    Single ConditionalSequenceTransformer trunk with two output heads:
    den_bin for E[Bin_t | X_t], and den_data for E[X_0 | X_t].
    Call configure_edm_den(...)/configure_edm_dend(...) once each before
    training/sampling, then forward_den(...)/forward_dend(...) to evaluate
    the corresponding head (each with its own EDM preconditioning).
    """
    def __init__(self, cfg):
        super().__init__()
        assert bool(getattr(cfg.model, 'edm_scaling', False)), \
            "SequenceTwoHeadDenoiser requires cfg.model.edm_scaling=True"
        self.edm_scaling = True
        self.core = ConditionalSequenceTransformer(cfg)
        d_model = self.core.output_linear.in_features
        # define output heads
        self.out_den_bin = nn.Linear(d_model, 1)
        self.out_den_data = nn.Linear(d_model, 1)
        del self.core.output_linear  # unused: heads above replace it
        self._edm_den_bin  = None
        self._edm_den_data = None

    def configure_edm_bin(self, forward_process, mdata: float, vdata: float):
        """ forward_process: forward.UpDownNdNoise instance (exposes edm_target_moments). """
        self._edm_den_bin = (forward_process, float(mdata), float(vdata))

    def configure_edm_data(self, forward_process, mdata: float, vdata: float):
        """ forward_process: forward.UpDownNdNoise instance (exposes edm_target_moments_data). """
        self._edm_den_data = (forward_process, float(mdata), float(vdata))

    def _predict(self, ctx_ids, x, t, edm_state, moments_fn_name, head):
        assert edm_state is not None, \
            f"configure_edm_{'den_bin' if head is self.out_den else 'den_data'}(...) must be called before forward"
        forward_process, mdata, vdata = edm_state
        moments_fn = getattr(forward_process, moments_fn_name)
        mdata0, vdata0, cdata, mdata_sigma, vdata_sigma = moments_fn(t, mdata, vdata)
        cin, sin, cskip, cout, w = edm_coefficients(mdata0, vdata0, cdata, mdata_sigma, vdata_sigma)
        cin, sin, cskip, cout = (c[:, None] for c in (cin, sin, cskip, cout))
        feats = self.core.forward_features(ctx_ids, cin * x + sin, t)  # [B, output_dim, d_model]
        net_out = head(feats).squeeze(-1)
        return cskip * x + cout * net_out, w

    def forward_den_bin(self, ctx_ids: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ returns: (B, output_dim) predicted E[Bin_t | P_t = x], unconstrained """
        out, wlambda = self._predict(ctx_ids, x, t, self._edm_den_bin, 'edm_target_moments_bin', self.out_den_bin)
        #self._edm_last_cout_den_bin, self._edm_last_w_den_bin = cout, w
        return out, wlambda

    def forward_den_data(self, ctx_ids: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ returns: (B, output_dim) predicted E[X_0 | P_t = x], unconstrained """
        out, wlambda = self._predict(ctx_ids, x, t, self._edm_den_data, 'edm_target_moments_data', self.out_den_data)
        #self._edm_last_cout_den_data, self._edm_last_w_den_data = cout, w
        return out, wlambda

    # def edm_lambda_den_bin(self) -> torch.Tensor:
    #     assert hasattr(self, '_edm_last_cout_den_bin'), "call forward_den_bin() before edm_lambda_den_bin()"
    #     return self._edm_last_w_den_bin / self._edm_last_cout_den_bin.squeeze(-1).pow(2)

    # def edm_lambda_den_data(self) -> torch.Tensor:
    #     assert hasattr(self, '_edm_last_cout_den_data'), "call forward_den_data() before edm_lambda_den_data()"
    #     return self._edm_last_w_den_data / self._edm_last_cout_den_data.squeeze(-1).pow(2)

    def as_den_module(self) -> nn.Module:
        """ (ctx, x, t) -> forward_den(...) callable, with a matching edm_lambda(); for
        model_utils.ReverseRates/ReverseRatesTwo, which expect single-target model objects. """
        return _SequenceTwoHeadView(self, 'den')

    def as_dend_module(self) -> nn.Module:
        """ Same as as_den_module(), but for the dend head. """
        return _SequenceTwoHeadView(self, 'dend')


class _SequenceTwoHeadView(nn.Module):
    """ Adapts one head of a SequenceTwoHeadDenoiser to the plain (ctx,x,t)-callable
    + edm_lambda() interface expected by loss functions and model_utils.ReverseRates*. """
    def __init__(self, two_head_model: SequenceTwoHeadDenoiser, head: str):
        super().__init__()
        assert head in ('den', 'dend')
        self.model = two_head_model
        self.head = head
        self.edm_scaling = True

    def forward(self, ctx_ids, x, t):
        fn = self.model.forward_den_bin if self.head == 'den' else self.model.forward_den_data
        out, _wlambda = fn(ctx_ids, x, t)
        return out

    def edm_lambda(self):
        fn = self.model.edm_lambda_den if self.head == 'den' else self.model.edm_lambda_dend
        return fn()


# ============================================================
# Sequence-conditioned "two denoiser" model for UpDownNdNoise, POSTERIOR
# (cross-entropy) variant: same ConditionalSequenceTransformer trunk as
# SequenceTwoHeadDenoiser (identical embeddings, cross-attention blocks,
# output_resid), but
# each head predicts a full categorical POSTERIOR p(Bin_t^(i) | P_t) /
# p(X_0^(i) | P_t) over {0,...,S-1} per output token i (a softmax head),
# instead of a single EDM-preconditioned conditional expectation. This is
# "one posterior for each head, for each output" -- output_dim independent
# per-token categoricals per head, since the shared Linear(d_model, S) head
# is applied per-token to the trunk's [B, output_dim, d_model] features.
# Trained with cross-entropy against the true class, SUMMED over the
# output_dim tokens
# rather than the quadratic Bregman divergence used for
# SequenceTwoHeadDenoiser's regression heads. The OUTPUT needs no EDM
# preconditioning: softmax already returns a properly normalized
# distribution on the right support, so there is no cskip/cout/w to derive.
# The INPUT can still use EDM's cin(t)/sin(t) rescaling (see
# edm_input_rescale/CrossEntropyMLP), shared by both heads since it only
# depends on P_t's own moments -- default when cfg.model.edm_scaling is
# True; the legacy fixed /S
# rescaling to [-1, 1] is used otherwise.
# ============================================================
class SequenceTwoHeadCategoricalDenoiser(nn.Module):
    """
    Single ConditionalSequenceTransformer trunk with two softmax heads:
    out_den_bin_logits for p(Bin_t | P_t), out_den_data_logits for
    p(X_0 | P_t). forward_den_bin/forward_den_data return each head's
    posterior MEAN (for model_utils.ReverseRatesTwo, which only consumes
    E[Bin_t|P_t]/E[X_0|P_t]); forward_den_bin_logits/forward_den_data_logits
    return the full (B, output_dim, S) logits for the cross-entropy loss.
    """
    def __init__(self, cfg):
        super().__init__()
        self.core = ConditionalSequenceTransformer(cfg)
        d_model = self.core.output_linear.in_features
        self.S = self.core.S
        # define output heads: one S-way softmax posterior per output token
        self.out_den_bin_logits = nn.Linear(d_model, self.S)
        self.out_den_data_logits = nn.Linear(d_model, self.S)
        del self.core.output_linear  # unused: heads above replace it
        self.register_buffer('_classes', torch.arange(self.S, dtype=torch.float32))
        self.edm_scaling = bool(getattr(cfg.model, 'edm_scaling', False))
        self._edm_bin = None

    def configure_edm_bin(self, forward_process, mdata: float, vdata: float):
        """
        Required before forward_*_logits()/forward_den_*()/training when
        cfg.model.edm_scaling is True. forward_process must expose
        edm_target_moments_bin(t, mdata, vdata) -- only its mdata_sigma/
        vdata_sigma outputs are used (see edm_input_rescale), which are
        shared by BOTH heads (unlike SequenceTwoHeadDenoiser's regression
        heads, there is no separate configure_edm_data: den/dend's INPUT
        rescaling is identical, since it depends only on P_t's own law,
        not on the Bin_t/X_0 target). mdata, vdata: population mean/
        variance of the data X_0.
        """
        self._edm_bin = (forward_process, float(mdata), float(vdata))

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        # Legacy fixed /S rescaling to [-1, 1] (same convention as e.g.
        # BinomialDenoiser's legacy, non-EDM path), used when
        # cfg.model.edm_scaling is False.
        return (x / float(self.S)) * 2.0 - 1.0

    def _input(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.edm_scaling:
            return self._normalize_input(x)
        assert self._edm_bin is not None, \
            "cfg.model.edm_scaling=True requires configure_edm_bin(...) before forward"
        forward_process, mdata, vdata = self._edm_bin
        return edm_input_rescale(x, t, lambda t: forward_process.edm_target_moments_bin(t, mdata, vdata))

    def _logits(self, ctx_ids: torch.Tensor, x: torch.Tensor, t: torch.Tensor, head: nn.Module) -> torch.Tensor:
        feats = self.core.forward_features(ctx_ids, self._input(x, t), t)  # [B, output_dim, d_model]
        return head(feats)  # [B, output_dim, S]

    def forward_den_bin_logits(self, ctx_ids: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ returns: (B, output_dim, S) logits of the posterior p(Bin_t | P_t = x) """
        return self._logits(ctx_ids, x, t, self.out_den_bin_logits)

    def forward_den_data_logits(self, ctx_ids: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ returns: (B, output_dim, S) logits of the posterior p(X_0 | P_t = x) """
        return self._logits(ctx_ids, x, t, self.out_den_data_logits)

    def _mean(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)  # (B, output_dim, S)
        return probs @ self._classes           # (B, output_dim)

    def forward_den_bin(self, ctx_ids: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ returns: (B, output_dim) E[Bin_t | P_t = x] under the predicted posterior --
        the scalar model_utils.ReverseRatesTwo needs for reverse-time sampling. """
        return self._mean(self.forward_den_bin_logits(ctx_ids, x, t))

    def forward_den_data(self, ctx_ids: torch.Tensor, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ returns: (B, output_dim) E[X_0 | P_t = x] under the predicted posterior. """
        return self._mean(self.forward_den_data_logits(ctx_ids, x, t))

    def as_den_module(self) -> nn.Module:
        """ (ctx, x, t) -> forward_den(...) callable (posterior mean); for
        model_utils.ReverseRatesTwo, which expects single-target model objects. """
        return _SequenceTwoHeadCategoricalView(self, 'den')

    def as_dend_module(self) -> nn.Module:
        """ Same as as_den_module(), but for the dend head. """
        return _SequenceTwoHeadCategoricalView(self, 'dend')


class _SequenceTwoHeadCategoricalView(nn.Module):
    """ Adapts one head's posterior MEAN to the plain (ctx,x,t)-callable interface
    expected by model_utils.ReverseRatesTwo (which never calls edm_lambda, unlike
    ReverseRates -- so unlike _SequenceTwoHeadView, no edm_lambda() is needed here). """
    def __init__(self, two_head_model: SequenceTwoHeadCategoricalDenoiser, head: str):
        super().__init__()
        assert head in ('den', 'dend')
        self.model = two_head_model
        self.head = head

    def forward(self, ctx_ids, x, t):
        fn = self.model.forward_den_bin if self.head == 'den' else self.model.forward_den_data
        return fn(ctx_ids, x, t)


# ============================================================
# Z-denoiser: outputs E[D_t | P_t] for the up_down_Zd (Skellam)
# noising process on Z^d (see forward.UpDownZdNoise)
# ============================================================
class ZDenoiser(_ResidualBackbone):
    """
    Predicts denbin(x, t) = E[D_t | P_t = x],
    where D_t ~ Pois(Rd(t)) is an independent (not thinned) subtractive
    Poisson latent. Unlike BinomialDenoiser, x can be negative and D_t is
    not bounded by x, so the output is only constrained to be non-negative
    (via softplus), not gated by x itself. cfg.data.S is used purely as an
    input-normalization scale here (e.g. a few multiples of
    sqrt(Rd_max + Ru_max)), not a hard bound on the state space.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.out = nn.Linear(self.D_MODEL, self.D)
        self._edm_forward_process = None

    def configure_edm(self, forward_process, latent: str, mdata: float, vdata: float):
        """
        Required before forward()/training when cfg.model.edm_scaling is
        True. forward_process must be a forward.UpDownZdNoise instance
        (exposes edm_target_moments(t, latent, mdata, vdata)). latent:
        'D' for denbin (Z_0 = D_t), 'U' for
        denpos (Z_0 = U_t). mdata, vdata: population mean/variance of the
        data X_0 (assumed i.i.d. coordinates).
        """
        assert latent in ('D', 'U'), f"latent must be 'D' or 'U', got {latent!r}"
        self._edm_forward_process = forward_process
        self._edm_latent = latent
        self._edm_mdata, self._edm_vdata = float(mdata), float(vdata)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.edm_scaling:
            assert self._edm_forward_process is not None, \
                "cfg.model.edm_scaling=True requires configure_edm(...) before forward()"

            def moments(t):
                return self._edm_forward_process.edm_target_moments(
                    t, self._edm_latent, self._edm_mdata, self._edm_vdata)
            # EDM's sin(t) (see edm_coefficients) replaces the legacy +S/2
            # recentering below, and the EDM parametrization has no positivity
            # constraint, unlike the softplus path.
            return self._edm_predict(x, t, moments)
        # returns: (B, D) predicted E[D_t | P_t = x], >= 0
        # _backbone.normalize_input assumes x in [0, S] (x/S*2-1 -> [-1,1]);
        # shift the symmetric-around-0 Z-domain input by S/2 first so that
        # range maps correctly ([-S/2, S/2] -> [-1, 1]) instead of badly
        # distorting/asymmetrically scaling negative x.
        h = self._backbone(x + self.S / 2.0, t)
        return F.softplus(self.out(h))


# ============================================================
# Two-head "two denoiser" models for the plain-MLP (non-sequence) synthetic
# experiments: share one
# _ResidualBackbone trunk between both denoiser targets and only branch
# into separate output heads/EDM preconditioning at the very end -- the
# MLP analogue of SequenceTwoHeadDenoiser above (see its docstring for the
# general rationale: one set of trunk parameters trained jointly from both
# losses, instead of two entirely separate networks/optimizers/EMAs).
# ============================================================
class TwoHeadNdDenoiser(_ResidualBackbone):
    """
    Shared trunk for the up_down_Nd pair: den(x,t) = E[Bin_t|P_t=x] (the
    BinomialDenoiser target) and dend(x,t) = E[X_0|P_t=x] (the
    DenoisingMLP target). Supports both cfg.model.edm_scaling=True (call
    configure_edm_den/dend before forward_den/dend) and the legacy non-EDM
    branch (den: sigmoid-gated by x as in BinomialDenoiser; dend:
    unconstrained affine as in DenoisingMLP -- no configure_* needed).
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.out_den = nn.Linear(self.D_MODEL, self.D)
        self.out_dend = nn.Linear(self.D_MODEL, self.D)
        self._edm_den = None
        self._edm_dend = None

    def configure_edm_den(self, forward_process, mdata: float, vdata: float):
        """ forward_process must expose edm_target_moments_bin(t, mdata, vdata) (row "den"). """
        self._edm_den = (forward_process, float(mdata), float(vdata))

    def configure_edm_dend(self, forward_process, mdata: float, vdata: float):
        """ forward_process must expose edm_target_moments_data(t, mdata, vdata) (row "dend"). """
        self._edm_dend = (forward_process, float(mdata), float(vdata))

    def _predict(self, x, t, edm_state, moments_fn_name, head):
        assert edm_state is not None, "call configure_edm_{den,dend}(...) before forward"
        forward_process, mdata, vdata = edm_state
        mdata0, vdata0, cdata, mdata_sigma, vdata_sigma = \
            getattr(forward_process, moments_fn_name)(t, mdata, vdata)
        cin, sin, cskip, cout, w = edm_coefficients(mdata0, vdata0, cdata, mdata_sigma, vdata_sigma)
        cin, sin, cskip, cout = (c[:, None] for c in (cin, sin, cskip, cout))
        z = cin * x + sin
        net_out = head(self._backbone_core(z, t))
        return cskip * x + cout * net_out, cout, w

    def forward_den(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.edm_scaling:
            h = self._backbone(x, t)
            return torch.sigmoid(self.out_den(h)) * x  # in [0, x], as BinomialDenoiser's legacy path
        out, cout, w = self._predict(x, t, self._edm_den, 'edm_target_moments_bin', self.out_den)
        self._edm_last_cout_den, self._edm_last_w_den = cout, w
        return out

    def forward_dend(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.edm_scaling:
            h = self._backbone(x, t)
            return self.unnormalize_input(self.out_dend(h))  # as DenoisingMLP's legacy path
        out, cout, w = self._predict(x, t, self._edm_dend, 'edm_target_moments_data', self.out_dend)
        self._edm_last_cout_dend, self._edm_last_w_dend = cout, w
        return out

    def edm_lambda_den(self) -> torch.Tensor:
        assert hasattr(self, '_edm_last_cout_den'), "call forward_den() before edm_lambda_den()"
        return self._edm_last_w_den / self._edm_last_cout_den.squeeze(-1).pow(2)

    def edm_lambda_dend(self) -> torch.Tensor:
        assert hasattr(self, '_edm_last_cout_dend'), "call forward_dend() before edm_lambda_dend()"
        return self._edm_last_w_dend / self._edm_last_cout_dend.squeeze(-1).pow(2)

    def as_den_module(self) -> nn.Module:
        """ (x, t) -> forward_den(...) callable, with a matching edm_lambda(), for
        model_utils.ReverseRates / ReverseRatesTwo. """
        return _TwoHeadMLPView(self, 'den')

    def as_dend_module(self) -> nn.Module:
        return _TwoHeadMLPView(self, 'dend')


class _TwoHeadMLPView(nn.Module):
    """ Adapts one head of a TwoHeadNdDenoiser/TwoHeadZDenoiser to the plain
    (x,t)-callable + edm_lambda() interface expected by loss functions and
    model_utils.ReverseRates / ReverseRatesTwo. """
    def __init__(self, two_head_model: nn.Module, head: str):
        super().__init__()
        assert head in ('den', 'dend', 'down', 'up')
        self.model = two_head_model
        self.head = head
        # mirrors the underlying model's actual edm_scaling (may be False for
        # TwoHeadZDenoiser's legacy softplus branch), so _apply_loss_weight(...)-
        # style callers only pick up edm_lambda() when it's actually available
        self.edm_scaling = bool(getattr(two_head_model, 'edm_scaling', True))

    def forward(self, x, t):
        return getattr(self.model, f'forward_{self.head}')(x, t)

    def edm_lambda(self):
        return getattr(self.model, f'edm_lambda_{self.head}')()


# ============================================================
# Fully-categorical two-head "two denoiser" model for the up_down_Nd pair:
# unlike TwoHeadNdDenoiser (both heads trained by regression), BOTH heads
# here are trained by cross-entropy -- den(x,t) = E[Bin_t|P_t=x] and
# dend(x,t) = E[X_0|P_t=x] are each recovered as the MEAN of their own full
# categorical posterior over {0,...,S-1} (two independent softmax heads),
# the MLP analogue of a two-head sequence categorical denoiser. No EDM preconditioning is needed for
# either head (as CrossEntropyMLP): softmax already returns a properly
# normalized distribution on the right support.
# ============================================================
class TwoHeadCategoricalNdDenoiser(_ResidualBackbone):
    """
    Shared trunk for the up_down_Nd pair, both heads via cross-entropy:
    out_den_logits for p(Bin_t | P_t), out_dend_logits for p(X_0 | P_t).
    forward_den/forward_dend return each head's posterior MEAN (for
    model_utils.ReverseRatesTwo); forward_den_logits/forward_dend_logits
    return the full (B, S) logits for the cross-entropy loss. Neither
    head's OUTPUT needs EDM preconditioning (see CrossEntropyMLP), but the
    shared trunk's INPUT can still use EDM's cin(t)/sin(t) rescaling (see
    _ResidualBackbone._edm_input_only) when cfg.model.edm_scaling is True
    -- identical for both heads, since cin/sin only depend on P_t's own
    moments, not on the Bin_t/X_0 target.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.out_den_logits = nn.Linear(self.D_MODEL, self.S)
        self.out_dend_logits = nn.Linear(self.D_MODEL, self.S)
        self.register_buffer('_classes', torch.arange(self.S, dtype=torch.float32))
        self._edm_forward_process = None

    def configure_edm(self, forward_process, mdata: float, vdata: float):
        """ See CrossEntropyMLP.configure_edm -- same moments, shared by both heads. """
        self._edm_forward_process = forward_process
        self._edm_mdata, self._edm_vdata = float(mdata), float(vdata)

    def _hidden(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.edm_scaling:
            return self._backbone(x, t)  # legacy (/S*2-1) input rescaling, same convention as CrossEntropyMLP
        assert self._edm_forward_process is not None, \
            "cfg.model.edm_scaling=True requires configure_edm(...) before forward"

        def moments(t):
            return self._edm_forward_process.edm_target_moments_bin(t, self._edm_mdata, self._edm_vdata)
        return self._backbone_core(self._edm_input_only(x, t, moments), t)

    def forward_den_logits(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ returns: (B, S) logits of the posterior p(Bin_t | P_t = x). """
        return self.out_den_logits(self._hidden(x, t))

    def forward_dend_logits(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ returns: (B, S) logits of the posterior p(X_0 | P_t = x). """
        return self.out_dend_logits(self._hidden(x, t))

    def _mean(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)  # (B, S)
        return probs @ self._classes[:, None]  # (B, 1)

    def forward_den(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ returns: (B, 1) E[Bin_t | P_t = x] under the predicted posterior --
        the scalar model_utils.ReverseRatesTwo needs for reverse-time sampling. """
        return self._mean(self.forward_den_logits(x, t))

    def forward_dend(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ returns: (B, 1) E[X_0 | P_t = x] under the predicted posterior. """
        return self._mean(self.forward_dend_logits(x, t))

    def as_den_module(self) -> nn.Module:
        """ (x, t) -> forward_den(...) callable (posterior mean); for
        model_utils.ReverseRatesTwo, which expects single-target model objects. """
        return _TwoHeadCategoricalMLPView(self, 'den')

    def as_dend_module(self) -> nn.Module:
        """ Same as as_den_module(), but for the dend head. """
        return _TwoHeadCategoricalMLPView(self, 'dend')

    def as_den_logits_module(self) -> nn.Module:
        """ (x, t) -> forward_den_logits(...) callable, for the cross-entropy
        loss used to train the den head. """
        return _TwoHeadCategoricalMLPLogitsView(self, 'den')

    def as_dend_logits_module(self) -> nn.Module:
        """ Same as as_den_logits_module(), but for the dend head. """
        return _TwoHeadCategoricalMLPLogitsView(self, 'dend')


class _TwoHeadCategoricalMLPView(nn.Module):
    """ Adapts one head's posterior MEAN of a TwoHeadCategoricalNdDenoiser to the
    plain (x,t)-callable interface expected by model_utils.ReverseRatesTwo (which
    never calls edm_lambda, unlike ReverseRates -- so unlike _TwoHeadMLPView, no
    edm_lambda() is needed here). """
    def __init__(self, two_head_model: TwoHeadCategoricalNdDenoiser, head: str):
        super().__init__()
        assert head in ('den', 'dend')
        self.model = two_head_model
        self.head = head
        self.edm_scaling = False

    def forward(self, x, t):
        return getattr(self.model, f'forward_{self.head}')(x, t)


class _TwoHeadCategoricalMLPLogitsView(nn.Module):
    """ Same as _TwoHeadCategoricalMLPView, but returns the full (B, S) logits
    instead of the posterior mean -- for the cross-entropy loss that trains
    that head. """
    def __init__(self, two_head_model: TwoHeadCategoricalNdDenoiser, head: str):
        super().__init__()
        assert head in ('den', 'dend')
        self.model = two_head_model
        self.head = head

    def forward(self, x, t):
        return getattr(self.model, f'forward_{self.head}_logits')(x, t)


class TwoHeadZDenoiser(_ResidualBackbone):
    """
    Shared trunk for the up_down_Zd pair: denbin(x,t) = E[D_t|P_t=x] and
    denpos(x,t) = E[U_t|P_t=x] (see ZDenoiser -- both heads use the same
    softplus-constrained-nonnegative output as ZDenoiser, only differing in
    which latent ('D' vs 'U') their EDM preconditioning targets). Supports
    both cfg.model.edm_scaling=True (call configure_edm_down/up before
    forward_down/up) and the legacy non-EDM softplus branch (same
    S/2-recentering as ZDenoiser's own legacy path, no configure_* needed).
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.out_down = nn.Linear(self.D_MODEL, self.D)
        self.out_up = nn.Linear(self.D_MODEL, self.D)
        self._edm_down = None
        self._edm_up = None

    def configure_edm_down(self, forward_process, mdata: float, vdata: float):
        """ forward_process must expose edm_target_moments(t, 'D', mdata, vdata). """
        self._edm_down = (forward_process, float(mdata), float(vdata))

    def configure_edm_up(self, forward_process, mdata: float, vdata: float):
        """ forward_process must expose edm_target_moments(t, 'U', mdata, vdata). """
        self._edm_up = (forward_process, float(mdata), float(vdata))

    def _predict(self, x, t, edm_state, latent, head):
        assert edm_state is not None, "call configure_edm_{down,up}(...) before forward"
        forward_process, mdata, vdata = edm_state
        mdata0, vdata0, cdata, mdata_sigma, vdata_sigma = \
            forward_process.edm_target_moments(t, latent, mdata, vdata)
        cin, sin, cskip, cout, w = edm_coefficients(mdata0, vdata0, cdata, mdata_sigma, vdata_sigma)
        cin, sin, cskip, cout = (c[:, None] for c in (cin, sin, cskip, cout))
        z = cin * x + sin
        net_out = head(self._backbone_core(z, t))
        return cskip * x + cout * net_out, cout, w

    def forward_down(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.edm_scaling:
            h = self._backbone(x + self.S / 2.0, t)
            return F.softplus(self.out_down(h))
        out, cout, w = self._predict(x, t, self._edm_down, 'D', self.out_down)
        self._edm_last_cout_down, self._edm_last_w_down = cout, w
        return out

    def forward_up(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.edm_scaling:
            h = self._backbone(x + self.S / 2.0, t)
            return F.softplus(self.out_up(h))
        out, cout, w = self._predict(x, t, self._edm_up, 'U', self.out_up)
        self._edm_last_cout_up, self._edm_last_w_up = cout, w
        return out

    def edm_lambda_down(self) -> torch.Tensor:
        assert hasattr(self, '_edm_last_cout_down'), "call forward_down() before edm_lambda_down()"
        return self._edm_last_w_down / self._edm_last_cout_down.squeeze(-1).pow(2)

    def edm_lambda_up(self) -> torch.Tensor:
        assert hasattr(self, '_edm_last_cout_up'), "call forward_up() before edm_lambda_up()"
        return self._edm_last_w_up / self._edm_last_cout_up.squeeze(-1).pow(2)

    def as_down_module(self) -> nn.Module:
        return _TwoHeadMLPView(self, 'down')

    def as_up_module(self) -> nn.Module:
        return _TwoHeadMLPView(self, 'up')


# ============================================================
# count-FM rate model: outputs local birth/death rates (lambda, mu) for
# the count-FM conditional bridge (see forward.CountFMBridge; Wei & Pearson
# 2026, "Flow Matching for Count Data", https://arxiv.org/pdf/2605.07746)
# ============================================================
class CountFMRates(_ResidualBackbone):
    """
    Predicts local birth rate lambda_theta(x, t) and death rate
    mu_theta(x, t) = x * beta_theta(x, t) (Sec. 2.1): both lambda and the
    death coefficient beta are constrained non-negative via softplus, and
    the x-multiplication on mu guarantees mu_theta(x, t) = 0 whenever
    x = 0 coordinatewise (no death below zero), matching the paper's
    boundary condition without needing an explicit mask.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.out_lambda = nn.Linear(self.D_MODEL, self.D)
        self.out_beta   = nn.Linear(self.D_MODEL, self.D)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # returns: (lambda, mu), each (B, D), both >= 0
        h = self._backbone(x, t)
        lam = F.softplus(self.out_lambda(h))
        beta = F.softplus(self.out_beta(h))
        mu = x * beta
        return lam, mu


# ============================================================
# Denoising model: outputs scalar E[X_T | X_t]
# ============================================================
class DenoisingMLP(_ResidualBackbone):
    """
    Predicts E[X_T | X_t] as a scalar in approximately [0, S].
    Trained with MSE loss against the true x0.
    """
    def __init__(self, cfg): #S: int):
        super().__init__(cfg)
        self.out = nn.Linear(self.D_MODEL, self.D)
        self._edm_forward_process = None

    def configure_edm(self, forward_process, mdata: float, vdata: float):
        """
        Required before forward()/training when cfg.model.edm_scaling is
        True. forward_process must expose edm_target_moments_data(t,
        mdata, vdata) (see forward.UpDownNdNoise,
        row "dend": Z_0 = X_0, Z_sigma = P_t) -- NOT every forward process
        this class is used with supports this (e.g. forward.PoissonFolmer
        does not; leave edm_scaling off / do not call configure_edm there).
        mdata, vdata: population mean/variance of the data X_0.
        """
        self._edm_forward_process = forward_process
        self._edm_mdata, self._edm_vdata = float(mdata), float(vdata)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.edm_scaling:
            assert self._edm_forward_process is not None, \
                "cfg.model.edm_scaling=True requires configure_edm(...) before forward()"

            def moments(t):
                return self._edm_forward_process.edm_target_moments_data(t, self._edm_mdata, self._edm_vdata)
            return self._edm_predict(x, t, moments)
        # returns: (B, 1)  predicted x0, in approximately (0, S)
        h = self._backbone(x, t)
        raw = self.out(h)  # (B, 1)
        # Linear rescaling: maps 0 -> S/2 (safe midpoint init), unbounded
        return self.unnormalize_input(raw)


# ============================================================
# Cross-entropy model: outputs logits over state space
# ============================================================
class CrossEntropyMLP(_ResidualBackbone):
    """
    Outputs logits over {0, ..., S-1} for CE training.
    E[X_T | X_t] is recovered as the expectation under softmax(logits).
    The OUTPUT has no EDM preconditioning (a softmax head already returns a
    properly normalized distribution on the right support), but the INPUT
    can still use EDM's cin(t)/sin(t) rescaling (see
    _ResidualBackbone._edm_input_only) instead of the legacy fixed
    x/S*2-1 rescaling, when cfg.model.edm_scaling is True.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.out = nn.Linear(self.D_MODEL, self.S)
        self._edm_forward_process = None

    def configure_edm(self, forward_process, mdata: float, vdata: float):
        """
        Required before forward()/training when cfg.model.edm_scaling is
        True. forward_process must expose edm_target_moments_bin(t, mdata,
        vdata) -- only its mdata_sigma/vdata_sigma outputs are used (see
        _edm_input_only), which are identical regardless of which row
        (den/dend) is queried, since they depend only on P_t's own law, not
        on the Bin_t/X_0 target. mdata, vdata: population mean/variance of
        the data X_0.
        """
        self._edm_forward_process = forward_process
        self._edm_mdata, self._edm_vdata = float(mdata), float(vdata)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # returns: (B, S) logits
        if self.edm_scaling:
            assert self._edm_forward_process is not None, \
                "cfg.model.edm_scaling=True requires configure_edm(...) before forward()"

            def moments(t):
                return self._edm_forward_process.edm_target_moments_bin(t, self._edm_mdata, self._edm_vdata)
            h = self._backbone_core(self._edm_input_only(x, t, moments), t)
        else:
            h = self._backbone(x, t)
        return self.out(h)

    def expected_x0(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x, t)                                         # (B, S)
        probs  = F.softmax(logits, dim=-1)                                  # (B, S)
        k      = torch.arange(self.S, dtype=torch.float32, device=x.device) # (S,)
        return (probs * k).sum(dim=-1, keepdim=True)                        # (B, 1)


# ============================================================
# Categorical concrete-score model (Lou, Meng & Ermon 2023, "Discrete
# Diffusion Modeling by Estimating the Ratio of Data Distributions",
# https://arxiv.org/abs/2310.16834): outputs s_theta(x,t)[.,y] ~=
# p_t(y)/p_t(x) for every category y in {0,...,S-1} and every coordinate,
# for the UNORDERED forward.CategoricalUniformNoise kernel (a complete
# graph on the S categories, so every y != x is a "neighbor" -- unlike the
# ordinal denoisers above, ratio matching needs a score at every category,
# not just a single conditional mean).
# ============================================================
class CategoricalScoreModel(_ResidualBackbone):
    """
    Same FiLM residual backbone as the other models in this file; only the
    output head differs, producing a full (dim, S) row of positive
    concrete scores per example rather than a single scalar/gated output.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.out = nn.Linear(self.D_MODEL, self.D * self.S)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # returns: (B, D, S) positive concrete scores s_theta(x,t)[.,y]
        h = self._backbone(x, t)
        raw = self.out(h).view(-1, self.D, self.S)
        return F.softplus(raw) + 1e-6


if __name__ == "__main__":

    # define model parameters
    import ml_collections
    cfg = ml_collections.ConfigDict()
    cfg.NUM_LAYERS         = 3   
    cfg.D_MODEL            = 128  
    cfg.HIDDEN_DIM         = 256  
    cfg.TEMB_DIM           = 128  
    cfg.S                  = 32  
    cfg.TIME_SCALE         = 100
    cfg.D                  = 1
    batch_size = 8

    # define inputs
    x = torch.randn(batch_size, cfg.D)
    t = torch.randint(0, 1000, (batch_size,)).float()
    print(x.shape)
    print(t.shape)

    # evaluate models
    model = DenoisingMLP(cfg)
    print(model(x, t).shape)  # (batch_size, spatial_dim)

    model = CrossEntropyMLP(cfg)
    print(model(x, t).shape)  # (batch_size, S)

    # BinomialDenoiser uses the nested cfg.data / cfg.model layout
    cfg2 = ml_collections.ConfigDict()
    cfg2.data                        = ml_collections.ConfigDict()
    cfg2.data.S                      = 32
    cfg2.data.dim                    = 1
    cfg2.model                       = ml_collections.ConfigDict()
    cfg2.model.num_layers            = 3
    cfg2.model.d_model               = 128
    cfg2.model.hidden_dim            = 256
    cfg2.model.temb_dim              = 128
    cfg2.model.time_scale_factor     = 100

    x = torch.rand(batch_size, cfg2.data.dim) * cfg2.data.S
    t = torch.rand(batch_size)
    model = BinomialDenoiser(cfg2)
    den = model(x, t)
    print(den.shape)  # (batch_size, dim)
    assert torch.all(den >= 0) and torch.all(den <= x)
