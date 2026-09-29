import torch
import numpy as np


def _auto_skellam_scales(dist, c: float = 1.2):
    """
    Default Rd_max/Ru_max for a Skellam-reference process (UpDownZdNoise,
    CountBridgeNoise), derived from the target distribution `dist` (any
    object exposing .probs(), an exact PMF over 0..S-1, e.g. datasets.py's
    UnivariateDiscreteDistribution) via two conditions:
      (1) spread: the Skellam std = sqrt(Rd_max+Ru_max) should be
          comparable to the data's own spread, so the noise actually
          washes out x0 rather than leaving sample_terminal()'s
          approximation data-dependent. Data spread is measured as the
          0.5-99.5 percentile range of the exact PMF (robust to
          multimodality, unlike raw std, which underestimates the gap
          between separated modes).
      (2) drift: the Skellam mean = Ru_max-Rd_max should match the data's
          mean, so the noise is centered where the data already is instead
          of also needing to "travel" there -- this keeps Rd_max/Ru_max
          from growing unnecessarily large just to shift the center.
    Solving Ru_max-Rd_max=mean, Ru_max+Rd_max=(c*range)^2 gives Ru_max,
    Rd_max below. A visual reference-PMF match at t=T can look fine well
    before the x0-dependence is actually gone -- a quantitative TV-distance
    check (P_T | low-x0 vs P_T | high-x0) on a hard bimodal target showed
    even c=1.0 (std comparable to the full data range) still leaves ~24% of
    the mass distinguishable by origin, so c=1.2 is used as a somewhat
    safer default, not a proven minimum -- verify with plot_bridge_marginals.py.
    """
    probs = dist.probs().detach().cpu().numpy().squeeze()
    xs = np.arange(len(probs))
    mean = float((xs * probs).sum())
    cdf = np.cumsum(probs)
    lo_idx = np.clip(np.searchsorted(cdf, 0.005), 0, len(cdf) - 1)
    hi_idx = np.clip(np.searchsorted(cdf, 0.995), 0, len(cdf) - 1)
    data_range = float(xs[hi_idx] - xs[lo_idx])
    spread_sq = (c * data_range) ** 2
    Ru_max = 0.5 * (spread_sq + mean)
    Rd_max = 0.5 * (spread_sq - mean)
    return max(Rd_max, 1.0), max(Ru_max, 1.0)


# ============================================================
# Forward process: Poisson-Folmer bridge
# ============================================================

class PoissonFolmer:
    def __init__(self, S: int, T0: float, T: float, dim: int):
        self.S   = S
        self.T0  = T0
        self.T   = T
        self.dim = dim

    def sample_initial(self, N: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(N, self.dim, dtype=torch.long, device=device)

    def conditional_sample(self, t: torch.Tensor, XT: torch.Tensor) -> torch.Tensor:
        # Sample from P[X_t = k|X_T = m]
        assert(XT.shape[1] == self.dim)
        assert(XT.shape[0] == len(t))
        return torch.distributions.Binomial(
            total_count=XT.T, probs=t / self.T
        ).sample().long().T

    def reference_up_rate(self, t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Known, state-independent birth "clock" rate R_t(x, x+1) of the fixed
        reference process underlying the Binomial bridge q_t(x|x0) =
        Bin(x0, t/T): with p(t) = t/T, each of the x0 particles is born at
        an i.i.d. time uniform on [0, T], so the hazard rate at time t is
        R_t = p'(t) / (1 - p(t)) = 1 / (T - t), independent of x0 and of the
        current count x. Used as the R_t term in the Appendix F.2
        predictor-corrector rate R^c_t = Rhat^theta_t + R_t (the R̂ term is
        supplied separately by the trained intensity model).
        """
        return 1.0 / (self.T - t).clamp(min=eps)


# ============================================================
# Forward process: genuine additive-Poisson noising (pure superposition,
# no thinning)
# ============================================================
#
#   P_t = X_0 + Pois(Ru(t)),   t in [T0, T]
#
# Unlike UpDownNdNoise, X_0 is never thinned -- the forward process only
# ever adds mass, so X_0 <= P_t always and X_0 is exactly recoverable from
# P_t in principle. This is the "genuine additive-Poisson process" that
# the reflected-binomial construction of "poisson_only" (see
# train_paths.py) was originally built to avoid, because
# P_T = X_0 + Pois(Ru(T)) is an exact translation of X_0 by however much
# noise has accumulated -- for ANY finite Ru(T), the terminal marginal is
# NEVER exactly independent of X_0 (unlike UpDownZdNoise/CountBridgeNoise,
# whose subtractive Poisson can exceed and erase X_0 entirely). But the
# total-variation distance between P_T | X_0=a and P_T | X_0=b shrinks like
# |a-b| / sqrt(Ru(T)) (a Poisson shifted by O(1) relative to its own std),
# so scaling Ru_max up relative to the spread of X_0 makes the terminal
# marginal APPROXIMATELY data-independent -- see auto_scale() below,
# analogous in spirit to UpDownZdNoise.auto_scales/CountBridgeNoise.
# auto_scales, but with the entire spread budget coming from Ru_max alone
# (no subtractive component to share the burden with).
#
# The reverse-time generative process this induces is pure DEATH: mass is
# only ever removed, from a (approximately) data-independent reference
# P_T down to X_0 (see model_utils.ReverseRatesAdditive).
# ============================================================
class AdditivePoissonNoise:
    def __init__(self, T0: float, T: float, dim: int, Ru_max: float, Ru_power: float = 1.0):
        self.T0        = T0
        self.T         = T
        self.dim       = dim
        self.Ru_max    = Ru_max
        self.Ru_power  = Ru_power

    @staticmethod
    def auto_scale(dist, c: float = 0.4) -> float:
        """
        Default Ru_max for this process's target `dist` (exposes .probs(),
        an exact PMF over 0..S-1, e.g. datasets.py's
        UnivariateDiscreteDistribution): the Poisson std sqrt(Ru_max)
        should be comparable to the data's own spread (0.5-99.5 percentile
        range of the exact PMF, robust to multimodality), so Ru_max =
        (c*range)^2 -- see the class docstring for why this only ever
        gives an approximate, not exact, x0-independent terminal marginal
        (unlike UpDownZdNoise.auto_scales/CountBridgeNoise.auto_scales,
        there is no subtractive Poisson here to also match the data's
        mean, so the reference is centered at Ru_max, not at the data).

        c=0.4 here (vs 1.2 for the two-Poisson Skellam recipes) is a
        deliberately looser bar than "washes out x0": Ru_max=(1.2*range)^2
        was empirically too large to *train against* in practice (a
        DenoisingMLP has to resolve x0's O(range)-scale variation on top
        of a much larger Ru_max-scale reference even with EDM rescaling)
        -- for heavier-tailed/more skewed targets (e.g.
        NegativeBinomialMixture) that pushed training into a persistent
        underfit regime where the reverse chain barely moves off its
        Ru_max-scale start, landing samples far outside the data's range.
        c=0.4 (lowered from an earlier 0.6) trades even more of the
        terminal marginal's x0-independence for a Ru_max small enough to
        actually fit at this script's default (n_epochs, num_steps);
        override with --Ru-max-Poisson if a particular target needs more
        of either.
        """
        probs = dist.probs().detach().cpu().numpy().squeeze()
        xs = np.arange(len(probs))
        cdf = np.cumsum(probs)
        lo_idx = np.clip(np.searchsorted(cdf, 0.005), 0, len(cdf) - 1)
        hi_idx = np.clip(np.searchsorted(cdf, 0.995), 0, len(cdf) - 1)
        data_range = float(xs[hi_idx] - xs[lo_idx])
        return max((c * data_range) ** 2, 1.0)

    def _frac(self, t: torch.Tensor) -> torch.Tensor:
        return ((t - self.T0) / (self.T - self.T0)).clamp(0.0, 1.0)

    def Ru(self, t: torch.Tensor) -> torch.Tensor:
        # cumulative additive Poisson rate: 0 -> Ru_max
        return self.Ru_max * self._frac(t).pow(self.Ru_power)

    def ru(self, t: torch.Tensor) -> torch.Tensor:
        # d/dt Ru(t)
        p = self.Ru_power
        return self.Ru_max * p / (self.T - self.T0) * self._frac(t).pow(p - 1.0)

    def conditional_sample(self, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """ Sample P_t = x0 + Pois(Ru(t)) given data x0. Single-tensor return
        (unlike UpDownNdNoise.conditional_sample) so this plugs directly into
        loss.denoising_loss / model.DenoisingMLP -- den(x,t) predicts X_0
        directly, since here bin_t == x0 exactly (no thinning). """
        assert x0.shape[1] == self.dim
        assert x0.shape[0] == len(t)
        ru_t = self.Ru(t)[:, None].expand_as(x0)
        return x0 + torch.poisson(ru_t)

    def sample_terminal(self, N: int, device: torch.device) -> torch.Tensor:
        """
        Approximate sample from the terminal marginal P_T, dropping the
        (approximately vanishing, for Ru_max large relative to the spread
        of X_0) dependence on x0: P_T ~approx Pois(Ru(T)).
        """
        T_tensor = torch.tensor([self.T], device=device)
        rate = self.Ru(T_tensor).expand(N, self.dim)
        return torch.poisson(rate)

    def reference_rates(self, x: torch.Tensor, t: torch.Tensor):
        """Forward birth/death rates: pure birth, forward never removes mass."""
        up_rate = self.ru(t)[:, None].expand_as(x).clamp(min=0.0)
        return up_rate, torch.zeros_like(up_rate)

    def edm_target_moments_data(self, t: torch.Tensor, mdata: float, vdata: float):
        """
        EDM target-side moments for DenoisingMLP's "dend" role
        (Z_0 = X_0, Z_sigma = P_t = X_0 + Pois(Ru(t))), mirroring
        UpDownNdNoise.edm_target_moments_data's case (3) with Rd_t == 1
        identically (no thinning here): X_0's own moments are exactly
        t-independent, and P_t's variance grows only through the additive
        Poisson term. This is what makes EDM scaling load-bearing for this
        arm (unlike PoissonFolmer, where a fixed /S rescaling suffices):
        Ru_max is scaled to (spread of X_0)^2 (auto_scale) to wash out X_0
        dependence, so P_t's dynamic range at t near T is orders of
        magnitude larger than X_0's own -- a fixed-scale input
        normalization drowns out X_0's entire range in float precision
        near the normalized input's boundary. EDM's per-t rescaling
        (cin(t)) keeps the network's effective input range calibrated to
        P_t's own (t-dependent) spread instead of a single global S.
        """
        Ru_t = self.Ru(t)
        mdata0 = torch.full_like(Ru_t, float(mdata))
        vdata0 = torch.full_like(Ru_t, float(vdata))
        cdata = vdata0  # Cov(X_0, P_t) = Var(X_0): P_t = X_0 + independent Pois_t
        mdata_sigma = mdata0 + Ru_t
        vdata_sigma = vdata0 + Ru_t
        return mdata0, vdata0, cdata, mdata_sigma, vdata_sigma


# ============================================================
# Forward process: up_down_Nd noising
# ============================================================
#
#   P_t = Bin(X_0, Rd(t)) + Pois(Ru(t)),   t in [T0, T]
#
# with Rd(T0) = 1, Rd(T) = 0, Ru(T0) = 0. Binomial thinning removes mass
# from the data over time ("jumping down") while the Poisson component
# superimposes new mass ("jumping up"), so that
#
#   P_{T0} = X_0 (data)   and   P_T ~ Pois(Ru(T)) (pure noise).
#
class UpDownNdNoise:
    def __init__(self, T0: float, T: float, dim: int, R_max: float,
                 Rd_power: float = 1.0, Ru_power: float = 1.0):
        self.T0        = T0
        self.T         = T
        self.dim       = dim
        self.R_max     = R_max
        self.Rd_power  = Rd_power
        self.Ru_power  = Ru_power

    def _frac(self, t: torch.Tensor) -> torch.Tensor:
        # fraction of the noise schedule elapsed, in [0, 1]
        return ((t - self.T0) / (self.T - self.T0)).clamp(0.0, 1.0)

    def Rd(self, t: torch.Tensor) -> torch.Tensor:
        # binomial retention probability: 1 -> 0
        return (1.0 - self._frac(t)).pow(self.Rd_power)

    def rd(self, t: torch.Tensor) -> torch.Tensor:
        # d/dt Rd(t)
        p = self.Rd_power
        return -p / (self.T - self.T0) * (1.0 - self._frac(t)).pow(p - 1.0)

    def Ru(self, t: torch.Tensor) -> torch.Tensor:
        # cumulative Poisson rate: 0 -> R_max
        return self.R_max * self._frac(t).pow(self.Ru_power)

    def ru(self, t: torch.Tensor) -> torch.Tensor:
        # d/dt Ru(t)
        p = self.Ru_power
        return self.R_max * p / (self.T - self.T0) * self._frac(t).pow(p - 1.0)

    def conditional_sample(self, t: torch.Tensor, x0: torch.Tensor):
        """
        Sample P_t = Bin(x0, Rd(t)) + Pois(Ru(t)) given data x0.

        Inputs
        ------
        t  : (B,) sampled noising times
        x0 : (B, D) data samples

        Returns
        -------
        pt    : (B, D) noised state
        bin_t : (B, D) Binomial component -- the denoising target E[Bin_t | P_t]
        """
        assert x0.shape[1] == self.dim
        assert x0.shape[0] == len(t)
        rd_t   = self.Rd(t)[:, None].expand_as(x0)
        bin_t  = torch.distributions.Binomial(total_count=x0, probs=rd_t).sample()
        ru_t   = self.Ru(t)[:, None].expand_as(x0)
        pois_t = torch.poisson(ru_t)
        return bin_t + pois_t, bin_t

    def sample_terminal(self, N: int, device: torch.device) -> torch.Tensor:
        """ Sample from the terminal (pure noise) marginal P_T ~ Pois(Ru(T)). """
        T_tensor = torch.tensor([self.T], device=device)
        rate = self.Ru(T_tensor).expand(N, self.dim)
        return torch.poisson(rate)

    def reference_rates(self, x: torch.Tensor, t: torch.Tensor, eps: float = 1e-6):
        """
        Known, closed-form birth/death rates R_t(x, x+1), R_t(x, x-1) of the
        fixed reference process underlying P_t = Bin(x0, Rd(t)) + Pois(Ru(t)):
        immigration at the state-independent Poisson rate ru(t), and
        independent per-particle thinning failure at rate -rd(t)/Rd(t) (so
        the death rate scales with the current count x). Used as the R_t
        term in the Appendix F.2 predictor-corrector rate
        R^c_t = Rhat^theta_t + R_t.
        """
        up_rate = self.ru(t)[:, None].expand_as(x).clamp(min=0.0)
        down_rate = x * (-self.rd(t)[:, None] / self.Rd(t)[:, None].clamp(min=eps))
        return up_rate, down_rate.clamp(min=0.0)

    def edm_target_moments_bin(self, t: torch.Tensor, mdata: float, vdata: float):
        """
        EDM target-side moments for the "binomial" denoiser,
        obtained by composing the binomial-thinning case
        (1) (X_0 -> Bin_t) with its additive-Poisson case (2) (Bin_t ->
        P_t). mdata, vdata: population mean/variance of the data X_0
        (assumed i.i.d. coordinates).

        Returns (mdata0, vdata0, cdata, mdata_sigma, vdata_sigma), each a
        (B,) tensor: the moments of Z_0=Bin_t, its covariance with
        Z_sigma=P_t, and Z_sigma's own moments.
        """
        Rd_t, Ru_t = self.Rd(t), self.Ru(t)
        mdata0 = Rd_t * mdata
        vdata0 = mdata * Rd_t * (1.0 - Rd_t) + Rd_t.pow(2) * vdata
        cdata = vdata0  # Cov(Bin_t, P_t) = Var(Bin_t), P_t = Bin_t + independent Pois_t
        mdata_sigma = mdata0 + Ru_t
        vdata_sigma = vdata0 + Ru_t
        return mdata0, vdata0, cdata, mdata_sigma, vdata_sigma

    def edm_target_moments_data(self, t: torch.Tensor, mdata: float, vdata: float):
        """
        EDM target-side moments for the "dend" denoiser, Z_sigma = Bin_t + Pois_t =
        P_t) -- the Bin_t + Pois_t case directly, with Z_0's own moments
        (mdata, vdata) genuinely t-independent here (unlike "den"'s Bin_t).
        """
        Rd_t, Ru_t = self.Rd(t), self.Ru(t)
        mdata0 = torch.full_like(Rd_t, float(mdata))
        vdata0 = torch.full_like(Rd_t, float(vdata))
        cdata = Rd_t * vdata  # Cov(X_0, P_t) = Cov(X_0, Bin_t) = Rd_t*Var(X_0)
        mdata_sigma = Rd_t * mdata + Ru_t
        vdata_sigma = mdata * Rd_t * (1.0 - Rd_t) + Rd_t.pow(2) * vdata + Ru_t
        return mdata0, vdata0, cdata, mdata_sigma, vdata_sigma


# ============================================================
# Forward process: up_down_Zd (Skellam) noising on Z^dim
# ============================================================
#
#   P_t = X_0 - D_t + U_t,   D_t ~ Pois(Rd(t)), U_t ~ Pois(Ru(t)),  t in [T0, T]
#
# with (X_0, D_t, U_t) independent and Rd(T0) = Ru(T0) = 0. Unlike
# UpDownNdNoise (binomial thinning, bounded to {0,...,S-1}), the
# subtracted component here is an independent Poisson (not a thinning of
# X_0), so P_t ranges over all of Z^dim -- this is the noising process for
# "up_down_Zd": both an
# additive and a subtractive Poisson perturbation are superimposed on the
# data, so P_T is approximately Skellam(Rd(T), Ru(T)), independent of X_0
# for Rd(T), Ru(T) large.
# ============================================================
class UpDownZdNoise:
    def __init__(self, T0: float, T: float, dim: int, Rd_max: float, Ru_max: float,
                 Rd_power: float = 1.0, Ru_power: float = 1.0):
        self.T0        = T0
        self.T         = T
        self.dim       = dim
        self.Rd_max    = Rd_max
        self.Ru_max    = Ru_max
        self.Rd_power  = Rd_power
        self.Ru_power  = Ru_power

    @staticmethod
    def auto_scales(dist, c: float = 1.2):
        """ Default (Rd_max, Ru_max) for this process's target `dist' -- see _auto_skellam_scales. """
        return _auto_skellam_scales(dist, c)

    def _frac(self, t: torch.Tensor) -> torch.Tensor:
        return ((t - self.T0) / (self.T - self.T0)).clamp(0.0, 1.0)

    def Rd(self, t: torch.Tensor) -> torch.Tensor:
        # cumulative subtractive Poisson rate: 0 -> Rd_max
        return self.Rd_max * self._frac(t).pow(self.Rd_power)

    def rd(self, t: torch.Tensor) -> torch.Tensor:
        # d/dt Rd(t)
        p = self.Rd_power
        return self.Rd_max * p / (self.T - self.T0) * self._frac(t).pow(p - 1.0)

    def Ru(self, t: torch.Tensor) -> torch.Tensor:
        # cumulative additive Poisson rate: 0 -> Ru_max
        return self.Ru_max * self._frac(t).pow(self.Ru_power)

    def ru(self, t: torch.Tensor) -> torch.Tensor:
        # d/dt Ru(t)
        p = self.Ru_power
        return self.Ru_max * p / (self.T - self.T0) * self._frac(t).pow(p - 1.0)

    def conditional_sample(self, t: torch.Tensor, x0: torch.Tensor):
        """
        Sample P_t = x0 - D_t + U_t given data x0, D_t ~ Pois(Rd(t)) i.i.d.
        per coordinate, U_t ~ Pois(Ru(t)) i.i.d. per coordinate.

        Inputs
        ------
        t  : (B,) sampled noising times
        x0 : (B, D) data samples (any integers, need not be non-negative)

        Returns
        -------
        pt  : (B, D) noised state, in Z^D
        d_t : (B, D) subtractive latent -- the denoising target E[D_t | P_t] (denbin)
        u_t : (B, D) additive latent    -- the denoising target E[U_t | P_t] (denpos)
        """
        assert x0.shape[1] == self.dim
        assert x0.shape[0] == len(t)
        rd_t = self.Rd(t)[:, None].expand_as(x0)
        d_t  = torch.poisson(rd_t)
        ru_t = self.Ru(t)[:, None].expand_as(x0)
        u_t  = torch.poisson(ru_t)
        return x0 - d_t + u_t, d_t, u_t

    def sample_terminal(self, N: int, device: torch.device) -> torch.Tensor:
        """
        Approximate sample from the terminal marginal P_T, ignoring the
        (vanishing, for Rd(T), Ru(T) large) dependence on x0:
        P_T ~approx Skellam(Rd(T), Ru(T)) = Pois(Ru(T)) - Pois(Rd(T)).
        """
        T_tensor = torch.tensor([self.T], device=device)
        rd_rate = self.Rd(T_tensor).expand(N, self.dim)
        ru_rate = self.Ru(T_tensor).expand(N, self.dim)
        return torch.poisson(ru_rate) - torch.poisson(rd_rate)

    def reference_rates(self, x: torch.Tensor, t: torch.Tensor):
        """Forward birth/death rates used by the optional PC corrector."""
        up_rate = self.ru(t)[:, None].expand_as(x).clamp(min=0.0)
        down_rate = self.rd(t)[:, None].expand_as(x).clamp(min=0.0)
        return up_rate, down_rate

    def edm_target_moments(self, t: torch.Tensor, latent: str, mdata: float, vdata: float):
        """
        EDM target-side moments for the up_down_Zd denoisers
        (P_t = X_0 - D_t + U_t, D_t ~ Pois(Rd(t)), U_t ~ Pois(Ru(t)), all
        mutually independent):
        (latent='D', Z_0 = D_t) or "denpos" (latent='U', Z_0 = U_t), both
        observing the same Z_sigma = P_t. mdata, vdata: population mean/
        variance of the data X_0 (assumed i.i.d. coordinates).

        Returns (mdata0, vdata0, cdata, mdata_sigma, vdata_sigma), each a
        (B,) tensor.
        """
        Rd_t, Ru_t = self.Rd(t), self.Ru(t)
        mdata_sigma = mdata - Rd_t + Ru_t
        vdata_sigma = vdata + Rd_t + Ru_t
        if latent == 'D':
            mdata0, vdata0 = Rd_t, Rd_t
            cdata = -Rd_t  # Cov(D_t, P_t) = -Var(D_t): D_t independent of (X_0, U_t)
        elif latent == 'U':
            mdata0, vdata0 = Ru_t, Ru_t
            cdata = Ru_t   # Cov(U_t, P_t) = Var(U_t): U_t independent of (X_0, D_t)
        else:
            raise ValueError(f"latent must be 'D' or 'U', got {latent!r}")
        return mdata0, vdata0, cdata, mdata_sigma, vdata_sigma


# ============================================================
# Forward process: categorical uniform noising (Lou, Meng & Ermon 2023,
# "Discrete Diffusion Modeling by Estimating the Ratio of Data
# Distributions", https://arxiv.org/abs/2310.16834, Appendix C.1's
# "uniform" transition matrix)
# ============================================================
#
# Every other forward process in this file is a birth-death CTMC on the
# integer LATTICE: jumps only ever go to a neighboring count x+-1, so the
# ordering of the state space matters (rate x -> x+2 requires two steps).
# CategoricalUniformNoise instead treats {0,...,S-1} as a complete graph of
# UNORDERED, exchangeable labels: at total noise level sigma_bar(t) each
# coordinate independently either stays at x0 (prob alpha(t) =
# exp(-sigma_bar(t))) or is resampled uniformly from all S categories (prob
# 1-alpha(t)), with closed-form marginal
#
#   p_t(y|x0) = alpha(t) * 1[y=x0] + (1-alpha(t))/S.
#
# This is the ordinal-blind baseline in this repo's comparisons: it is
# trained not with a conditional-mean denoiser (which implicitly assumes
# "closer counts are more likely confused") but with the paper's score
# entropy / density-ratio matching loss (see
# loss.denoising_loss_categorical), using a backbone from the same family as the
# other arms (model.CategoricalScoreModel).
# ============================================================
class CategoricalUniformNoise:
    def __init__(self, T0: float, T: float, dim: int, S: int, sigma_max: float = 8.0):
        self.T0 = T0
        self.T = T
        self.dim = dim
        self.S = S
        self.sigma_max = sigma_max  # total noise level at t=T; alpha(T) = exp(-sigma_max) ~ 0

    def _frac(self, t: torch.Tensor) -> torch.Tensor:
        return ((t - self.T0) / (self.T - self.T0)).clamp(0.0, 1.0)

    def sigma_bar(self, t: torch.Tensor) -> torch.Tensor:
        """ Cumulative noise level: 0 -> sigma_max, linear in the elapsed fraction. """
        return self.sigma_max * self._frac(t)

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """ Probability of staying at x0: exp(-sigma_bar(t)), 1 -> ~0. """
        return torch.exp(-self.sigma_bar(t))

    def kernel(self, y: torch.Tensor, x0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """ p_t(y|x0) = alpha(t)*1[y=x0] + (1-alpha(t))/S. Broadcasts t over y/x0's shape. """
        a = self.alpha(t)
        while a.dim() < y.dim():
            a = a.unsqueeze(-1)
        return a * (y == x0).float() + (1.0 - a) / self.S

    def conditional_sample(self, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """ Sample X_t: keep x0 w.p. alpha(t), else resample uniformly from {0,...,S-1}. """
        assert x0.shape[1] == self.dim
        a = self.alpha(t)[:, None].expand_as(x0)
        keep = torch.bernoulli(a).bool()
        rand_cat = torch.randint(0, self.S, x0.shape, device=x0.device).float()
        return torch.where(keep, x0, rand_cat)

    def sample_terminal(self, N: int, device: torch.device) -> torch.Tensor:
        """ Terminal marginal P_T is (approximately, for sigma_max large) discrete-uniform on {0,...,S-1}. """
        return torch.randint(0, self.S, (N, self.dim), device=device).float()


def _poisson_diff_conditional(d: torch.Tensor, lam_plus: torch.Tensor, lam_minus: torch.Tensor,
                               max_tries: int = 60):
    """
    Exact rejection sampling of (U, D) ~ Poisson(lam_plus), Poisson(lam_minus)
    independent, conditioned on U - D = d. This is the elementary identity
    underlying the "Count Bridges" Poisson birth-death bridge kernel
    (Fishman et al. 2026, https://arxiv.org/pdf/2603.04730, Sec. 3, eq. 8-9):
    conditioning two independent Poissons on their difference reproduces
    exactly the Bessel-distributed slack M = min(U, D) they sample directly
    via a dedicated (CUDA) Bessel sampler -- min(U, D) ~ Bes(|d|; lam_plus,
    lam_minus). Rejection sampling is exact but its acceptance probability
    is the Skellam pmf P(U - D = d), so it is only practical for the modest
    rates/gaps used in this repo's synthetic comparisons, not at the scale
    the original paper targets (hence their custom sampler).
    """
    U = torch.zeros_like(d)
    D = torch.zeros_like(d)
    pending = torch.ones_like(d, dtype=torch.bool)
    for _ in range(max_tries):
        if not pending.any():
            break
        u = torch.poisson(lam_plus.expand_as(d))
        v = torch.poisson(lam_minus.expand_as(d))
        accept = pending & ((u - v) == d)
        U = torch.where(accept, u, U)
        D = torch.where(accept, v, D)
        pending = pending & ~accept
    # Fallback for any entries that failed to accept within max_tries (rare,
    # only for extreme gaps relative to lam_plus/lam_minus): the minimal
    # (M=0) split consistent with the required difference.
    U = torch.where(pending, d.clamp(min=0.0), U)
    D = torch.where(pending, (-d).clamp(min=0.0), D)
    return U, D


def _bridge_thin(x_from: torch.Tensor, x_ref: torch.Tensor,
                  lam_plus_ref: torch.Tensor, lam_minus_ref: torch.Tensor,
                  w_query: torch.Tensor, w_ref: torch.Tensor, eps: float = 1e-8):
    """
    One Poisson birth-death bridge step (Fishman et al. 2026, Algorithm 1
    lines 4-10 / Algorithm 2 lines 3-10): given the state x_ref at schedule
    value w_ref (cumulative birth/death intensities lam_plus_ref,
    lam_minus_ref since the OTHER endpoint x_from, at schedule value 0),
    returns an exact sample of the state at intermediate schedule value
    w_query in [0, w_ref].
    """
    d_ref = x_ref - x_from
    U_ref, D_ref = _poisson_diff_conditional(d_ref, lam_plus_ref, lam_minus_ref)
    N_ref, B_ref = U_ref + D_ref, U_ref

    r = (w_query / w_ref.clamp(min=eps)).clamp(0.0, 1.0).expand_as(N_ref)
    N_query = torch.distributions.Binomial(total_count=N_ref, probs=r).sample()

    # Hypergeometric(N_ref, B_ref, N_query): torch has no vectorized
    # hypergeometric sampler, so batch through numpy (exact, just not GPU).
    import numpy as _np
    rng = _np.random.default_rng()
    N_ref_np, B_ref_np, N_query_np = (t.detach().cpu().numpy().astype(_np.int64)
                                       for t in (N_ref, B_ref, N_query))
    bad_ref_np = N_ref_np - B_ref_np
    B_query_np = rng.hypergeometric(ngood=B_ref_np, nbad=bad_ref_np, nsample=N_query_np)
    B_query = torch.as_tensor(B_query_np, dtype=x_ref.dtype, device=x_ref.device)

    return x_ref - 2.0 * (B_ref - B_query) + (N_ref - N_query)


# ============================================================
# Forward process: Count Bridges (Poisson birth-death bridge on Z^dim)
# ============================================================
#
# Simplified reproduction of the core forward process of "Count Bridges"
# (Fishman et al. 2026, https://arxiv.org/pdf/2603.04730): an EXACT
# Poisson birth-death bridge between data X_0 and a noise/source endpoint
# X_1 (Algorithm 1), as opposed to UpDownZdNoise above, which only
# specifies independent marginal noise at each t (an approximate bridge,
# see its "approx." terminal marginal). Two simplifications relative to
# the paper, both documented at point of use: (i) the denoiser here is
# trained as a conditional-mean regressor (matching this repo's other
# arms, via the shared model.DenoisingMLP/loss.denoising_loss machinery)
# rather than their full distributional model trained with a strictly
# proper energy score; (ii) the exact Bessel-distributed bridge slack is
# obtained here via rejection sampling (see _poisson_diff_conditional)
# rather than their dedicated (CUDA) Bessel sampler, which is exact but
# does not scale to the rates/dimensions the paper targets.
# ============================================================
class CountBridgeNoise:
    def __init__(self, T0: float, T: float, dim: int, Rd_max: float, Ru_max: float, power: float = 1.0):
        self.T0     = T0
        self.T      = T
        self.dim    = dim
        self.Rd_max = Rd_max
        self.Ru_max = Ru_max
        self.power  = power

    @staticmethod
    def auto_scales(dist, c: float = 1.2):
        """ Default (Rd_max, Ru_max) for this process's target `dist' -- see _auto_skellam_scales. """
        return _auto_skellam_scales(dist, c)

    def w(self, t: torch.Tensor) -> torch.Tensor:
        """ Shared schedule 0 -> 1 governing both the up- and down-intensity. """
        return ((t - self.T0) / (self.T - self.T0)).clamp(0.0, 1.0).pow(self.power)

    def sample_reference(self, N: int, device: torch.device) -> torch.Tensor:
        """ Draw the noise/source endpoint X_1 ~ Skellam(Rd_max, Ru_max). """
        return (torch.poisson(torch.full((N, self.dim), self.Ru_max, device=device))
                - torch.poisson(torch.full((N, self.dim), self.Rd_max, device=device)))

    def conditional_sample(self, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        """
        Sample X_t from the exact bridge between x0 (endpoint at w=0) and a
        freshly-drawn reference X_1 (endpoint at w=1), following Algorithm 1.
        Returns only X_t: the denoising target is x0 itself (trained with
        the shared loss.denoising_loss, same as PoissonFolmer/DenoisingMLP).
        """
        assert x0.shape[1] == self.dim
        x1 = self.sample_reference(x0.shape[0], device=x0.device)
        lam_plus = torch.full((x0.shape[0], 1), self.Ru_max, device=x0.device)
        lam_minus = torch.full((x0.shape[0], 1), self.Rd_max, device=x0.device)
        w_t = self.w(t)[:, None]
        w_1 = torch.ones_like(w_t)
        return _bridge_thin(x0, x1, lam_plus, lam_minus, w_t, w_1)

    def sample_terminal(self, N: int, device: torch.device) -> torch.Tensor:
        return self.sample_reference(N, device)

    def bridge_step(self, x0_hat: torch.Tensor, x_tk: torch.Tensor, t_k: torch.Tensor, t_km1: torch.Tensor):
        """
        One reverse-time (generative) bridge step (Algorithm 2, lines 3-10):
        given the current noisy state x_tk and a denoiser prediction x0_hat
        of the data endpoint, sample the state at the earlier time t_km1.
        """
        lam_plus = self.Ru_max * self.w(t_k)[:, None]
        lam_minus = self.Rd_max * self.w(t_k)[:, None]
        w_km1 = self.w(t_km1)[:, None]
        w_k = self.w(t_k)[:, None]
        return _bridge_thin(x0_hat, x_tk, lam_plus, lam_minus, w_km1, w_k)


# ============================================================
# Forward process: count-FM conditional bridge (Wei & Pearson 2026,
# "Flow Matching for Count Data", https://arxiv.org/pdf/2605.07746)
# ============================================================
#
# A birth-death CTMC on N_0^dim with LOCAL (+-1) jumps, transporting an
# arbitrary source distribution p0 (here: discrete-uniform on {0,...,S-1},
# matching the paper's own simulation baseline) to a target p1 (the data),
# via the coordinatewise conditional binomial bridge (eq. 1):
#
#   X_t^(i) = x0^(i) + sgn(x1^(i) - x0^(i)) * Binomial(|x1^(i) - x0^(i)|, t)
#
# with t in [0, 1] (t=0 -> source/prior, t=1 -> target/data -- the SAME
# convention as PoissonFolmer above, opposite of UpDownNdNoise /
# UpDownZdNoise / CountBridgeNoise, which all use t=T0 -> data).
# Unlike those other forward classes, this one is not a fixed noising
# process for a single data distribution; it is a bridge CONSTRUCTOR
# between whatever two count-valued endpoints (x0, x1) are passed in, so
# there is no analogue of conditional_sample(t, x0) alone -- see
# conditional_sample(t, x0, x1) and conditional_rates below (paper's eq. 1
# and the display equation directly following eq. 2, respectively).
# This class only implements the process; the KL rate-matching training
# loss (eq. 4) and the first-order birth-death sampler (Sec. 2.2) are
# orchestration code, kept in loss.py and sampling.py alongside the analogous
# code for the other forward processes.
# ============================================================
class CountFMBridge:
    def __init__(self, dim: int, S: int):
        self.dim = dim
        self.S = S      # bound for the discrete-uniform source distribution
        self.T0 = 0.0   # t=0 -> source (prior)
        self.T = 1.0    # t=1 -> target (data)

    def sample_source(self, N: int, device: torch.device) -> torch.Tensor:
        """ p0: discrete-uniform on {0,...,S-1}^dim (paper's own simulation baseline). """
        return torch.randint(0, self.S, (N, self.dim), device=device).float()

    def conditional_sample(self, t: torch.Tensor, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """ X_t from the conditional binomial bridge between x0 (t=0) and x1 (t=1), eq. 1. """
        d = x1 - x0
        t_exp = t[:, None].expand_as(d).clamp(0.0, 1.0)
        B_t = torch.distributions.Binomial(total_count=d.abs(), probs=t_exp).sample()
        return x0 + torch.sign(d) * B_t

    def conditional_rates(self, x: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, eps_t: float = 1e-3):
        """
        Analytic conditional birth/death rates (display equation following
        eq. 2): lambda_t,i(x|x0,x1) = (x1-x)_+ / (1-t+eps_t), mu_t,i(x|x0,x1)
        = (x-x1)_+ / (1-t+eps_t). Note these depend on x1 and the current
        state x only, not on x0 directly (x0 only shapes which x's are
        reachable via the bridge).
        """
        denom = (1.0 - t[:, None] + eps_t)
        lam = (x1 - x).clamp(min=0.0) / denom
        mu = (x - x1).clamp(min=0.0) / denom
        return lam, mu
