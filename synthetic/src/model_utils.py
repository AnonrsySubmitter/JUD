import torch
import torch.nn as nn

# define intensity model
class Intensity(nn.Module):
    def __init__(self, model, max_time, S, denormalize):
        super(Intensity, self).__init__()
        self.model = model
        self.max_time = max_time
        self.S = S
        self.normalize_targets = denormalize

    def forward(self, x, t, ctx=None):
        assert(t.ndim == 1)
        if ctx is not None:
            model_output = self.model(ctx, x, t)
        else:
            model_output = self.model(x, t)
        if self.normalize_targets:
            model_output *= self.S
        return (model_output - x.float()) / (self.max_time - t[:,None])


# define reverse-time rates for the up_down_Nd noising process
class ReverseRates(nn.Module):
    """
    Computes the reverse-time birth/death rates of the up_down_Nd
    noising process (forward.UpDownNdNoise) from a trained denoiser
    den(x, t) = E[Bin_t | P_t = x]:

        rate(x -> x-1) = (ru(t)/Ru(t)) * (x - den(x, t))
        rate(x -> x+1) = -rd(t) * (Ru(t)/Rd(t)) * den(x+1,t) / (x+1-den(x+1,t))

    Returns a tuple (up_rate, down_rate) for use with the birth-death
    samplers in sampling.py.
    """
    def __init__(self, model, forward, eps: float = 1e-6):
        super(ReverseRates, self).__init__()
        self.model = model
        self.forward_process = forward
        self.eps = eps

    def forward(self, x, t, ctx=None):
        assert(t.ndim == 1)

        ru_t = self.forward_process.ru(t)[:, None]
        Ru_t = self.forward_process.Ru(t)[:, None].clamp(min=self.eps)
        rd_t = self.forward_process.rd(t)[:, None]
        Rd_t = self.forward_process.Rd(t)[:, None].clamp(min=self.eps)

        den_x = self.model(ctx, x, t) if ctx is not None else self.model(x, t)
        down_rate = (ru_t / Ru_t) * (x - den_x).clamp(min=0.0)

        x_plus = x + 1.0
        den_x_plus = self.model(ctx, x_plus, t) if ctx is not None else self.model(x_plus, t)
        denom = (x_plus - den_x_plus).clamp(min=self.eps)
        up_rate = (-rd_t) * (Ru_t / Rd_t) * den_x_plus / denom

        return up_rate.clamp(min=0.0), down_rate


# define division-free reverse-time rates for the up_down_Nd noising
# process from TWO directly-trained denoisers (den, dend)
class ReverseRatesTwo(nn.Module):
    """
    Computes the reverse-time birth/death rates of the up_down_Nd noising
    process (forward.UpDownNdNoise) from two directly-trained denoisers,
    den(x, t) = E[Bin_t | P_t = x] and dend(x, t) = E[X_0 | P_t = x],
    using the division-free (in network outputs) up_rate:

        down_rate(x -> x-1) = (ru(t)/Ru(t)) * (x - den(x, t))
        up_rate(x -> x+1)   = -rd(t)/(1-Rd(t)) * (dend(x, t) - den(x, t))

    Unlike ReverseRates (single denoiser den), up_rate here never divides
    by a noisy network output -- only by the deterministic schedule
    quantity 1-Rd(t) -- so it does not suffer ReverseRates' numerical
    fragility (den(x+1,t) evaluated at a shifted input, with a
    denominator that can collapse toward 0). den and dend are evaluated
    at the SAME x (no x+1 shift needed).

    Returns a tuple (up_rate, down_rate) for use with the birth-death
    samplers in sampling.py.
    """
    def __init__(self, model_den, model_dend, forward, eps: float = 1e-6):
        super(ReverseRatesTwo, self).__init__()
        self.model_den = model_den
        self.model_dend = model_dend
        self.forward_process = forward
        self.eps = eps

    def forward(self, x, t, ctx=None):
        assert(t.ndim == 1)

        ru_t = self.forward_process.ru(t)[:, None]
        Ru_t = self.forward_process.Ru(t)[:, None].clamp(min=self.eps)
        rd_t = self.forward_process.rd(t)[:, None]
        Rd_t = self.forward_process.Rd(t)[:, None]

        den_x = self.model_den(ctx, x, t) if ctx is not None else self.model_den(x, t)
        down_rate = (ru_t / Ru_t) * (x - den_x).clamp(min=0.0)

        dend_x = self.model_dend(ctx, x, t) if ctx is not None else self.model_dend(x, t)
        denom = (1.0 - Rd_t).clamp(min=self.eps)
        up_rate = (-rd_t / denom) * (dend_x - den_x)

        return up_rate.clamp(min=0.0), down_rate


class ReverseRatesZTwo(nn.Module):
    """Stable Z^d reverse rates using directly trained down/up latent denoisers
    (see model.TwoHeadZDenoiser: model_down = as_down_module(), model_up =
    as_up_module())."""
    def __init__(self, model_down, model_up, forward, eps: float = 1e-6):
        super(ReverseRatesZTwo, self).__init__()
        self.model_down = model_down
        self.model_up = model_up
        self.forward_process = forward
        self.eps = eps

    def forward(self, x, t, ctx=None):
        rd = self.forward_process.rd(t)[:, None]
        Rd = self.forward_process.Rd(t)[:, None].clamp(min=self.eps)
        ru = self.forward_process.ru(t)[:, None]
        Ru = self.forward_process.Ru(t)[:, None].clamp(min=self.eps)
        up = (rd / Rd) * self.model_down(x, t)
        down = (ru / Ru) * self.model_up(x, t)
        return up.clamp(min=0.0), down.clamp(min=0.0)


# define reverse-time rates for the genuine additive-Poisson noising process
class ReverseRatesAdditive(nn.Module):
    """
    Computes the reverse-time birth/death rates of the genuine
    additive-Poisson noising process (forward.AdditivePoissonNoise,
    P_t = X_0 + Pois(Ru(t)), no thinning) from a trained denoiser
    den(x, t) = E[X_0 | P_t = x] (model.DenoisingMLP -- here bin_t == x0
    exactly, so this is the same architecture/loss as "binomial"):

        rate(x -> x-1) = (ru(t)/Ru(t)) * (x - den(x, t))
        rate(x -> x+1) = 0   (forward never adds mass in reverse; the
                              process only ever removes the superimposed
                              Poisson noise, ending at X_0)

    Returns a tuple (up_rate, down_rate) for use with the birth-death
    samplers in sampling.py (clamp_min=0.0, N_0-valued state space).
    """
    def __init__(self, model, forward, eps: float = 1e-6):
        super(ReverseRatesAdditive, self).__init__()
        self.model = model
        self.forward_process = forward
        self.eps = eps

    def forward(self, x, t, ctx=None):
        assert(t.ndim == 1)

        ru_t = self.forward_process.ru(t)[:, None]
        Ru_t = self.forward_process.Ru(t)[:, None].clamp(min=self.eps)

        den_x = self.model(ctx, x, t) if ctx is not None else self.model(x, t)
        down_rate = (ru_t / Ru_t) * (x - den_x).clamp(min=0.0)

        return torch.zeros_like(down_rate), down_rate


# define reverse-time rates for the up_down_Zd (Skellam) noising process on Z^d
class ReverseRatesZ(nn.Module):
    """
    Computes the reverse-time birth/death rates on Z^d of the up_down_Zd
    noising process (forward.UpDownZdNoise), P_t = x0 - D_t + U_t, from
    a single trained latent denoiser denbin(x, t) = E[D_t | P_t = x]
   :

        rate(x -> x+e_i) = (rd^i(t) / Rd^i(t)) * denbin^i(x, t)
        rate(x -> x-e_i) = (ru^i(t) / Ru^i(t)) * denpos^i(x, t)

    where the companion latent denoiser denpos is recovered from denbin via
    the Tweedie-type identity

        denpos^i(x, t) = Ru^i(t) * Rd^i(t) / denbin^i(x - e_i, t).

    Returns a tuple (up_rate, down_rate) for use with the birth-death
    samplers in sampling.py (TauLeapingBirthDeath/EulerBirthDeath with
    clamp_min=None, since the state space is Z^d, not N_0^d).
    """
    def __init__(self, model, forward, eps: float = 1e-6):
        super(ReverseRatesZ, self).__init__()
        self.model = model
        self.forward_process = forward
        self.eps = eps

    def forward(self, x, t, ctx=None):
        assert(t.ndim == 1)

        ru_t = self.forward_process.ru(t)[:, None]
        Ru_t = self.forward_process.Ru(t)[:, None].clamp(min=self.eps)
        rd_t = self.forward_process.rd(t)[:, None]
        Rd_t = self.forward_process.Rd(t)[:, None].clamp(min=self.eps)

        denbin_x = self.model(ctx, x, t) if ctx is not None else self.model(x, t)
        up_rate = (rd_t / Rd_t) * denbin_x.clamp(min=0.0)

        x_minus = x - 1.0
        denbin_x_minus = self.model(ctx, x_minus, t) if ctx is not None else self.model(x_minus, t)
        denpos_x = (Ru_t * Rd_t) / denbin_x_minus.clamp(min=self.eps)
        down_rate = (ru_t / Ru_t) * denpos_x

        return up_rate.clamp(min=0.0), down_rate.clamp(min=0.0)

