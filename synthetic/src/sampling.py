import torch
from torch import Tensor
from typing import Optional

# ------------------------------------------------------------
# Measure-preserving corrector for a pure-birth rate(t) = vec_rate(t)
# (Prop. 4 in Campbell): corr(t) = vec_rate(t) + cev_rate(t), so the
# corrected process has rate(t) + corr(t) = 2*vec_rate(t) + cev_rate(t)
# -- the birth rate doubles and a companion death rate cev_rate is added
# -- while still preserving the marginals of the original pure-birth
# process.
#
# For the Poisson-Follmer intensity vec_rate(t,x) = ints(t,x) used by
# Euler/TauLeaping below (model_utils.Intensity), the discrete score
# identity flow_t(x+e_i)/flow_t(x) = ints^i(t,x) * t/(x^i+1) (see
# the accompanying paper) gives, via
# cev_rate(x) = vec_rate(x-e_i) * flow_t(x-e_i)/flow_t(x),
# the closed form
#
#     cev_rate(t, x) = x / t
#
# with no model evaluation and no free hyperparameter -- x/t is exact
# (given an exact intensity model). corrector_rate scales corr(t) as a
# whole: 0 disables it, 1 applies the literal Prop. 4 corrector.
# ------------------------------------------------------------

def vec_cev_corrector_rate(x: Tensor, t_tensor: Tensor, eps: float = 1e-6) -> Tensor:
    """ cev_rate(t, x) = x / t for the Poisson-Follmer pure-birth intensity. """
    down_rate = x.float() / t_tensor[:, None].clamp(min=eps)
    return torch.where(x > 0, down_rate, torch.zeros_like(down_rate))


def _euler_birth_death_step(
    x: Tensor,
    up_rate: Tensor,
    down_rate: Tensor,
    h: Tensor,
    S: int,
    device,
) -> Tensor:
    """
    Single Euler update for a birth-death CTMC on {0,...,S-1}: builds the
    (B, D, S) rate matrix implied by (up_rate, down_rate) at the current
    state x and samples the next state from the resulting one-step
    transition probabilities. Shared by the predictor step and the F.2
    corrector sub-steps in Euler().
    """
    B, D = x.shape[0], x.shape[1]

    x_curr = x.long()                                          # (B, D)
    x_plus = (x_curr + 1).clamp(max=S - 1)                     # (B, D)
    x_minus = (x_curr - 1).clamp(min=0)                        # (B, D)

    rates = torch.zeros((B, D, S), device=device)
    batch_idx = torch.arange(B, device=device)[:, None]        # (B, 1)
    dim_idx   = torch.arange(D, device=device)[None, :]        # (1, D)
    rates[batch_idx, dim_idx, x_plus.long()]  += up_rate
    rates[batch_idx, dim_idx, x_minus.long()] += down_rate
    rates[batch_idx, dim_idx, x_curr.long()]  += -(up_rate + down_rate)

    x_next = x.clone()
    for i in range(D):
        # One-hot over current state: (B, S)
        p_cond_i = torch.eye(S, device=device)[x[:, i].long()]    # (B, S)
        p_cond_i = p_cond_i + h * rates[:, i, :]
        p_cond_i = p_cond_i.clamp(min=0.0)                        # ensure non-negative
        p_cond_i = p_cond_i / p_cond_i.sum(dim=-1, keepdim=True)  # normalize over the state space
        # Sample new state and increment time
        x_next[:, i] = torch.multinomial(p_cond_i, num_samples=1).squeeze(-1)

    return x_next


def Euler(
    forward,
    model,
    time_grid: Tensor,
    x0: Tensor,
    device,
    conditioner=None,
    corrector_rate: float = 0.0,
    pc_corrector_steps: int = 0,
    pc_late_stage_start: float = 0.9,
    pc_corrector_tau_scale: float = 0.1,
):
    """
    Euler scheme for sampling CMTC, with an optional measure-preserving
    corrector (Prop. 4 in Campbell, see module docstring above) added to
    the pure-birth intensity, and an optional literal Appendix F.2
    predictor-corrector scheme (Campbell et al.,
    https://arxiv.org/pdf/2205.14987): each corrector sub-step builds the
    Euler transition probabilities from R^c_t = Rhat^theta_t + R_t, i.e.
    the model intensity plus the *known* closed-form reference-process
    rate (forward.reference_up_rate), rather than the ad hoc Prop. 4
    corrector above. See TauLeaping for the tau-leaping version of the
    same F.2 scheme.

    Inputs:
      forward: Properties of forward process
      conditioner: Second model input
      model : Provides transition(), rate(), and p(x0|xt)
      time_grid : Tensor Monotonically increasingg time grid [0,...,T]
      corrector_rate : magnitude of corr(t) = vec_rate(t) + cev_rate(t) to add
                        (0 disables it, 1 applies the full Prop. 4 corrector)
      pc_corrector_steps : Number of literal Appendix F.2 corrector sub-steps per
                            predictor step in the late stage
      pc_late_stage_start : Fraction of sampling trajectory after which the F.2 corrector starts
      pc_corrector_tau_scale : Step-size scaling (relative to h) for the F.2 corrector sub-steps

    Returns
    -------
      x0_hat : ndarray (N, D)
      x_hist : ndarray (K, N, D)
    """


    # define
    S = forward.S

    # initialize x and t
    x_hist = [x0.clone()]
    x = x0.clone()
    B, D = x.shape[0], x.shape[1]
    total_steps = max(len(time_grid) - 1, 1)

    for step in range(len(time_grid)-1):
        h = time_grid[step+1] - time_grid[step]

        # Evaluate model at (x, t) to get intensities at x
        t_tensor = torch.tensor([time_grid[step]], dtype=torch.float32, device=device)
        t_tensor = t_tensor.repeat(x.shape[0])                     # (B,)
        with torch.no_grad():
            if conditioner is None:
                intensities = model(x.float(), t_tensor)
            else:
                intensities = model(x.float(), t_tensor, conditioner)

        # Mask out jumps that would exceed S-1
        state_mask = (x.long() + 1) < S                            # (B, D)
        intensities = intensities * state_mask.to(intensities)

        up_rate = intensities * (1.0 + corrector_rate)
        if corrector_rate != 0.0:
            down_rate = corrector_rate * vec_cev_corrector_rate(x, t_tensor)
        else:
            down_rate = torch.zeros_like(intensities)

        x = _euler_birth_death_step(x, up_rate, down_rate, h, S, device)

        # Literal Appendix F.2 predictor-corrector sub-steps: R^c_t = Rhat^theta_t + R_t
        frac_done = (step + 1) / total_steps
        if pc_corrector_steps > 0 and frac_done >= pc_late_stage_start:
            h_c = h * pc_corrector_tau_scale
            for _ in range(pc_corrector_steps):
                with torch.no_grad():
                    if conditioner is None:
                        pc_rate = model(x.float(), t_tensor)
                    else:
                        pc_rate = model(x.float(), t_tensor, conditioner)
                state_mask = (x.long() + 1) < S
                pc_rate = pc_rate * state_mask.to(pc_rate)
                pc_rate = pc_rate + forward.reference_up_rate(t_tensor)[:, None]
                x = _euler_birth_death_step(x, pc_rate, torch.zeros_like(pc_rate), h_c, S, device)

        x_hist.append(x.clone())

    return (
        x.detach().cpu(),
        torch.stack(x_hist).detach().cpu()
    )

def _tau_leap_step(
    x: Tensor,
    rates: Tensor,
    tau: float,
    max_state: Optional[int] = None,
    reject_multi_jumps: bool = False,
    max_expected_jumps: float = 0.5,
) -> Tensor:
    """Single tau-leaping state update."""
    rates_pos = rates.clamp(min=0.0)

    # Stabilize numerics so late-time intensities do not explode.
    if max_expected_jumps is not None:
        rates_pos = rates_pos.clamp(max=max_expected_jumps / max(tau, 1e-8))

    # Categorical setting: reject multi-jumps by sampling at most one jump.
    if reject_multi_jumps:
        jump_prob = 1.0 - torch.exp(-rates_pos * tau)
        jumps = torch.bernoulli(jump_prob)
    else:
        jumps = torch.poisson(rates_pos * tau)

    # Update state
    x_next = x + jumps
    if max_state is not None:
        x_next = x_next.clamp(max=max_state)
    return x_next

def TauLeaping(
    forward,
    model,
    time_grid: Tensor,
    x0: Tensor,
    device,
    conditioner=None,
    corrector_steps: int = 0,
    late_stage_start: float = 0.9,
    corrector_tau_scale: float = 0.1,
    reject_multi_jumps: bool = False,
    corrector_noise_scale: float = 0.0,
    max_expected_jumps: float = 0.5,
    corrector_rate: float = 0.0,
    pc_corrector_steps: int = 0,
    pc_late_stage_start: float = 0.9,
    pc_corrector_tau_scale: float = 0.1,
):
    """
    Tau-leaping sampler with optional late-stage predictor-corrector updates,
    an optional measure-preserving corrector (Prop. 4 in Campbell, see
    the docstring above Euler()) added to the pure-birth intensity, and an
    optional literal Appendix F.2 predictor-corrector scheme (Campbell et
    al., https://arxiv.org/pdf/2205.14987): each corrector step simulates
    the CTMC with rate R^c_t = Rhat^theta_t + R_t, i.e. the learned reverse
    rate plus the *known* closed-form reference-process rate
    (forward.reference_up_rate), rather than an ad hoc rate as in the
    `corrector_steps` scheme above. For a pure-birth reference process the
    reference rate has no death channel, so this corrector only speeds up
    the birth clock -- it cannot add up/down exploration the way it does
    for a birth-death reference process (see TauLeapingBirthDeath).

    Inputs:
      forward: Properties of forward process
      conditioner: Second model input
      model : Provides transition(), rate(), and p(x0|xt)
      time_grid : Tensor Monotonically increasingg time grid [0,...,T]
      corrector_steps : Number of ad hoc corrector steps per predictor step in late stage
      late_stage_start : Fraction of sampling trajectory after which the ad hoc corrector starts
      corrector_tau_scale : Tau scaling for the ad hoc corrector steps (usually smaller than predictor tau)
      reject_multi_jumps : If True, reject >1 jumps per dimension per step (categorical stability)
      corrector_noise_scale : Uniform exploration rate added during the ad hoc corrector step
      max_expected_jumps : Upper bound for rate*tau per dimension for numerical stability
      corrector_rate : magnitude of corr(t) = vec_rate(t) + cev_rate(t) to add
                        (0 disables it, 1 applies the full Prop. 4 corrector)
      pc_corrector_steps : Number of literal Appendix F.2 corrector steps per predictor step in late stage
      pc_late_stage_start : Fraction of sampling trajectory after which the F.2 corrector starts
      pc_corrector_tau_scale : Tau scaling for the F.2 corrector steps

    Returns
    -------
      x0_hat : ndarray (N, D)
      x_hist : ndarray (K, N, D)
    """

    # initialize x and t
    x_hist = [x0.clone()]
    x = x0.clone()

    total_steps = max(len(time_grid) - 1, 1)
    max_state = getattr(forward, "S", None)
    if max_state is not None:
        max_state = int(max_state) - 1

    for step, (t_prev, t) in enumerate(zip(time_grid[:-1], time_grid[1:])):
        tau = float(t - t_prev)

        # Compute rates Q(x -> y). Tau-leaping freezes the rates at the
        # left endpoint t_prev, the time at which the current state lives
        # (evaluating at the right endpoint t systematically underestimates
        # the corrector's down rate x/t and drifts the samples upward).
        t_tensor = torch.tensor([t_prev], dtype=torch.float32, device=device)
        t_tensor = t_tensor.repeat(x.shape[0])  # (B,)
        with torch.no_grad():
            if conditioner is None:
                rates = model(x.float(), t_tensor)
            else:
                rates = model(x.float(), t_tensor, conditioner)

        # Predictor step
        if corrector_rate != 0.0:
            up_rate = rates * (1.0 + corrector_rate)
            down_rate = corrector_rate * vec_cev_corrector_rate(x, t_tensor)
            x = _tau_leap_birth_death_step(x, up_rate, down_rate, tau, max_expected_jumps)
            if max_state is not None:
                x = x.clamp(max=max_state)
        else:
            x = _tau_leap_step(
                x=x,
                rates=rates,
                tau=tau,
                max_state=max_state,
                reject_multi_jumps=reject_multi_jumps,
                max_expected_jumps=max_expected_jumps,
            )

        # Late-stage corrector steps (paper-style schedule near end of trajectory)
        frac_done = (step + 1) / total_steps
        if corrector_steps > 0 and frac_done >= late_stage_start:
            corrector_tau = tau * corrector_tau_scale
            for _ in range(corrector_steps):
                with torch.no_grad():
                    if conditioner is None:
                        corr_rates = model(x.float(), t_tensor)
                    else:
                        corr_rates = model(x.float(), t_tensor, conditioner)
                if corrector_noise_scale > 0.0:
                    # Proxy for forward-rate mixing in discrete PC samplers.
                    corr_rates = corr_rates + corrector_noise_scale * torch.ones_like(corr_rates)
                x = _tau_leap_step(
                    x=x,
                    rates=corr_rates,
                    tau=corrector_tau,
                    max_state=max_state,
                    reject_multi_jumps=reject_multi_jumps,
                    max_expected_jumps=max_expected_jumps,
                )

        # Literal Appendix F.2 predictor-corrector steps: R^c_t = Rhat^theta_t + R_t
        if pc_corrector_steps > 0 and frac_done >= pc_late_stage_start:
            pc_tau = tau * pc_corrector_tau_scale
            for _ in range(pc_corrector_steps):
                with torch.no_grad():
                    if conditioner is None:
                        pc_rates = model(x.float(), t_tensor)
                    else:
                        pc_rates = model(x.float(), t_tensor, conditioner)
                pc_rates = pc_rates + forward.reference_up_rate(t_tensor)[:, None]
                x = _tau_leap_step(
                    x=x,
                    rates=pc_rates,
                    tau=pc_tau,
                    max_state=max_state,
                    reject_multi_jumps=reject_multi_jumps,
                    max_expected_jumps=max_expected_jumps,
                )

        x_hist.append(x.clone())

    return (
        x.detach().cpu(),
        torch.stack(x_hist).detach().cpu()
    )


# ============================================================
# Reverse-time samplers for the up_down_Nd noising process
# (birth-death CTMC: jumps to both x+1 and x-1). Use with
# model_utils.ReverseRates as rates_fn. time_grid must be
# monotonically DECREASING, e.g. torch.linspace(T-eps, T0+eps, K),
# and x_init should be drawn from forward.sample_terminal(...).
# ============================================================

def birth_death_time_grid(forward, num_steps: int, eps_t: float = 1e-4) -> Tensor:
    """
    Build the monotonically-decreasing time_grid expected by
    EulerBirthDeath/TauLeapingBirthDeath: torch.linspace(T-eps_t,
    T0+eps_t, num_steps). eps_t is an early-stopping floor -- the reverse
    process is only integrated down to t=T0+eps_t (and up from t=T-eps_t),
    not all the way to the t=T0/T boundary, avoiding the 1/t-type
    singularities in the rate/EDM-moment formulas there (see e.g.
    vec_cev_corrector_rate above). eps_t=1e-4 (the default) only backs
    away from the boundary enough to avoid that singularity; a much larger
    eps_t (e.g. 0.01, as used for min_t in Campbell et al. 2022's tauLDR
    release, github.com/andrew-cr/tauLDR) trades a small amount of
    distributional accuracy for a reverse process that never has to
    resolve the highest-noise/lowest-noise regime, where the model is
    least reliable -- pass cfg.sampling.early_stopping here to make that
    trade-off a config-level choice instead of a hardcoded literal.
    """
    return torch.linspace(forward.T - eps_t, forward.T0 + eps_t, num_steps)


# ------------------------------------------------------------
# Measure-preserving corrector.
#
# Given a CTMC with rate(t) and marginals trans_t, corr(t) is
# trans_t-marginal-preserving if rate(t) + corr(t) has trans_t as its
# marginals. rates_fn(x, t) = (up_rate, down_rate) is already built (via
# the discrete Tweedie formula in ReverseRates) so that
#
#     up_rate(x, t) / down_rate(x+1, t)  ~  trans_t(x+1) / trans_t(x)
#     down_rate(x, t) / up_rate(x-1, t)  ~  trans_t(x-1) / trans_t(x)
#
# We reuse these self-consistent ratio estimates to add the psi-corrector
#
#     corr_up(x, t)   = a(t) * psi( trans_t(x+1) / trans_t(x) )
#     corr_down(x, t) = a(t) * psi( trans_t(x-1) / trans_t(x) )
#
# with a_t(x) = a(t) taken constant across x (so a_t(x) = a_t(x+e_i) =
# a_t(x-e_i) holds trivially) and psi one of the two examples below
# (both satisfy psi(r) = r * psi(1/r)).
# ------------------------------------------------------------

PSI_FUNCS = {
    'sqrt': lambda r: r.clamp(min=0.0).sqrt(),
    'frac': lambda r: r / (1.0 + r),
}


def birth_death_corrector_rates(
    rates_fn,
    x: Tensor,
    t_tensor: Tensor,
    conditioner=None,
    corrector_rate: float = 0.0,
    corrector_psi: str = 'sqrt',
    max_state: Optional[int] = None,
    clamp_min: Optional[float] = 0.0,
    eps: float = 1e-6,
):
    """
    Extra (corr_up, corr_down) rates to add to (up_rate, down_rate) so the
    combined birth-death CTMC still has trans_t as its marginals. Returns
    zeros if corrector_rate == 0. clamp_min=0.0 (default) reflects the
    lower boundary at 0 for state spaces N_0^D; pass clamp_min=None for an
    unbounded Z^D state space (e.g. ReverseRatesZ), where there is no lower
    boundary to protect.
    """
    if corrector_rate == 0.0:
        zeros = torch.zeros_like(x)
        return zeros, zeros

    psi = PSI_FUNCS[corrector_psi]

    def _eval(state):
        with torch.no_grad():
            if conditioner is None:
                return rates_fn(state.float(), t_tensor)
            return rates_fn(state.float(), t_tensor, conditioner)

    up_x, down_x = _eval(x)

    x_plus = x + 1.0
    if max_state is not None:
        x_plus = x_plus.clamp(max=float(max_state))
    _, down_xplus = _eval(x_plus)

    x_minus = x - 1.0
    if clamp_min is not None:
        x_minus = x_minus.clamp(min=clamp_min)
    up_xminus, _ = _eval(x_minus)

    ratio_up   = up_x   / down_xplus.clamp(min=eps)
    ratio_down = down_x / up_xminus.clamp(min=eps)

    corr_up   = corrector_rate * psi(ratio_up)
    corr_down = corrector_rate * psi(ratio_down)

    # the corrector cannot move mass past the state-space boundaries either
    if clamp_min is not None:
        corr_down = torch.where(x > clamp_min, corr_down, torch.zeros_like(corr_down))
    if max_state is not None:
        corr_up = torch.where(x < max_state, corr_up, torch.zeros_like(corr_up))

    return corr_up, corr_down


def _euler_up_down_step(x: Tensor, up_rate: Tensor, down_rate: Tensor, h: float,
                        clamp_min: Optional[float] = 0.0,
                        clamp_max: Optional[float] = None) -> Tensor:
    """
    Single Euler update for a birth-death process: samples stay/up/down from
    the one-step transition probabilities implied by (up_rate, down_rate).
    Shared by the predictor step and the F.2 corrector sub-steps in
    EulerBirthDeath(). clamp_min=0.0 (default) reflects at 0, for state
    spaces N_0^D; pass clamp_min=None for an unbounded Z^D state space
    (e.g. ReverseRatesZ). clamp_max=None (default) leaves the state space
    unbounded above; pass e.g. cfg.data.S-1 for a state space with a known
    finite upper bound (e.g. forward.UpDownNdNoise on a fixed-vocabulary
    sequence problem, which has no upper bound of its own -- without this,
    a run of up-jumps outnumbering down-jumps for some trajectory has
    nothing to stop it drifting arbitrarily high).
    """
    p_up   = (h * up_rate).clamp(min=0.0, max=1.0)
    p_down = (h * down_rate).clamp(min=0.0, max=1.0)
    if clamp_min is not None:
        p_down = torch.where(x > clamp_min, p_down, torch.zeros_like(p_down))  # cannot jump below clamp_min
    if clamp_max is not None:
        p_up = torch.where(x < clamp_max, p_up, torch.zeros_like(p_up))  # cannot jump above clamp_max
    p_stay = (1.0 - p_up - p_down).clamp(min=0.0)

    probs  = torch.stack([p_stay, p_up, p_down], dim=-1)  # (B, D, 3)
    probs  = probs / probs.sum(dim=-1, keepdim=True)
    choice = torch.multinomial(probs.reshape(-1, 3), num_samples=1).reshape(x.shape)

    delta = torch.zeros_like(x)
    delta = torch.where(choice == 1, torch.ones_like(x), delta)
    delta = torch.where(choice == 2, -torch.ones_like(x), delta)
    x_next = x + delta
    if clamp_min is not None or clamp_max is not None:
        x_next = x_next.clamp(min=clamp_min, max=clamp_max)
    return x_next


def EulerBirthDeath(
    rates_fn,
    time_grid: Tensor,
    x_init: Tensor,
    device,
    conditioner=None,
    corrector_rate: float = 0.0,
    corrector_psi: str = 'sqrt',
    pc_corrector_steps: int = 0,
    pc_late_stage_start: float = 0.9,
    pc_corrector_tau_scale: float = 0.1,
    clamp_min: Optional[float] = 0.0,
    clamp_max: Optional[float] = None,
):
    """
    Euler scheme for sampling the reverse-time birth-death CTMC, with an
    optional measure-preserving corrector added to the rates at every step,
    and an optional literal Appendix F.2 predictor-corrector scheme
    (Campbell et al., https://arxiv.org/pdf/2205.14987): each corrector
    sub-step builds the Euler transition probabilities from
    R^c_t = Rhat^theta_t + R_t, i.e. the learned reverse rate (up_rate,
    down_rate from rates_fn) plus the *known* closed-form reference-process
    rate (forward_process.reference_rates), rather than the self-consistent
    ratio-based corrector implemented by birth_death_corrector_rates above.
    See TauLeapingBirthDeath for the tau-leaping version of the same F.2
    scheme.

    Inputs:
      rates_fn        : callable (x, t) -> (up_rate, down_rate), e.g. a ReverseRates module
      time_grid       : Tensor, monotonically decreasing time grid [T, ..., T0]
      x_init          : (N, D) samples from the terminal (noise) distribution
      corrector_rate  : magnitude a(t) of the marginal-preserving corrector (0 disables it)
      corrector_psi   : 'sqrt' or 'frac', see birth_death_corrector_rates
      pc_corrector_steps      : Number of literal Appendix F.2 corrector sub-steps per
                                 predictor step in the late stage
      pc_late_stage_start     : Fraction of sampling trajectory after which the F.2 corrector starts
      pc_corrector_tau_scale  : Step-size scaling (relative to h) for the F.2 corrector sub-steps
      clamp_min       : lower reflecting boundary, 0.0 (default) for N_0^D state spaces;
                         pass None for an unbounded Z^D state space (e.g. ReverseRatesZ)
      clamp_max       : upper reflecting boundary. Default None auto-detects one from
                         forward_process.S (rates_fn.forward_process.S), if that attribute
                         exists (e.g. forward.PoissonFolmer/CategoricalUniformNoise do,
                         forward.UpDownNdNoise does not); pass an explicit value (e.g.
                         cfg.data.S - 1) for a forward process that has no .S of its own but
                         is still being used on a fixed-vocabulary problem (e.g. UpDownNdNoise) -- without this, an up-jump-dominated
                         trajectory has nothing stopping it drifting arbitrarily high.

    Returns
    -------
      x0_hat : (N, D)
      x_hist : (K, N, D)
    """
    x_hist = [x_init.clone()]
    x = x_init.clone()
    forward_process = getattr(rates_fn, "forward_process", None)
    max_state = getattr(forward_process, "S", None)
    if max_state is not None:
        max_state = int(max_state) - 1
    if clamp_max is not None:
        max_state = clamp_max  # explicit override takes priority over auto-detection

    total_steps = max(len(time_grid) - 1, 1)

    for step in range(len(time_grid) - 1):
        h = float(time_grid[step] - time_grid[step + 1])  # positive: time decreases
        t_tensor = time_grid[step].to(device).repeat(x.shape[0])

        with torch.no_grad():
            if conditioner is None:
                up_rate, down_rate = rates_fn(x.float(), t_tensor)
            else:
                up_rate, down_rate = rates_fn(x.float(), t_tensor, conditioner)

        corr_up, corr_down = birth_death_corrector_rates(
            rates_fn, x, t_tensor, conditioner=conditioner,
            corrector_rate=corrector_rate, corrector_psi=corrector_psi,
            max_state=max_state, clamp_min=clamp_min,
        )
        up_rate   = up_rate + corr_up
        down_rate = down_rate + corr_down

        x = _euler_up_down_step(x, up_rate, down_rate, h, clamp_min=clamp_min, clamp_max=max_state)

        # Literal Appendix F.2 predictor-corrector sub-steps: R^c_t = Rhat^theta_t + R_t
        frac_done = (step + 1) / total_steps
        if pc_corrector_steps > 0 and frac_done >= pc_late_stage_start and forward_process is not None:
            h_c = h * pc_corrector_tau_scale
            for _ in range(pc_corrector_steps):
                with torch.no_grad():
                    if conditioner is None:
                        pc_up, pc_down = rates_fn(x.float(), t_tensor)
                    else:
                        pc_up, pc_down = rates_fn(x.float(), t_tensor, conditioner)
                ref_up, ref_down = forward_process.reference_rates(x, t_tensor)
                pc_up = pc_up + ref_up
                pc_down = pc_down + ref_down
                x = _euler_up_down_step(x, pc_up, pc_down, h_c, clamp_min=clamp_min, clamp_max=max_state)

        x_hist.append(x.clone())

    return (
        x.detach().cpu(),
        torch.stack(x_hist).detach().cpu()
    )


# ============================================================
# Samplers that don't fit the birth-death (up_rate/down_rate) interface
# above: each corresponds to a different forward process / model output /
# paper-specified discretization (see forward.CountBridgeNoise,
# forward.CountFMBridge, forward.CategoricalUniformNoise, and
# loss.denoising_loss_count_fm / loss.denoising_loss_categorical for the
# matching training losses).
# ============================================================

def sample_count_bridge(model, forward, time_grid, x_init, device):
    """
    Ancestral bridge sampler for forward.CountBridgeNoise (Fishman et al.
    2026, "Count Bridges", Algorithm 2): at each step, predict x0_hat with
    the conditional-mean denoiser, then draw the earlier bridge state
    exactly between x0_hat and the current state via forward.bridge_step.
    time_grid must be monotonically decreasing, [T, ..., T0], matching the
    other reverse-time samplers above.
    """
    x = x_init.clone()
    x_hist = [x.clone()]
    for k in range(len(time_grid) - 1):
        t_k = time_grid[k].to(device).repeat(x.shape[0])
        t_km1 = time_grid[k + 1].to(device).repeat(x.shape[0])
        with torch.no_grad():
            x0_hat = model(x.float(), t_k)
        # the bridge kernel requires an integer endpoint (d_ref = x_ref - x0_hat
        # must be an integer gap for the Poisson-difference rejection sampler);
        # the denoiser's raw output is continuous, so round it first.
        x0_hat = torch.round(x0_hat)
        x = forward.bridge_step(x0_hat, x, t_k, t_km1)
        x_hist.append(x.clone())
    return x, torch.stack(x_hist)


def sample_count_fm(model, forward, num_samples, num_steps, device, eps_t: float = 1e-3, eps_r: float = 1e-8):
    """
    First-order local-jump discretization (Wei & Pearson 2026, "Flow
    Matching for Count Data", Sec. 2.2): simulates the learned birth-death
    process forward from t=eps_t to t=1-eps_t in num_steps equal
    increments, starting from forward.sample_source. At each step, each
    coordinate independently stays / births (+1) / deaths (-1) with
    probabilities derived from the exact no-jump probability exp(-r*Delta)
    under the model's current total rate r = lambda_theta + mu_theta (as
    opposed to TauLeapingBirthDeath/EulerBirthDeath above, which use a
    Poisson-count or linear-in-h approximation instead).
    """
    x = forward.sample_source(num_samples, device=device)
    x_hist = [x.clone()]
    delta = (1.0 - 2.0 * eps_t) / num_steps
    for k in range(num_steps):
        t = eps_t + k * delta
        t_batch = torch.full((x.shape[0],), t, device=device)
        with torch.no_grad():
            lam, mu = model(x.float(), t_batch)
        r = lam + mu
        p_stay = torch.exp(-r * delta)
        p_birth = (1.0 - p_stay) * lam / (r + eps_r)
        p_death = (1.0 - p_stay) * mu / (r + eps_r)

        probs = torch.stack([p_stay, p_birth, p_death], dim=-1)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(probs.reshape(-1, 3), num_samples=1).reshape(x.shape)

        step = torch.zeros_like(x)
        step = torch.where(choice == 1, torch.ones_like(x), step)
        step = torch.where(choice == 2, -torch.ones_like(x), step)
        x = (x + step).clamp(min=0.0)
        x_hist.append(x.clone())
    return x, torch.stack(x_hist)


def sample_categorical(model, forward, num_samples, num_steps, device, eps_t: float = 1e-3):
    """
    Reverse-time Euler sampler for forward.CategoricalUniformNoise (Lou,
    Meng & Ermon 2023, "Discrete Diffusion Modeling by Estimating the Ratio
    of Data Distributions", Sec. 4.2 / Appendix B): the reverse rate
    R_t(x,y) = Q_t(x,y) * s_theta(x,t)[y] for y != x (their Eq. 5). Q_t is
    the *instantaneous* forward uniform rate implied by forward.kernel,
    p_t(y|x0) = alpha(t)*1[y=x0] + (1-alpha(t))/S with alpha=exp(-sigma_bar(t)):
    differentiating gives Q_t(x,y) = sigma_bar_dot(t)/S for y != x, i.e. the
    forward noise rate scales with the noise schedule's instantaneous rate,
    not a fixed constant. Discretized into a one-step categorical transition
    distribution per coordinate, analogous to _euler_birth_death_step above
    but over the full S-category simplex rather than just +-1 moves -- there
    is no ordinal neighbor structure to exploit here, by construction.
    """
    S = forward.S
    x = forward.sample_terminal(num_samples, device=device)
    x_hist = [x.clone()]
    time_grid = torch.linspace(forward.T - eps_t, forward.T0 + eps_t, num_steps)
    sigma_bar_dot = forward.sigma_max / (forward.T - forward.T0)  # d(sigma_bar)/dt
    B, D = x.shape

    for step in range(len(time_grid) - 1):
        h = float(time_grid[step] - time_grid[step + 1])
        t_batch = time_grid[step].to(device).repeat(B)
        with torch.no_grad():
            score = model(x.float(), t_batch)  # (B, D, S)

        q_t = sigma_bar_dot / S
        rate = (q_t * h) * score
        idx = x.long().unsqueeze(-1)
        rate = rate.scatter(-1, idx, torch.zeros_like(idx, dtype=rate.dtype))  # no self-transition rate
        probs = rate.clamp(min=0.0)
        stay_prob = (1.0 - probs.sum(dim=-1, keepdim=True)).clamp(min=0.0)
        probs = torch.cat([stay_prob, probs], dim=-1)  # (B, D, S+1), index 0 = stay
        probs = probs / probs.sum(dim=-1, keepdim=True)
        choice = torch.multinomial(probs.reshape(-1, S + 1), num_samples=1).reshape(B, D)
        x = torch.where(choice == 0, x, (choice - 1).float())
        x_hist.append(x.clone())

    return x, torch.stack(x_hist)


def _tau_leap_birth_death_step(
    x: Tensor,
    up_rate: Tensor,
    down_rate: Tensor,
    tau: float,
    max_expected_jumps: float = 0.5,
    clamp_min: Optional[float] = 0.0,
    clamp_max: Optional[float] = None,
    reject_multi_jumps: bool = False,
) -> Tensor:
    """
    Single tau-leaping update for a birth-death process. clamp_min=0.0
    (default) reflects at 0, for state spaces N_0^D; pass clamp_min=None
    for an unbounded Z^D state space (e.g. ReverseRatesZ). clamp_max=None
    (default) leaves the state space unbounded above; pass e.g.
    cfg.data.S-1 for a forward process with a known finite upper bound but
    no .S of its own (e.g. forward.UpDownNdNoise on a fixed-vocabulary
    sequence problem) -- see _euler_up_down_step's docstring for why this
    matters (an up-jump-dominated trajectory otherwise has nothing to stop
    it drifting arbitrarily high).

    reject_multi_jumps: tauLDR's (Campbell et al., github.com/andrew-cr/
    tauLDR) reject_multiple_jumps=True safeguard, adapted from their
    single-channel categorical tau-leap (Bernoulli(1-exp(-rate*tau)) event
    indicator instead of Poisson(rate*tau) event count, so a channel fires
    at most once per tau step) to this two-channel (up/down) birth-death
    step: each channel independently draws a Bernoulli jump indicator
    instead of a Poisson count, capping |x_next - x| <= 1 per coordinate
    per step (both channels firing simultaneously nets to a possible
    2-state round trip, same corner case tauLDR's own version has when its
    single categorical channel is applied per-coordinate here). This
    removes tau-leaping's main discretization bias -- large multi-jumps
    that overshoot past states the reverse rate was only evaluated
    at -- which is likely why our tau-leaping@200 Hellinger score
    degraded sharply for a large R_max in the earlier debug-scale sweep
    (large R_max -> large rates -> large expected per-step jump counts).
    """
    up_rate   = up_rate.clamp(min=0.0)
    down_rate = down_rate.clamp(min=0.0)

    if max_expected_jumps is not None:
        cap = max_expected_jumps / max(tau, 1e-8)
        up_rate   = up_rate.clamp(max=cap)
        down_rate = down_rate.clamp(max=cap)

    # up_rate is masked to 0 at the upper boundary (rather than relying on
    # clamping x_next alone), so a state already at clamp_max doesn't draw a
    # nonzero Poisson up-count only to have it silently discarded by the
    # clamp -- consistent with _euler_up_down_step's p_up boundary masking.
    # (clamp_min's existing behavior, relying on the post-hoc x_next clamp
    # alone, is left exactly as it was to avoid changing default behavior
    # for every other caller of this function.)
    if clamp_max is not None:
        up_rate = torch.where(x < clamp_max, up_rate, torch.zeros_like(up_rate))

    if reject_multi_jumps:
        up_jumps   = torch.bernoulli(1.0 - torch.exp(-up_rate * tau))
        down_jumps = torch.bernoulli(1.0 - torch.exp(-down_rate * tau))
    else:
        # One Poisson event count per reaction channel (up and down) per
        # coordinate. Net the two channels before clamping at 0: truncating
        # down_jumps at x before adding up_jumps discards downward moves near
        # the boundary and biases the state upward.
        up_jumps   = torch.poisson(up_rate * tau)
        down_jumps = torch.poisson(down_rate * tau)

    x_next = x + up_jumps - down_jumps
    if clamp_min is not None or clamp_max is not None:
        x_next = x_next.clamp(min=clamp_min, max=clamp_max)
    return x_next


def TauLeapingBirthDeath(
    rates_fn,
    time_grid: Tensor,
    x_init: Tensor,
    device,
    conditioner=None,
    max_expected_jumps: float = 0.5,
    corrector_rate: float = 0.0,
    corrector_psi: str = 'sqrt',
    pc_corrector_steps: int = 0,
    pc_late_stage_start: float = 0.9,
    pc_corrector_tau_scale: float = 0.1,
    clamp_min: Optional[float] = 0.0,
    clamp_max: Optional[float] = None,
    reject_multi_jumps: bool = False,
):
    """
    Tau-leaping sampler for the reverse-time birth-death CTMC, with an
    optional measure-preserving corrector added to the rates at every step,
    and an optional literal Appendix F.2 predictor-corrector scheme
    (Campbell et al., https://arxiv.org/pdf/2205.14987): each corrector step
    simulates the CTMC with rate R^c_t = Rhat^theta_t + R_t, i.e. the
    learned reverse rate (up_rate, down_rate from rates_fn) plus the *known*
    closed-form reference-process rate (forward_process.reference_rates),
    rather than the self-consistent ratio-based corrector implemented by
    birth_death_corrector_rates above.

    Inputs:
      rates_fn            : callable (x, t) -> (up_rate, down_rate), e.g. a ReverseRates module
      time_grid           : Tensor, monotonically decreasing time grid [T, ..., T0]
      x_init               : (N, D) samples from the terminal (noise) distribution
      max_expected_jumps  : upper bound on rate*tau per dimension for numerical stability
      corrector_rate       : magnitude a(t) of the marginal-preserving corrector (0 disables it)
      corrector_psi        : 'sqrt' or 'frac', see birth_death_corrector_rates
      pc_corrector_steps      : Number of literal Appendix F.2 corrector steps per predictor step in late stage
      pc_late_stage_start     : Fraction of sampling trajectory after which the F.2 corrector starts
      pc_corrector_tau_scale  : Tau scaling for the F.2 corrector steps
      clamp_min                : lower reflecting boundary for the state space (default 0.0, for
                                  N_0^D state spaces). Pass None for an unbounded Z^D state space
                                  (e.g. ReverseRatesZ / forward.UpDownZdNoise).
      clamp_max                : upper reflecting boundary. Default None auto-detects one from
                                  forward_process.S, if that attribute exists; pass an explicit
                                  value (e.g. cfg.data.S - 1) for a forward process with no .S of
                                  its own but a known finite upper bound (e.g. UpDownNdNoise) -- see _tau_leap_birth_death_step's
                                  docstring for why this matters.
      reject_multi_jumps       : tauLDR's reject_multiple_jumps=True safeguard -- see
                                  _tau_leap_birth_death_step's docstring. Applied to both the
                                  predictor steps and the F.2 corrector sub-steps below.

    Returns
    -------
      x0_hat : (N, D)
      x_hist : (K, N, D)
    """
    x_hist = [x_init.clone()]
    x = x_init.clone()
    forward_process = getattr(rates_fn, "forward_process", None)
    max_state = getattr(forward_process, "S", None)
    if max_state is not None:
        max_state = int(max_state) - 1
    if clamp_max is not None:
        max_state = clamp_max  # explicit override takes priority over auto-detection

    total_steps = max(len(time_grid) - 1, 1)

    for step in range(len(time_grid) - 1):
        tau = float(time_grid[step] - time_grid[step + 1])
        t_tensor = time_grid[step].to(device).repeat(x.shape[0])

        with torch.no_grad():
            if conditioner is None:
                up_rate, down_rate = rates_fn(x.float(), t_tensor)
            else:
                up_rate, down_rate = rates_fn(x.float(), t_tensor, conditioner)

        corr_up, corr_down = birth_death_corrector_rates(
            rates_fn, x, t_tensor, conditioner=conditioner,
            corrector_rate=corrector_rate, corrector_psi=corrector_psi,
            max_state=max_state, clamp_min=clamp_min,
        )
        up_rate   = up_rate + corr_up
        down_rate = down_rate + corr_down

        x = _tau_leap_birth_death_step(x, up_rate, down_rate, tau, max_expected_jumps,
                                       clamp_min=clamp_min, clamp_max=max_state,
                                       reject_multi_jumps=reject_multi_jumps)

        # Literal Appendix F.2 predictor-corrector steps: R^c_t = Rhat^theta_t + R_t
        frac_done = (step + 1) / total_steps
        if pc_corrector_steps > 0 and frac_done >= pc_late_stage_start and forward_process is not None:
            pc_tau = tau * pc_corrector_tau_scale
            for _ in range(pc_corrector_steps):
                with torch.no_grad():
                    if conditioner is None:
                        pc_up, pc_down = rates_fn(x.float(), t_tensor)
                    else:
                        pc_up, pc_down = rates_fn(x.float(), t_tensor, conditioner)
                ref_up, ref_down = forward_process.reference_rates(x, t_tensor)
                pc_up = pc_up + ref_up
                pc_down = pc_down + ref_down
                x = _tau_leap_birth_death_step(x, pc_up, pc_down, pc_tau, max_expected_jumps,
                                               clamp_min=clamp_min, clamp_max=max_state,
                                               reject_multi_jumps=reject_multi_jumps)

        x_hist.append(x.clone())

    return (
        x.detach().cpu(),
        torch.stack(x_hist).detach().cpu()
    )
