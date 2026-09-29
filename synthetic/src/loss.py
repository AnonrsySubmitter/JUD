import torch

def bregman_divergence_quadratic(a, b):
    """Elementwise quadratic Bregman divergence.

    Callers decide whether to sum or average coordinates.  Returning an
    already-reduced tensor here made their subsequent ``sum(dim=-1)`` sum
    over the batch instead, breaking per-example time weighting.
    """
    return (a - b).pow(2)

def bregman_divergence_xlogx(a, b, eps=1e-8):
    # Clamp to avoid log(0) and division by zero
    a_safe = torch.clamp(a, min=eps)
    b_safe = torch.clamp(b, min=eps)
    # Stable computation
    loss = a_safe * (torch.log(a_safe) - torch.log(b_safe)) - (a_safe - b_safe)
    return loss

def _apply_loss_weight(loss, model, cfg, ts):
    """
    If cfg.model.edm_scaling is on, reweight by lambda(t) = w(t)/cout(t)^2
    -- computed from the model's last forward() call, so
    this must be called right after model(...) with no intervening forward
    -- so that the training loss's expectation under net_theta=0 is roughly
    constant in t; this takes priority over (and replaces)
    the plain ts^power weighting below, since the two are redundant.
    Otherwise falls back to the existing ts^loss_weight_power scheme.
    """
    if getattr(model, 'edm_scaling', False):
        return loss * model.edm_lambda()
    if cfg.training.use_loss_weighting:
        return loss * torch.clamp(ts.pow(cfg.training.loss_weight_power), min=cfg.training.loss_weight_floor)
    return loss

def denoising_loss_z(forward, model, cfg, minibatch, divergence='xlogx'):
    """
    Trains model to predict denbin(x, t) = E[D_t | P_t] (see model.ZDenoiser).
    D_t is Poisson-distributed, so the matching Bregman divergence is the
    generalized-KL / xlogx divergence (default here), not the quadratic one
    -- it weights errors relative to the (mean-scaling) Poisson variance
    instead of uniformly, which matters a lot for targets that mix sharp
    point masses (small residual noise) with a diffuse bump (large counts
    near t=T).
    """
    B, _ = minibatch.shape
    device = next(model.parameters()).device
    x0 = minibatch.to(device).float()

    ts = torch.rand((B,), device=device) * (cfg.data.T - cfg.data.T0) + cfg.data.T0
    pt, d_t, u_t = forward.conditional_sample(ts, x0)
    model_d = model(pt, ts)

    if divergence == 'quad':
        loss = bregman_divergence_quadratic(model_d, d_t).sum(dim=-1)
    elif divergence == 'xlogx':
        loss = bregman_divergence_xlogx(model_d, d_t).sum(dim=-1)
    else:
        raise ValueError("Unsupported divergence type")
    loss = _apply_loss_weight(loss, model, cfg, ts)
    return loss.mean()

def denoising_loss_bd(forward, model, cfg, minibatch, divergence='quad'):
    """
    Trains model to predict den(x, t) = E[Bin_t | P_t] for the bounded
    up_down_Nd process on N_0^d (see forward.UpDownNdNoise,
    model_utils.ReverseRates).
    """
    B, _ = minibatch.shape
    device = next(model.parameters()).device
    x0 = minibatch.to(device).float()

    ts = torch.rand((B,), device=device) * (cfg.data.T - cfg.data.T0) + cfg.data.T0
    pt, bin_t = forward.conditional_sample(ts, x0)
    model_bin = model(pt, ts)

    loss = bregman_divergence_quadratic(model_bin, bin_t).sum(dim=-1)
    loss = _apply_loss_weight(loss, model, cfg, ts)
    return loss.mean()

def denoising_loss_bd_data(forward, model, cfg, minibatch):
    """Train the companion data denoiser E[X_0 | P_t] for Eq. 14."""
    B = minibatch.shape[0]
    device = next(model.parameters()).device
    x0 = minibatch.to(device).float()
    ts = torch.rand((B,), device=device) * (cfg.data.T - cfg.data.T0) + cfg.data.T0
    pt, _ = forward.conditional_sample(ts, x0)
    loss = bregman_divergence_quadratic(model(pt, ts), x0).sum(dim=-1)
    loss = _apply_loss_weight(loss, model, cfg, ts)
    return loss.mean()

def denoising_loss_z_up(forward, model, cfg, minibatch, divergence='xlogx'):
    """Train the companion Poisson latent denoiser E[U_t | P_t]. See
    denoising_loss_z for why xlogx (generalized-KL) is the default."""
    B = minibatch.shape[0]
    device = next(model.parameters()).device
    x0 = minibatch.to(device).float()
    ts = torch.rand((B,), device=device) * (cfg.data.T - cfg.data.T0) + cfg.data.T0
    pt, _, u_t = forward.conditional_sample(ts, x0)
    model_u = model(pt, ts)
    if divergence == 'quad':
        loss = bregman_divergence_quadratic(model_u, u_t).sum(dim=-1)
    elif divergence == 'xlogx':
        loss = bregman_divergence_xlogx(model_u, u_t).sum(dim=-1)
    else:
        raise ValueError("Unsupported divergence type")
    loss = _apply_loss_weight(loss, model, cfg, ts)
    return loss.mean()

def denoising_loss(forward, model, cfg, minibatch, divergence='quad'):
    
    # get batch size and device
    B, _ = minibatch.shape
    device = next(model.parameters()).device

    # split data and move to device
    if cfg.data.condition_dim > 0:
        conditioner = minibatch[:, 0:cfg.data.condition_dim].to(device)
        x0 = minibatch[:, cfg.data.condition_dim:].to(device).float()
    else:
        x0 = minibatch.to(device).float()

    # sample times uniformly from [min_time, max_time] and convert to sigma
    ts = torch.rand((B,), device=device) * (cfg.data.T - cfg.data.T0) + cfg.data.T0

    # sample X_t | x0 
    xt = forward.conditional_sample(ts, x0)

    # evaluate model 
    if cfg.data.condition_dim > 0:
        model_x0 = model(ctx_ids=conditioner, xt_ids=xt, t=ts)
    else:
        model_x0 = model(xt, ts)

    # compute loss per sample
    if divergence == 'quad':
        loss = bregman_divergence_quadratic(model_x0, x0).sum(dim=-1)
    elif divergence == 'xlogx':
        loss = bregman_divergence_xlogx(model_x0, x0).sum(dim=-1)
    else:
        raise ValueError("Unsupported divergence type")

    if cfg.training.use_loss_weighting:
        loss *= torch.clamp(ts.pow(cfg.training.loss_weight_power), min=cfg.training.loss_weight_floor)

    return loss.mean()


def denoising_loss_count_fm(forward, model, cfg, minibatch, eps_ell: float = 1e-6):
    """
    Trains lambda_theta, mu_theta against the analytic conditional bridge
    rates via the generalized-KL loss ell(u,v) = v - u*log(v) (Wei & Pearson
    2026, "Flow Matching for Count Data", eq. 4): endpoints are (x0, x1)
    with x0 ~ forward.sample_source (a fixed, uninformative discrete-uniform
    prior, matching the paper's own simulation baseline) and x1 = the data
    minibatch, coupled independently (pi_ind(x0,x1) = p0(x0) p1(x1) --
    "count-FM" in the paper, as opposed to their optional minibatch-OT
    coupling "count-FM-OT", not implemented here).
    """
    B, _ = minibatch.shape
    device = next(model.parameters()).device
    x1 = minibatch.to(device).float()
    x0 = forward.sample_source(B, device=device)

    ts = torch.rand((B,), device=device)
    xt = forward.conditional_sample(ts, x0, x1)
    lam_true, mu_true = forward.conditional_rates(xt, x1, ts)

    lam_pred, mu_pred = model(xt, ts)

    def ell(u, v):
        return v - u * torch.log(v + eps_ell)

    loss = (ell(lam_true, lam_pred) + ell(mu_true, mu_pred)).sum(dim=-1)
    return loss.mean()


def denoising_loss_categorical(forward, model, cfg, minibatch):
    """
    Denoising score-entropy loss (Lou, Meng & Ermon 2023, "Discrete
    Diffusion Modeling by Estimating the Ratio of Data Distributions",
    https://arxiv.org/abs/2310.16834, Eq. 6) for forward.
    CategoricalUniformNoise: the uniform kernel makes every category y !=
    xt a graph neighbor with the SAME edge weight, so the neighbor sum is
    just "all S categories except xt", and the target ratio p_t(y|x0)/
    p_t(xt|x0) is available in closed form via forward.kernel -- no need
    for the paper's more general transition-rate-matrix machinery, since S
    is small and known here.
    """
    B, D = minibatch.shape
    device = next(model.parameters()).device
    x0 = minibatch.to(device).float()
    S = forward.S

    ts = torch.rand((B,), device=device) * (cfg.data.T - cfg.data.T0) + cfg.data.T0
    xt = forward.conditional_sample(ts, x0)

    cats = torch.arange(S, device=device).float().view(1, 1, S).expand(B, D, S)
    x0_exp = x0.unsqueeze(-1).expand(B, D, S)
    xt_exp = xt.unsqueeze(-1).expand(B, D, S)
    p_y = forward.kernel(cats, x0_exp, ts)                                        # (B, D, S)
    p_xt = forward.kernel(xt.unsqueeze(-1), x0.unsqueeze(-1), ts).squeeze(-1)      # (B, D)
    ratio = p_y / p_xt.unsqueeze(-1).clamp(min=1e-12)                             # (B, D, S)

    score = model(xt, ts)                                                        # (B, D, S), > 0
    not_self = (cats != xt_exp).float()
    # score-entropy summand score - ratio*log(score) (dropping the additive,
    # theta-independent constant ratio*log(ratio) - ratio), zeroed on the
    # self entry y = xt (excluded from the neighbor sum).
    per_entry = (score - ratio * torch.log(score.clamp(min=1e-12))) * not_self
    loss = per_entry.sum(dim=-1).sum(dim=-1)  # sum over S then over D -> (B,)
    if cfg.training.use_loss_weighting:
        loss = loss * torch.clamp(ts.pow(cfg.training.loss_weight_power), min=cfg.training.loss_weight_floor)
    return loss.mean()
