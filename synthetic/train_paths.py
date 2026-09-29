import torch
import ml_collections
import numpy as np
import yaml
import copy
import os
import argparse

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from scipy.stats import poisson as poisson_dist, skellam as skellam_dist

from model import DenoisingMLP, BinomialDenoiser, TwoHeadNdDenoiser, ZDenoiser, CountFMRates, CategoricalScoreModel
from loss import (denoising_loss, bregman_divergence_quadratic, bregman_divergence_xlogx,
                   denoising_loss_count_fm, denoising_loss_categorical)
from forward import PoissonFolmer, AdditivePoissonNoise, UpDownNdNoise, UpDownZdNoise, CountBridgeNoise, CountFMBridge, CategoricalUniformNoise
from sampling import (TauLeaping, TauLeapingBirthDeath, Euler, EulerBirthDeath,
                       sample_count_bridge, sample_count_fm, sample_categorical)
from model_utils import Intensity, ReverseRates, ReverseRatesTwo, ReverseRatesAdditive, ReverseRatesZ
from utils import create_run_dir
import datasets

# ============================================================
# Trains a denoiser and draws sampling trajectories x_t (vs. generative
# progress) for each of the noising processes below, then saves everything
# plot_paths.py needs (trajectories, losses, reference/target PMFs) to a
# single .pt file -- no plotting here, see plot_paths.py for that:
#
#   "binomial"  : pure Binomial-thinning noise (forward.PoissonFolmer).
#                 Its reverse-time generative process is a pure-birth CTMC
#                 (TauLeaping): x can only ever go UP. Once a step jumps too
#                 high (denoiser error / stochastic overshoot), there is no
#                 mechanism to correct it -- the trajectory is stuck above
#                 its true value for the rest of sampling.
#
#   "poisson_only" : genuine additive-Poisson process, P_t = x0 + Pois(Ru(t))
#                 (forward.AdditivePoissonNoise) -- a pure superposition with
#                 NO thinning/subtraction of x0 at all, so x0 <= P_t always.
#                 For any finite Ru(T) this is an exact translation of x0,
#                 so its terminal marginal is never EXACTLY independent of
#                 x0 (unlike up_down_Zd/count_bridge, whose subtractive
#                 Poisson can exceed and erase x0). Instead Ru_max is
#                 adaptively scaled (AdditivePoissonNoise.auto_scale, same
#                 c*range^2 recipe as UpDownZdNoise.auto_scales/
#                 CountBridgeNoise.auto_scales) so that Ru_max is large
#                 relative to the spread of x0: the TV distance between
#                 P_T|x0=a and P_T|x0=b shrinks like |a-b|/sqrt(Ru_max), so
#                 this only makes the terminal marginal APPROXIMATELY
#                 data-independent, not exactly. Trained with the same
#                 DenoisingMLP/denoising_loss as "binomial" (den(x,t) =
#                 E[x0|P_t=x] directly, since there is no thinning to
#                 distinguish x0 from the "Bin_t" target of up_down_Nd), and
#                 reverse-generated via pure death (model_utils.
#                 ReverseRatesAdditive, TauLeapingBirthDeath/EulerBirthDeath
#                 with clamp_min=0.0): x can only ever go DOWN, starting
#                 from x_init ~approx Pois(Ru_max) and ending at x0.
#
#   "up_down_Nd"    : Binomial thinning + Poisson superposition
#                 (forward.UpDownNdNoise). Its reverse-time
#                 generative process is a birth-DEATH CTMC
#                 (TauLeapingBirthDeath): x can go up AND down, so an
#                 overshoot can be corrected by later down-moves.
#
#   "up_down_Zd"    : "up_down_Zd",
#                 P_t = X_0 - D_t + U_t with D_t, U_t independent Poisson
#                 latents (not a thinning of X_0), so P_t ranges over all
#                 of Z rather than being capped at X_0 like "up_down_Nd"
#                 (forward.UpDownZdNoise, ZDenoiser, ReverseRatesZ,
#                 TauLeapingBirthDeath with clamp_min=None -- paths can dip
#                 below 0, unlike every N_0-valued arm here).
#
#   "count_bridge" : simplified reproduction of Fishman et al. (2026)
#                 "Count Bridges" (https://arxiv.org/pdf/2603.04730) -- an
#                 exact Poisson birth-death BRIDGE between data and a
#                 Skellam reference endpoint (forward.CountBridgeNoise),
#                 trained with the same conditional-mean DenoisingMLP/loss
#                 as "binomial", but sampled via the exact ancestral bridge
#                 sampler (sampling.sample_count_bridge) rather than
#                 rate-based tau-leaping.
#
#   "count_fm"  : reproduction of Wei & Pearson (2026) "Flow Matching for
#                 Count Data" (https://arxiv.org/pdf/2605.07746) -- no
#                 public code available for this paper. A local (+-1)-jump
#                 birth-death CTMC bridging data to a fixed discrete-uniform
#                 source (forward.CountFMBridge), trained with a
#                 generalized-KL rate-matching loss
#                 (loss.denoising_loss_count_fm), and sampled
#                 with its own first-order local-jump discretization
#                 (sampling.sample_count_fm) rather than tau-leaping.
#
#   "categorical" : reproduction of Lou, Meng & Ermon (2023) "Discrete
#                 Diffusion Modeling by Estimating the Ratios of the Data
#                 Distribution" (https://arxiv.org/abs/2310.16834) --
#                 treats {0,...,S-1} as an UNORDERED complete graph of
#                 exchangeable labels rather than a lattice, so "up"/"down"
#                 don't apply: at each step every coordinate stays at its
#                 current value or jumps directly to any other category
#                 (forward.CategoricalUniformNoise), trained with the
#                 paper's denoising score-entropy loss
#                 (loss.denoising_loss_categorical) and
#                 sampled via its reverse-time Euler discretization
#                 (sampling.sample_categorical). Included as the
#                 ordinal-blind baseline: unlike every other arm here, it
#                 does not exploit the fact that closer counts are more
#                 likely confused.
# ============================================================

parser = argparse.ArgumentParser(description="Train denoisers and sample paths for binomial-only, "
                                              "poisson-only, up-and-down (up_down_Nd/up_down_Zd), "
                                              "count-bridge, count-fm, and categorical (SEDD) "
                                              "generative processes. See plot_paths.py to render "
                                              "the saved output.")
parser.add_argument('--distribution', type=str, default='PoissonMixture',
                     choices=['Poisson', 'PoissonMixture', 'ZeroInflatedPoisson',
                              'NegativeBinomialMixture', 'NegativeBinomialMixtureBalanced',
                              'BetaNegativeBinomial',
                              'ZipfDistribution', 'YuleSimonDistribution'],
                     help="Target distribution to train on. Bimodal/heavy-tailed distributions "
                          "(e.g. PoissonMixture) make overshoot-and-correct behavior most visible.")
parser.add_argument('--num-paths', type=int, default=60, help="Number of sampling paths to draw.")
parser.add_argument('--num-hist-samples', type=int, default=1000,
                     help="Number of generated samples to draw (independently of --num-paths) "
                          "for the empirical histogram in the target-side panel of Plot 1.")
parser.add_argument('--Rd-max-Zd', type=float, default=None,
                     help="Override Rd_max for up_down_Zd/count_bridge (default: auto-derived "
                          "per distribution via UpDownZdNoise.auto_scales/CountBridgeNoise."
                          "auto_scales in forward.py). Their sample_terminal() drops the x0 "
                          "term and approximates P_T ~ Skellam(Ru_max, Rd_max), valid only when "
                          "Ru_max/Rd_max are large relative to the spread of the target "
                          "distribution -- too small and x_init clusters near 0 regardless of "
                          "the data, silently dropping any mode far from 0.")
parser.add_argument('--Ru-max-Zd', type=float, default=None,
                     help="Override Ru_max for up_down_Zd/count_bridge (see --Rd-max-Zd).")
parser.add_argument('--Ru-max-Poisson', type=float, default=None,
                     help="Override Ru_max for the genuine additive-Poisson 'poisson_only' arm "
                          "(default: auto-derived per distribution via "
                          "forward.AdditivePoissonNoise.auto_scale). Its sample_terminal() drops "
                          "the x0 term and approximates P_T ~ Pois(Ru_max) -- unlike up_down_Zd/"
                          "count_bridge there is no subtractive Poisson to help erase x0, so this "
                          "is only ever an approximation (TV distance ~ spread(x0)/sqrt(Ru_max)), "
                          "never exact; too small and the reverse chain starts from a distribution "
                          "the model was never trained to start from.")
parser.add_argument('--n-epochs', type=int, default=None, help="Override config.yaml training.n_epochs.")
parser.add_argument('--num-train-samples', type=int, default=None,
                     help="Override config.yaml data.N (total samples drawn from the target "
                          "distribution before the train/val split, see --val-frac).")
parser.add_argument('--val-frac', type=float, default=None,
                     help="Override config.yaml training.val_frac -- fraction of "
                          "--num-train-samples held out for validation. Pass 0 to use ALL samples "
                          "for training with no held-out validation set -- the reported 'val' loss "
                          "then falls back to the training set itself (no genuine held-out "
                          "estimate), since the loss-curve/diagnose plumbing downstream expects a "
                          "val_data split.")
parser.add_argument('--num-steps', type=int, default=None,
                     help="Override config.yaml sampling.num_steps (reverse-time discretization "
                          "steps per trajectory).")
parser.add_argument('--sampler', type=str, default=None, choices=['tau_leaping', 'euler'],
                     help="Override config.yaml sampling.ode_solver -- reverse-time "
                          "discretization for the lattice-based arms (binomial, poisson_only, "
                          "up_down_Nd, up_down_Zd): tau-leaping (sampling.TauLeaping/"
                          "TauLeapingBirthDeath) or Euler (sampling.Euler/EulerBirthDeath). "
                          "count_bridge/count_fm/categorical each use their own bespoke sampler "
                          "regardless of this flag.")
ALL_METHODS = ['binomial', 'poisson_only', 'up_down_Nd', 'up_down_Zd', 'count_bridge', 'count_fm', 'categorical']
parser.add_argument('--methods', type=str, nargs='+', default=ALL_METHODS, choices=ALL_METHODS,
                     help="Subset of processes to train/sample, e.g. '--methods binomial "
                          "up_down_Nd' to reproduce the original two-panel (binomial vs. "
                          "up_down_Nd) comparison.")
args = parser.parse_args()
methods = args.methods

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("./config.yaml", "r") as f:
     cfg_dict = yaml.safe_load(f)
cfg = ml_collections.ConfigDict(cfg_dict)
# EDM scaling is now config.yaml's default, but BinomialDenoiser/
# ZDenoiser here are never configure_edm(...)'d, which forward() requires
# when edm_scaling=True -- keep this script's prior (non-EDM) behavior.
cfg.model.edm_scaling = False
if args.n_epochs is not None:
    cfg.training.n_epochs = args.n_epochs
if args.num_train_samples is not None:
    cfg.data.N = args.num_train_samples
if args.num_steps is not None:
    cfg.sampling.num_steps = args.num_steps
if args.val_frac is not None:
    cfg.training.val_frac = args.val_frac
if args.sampler is not None:
    cfg.sampling.ode_solver = args.sampler

if not os.path.exists('./runs'):
    os.makedirs('./runs')
run_dir = create_run_dir('./runs')

DIST_CLASSES = {
    'Poisson': lambda: datasets.Poisson(5.0, 40, device),
    'PoissonMixture': lambda: datasets.PoissonMixture(device),
    'ZeroInflatedPoisson': lambda: datasets.ZeroInflatedPoisson(device),
    'NegativeBinomialMixture': lambda: datasets.NegativeBinomialMixture(device),
    'NegativeBinomialMixtureBalanced': lambda: datasets.NegativeBinomialMixtureBalanced(device),
    'BetaNegativeBinomial': lambda: datasets.BetaNegativeBinomial(device),
    'ZipfDistribution': lambda: datasets.ZipfDistribution(device),
    'YuleSimonDistribution': lambda: datasets.YuleSimonDistribution(device),
}
dist = DIST_CLASSES[args.distribution]()
name = dist.__class__.__name__


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        msd = model.state_dict()
        for k, ema_v in ema_model.state_dict().items():
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1. - decay) * model_v)


def denoising_loss_up_down_Nd(forward, model, cfg, minibatch, divergence='quad'):
    B, _ = minibatch.shape
    device = next(model.parameters()).device
    x0 = minibatch.to(device).float()

    ts = torch.rand((B,), device=device) * (cfg.data.T - cfg.data.T0) + cfg.data.T0
    pt, bin_t = forward.conditional_sample(ts, x0)
    model_bin = model(pt, ts)

    if divergence == 'quad':
        loss = bregman_divergence_quadratic(model_bin, bin_t).sum(dim=-1)
    elif divergence == 'xlogx':
        loss = bregman_divergence_xlogx(model_bin, bin_t).sum(dim=-1)
    else:
        raise ValueError("Unsupported divergence type")

    if cfg.training.use_loss_weighting:
        loss *= torch.clamp(ts.pow(cfg.training.loss_weight_power), min=cfg.training.loss_weight_floor)

    return loss.mean()


def denoising_loss_up_down_Nd_data(forward, model, cfg, minibatch):
    """dend(x, t) = E[X_0 | P_t = x] companion loss for TwoHeadNdDenoiser's
    as_dend_module() view."""
    B, _ = minibatch.shape
    device = next(model.parameters()).device
    x0 = minibatch.to(device).float()

    ts = torch.rand((B,), device=device) * (cfg.data.T - cfg.data.T0) + cfg.data.T0
    pt, _ = forward.conditional_sample(ts, x0)
    loss = bregman_divergence_quadratic(model(pt, ts), x0).sum(dim=-1)

    if cfg.training.use_loss_weighting:
        loss *= torch.clamp(ts.pow(cfg.training.loss_weight_power), min=cfg.training.loss_weight_floor)

    return loss.mean()


def train(model, forward, loss_fn, cfg, train_data, val_data):
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=cfg.training.batch_size,
                                              shuffle=cfg.data.shuffle)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.training.batch_size,
                                              shuffle=False)

    losses = []
    train_epoch_losses, val_epoch_losses = [], []
    for _ in range(cfg.training.n_epochs):
        epoch_losses = []
        for minibatch in dataloader:
            optimizer.zero_grad()
            l = loss_fn(forward, model, cfg, minibatch)
            l.backward()
            if cfg.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(model, ema, cfg.model.ema_decay)
            losses.append(l.item())
            epoch_losses.append(l.item())
        train_epoch_losses.append(float(np.mean(epoch_losses)))

        with torch.no_grad():
            val_losses = [loss_fn(forward, ema, cfg, mb).item() for mb in val_loader]
        val_epoch_losses.append(float(np.mean(val_losses)))

    return ema, losses, train_epoch_losses, val_epoch_losses


def train_two_head(model, forward, loss_fn_den, loss_fn_dend, cfg, train_data, val_data):
    """
    Joint-training analogue of train(...) for TwoHeadNdDenoiser (shared
    trunk, den + dend heads; both heads trained jointly, one
    backward()/optimizer.step() per minibatch). Returns (ema, losses, train_epoch_losses,
    val_epoch_losses) with the SAME flat-list shape as train(...) (den+dend
    summed), so the rest of this script's saving code needs no
    per-head awareness.
    """
    view_den, view_dend = model.as_den_module(), model.as_dend_module()
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)
    ema_den, ema_dend = ema.as_den_module(), ema.as_dend_module()

    optimizer = torch.optim.Adam(model.parameters(), cfg.training.lr,
                                  weight_decay=cfg.training.weight_decay)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=cfg.training.batch_size,
                                              shuffle=cfg.data.shuffle)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=cfg.training.batch_size,
                                              shuffle=False)

    losses = []
    train_epoch_losses, val_epoch_losses = [], []
    for _ in range(cfg.training.n_epochs):
        epoch_losses = []
        for minibatch in dataloader:
            optimizer.zero_grad()
            l_den = loss_fn_den(forward, view_den, cfg, minibatch)
            l_dend = loss_fn_dend(forward, view_dend, cfg, minibatch)
            l = l_den + l_dend
            l.backward()
            if cfg.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(model, ema, cfg.model.ema_decay)
            losses.append(l.item())
            epoch_losses.append(l.item())
        train_epoch_losses.append(float(np.mean(epoch_losses)))

        with torch.no_grad():
            val_losses = [(loss_fn_den(forward, ema_den, cfg, mb) +
                            loss_fn_dend(forward, ema_dend, cfg, mb)).item() for mb in val_loader]
        val_epoch_losses.append(float(np.mean(val_losses)))

    return ema, losses, train_epoch_losses, val_epoch_losses


print(f"Training on distribution: {name}")
data = dist.sample(cfg.data.N)
cfg.data.S = int(dist.S + cfg.data.R_max)

# held-out split, shared by both models, to monitor over/underfitting
n_val = max(int(cfg.training.val_frac * data.shape[0]), 1) if cfg.training.val_frac > 0 else 0
perm = torch.randperm(data.shape[0])
val_data, train_data = data[perm[:n_val]], data[perm[n_val:]]
if n_val == 0:
    # --val-frac 0: no held-out set -- train on everything, and fall back
    # to evaluating "val" loss on the training set itself so the existing
    # loss-curve/diagnose plumbing (which expects a val_data split) still runs.
    val_data = train_data

# --- train the "binomial" (pure Binomial-thinning, pure-birth) denoiser ---
if 'binomial' in methods:
    forward_b = PoissonFolmer(S=cfg.data.S, T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim)
    model_b = DenoisingMLP(cfg).to(device)
    ema_b, losses_b, train_epoch_b, val_epoch_b = train(
        model_b, forward_b, denoising_loss, cfg, train_data, val_data)

# --- train the "up_down_Nd" (Binomial thinning + Poisson superposition, birth-death) denoiser ---
if 'up_down_Nd' in methods:
    forward_jn = UpDownNdNoise(T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim,
                                       R_max=cfg.data.R_max, Rd_power=cfg.data.Rd_power,
                                       Ru_power=cfg.data.Ru_power)
    # den + dend share one TwoHeadNdDenoiser trunk, trained jointly by
    # train_two_head and reverse-sampled via ReverseRatesTwo (division-free
    # up_rate).
    model_jn = TwoHeadNdDenoiser(cfg).to(device)
    ema_jn, losses_jn, train_epoch_jn, val_epoch_jn = train_two_head(
        model_jn, forward_jn, denoising_loss_up_down_Nd, denoising_loss_up_down_Nd_data,
        cfg, train_data, val_data)

# --- train the "poisson_only" (genuine additive-Poisson, pure-death) denoiser ---
# P_t = x0 + Pois(Ru(t)), no thinning of x0 at all (forward.
# AdditivePoissonNoise). Ru_max is scaled up (adaptively, per distribution)
# so that the Poisson std sqrt(Ru_max) dominates the spread of x0, making
# the terminal marginal P_T ~approx Pois(Ru_max) APPROXIMATELY (never
# exactly) independent of x0 -- see the class/module docstrings above.
# den(x,t) = E[x0|P_t=x] directly (bin_t == x0 exactly here, no thinning to
# distinguish them), so this reuses "binomial"'s DenoisingMLP/denoising_loss
# unchanged; only the forward process (and hence the induced reverse rates)
# differs. Ru_max can push P_t well outside the shared cfg.data.S state-space
# scale used by every other arm, so DenoisingMLP is built against its own,
# separately-scaled cfg copy (cfg_po) rather than the global cfg.
if 'poisson_only' in methods:
    Ru_max_po = args.Ru_max_Poisson if args.Ru_max_Poisson is not None else AdditivePoissonNoise.auto_scale(dist, c=0.1)
    S_po = int(np.ceil(dist.S + Ru_max_po + 6.0 * np.sqrt(Ru_max_po)))
    cfg_po = copy.deepcopy(cfg)
    cfg_po.data.S = S_po
    # Ru_max_po is typically orders of magnitude larger than the data's own
    # range (it must be, to wash out x0's contribution to P_T -- see class
    # docstring), so a fixed /S_po input normalization would drown the
    # entire data range into a sliver near the normalized input's boundary.
    # EDM scaling (forward_po.edm_target_moments_data) instead rescales the
    # network's input per-t to P_t's own (t-dependent) spread, so this arm
    # forces edm_scaling on regardless of cfg.model.edm_scaling.
    cfg_po.model.edm_scaling = True
    forward_po = AdditivePoissonNoise(T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim,
                                       Ru_max=Ru_max_po, Ru_power=cfg.data.Ru_power)
    model_po = DenoisingMLP(cfg_po).to(device)
    mdata_po = train_data.float().mean().item()
    vdata_po = train_data.float().var(unbiased=False).item()
    model_po.configure_edm(forward_po, mdata_po, vdata_po)
    ema_po, losses_po, train_epoch_po, val_epoch_po = train(
        model_po, forward_po, denoising_loss, cfg_po, train_data, val_data)


def denoising_loss_up_down_Zd(forward, model, cfg, minibatch, divergence='quad'):
    # ZDenoiser is trained to predict d_t = E[D_t | P_t] from the noised
    # state pt = x0 - D_t + U_t (UpDownZdNoise.conditional_sample),
    # an independent subtractive Poisson latent rather than a thinning of
    # x0, so like "up_down_Nd" this needs its own training loss here.
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

    if cfg.training.use_loss_weighting:
        loss *= torch.clamp(ts.pow(cfg.training.loss_weight_power), min=cfg.training.loss_weight_floor)

    return loss.mean()


# --- train the "up_down_Zd" ("up_down_Zd", birth-death) denoiser ---
if 'up_down_Zd' in methods:
    if args.Rd_max_Zd is not None or args.Ru_max_Zd is not None:
        Rd_max_zd = args.Rd_max_Zd if args.Rd_max_Zd is not None else cfg.data.R_max
        Ru_max_zd = args.Ru_max_Zd if args.Ru_max_Zd is not None else cfg.data.R_max
    else:
        Rd_max_zd, Ru_max_zd = UpDownZdNoise.auto_scales(dist)
    forward_jz = UpDownZdNoise(T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim,
                                      Rd_max=Rd_max_zd, Ru_max=Ru_max_zd,
                                      Rd_power=cfg.data.Rd_power, Ru_power=cfg.data.Ru_power)
    model_jz = ZDenoiser(cfg).to(device)
    ema_jz, losses_jz, train_epoch_jz, val_epoch_jz = train(
        model_jz, forward_jz, denoising_loss_up_down_Zd, cfg, train_data, val_data)

# --- train the "count_bridge" (exact Poisson birth-death bridge, Fishman et al. 2026) denoiser ---
if 'count_bridge' in methods:
    if args.Rd_max_Zd is not None or args.Ru_max_Zd is not None:
        Rd_max_cb = args.Rd_max_Zd if args.Rd_max_Zd is not None else cfg.data.R_max
        Ru_max_cb = args.Ru_max_Zd if args.Ru_max_Zd is not None else cfg.data.R_max
    else:
        Rd_max_cb, Ru_max_cb = CountBridgeNoise.auto_scales(dist)
    forward_cb = CountBridgeNoise(T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim,
                                   Rd_max=Rd_max_cb, Ru_max=Ru_max_cb)
    model_cb = DenoisingMLP(cfg).to(device)
    ema_cb, losses_cb, train_epoch_cb, val_epoch_cb = train(
        model_cb, forward_cb, denoising_loss, cfg, train_data, val_data)

# --- train the "count_fm" (local-jump birth-death, Wei & Pearson 2026) denoiser ---
if 'count_fm' in methods:
    forward_fm = CountFMBridge(dim=cfg.data.dim, S=cfg.data.S)
    model_fm = CountFMRates(cfg).to(device)
    ema_fm, losses_fm, train_epoch_fm, val_epoch_fm = train(
        model_fm, forward_fm, denoising_loss_count_fm, cfg, train_data, val_data)

# --- train the "categorical" (SEDD-style, Lou, Meng & Ermon 2023) score model ---
if 'categorical' in methods:
    forward_cat = CategoricalUniformNoise(T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim, S=cfg.data.S)
    model_cat = CategoricalScoreModel(cfg).to(device)
    ema_cat, losses_cat, train_epoch_cat, val_epoch_cat = train(
        model_cat, forward_cat, denoising_loss_categorical, cfg, train_data, val_data)

N = args.num_paths

# exact reference (t=T / noise) PMF per method -- (grid, pmf) arrays -- for
# plot_paths.py's reference-distribution PMF panel, computed analytically
# from each process's known terminal/source distribution rather than an
# empirical histogram from samples.
def _poisson_pmf_grid(rate, n_std=6):
    std = np.sqrt(max(rate, 1e-8))
    lo, hi = max(0, int(np.floor(rate - n_std * std))), int(np.ceil(rate + n_std * std))
    grid = np.arange(lo, hi + 1)
    return grid, poisson_dist.pmf(grid, mu=rate)


def _skellam_pmf_grid(mu_up, mu_down, n_std=6):
    mean, std = mu_up - mu_down, np.sqrt(mu_up + mu_down)
    lo, hi = int(np.floor(mean - n_std * std)), int(np.ceil(mean + n_std * std))
    grid = np.arange(lo, hi + 1)
    return grid, skellam_dist.pmf(grid, mu1=mu_up, mu2=mu_down)


ref_pmfs = {}

_birth_sampler = TauLeaping if cfg.sampling.ode_solver == 'tau_leaping' else Euler
_birth_death_sampler = TauLeapingBirthDeath if cfg.sampling.ode_solver == 'tau_leaping' else EulerBirthDeath

# --- sample paths from the pure-birth ("binomial") process ---
if 'binomial' in methods:
    intensity = Intensity(ema_b, max_time=cfg.data.T, S=cfg.data.S, denormalize=False)
    x_init_b = forward_b.sample_initial(N, device=device)
    time_grid_b = torch.linspace(forward_b.T0, forward_b.T - 1e-4, cfg.sampling.num_steps)
    _, x_hist_b = _birth_sampler(forward_b, intensity, time_grid_b, x_init_b, device=device)
    x_hist_b = x_hist_b.squeeze(-1).numpy()          # (K, N)
    progress_b = ((time_grid_b - forward_b.T0) / (forward_b.T - forward_b.T0)).numpy()
    # sample_initial is deterministically 0 (pure-birth starts empty)
    ref_pmfs['binomial'] = (np.array([0]), np.array([1.0]))

# --- sample paths from the birth-death ("up_down_Nd") process ---
if 'up_down_Nd' in methods:
    rates = ReverseRatesTwo(ema_jn.as_den_module(), ema_jn.as_dend_module(), forward_jn)
    x_init_jn = forward_jn.sample_terminal(N, device=device)
    time_grid_jn = torch.linspace(forward_jn.T - 1e-4, forward_jn.T0 + 1e-4, cfg.sampling.num_steps)
    _, x_hist_jn = _birth_death_sampler(rates, time_grid_jn, x_init_jn, device=device)
    x_hist_jn = x_hist_jn.squeeze(-1).numpy()          # (K, N)
    progress_jn = ((forward_jn.T - time_grid_jn) / (forward_jn.T - forward_jn.T0)).numpy()
    # sample_terminal ~ Pois(Ru(T)) exactly
    ru_T = float(forward_jn.Ru(torch.tensor([forward_jn.T])))
    ref_pmfs['up_down_Nd'] = _poisson_pmf_grid(ru_T)

# --- sample paths from the genuine additive-Poisson ("poisson_only") process ---
if 'poisson_only' in methods:
    # pure death: reverse rates only ever remove mass (model_utils.
    # ReverseRatesAdditive), starting from x_init ~approx Pois(Ru_max)
    # (sample_terminal drops the vanishing-for-large-Ru_max x0 term) and
    # ending at x0. Ru_max here is scaled to the data's spread SQUARED
    # (AdditivePoissonNoise.auto_scale), so the reverse chain must remove
    # O(Ru_max) mass over the trajectory -- far more than Euler's
    # at-most-one-jump-per-step can manage in a reasonable step budget, so
    # (unlike every other lattice arm here) this ignores --sampler and
    # always uses uncapped tau-leaping, letting each step's jump count be
    # the natural Poisson(rate*tau) implied by the learned rate.
    rates_po = ReverseRatesAdditive(ema_po, forward_po)
    x_init_po = forward_po.sample_terminal(N, device=device)
    time_grid_po = torch.linspace(forward_po.T - 1e-4, forward_po.T0 + 1e-4, cfg.sampling.num_steps)
    _, x_hist_po = TauLeapingBirthDeath(rates_po, time_grid_po, x_init_po, device=device,
                                         max_expected_jumps=None)
    x_hist_po = x_hist_po.squeeze(-1).numpy()          # (K, N)
    progress_po = ((forward_po.T - time_grid_po) / (forward_po.T - forward_po.T0)).numpy()
    # sample_terminal ~approx Pois(Ru(T)) (dropping x0's vanishing contribution)
    ru_T_po = float(forward_po.Ru(torch.tensor([forward_po.T])))
    ref_pmfs['poisson_only'] = _poisson_pmf_grid(ru_T_po)

# --- sample paths from the "up_down_Zd" (birth-death on Z) process ---
if 'up_down_Zd' in methods:
    rates_jz = ReverseRatesZ(ema_jz, forward_jz)
    x_init_jz = forward_jz.sample_terminal(N, device=device)
    time_grid_jz = torch.linspace(forward_jz.T - 1e-4, forward_jz.T0 + 1e-4, cfg.sampling.num_steps)
    _, x_hist_jz = _birth_death_sampler(rates_jz, time_grid_jz, x_init_jz, device=device, clamp_min=None)
    x_hist_jz = x_hist_jz.squeeze(-1).numpy()          # (K, N)
    progress_jz = ((forward_jz.T - time_grid_jz) / (forward_jz.T - forward_jz.T0)).numpy()
    # sample_terminal ~approx Skellam(Ru(T), Rd(T))
    ru_T_jz = float(forward_jz.Ru(torch.tensor([forward_jz.T])))
    rd_T_jz = float(forward_jz.Rd(torch.tensor([forward_jz.T])))
    ref_pmfs['up_down_Zd'] = _skellam_pmf_grid(ru_T_jz, rd_T_jz)

# --- sample paths from the "count_bridge" (exact ancestral bridge) process ---
if 'count_bridge' in methods:
    x_init_cb = forward_cb.sample_terminal(N, device=device)
    time_grid_cb = torch.linspace(forward_cb.T - 1e-4, forward_cb.T0, cfg.sampling.num_steps)
    _, x_hist_cb = sample_count_bridge(ema_cb, forward_cb, time_grid_cb, x_init_cb, device)
    x_hist_cb = x_hist_cb.squeeze(-1).numpy()          # (K, N)
    progress_cb = ((forward_cb.T - time_grid_cb) / (forward_cb.T - forward_cb.T0)).numpy()
    # sample_reference ~ Skellam(Ru_max, Rd_max)
    ref_pmfs['count_bridge'] = _skellam_pmf_grid(forward_cb.Ru_max, forward_cb.Rd_max)

# --- sample paths from the "count_fm" (local-jump discretization) process ---
if 'count_fm' in methods:
    _, x_hist_fm = sample_count_fm(ema_fm, forward_fm, N, cfg.sampling.num_steps, device)
    x_hist_fm = x_hist_fm.squeeze(-1).numpy()          # (K, N)
    progress_fm = np.linspace(0.0, 1.0, x_hist_fm.shape[0])
    # sample_source ~ discrete-uniform on {0,...,S-1}
    ref_pmfs['count_fm'] = (np.arange(forward_fm.S), np.full(forward_fm.S, 1.0 / forward_fm.S))

# --- sample paths from the "categorical" (SEDD-style, unordered) process ---
if 'categorical' in methods:
    _, x_hist_cat = sample_categorical(ema_cat, forward_cat, N, cfg.sampling.num_steps, device)
    x_hist_cat = x_hist_cat.squeeze(-1).numpy()          # (K, N)
    progress_cat = np.linspace(0.0, 1.0, x_hist_cat.shape[0])
    # sample_terminal ~ discrete-uniform on {0,...,S-1}
    ref_pmfs['categorical'] = (np.arange(forward_cat.S), np.full(forward_cat.S, 1.0 / forward_cat.S))

# --- independently sample N_HIST final draws per method (not saved as
# trajectories) to get a well-populated empirical histogram of generated
# samples for plot_paths.py's target-side panel, decoupled from the (much
# smaller) number of trajectories actually drawn ---
N_HIST = args.num_hist_samples
hist_samples = {}

if 'binomial' in methods:
    x_init_b_h = forward_b.sample_initial(N_HIST, device=device)
    _, x_hist_b_h = _birth_sampler(forward_b, intensity, time_grid_b, x_init_b_h, device=device)
    hist_samples['binomial'] = x_hist_b_h.squeeze(-1).numpy()[-1, :]

if 'up_down_Nd' in methods:
    x_init_jn_h = forward_jn.sample_terminal(N_HIST, device=device)
    _, x_hist_jn_h = _birth_death_sampler(rates, time_grid_jn, x_init_jn_h, device=device)
    hist_samples['up_down_Nd'] = x_hist_jn_h.squeeze(-1).numpy()[-1, :]

if 'poisson_only' in methods:
    x_init_po_h = forward_po.sample_terminal(N_HIST, device=device)
    _, x_hist_po_h = TauLeapingBirthDeath(rates_po, time_grid_po, x_init_po_h, device=device,
                                           max_expected_jumps=None)
    hist_samples['poisson_only'] = x_hist_po_h.squeeze(-1).numpy()[-1, :]

if 'up_down_Zd' in methods:
    x_init_jz_h = forward_jz.sample_terminal(N_HIST, device=device)
    _, x_hist_jz_h = _birth_death_sampler(rates_jz, time_grid_jz, x_init_jz_h, device=device, clamp_min=None)
    hist_samples['up_down_Zd'] = x_hist_jz_h.squeeze(-1).numpy()[-1, :]

if 'count_bridge' in methods:
    x_init_cb_h = forward_cb.sample_terminal(N_HIST, device=device)
    _, x_hist_cb_h = sample_count_bridge(ema_cb, forward_cb, time_grid_cb, x_init_cb_h, device)
    hist_samples['count_bridge'] = x_hist_cb_h.squeeze(-1).numpy()[-1, :]

if 'count_fm' in methods:
    _, x_hist_fm_h = sample_count_fm(ema_fm, forward_fm, N_HIST, cfg.sampling.num_steps, device)
    hist_samples['count_fm'] = x_hist_fm_h.squeeze(-1).numpy()[-1, :]

if 'categorical' in methods:
    _, x_hist_cat_h = sample_categorical(ema_cat, forward_cat, N_HIST, cfg.sampling.num_steps, device)
    hist_samples['categorical'] = x_hist_cat_h.squeeze(-1).numpy()[-1, :]

# --- true samples from the target data distribution, for reference ---
true_samples = dist.sample(N).float().cpu().numpy().squeeze()

# Per-method info needed by plot_paths.py, keyed in the same order as
# ALL_METHODS so panels always appear in a fixed order regardless of the
# order --methods was passed in.
METHOD_INFO = {
    'binomial':     dict(title='Binomial-only',
                          loss_title='Binomial-only: training loss', label='Binomial-only'),
    'poisson_only': dict(title='Poisson-only',
                          loss_title='Poisson-only: training loss', label='Poisson-only'),
    'up_down_Nd':   dict(title='Binomial-Poisson',
                          loss_title='Binomial-Poisson: training loss', label='Binomial-Poisson'),
    'up_down_Zd':   dict(title='Up-and-down (Z, birth-death)',
                          loss_title='Up-and-down (Z): training loss', label='Up-and-down (Z)'),
    'count_bridge': dict(title='Count Bridges',
                          loss_title='Count Bridges: training loss', label='Count Bridges'),
    'count_fm':     dict(title='Count-FM',
                          loss_title='Count-FM: training loss (generalized-KL)', label='Count-FM'),
    'categorical':  dict(title='Categorical (SEDD)',
                          loss_title='Categorical (SEDD): training loss (score entropy)', label='Categorical (SEDD)'),
}
METHOD_VARS = {
    'binomial':     dict(x_hist='x_hist_b', progress='progress_b',
                          losses='losses_b', train_epoch='train_epoch_b', val_epoch='val_epoch_b'),
    'poisson_only': dict(x_hist='x_hist_po', progress='progress_po',
                          losses='losses_po', train_epoch='train_epoch_po', val_epoch='val_epoch_po'),
    'up_down_Nd':   dict(x_hist='x_hist_jn', progress='progress_jn',
                          losses='losses_jn', train_epoch='train_epoch_jn', val_epoch='val_epoch_jn'),
    'up_down_Zd':   dict(x_hist='x_hist_jz', progress='progress_jz',
                          losses='losses_jz', train_epoch='train_epoch_jz', val_epoch='val_epoch_jz'),
    'count_bridge': dict(x_hist='x_hist_cb', progress='progress_cb',
                          losses='losses_cb', train_epoch='train_epoch_cb', val_epoch='val_epoch_cb'),
    'count_fm':     dict(x_hist='x_hist_fm', progress='progress_fm',
                          losses='losses_fm', train_epoch='train_epoch_fm', val_epoch='val_epoch_fm'),
    'categorical':  dict(x_hist='x_hist_cat', progress='progress_cat',
                          losses='losses_cat', train_epoch='train_epoch_cat', val_epoch='val_epoch_cat'),
}
selected = [m for m in ALL_METHODS if m in methods]
local_vars = dict(locals())
results = {m: {**METHOD_INFO[m], **{k: local_vars[v] for k, v in METHOD_VARS[m].items()}}
           for m in selected}

target_pmf = dist.probs().detach().cpu().numpy().squeeze()
target_x = np.arange(len(target_pmf))


def diagnose(train_epoch, val_epoch, label):
    final_gap = val_epoch[-1] - train_epoch[-1]
    best_val_epoch = int(np.argmin(val_epoch)) + 1
    still_improving = val_epoch[-1] <= min(val_epoch[:-1] or [np.inf]) + 1e-8
    if still_improving:
        verdict = "still improving at the last epoch -- likely UNDERFIT (train longer)"
    elif final_gap > 0.1 * abs(train_epoch[-1]):
        verdict = f"validation loss rising away from training loss -- likely OVERFIT (best epoch {best_val_epoch}/{len(val_epoch)})"
    else:
        verdict = f"train/val losses track each other -- reasonable fit (best epoch {best_val_epoch}/{len(val_epoch)})"
    print(f"  {label:15s} final train={train_epoch[-1]:.4f} val={val_epoch[-1]:.4f} ({verdict})")


print("\nOver/underfitting diagnosis:")
for m in selected:
    r = results[m]
    diagnose(r['train_epoch'], r['val_epoch'], r['label'])

# --- save everything plot_paths.py needs ---
save_dict = {
    'distribution': name,
    'selected': selected,
    'n_epochs': cfg.training.n_epochs,
    'target_pmf': target_pmf,
    'target_x': target_x,
    'true_samples': true_samples,
    'method_info': {m: METHOD_INFO[m] for m in selected},
    'ref_pmfs': {m: ref_pmfs[m] for m in selected},
    'hist_samples': {m: hist_samples[m] for m in selected},
}
for m in selected:
    r = results[m]
    save_dict[f'x_hist_{m}'] = r['x_hist']
    save_dict[f'progress_{m}'] = r['progress']
    save_dict[f'losses_{m}'] = r['losses']
    save_dict[f'train_epoch_{m}'] = r['train_epoch']
    save_dict[f'val_epoch_{m}'] = r['val_epoch']
torch.save(save_dict, f'{run_dir}/paths_{name}.pt')

print(f"\nSaved training/sampling output to {run_dir}/paths_{name}.pt for: {', '.join(selected)}")
print(f"Plot it with: python plot_paths.py --run {run_dir}/paths_{name}.pt")
