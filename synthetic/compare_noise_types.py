"""Compare accuracy of different noising processes trained on the same 1-D
synthetic target distributions.

Methods (see METHODS below for the forward/model/loss/sampler used by each):
  binomial     : pure Binomial-thinning noise (forward.PoissonFolmer)
  up_down_Nd   : Binomial thinning + Poisson superposition on N_0^d
                 (forward.UpDownNdNoise)
  up_down_Zd   : same but on Z^d, P_t = X_0 - D_t + U_t (forward.UpDownZdNoise)
  count_bridge : exact Poisson birth-death bridge (Fishman et al. 2026,
                 "Count Bridges", forward.CountBridgeNoise)
  count_fm     : local-jump birth-death bridge (Wei & Pearson 2026,
                 "Flow Matching for Count Data", forward.CountFMBridge)
  categorical  : SEDD-style ratio matching over unordered categories
                 (forward.CategoricalUniformNoise)

For each target distribution and trial: train every method on the same data,
sample from each, and report TV distance (to the true PMF) and Wasserstein-1
distance (to held-out true samples) as accuracy metrics.
"""

import argparse
import copy
import os
import sys

import matplotlib.pyplot as plt
import ml_collections
import numpy as np
import torch
import yaml
from scipy.stats import wasserstein_distance

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from model import DenoisingMLP, BinomialDenoiser, ZDenoiser, CountFMRates, CategoricalScoreModel
from loss import denoising_loss, bregman_divergence_quadratic, denoising_loss_count_fm, denoising_loss_categorical
from forward import PoissonFolmer, UpDownNdNoise, UpDownZdNoise, CountBridgeNoise, CountFMBridge, CategoricalUniformNoise
from sampling import (Euler, TauLeaping, EulerBirthDeath, TauLeapingBirthDeath,
                       sample_count_bridge, sample_count_fm, sample_categorical)
from model_utils import Intensity, ReverseRates, ReverseRatesZ
from utils import create_run_dir, tv_distance
import datasets
import datasets_highdim

parser = argparse.ArgumentParser()
parser.add_argument('--per_method_hist', action='store_true',
                     help='Also save a figure with one subplot per method (in a row), each showing '
                          'that method\'s sample histogram overlaid on the true PMF.')
parser.add_argument('--methods', nargs='+', default=None,
                     help='Subset of method names to run (see METHODS below for valid names, '
                          'e.g. --methods up_down_Zd). Default: run all methods.')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

pwd = os.path.dirname(os.path.abspath(__file__)) + '/'

with open(pwd + 'config.yaml', "r") as f:
    cfg_dict = yaml.safe_load(f)
cfg = ml_collections.ConfigDict(cfg_dict)
# EDM scaling is enabled
# per-method below (METHODS[...]['use_edm']): only up_down_Nd/up_down_Zd's
# BinomialDenoiser/ZDenoiser support it (their forward processes expose the
# needed edm_target_moments_*); DenoisingMLP with PoissonFolmer/
# CountBridgeNoise and the count-FM/categorical models stay in legacy mode
# (PoissonFolmer/CountBridgeNoise don't expose edm_target_moments_data, and
# CountFMRates/CategoricalScoreModel never consult edm_scaling at all).

run_dir = create_run_dir(pwd + 'runs')

dists = [datasets.Poisson(5.0, 40, device),
         datasets.PoissonMixture(device),
         datasets.ZeroInflatedPoisson(device),
         datasets.NegativeBinomialMixture(device),
         datasets.BetaNegativeBinomial(device),
         datasets.ZipfDistribution(device),
         datasets.YuleSimonDistribution(device),
         datasets_highdim.CorrelatedBetaBinomial(dim=1, centered=False, device=device)]
num_trials = 5


def update_ema(model, ema_model, decay):
    with torch.no_grad():
        msd = model.state_dict()
        for k, ema_v in ema_model.state_dict().items():
            model_v = msd[k].detach()
            ema_v.copy_(ema_v * decay + (1. - decay) * model_v)


def denoising_loss_up_down_Nd(forward, model, cfg, minibatch):
    """ BinomialDenoiser is trained to predict bin_t = E[Bin_t | P_t] from pt.

    When cfg.model.edm_scaling is on, model_bin is already in raw (cskip*x +
    cout*net) prediction space, so the raw squared error here scales with
    however large bin_t itself is (e.g. thousands, for auto-scaled R_max).
    Weighting by model.edm_lambda() = w(t)/cout(t)^2 converts this back into
    the intended, t-uniform loss on the underlying net_theta residual (see
    model.py's edm_lambda docstring), instead of on the raw target scale.
    """
    B, _ = minibatch.shape
    x0 = minibatch.to(device).float()
    ts = torch.rand((B,), device=device) * (cfg.data.T - cfg.data.T0) + cfg.data.T0
    pt, bin_t = forward.conditional_sample(ts, x0)
    model_bin = model(pt, ts)
    loss = bregman_divergence_quadratic(model_bin, bin_t).sum(dim=-1)
    if cfg.model.edm_scaling:
        loss = loss * model.edm_lambda()
    return loss.mean()


def denoising_loss_up_down_Zd(forward, model, cfg, minibatch):
    """ ZDenoiser is trained to predict d_t = E[D_t | P_t] from pt.

    Same edm_lambda reweighting as denoising_loss_up_down_Nd above: without
    it, distributions whose auto-scaled Rd_max/Ru_max are large (heavy-tailed
    or zero-spiked targets) push d_t/u_t into the thousands, and the raw
    squared error explodes accordingly (observed loss spikes up to ~1e7).
    """
    B, _ = minibatch.shape
    x0 = minibatch.to(device).float()
    ts = torch.rand((B,), device=device) * (cfg.data.T - cfg.data.T0) + cfg.data.T0
    pt, d_t, u_t = forward.conditional_sample(ts, x0)
    model_d = model(pt, ts)
    loss = bregman_divergence_quadratic(model_d, d_t).sum(dim=-1)
    if cfg.model.edm_scaling:
        loss = loss * model.edm_lambda()
    return loss.mean()


def train(model, forward, loss_fn, cfg, data):
    ema = copy.deepcopy(model)
    for p in ema.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(model.parameters(), cfg.training.lr, weight_decay=cfg.training.weight_decay)
    dataloader = torch.utils.data.DataLoader(data, batch_size=cfg.training.batch_size, shuffle=cfg.data.shuffle)

    losses = []
    for _ in range(cfg.training.n_epochs):
        for minibatch in dataloader:
            optimizer.zero_grad()
            loss = loss_fn(forward, model, cfg, minibatch)
            loss.backward()
            if cfg.training.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            update_ema(model, ema, cfg.model.ema_decay)
            losses.append(loss.item())
    return ema, losses


def sample_binomial(forward, ema, cfg, num_samples):
    intensity = Intensity(ema, max_time=cfg.data.T, S=cfg.data.S, denormalize=False)
    x_init = forward.sample_initial(num_samples, device=device)
    time_grid = torch.linspace(forward.T0, forward.T - 1e-4, cfg.sampling.num_steps)
    sampler = Euler if cfg.sampling.ode_solver == 'euler' else TauLeaping
    samples, _ = sampler(forward, intensity, time_grid, x_init, device=device)
    return samples


def sample_jump_N(forward, ema, cfg, num_samples):
    rates = ReverseRates(ema, forward)
    x_init = forward.sample_terminal(num_samples, device=device)
    time_grid = torch.linspace(forward.T - 1e-4, forward.T0 + 1e-4, cfg.sampling.num_steps)
    sampler = EulerBirthDeath if cfg.sampling.ode_solver == 'euler' else TauLeapingBirthDeath
    samples, _ = sampler(rates, time_grid, x_init, device=device)
    return samples


def sample_jump_Z(forward, ema, cfg, num_samples):
    rates = ReverseRatesZ(ema, forward)
    x_init = forward.sample_terminal(num_samples, device=device)
    # Log-spaced time grid (geometric in distance-from-T0, i.e. denser steps
    # near the data end T0 and coarser ones near the noise end T) rather than
    # linspace. NOTE: this is not a verified accuracy fix -- in a head-to-head
    # comparison, linear vs. log spacing traded off differently on
    # heavy-tailed targets (log: worse TV, better W1) rather than either
    # being a clear win.
    eps = 1e-4
    dist_from_T0 = np.geomspace(forward.T - forward.T0 - eps, eps, cfg.sampling.num_steps)
    time_grid = torch.as_tensor(forward.T0 + dist_from_T0, dtype=torch.float32)
    sampler = EulerBirthDeath if cfg.sampling.ode_solver == 'euler' else TauLeapingBirthDeath
    # state space is Z^d (unbounded below), so clamp_min=None
    samples, _ = sampler(rates, time_grid, x_init, device=device, clamp_min=None)
    return samples


def sample_bridge(forward, ema, cfg, num_samples):
    x_init = forward.sample_terminal(num_samples, device=device)
    time_grid = torch.linspace(forward.T - 1e-4, forward.T0, cfg.sampling.num_steps)
    samples, _ = sample_count_bridge(ema, forward, time_grid, x_init, device)
    return samples


def sample_fm(forward, ema, cfg, num_samples):
    samples, _ = sample_count_fm(ema, forward, num_samples, cfg.sampling.num_steps, device)
    return samples


def sample_categorical_method(forward, ema, cfg, num_samples):
    samples, _ = sample_categorical(ema, forward, num_samples, cfg.sampling.num_steps, device)
    return samples


# Each method: how to build its forward process + model (from cfg, needs
# cfg.data.S set), its training loss, and its reverse sampler.
METHODS = [
    dict(name='binomial', label='Binomial-only',
         build_forward=lambda cfg, dist: PoissonFolmer(S=cfg.data.S, T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim),
         build_model=lambda cfg: DenoisingMLP(cfg).to(device),
         loss_fn=denoising_loss,
         sample_fn=sample_binomial),
    dict(name='up_down_Nd', label='Up-and-down (N^d)',
         build_forward=lambda cfg, dist: UpDownNdNoise(T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim,
                                                        R_max=cfg.data.R_max, Rd_power=cfg.data.Rd_power,
                                                        Ru_power=cfg.data.Ru_power),
         build_model=lambda cfg: BinomialDenoiser(cfg).to(device),
         loss_fn=denoising_loss_up_down_Nd,
         sample_fn=sample_jump_N,
         use_edm=True,
         configure_edm=lambda model, forward, mean, var: model.configure_edm(forward, mean, var)),
    dict(name='up_down_Zd', label='Up-and-down (Z^d)',
         build_forward=lambda cfg, dist: UpDownZdNoise(T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim,
                                                        **dict(zip(('Rd_max', 'Ru_max'), UpDownZdNoise.auto_scales(dist))),
                                                        Rd_power=cfg.data.Rd_power, Ru_power=cfg.data.Ru_power),
         build_model=lambda cfg: ZDenoiser(cfg).to(device),
         loss_fn=denoising_loss_up_down_Zd,
         sample_fn=sample_jump_Z,
         use_edm=True,
         # ZDenoiser is trained to predict D_t (denbin), see denoising_loss_up_down_Zd
         configure_edm=lambda model, forward, mean, var: model.configure_edm(forward, 'D', mean, var)),
    dict(name='count_bridge', label='Count Bridges (Fishman et al. 2026)',
         build_forward=lambda cfg, dist: CountBridgeNoise(T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim,
                                                           Rd_max=cfg.data.R_max, Ru_max=cfg.data.R_max),
         build_model=lambda cfg: DenoisingMLP(cfg).to(device),
         loss_fn=denoising_loss,
         sample_fn=sample_bridge),
    dict(name='count_fm', label='count-FM (Wei & Pearson 2026)',
         build_forward=lambda cfg, dist: CountFMBridge(dim=cfg.data.dim, S=cfg.data.S),
         build_model=lambda cfg: CountFMRates(cfg).to(device),
         loss_fn=denoising_loss_count_fm,
         sample_fn=sample_fm),
    dict(name='categorical', label='Categorical / SEDD (Lou et al. 2023)',
         build_forward=lambda cfg, dist: CategoricalUniformNoise(T0=cfg.data.T0, T=cfg.data.T, dim=cfg.data.dim, S=cfg.data.S),
         build_model=lambda cfg: CategoricalScoreModel(cfg).to(device),
         loss_fn=denoising_loss_categorical,
         sample_fn=sample_categorical_method),
]

if args.methods is not None:
    valid_names = {m['name'] for m in METHODS}
    unknown = set(args.methods) - valid_names
    if unknown:
        raise ValueError(f"Unknown method name(s) {sorted(unknown)}; valid choices are {sorted(valid_names)}")
    METHODS = [m for m in METHODS if m['name'] in args.methods]

####################################
# RUN COMPARISON
####################################

results = {dist.__class__.__name__: {m['name']: {'tv': [], 'w1': []} for m in METHODS} for dist in dists}

for dist in dists:
    name = dist.__class__.__name__

    for trial in range(num_trials):
        print('============================================')
        print(f"Distribution: {name}, Trial: {trial + 1}")

        data = dist.sample(cfg.data.N)
        # normalization scale must cover the data range plus the injected Poisson noise
        cfg.data.S = int(dist.S + cfg.data.R_max)
        # EDM target moments are population mean/var of the data itself
        mean_data, var_data = float(data.float().mean()), float(data.float().var())

        samples_np, losses = {}, {}
        for m in METHODS:
            cfg.model.edm_scaling = m.get('use_edm', False)
            forward = m['build_forward'](cfg, dist)
            model = m['build_model'](cfg)
            if m.get('configure_edm'):
                m['configure_edm'](model, forward, mean_data, var_data)
            ema, losses[m['name']] = train(model, forward, m['loss_fn'], cfg, data)
            samples = m['sample_fn'](forward, ema, cfg, cfg.sampling.num_samples)
            samples_np[m['name']] = samples.float().cpu().numpy().squeeze()

        # --- evaluation ---
        eval_data = dist.sample(cfg.sampling.num_samples)
        eval_data_np = eval_data.float().cpu().numpy().squeeze()
        true_probs = dist.probs().cpu().numpy()

        for m in METHODS:
            tv = tv_distance(samples_np[m['name']], true_probs, dist.S)
            w1 = wasserstein_distance(samples_np[m['name']], eval_data_np).item()
            print(f"  {m['label']:38s} | TV: {tv:.4f}  W1: {w1:.4f}")
            results[name][m['name']]['tv'].append(tv)
            results[name][m['name']]['w1'].append(w1)

        # --- plots: training loss ---
        plt.figure()
        for m in METHODS:
            plt.plot(losses[m['name']], label=m['label'], alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('Denoising Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{run_dir}/loss_{name}_trial{trial + 1}.pdf', bbox_inches='tight')
        plt.close()

        # --- plots: sample histograms vs. true PMF ---
        bins = np.arange(0, dist.S + 0.5) - 0.5
        plt.figure()
        plt.hist(eval_data_np, bins, density=True, alpha=0.4, label='True Samples')
        for m in METHODS:
            plt.hist(samples_np[m['name']], bins, density=True, alpha=0.4, label=m['label'])
        plt.plot(bins[:-1] + 0.5, true_probs, '-k', lw=1.5, label='True PMF')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{run_dir}/samples_{name}_trial{trial + 1}.pdf', bbox_inches='tight')
        plt.close()

        # --- plots (optional): per-method sample histogram vs. true PMF, one subplot per method ---
        if args.per_method_hist:
            fig, axes = plt.subplots(1, len(METHODS), figsize=(4 * len(METHODS), 3.5), sharey=True, squeeze=False)
            axes = axes[0]
            for ax, m in zip(axes, METHODS):
                ax.hist(eval_data_np, bins, density=True, alpha=0.4, label='True Samples')
                ax.hist(samples_np[m['name']], bins, density=True, alpha=0.4, label=m['label'])
                ax.plot(bins[:-1] + 0.5, true_probs, '-k', lw=1.5, label='True PMF')
                ax.set_title(m['label'], fontsize=9)
                ax.set_xlabel('x')
            axes[0].set_ylabel('Density')
            axes[0].legend(fontsize=7)
            plt.tight_layout()
            plt.savefig(f'{run_dir}/samples_per_method_{name}_trial{trial + 1}.pdf', bbox_inches='tight')
            plt.close()

        # --- save raw results ---
        torch.save({
            'data': eval_data,
            'samples': {m['name']: samples_np[m['name']] for m in METHODS},
            'losses': losses,
        }, f'{run_dir}/results_{name}_trial{trial + 1}.pt')

####################################
# SUMMARY
####################################
print()
print("Summary of Results")
for name in results:
    print(f"\nDistribution: {name}")
    for m in METHODS:
        tv_arr = np.array(results[name][m['name']]['tv'])
        w1_arr = np.array(results[name][m['name']]['w1'])
        print(f"  {m['label']:38s} TV: {tv_arr.mean():.4f} +/- {tv_arr.std() / np.sqrt(len(tv_arr)):.4f}"
              f"   W1: {w1_arr.mean():.4f} +/- {w1_arr.std() / np.sqrt(len(w1_arr)):.4f}")
