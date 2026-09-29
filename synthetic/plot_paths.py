import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
plt.rcParams.update({'text.usetex': True, 'font.family': 'serif'})
import numpy as np
import os
import argparse

import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from utils import create_run_dir, tv_distance, w1_distance

# ============================================================
# Renders the three plot sets for a run saved by train_paths.py:
#
#   Plot 1 ("paths_<dist>_<method>.pdf"): individual trajectories, one
#   figure per selected process, stacked vertically with the state value x
#   on the shared (horizontal) x-axis and generative progress on the
#   (vertical) y-axis, increasing downward. The trajectory panel is flanked
#   by a line-PMF strip on top (exact PMF of the reverse chain's x_init,
#   i.e. the noising process's reference/terminal distribution -- this is
#   what silently loses a mode if it's a bad approximation) and on the
#   bottom (exact PMF of the target data distribution, overlaid with a
#   histogram of generated samples), sharing the trajectory panel's x-axis
#   so mismatches are visible at a glance.
#
#   Plot 2 ("loss_<dist>[_<method>].pdf"): train vs. validation loss, to
#   diagnose over/underfitting.
#
#   Plot 3 ("hist_<dist>[_<method>].pdf"): histograms of final generated
#   samples vs. true data samples, to compare marginal fit across processes.
#
# See train_paths.py for the training/sampling step that produces the .pt
# file this script loads.
# ============================================================

parser = argparse.ArgumentParser(description="Plot sampling paths/losses/histograms from a "
                                              ".pt file saved by train_paths.py.")
parser.add_argument('--run', type=str, required=True,
                     help="Path to the paths_<dist>.pt file saved by train_paths.py.")
parser.add_argument('--methods', type=str, nargs='+', default=None,
                     help="Subset of the methods saved in --run to plot (default: all of them).")
parser.add_argument('--show-titles', action='store_true', default=False,
                     help="Show each panel's method title (e.g. 'Binomial-only', 'Categorical "
                          "(SEDD)') above the trajectory plot in Plot 1. Off by default.")
parser.add_argument('--out-dir', type=str, default=None,
                     help="Directory to save plots into (default: a fresh run dir under ./runs).")
parser.add_argument('--num-paths', type=int, default=None,
                     help="Number of individual trajectories to draw in Plot 1 (default: all "
                          "trajectories saved in --run). Capped at the number available.")
args = parser.parse_args()

saved = torch.load(args.run, weights_only=False)
name = saved['distribution']
selected = [m for m in saved['selected'] if args.methods is None or m in args.methods]
if not selected:
    raise ValueError(f"No requested methods found in {args.run} (available: {saved['selected']})")

if args.out_dir is not None:
    run_dir = args.out_dir
    os.makedirs(run_dir, exist_ok=True)
else:
    if not os.path.exists('./runs'):
        os.makedirs('./runs')
    run_dir = create_run_dir('./runs')

target_pmf = saved['target_pmf']
target_x = saved['target_x']
ref_pmfs = saved['ref_pmfs']
hist_samples = saved['hist_samples']
method_info = saved['method_info']
n_epochs = saved['n_epochs']

results = {}
for m in selected:
    r = dict(method_info[m])
    r['x_hist'] = saved[f'x_hist_{m}']
    r['progress'] = saved[f'progress_{m}']
    r['losses'] = saved[f'losses_{m}']
    r['train_epoch'] = saved[f'train_epoch_{m}']
    r['val_epoch'] = saved[f'val_epoch_{m}']
    results[m] = r

N = results[selected[0]]['x_hist'].shape[1]
if args.num_paths is not None:
    N = min(N, args.num_paths)
n_panels = len(selected)

# ============================================================
# Metrics: total-variation and 1-Wasserstein distance between the target
# distribution and the generated samples' final marginal (hist_samples[m]),
# one row per selected process. TV compares the exact target PMF against
# the empirical histogram of generated samples over the same support; W1
# compares generated samples against samples drawn i.i.d. from the exact
# target PMF (so it doesn't need a matched sample count on the target side).
# ============================================================
S = len(target_pmf)
rng = np.random.default_rng(0)
target_samples = rng.choice(target_x, size=100_000, p=target_pmf / target_pmf.sum())

metrics = {}
for m in selected:
    samples = hist_samples[m]
    metrics[m] = {
        'tv': tv_distance(samples, target_pmf, S),
        'w1': w1_distance(samples, target_samples),
    }

metrics_lines = [f"{results[m]['title']:<30s} TV = {metrics[m]['tv']:.4f}   "
                  f"W1 = {metrics[m]['w1']:.4f}" for m in selected]
print("Target-vs-generated metrics:")
print('\n'.join(metrics_lines))
with open(f'{run_dir}/metrics_{name}.txt', 'w') as f:
    f.write('\n'.join(metrics_lines) + '\n')

# ============================================================
# Plot 1: individual trajectories, one panel per selected process. Saved as
# one figure per method.
# ============================================================

# per-distribution (0, ymax) overrides for the target panel's y-axis,
# keyed by dist.__class__.__name__ (= saved['distribution']), used in
# place of the generic max(0.03, 1.2*target_pmf.max()) formula below.
TARGET_YLIM_OVERRIDES = {
    'NegativeBinomialMixture': (0, 0.02),
}

# per-method overrides for the 'Reference' label's position in the
# reference panel (top of Plot 1), keyed by method name (as in `selected`).
# Default is the upper-right corner; some methods' reference PMF is
# concentrated on the right side of the panel, which collides with the
# label there, so move it to the upper-left instead.
REFERENCE_LABEL_POS_OVERRIDES = {
    'poisson_only': dict(x=0.02, ha='left'),
}

# y-range fixed from the target data distribution's support alone (its
# effective range where the PMF is non-negligible), the same for every
# method's panel regardless of that method's reference/trajectory range.
target_support = target_x[target_pmf > 1e-4]
y_pad = 0.05 * (target_support.max() - target_support.min() + 1)
y_lim = (target_support.min() - y_pad, target_support.max() + y_pad)

cmap = plt.get_cmap('viridis', N)

TICK_FONTSIZE = 5
TITLE_FONTSIZE = 7
AXIS_LABEL_FONTSIZE = 6


def _support_range(centers, counts, thresh=1e-4):
    # value-range spanning where the PMF is non-negligible (thresholded, as
    # for target_support below), falling back to the full nonzero support
    # if thresholding would leave it empty (e.g. a support already small
    # enough -- few, low-probability grid points -- that thresholding
    # wipes it out).
    sel = centers[counts > thresh]
    if sel.size == 0:
        sel = centers[counts > 0]
    lo, hi = float(sel.min()), float(sel.max())
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return (lo, hi)


def _draw_target(ax, m, title, xlim, show_xticklabels):
    # exact PMF of the target data distribution, overlaid with a (vertical)
    # histogram of generated samples (hist_samples[m]) to show how the
    # generated marginal lines up against the true density.
    ax.plot(target_x, target_pmf, color='k', lw=1.0, label='True data')
    samples = hist_samples[m]
    # bin width 4 (widened from 2) to further smooth out the empirical
    # histogram's sampling noise.
    x_max = int(max(samples.max(), target_x[target_pmf > 1e-4].max()))
    bins = np.arange(0, x_max + 5, 4) - 0.5
    ax.hist(samples, bins=bins, density=True, color='tab:blue', alpha=0.5, label='Generated')
    ax.set_xlim(*xlim)
    # fixed density-axis range so panels are directly comparable across
    # methods; padded above target_pmf's own max so wider/taller reference
    # PMFs (e.g. PoissonMixture) aren't cut off at the top. Some
    # distributions (TARGET_YLIM_OVERRIDES) have a narrow enough PMF that
    # this auto-padding leaves too much unused headroom, so use a fixed
    # range instead.
    ax.set_ylim(*TARGET_YLIM_OVERRIDES.get(name, (0, max(0.03, 1.2 * target_pmf.max()))))
    ax.set_yticks([])
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
    ax.text(0.98, 0.95, title, transform=ax.transAxes, ha='right', va='top',
            fontsize=TITLE_FONTSIZE)
    ax.text(0.98, 0.50, f"$W_1$ = {metrics[m]['w1']:.2f}", transform=ax.transAxes,
            ha='right', va='top', fontsize=TITLE_FONTSIZE)
    ax.tick_params(labelbottom=show_xticklabels, labelsize=TICK_FONTSIZE)


def draw_path_panel(ref_ax, main_ax, target_ax, m, show_xlabel):
    r = results[m]

    # reference PMF panel (top): exact PMF of x_init, x-limits set to the
    # UNION of the (thresholded) reference support and the target-window
    # y_lim, so both the reference and target distributions fit fully.
    # main_ax and target_ax are sharex'd with ref_ax (set at figure
    # construction), so setting ref_ax's xlim here aligns all three.
    centers, counts = ref_pmfs[m]
    if m == 'binomial':
        # x_init is a point mass at 0 for the binomial-only method; a plain
        # line plot is invisible for a single point, so mark it explicitly.
        ref_ax.plot(centers, counts, color='maroon', marker='o', markersize=3, lw=0)
    else:
        ref_ax.plot(centers, counts, color='maroon', lw=1.0)
    ref_support = _support_range(centers, counts)
    ref_lim = (min(ref_support[0], y_lim[0]), max(ref_support[1], y_lim[1]))
    ref_ax.set_xlim(*ref_lim)
    ref_ax.set_ylim(bottom=0)
    ref_ax.set_yticks([])
    ref_ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4, integer=True))
    label_pos = REFERENCE_LABEL_POS_OVERRIDES.get(m, dict(x=0.98, ha='right'))
    ref_ax.text(label_pos['x'], 0.95, 'Reference', transform=ref_ax.transAxes,
                ha=label_pos['ha'], va='top', fontsize=TITLE_FONTSIZE, color='maroon')
    ref_ax.tick_params(labelbottom=False, labelsize=TICK_FONTSIZE)

    # trajectory panel (middle), rotated relative to the old layout: state
    # value x on the (shared) x-axis, progress on the y-axis, inverted so it
    # stacks visually between the reference panel (top, progress=0) and the
    # target panel (bottom, progress=1).
    for i in range(N):
        main_ax.plot(r['x_hist'][:, i], r['progress'], color=cmap(i), alpha=0.6, lw=1.0)
    if args.show_titles:
        main_ax.set_title(r['title'])
    main_ax.set_ylabel('Time $t$', fontsize=AXIS_LABEL_FONTSIZE, labelpad=1)
    main_ax.set_ylim(1.0, 0.0)
    main_ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    main_ax.tick_params(labelbottom=False, labelsize=TICK_FONTSIZE)

    # target panel (bottom), on the same x-scale as the reference/trajectory
    # panels (ref_lim); its x tick labels are the ones shown, since it's the
    # bottom-most panel.
    _draw_target(target_ax, m, 'Target', ref_lim, show_xticklabels=True)
    if show_xlabel:
        target_ax.set_xlabel('$x$', fontsize=AXIS_LABEL_FONTSIZE, labelpad=1)


# one figure per method
for m in selected:
    fig = plt.figure(figsize=(2.4, 2.3))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.3, 1.0, 0.3], hspace=0.06)
    ref_ax = fig.add_subplot(gs[0, 0])
    main_ax = fig.add_subplot(gs[1, 0], sharex=ref_ax)
    target_ax = fig.add_subplot(gs[2, 0], sharex=ref_ax)
    draw_path_panel(ref_ax, main_ax, target_ax, m, show_xlabel=True)
    plt.tight_layout(pad=0.02)
    plt.savefig(f'{run_dir}/paths_{name}_{m}.pdf', bbox_inches='tight', pad_inches=0.01)
    plt.close()

# ============================================================
# Plot 2: train vs. validation loss, one panel per selected process, to
# diagnose over/underfitting. Saved both as one combined figure and as one
# figure per method.
# ============================================================
epochs = np.arange(1, n_epochs + 1)


def plot_loss(ax, train_epoch, val_epoch, title):
    ax.semilogy(epochs, train_epoch, label='Train', color='tab:blue')
    ax.semilogy(epochs, val_epoch, label='Validation', color='tab:orange')
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.legend()


fig, axes = plt.subplots(1, n_panels, figsize=(5.4 * n_panels, 5), squeeze=False)
axes = axes[0]
for ax, m in zip(axes, selected):
    r = results[m]
    plot_loss(ax, r['train_epoch'], r['val_epoch'], r['loss_title'])
axes[0].set_ylabel('Denoising loss')

plt.tight_layout(pad=0.2)
plt.savefig(f'{run_dir}/loss_{name}.pdf', bbox_inches='tight', pad_inches=0.02)
plt.close()

for m in selected:
    r = results[m]
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    plot_loss(ax, r['train_epoch'], r['val_epoch'], r['loss_title'])
    ax.set_ylabel('Denoising loss')
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{run_dir}/loss_{name}_{m}.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close()

# ============================================================
# Plot 3: histograms of final generated samples vs. true data samples,
# in the same 1xN layout as Plot 1, to compare marginal fit across
# processes. Saved both as one combined figure and as one figure per method.
# ============================================================
def plot_hist(ax, x_hist, title, m, ymax=None):
    final = x_hist[-1, :]
    x_max = int(max(final.max(), target_x[target_pmf > 1e-4].max()))
    ax.bar(target_x, target_pmf, width=1.0, alpha=0.5, color='k', label='True data (exact PMF)')
    # bin width 4 (widened from 2) to further smooth out the empirical
    # histogram's sampling noise.
    bins = np.arange(0, x_max + 5, 4) - 0.5
    ax.hist(final, bins=bins, density=True, alpha=0.5, color='tab:blue', label='Generated')
    ax.set_xlim(-1, x_max + 1)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.set_title(f"{title}\nTV = {metrics[m]['tv']:.4f}, $W_1$ = {metrics[m]['w1']:.2f}")
    ax.set_xlabel('x')
    ax.legend(fontsize=8)


fig, axes = plt.subplots(1, n_panels, figsize=(5.4 * n_panels, 5), sharey=True, squeeze=False)
axes = axes[0]
for ax, m in zip(axes, selected):
    r = results[m]
    plot_hist(ax, r['x_hist'], r['title'], m)
axes[0].set_ylabel('Density')

plt.tight_layout(pad=0.2)
plt.savefig(f'{run_dir}/hist_{name}.pdf', bbox_inches='tight', pad_inches=0.02)
plt.close()

for m in selected:
    r = results[m]
    fig, ax = plt.subplots(figsize=(5.4, 2.5))
    plot_hist(ax, r['x_hist'], r['title'], m, ymax=0.1)
    ax.set_ylabel('Density')
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{run_dir}/hist_{name}_{m}.pdf', bbox_inches='tight', pad_inches=0.02)
    plt.close()

print(f"Saved combined plots to {run_dir}/loss_{name}.pdf, {run_dir}/hist_{name}.pdf")
print(f"Saved per-method plots to {run_dir}/{{paths,loss,hist}}_{name}_<method>.pdf for: {', '.join(selected)}")
print(f"Saved target-vs-generated metrics to {run_dir}/metrics_{name}.txt")
