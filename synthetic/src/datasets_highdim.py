import math
import torch

# Higher-dimensional / 2-D synthetic target distributions, reproducing
# Appendix D of Fishman et al. (2026) "Count Bridges"
# (https://arxiv.org/pdf/2603.04730). Each class exposes .sample(N) ->
# (N, dim) integer-valued float tensor, .dim, and .S (state-space size
# per coordinate).


def _integerize_clip(x: torch.Tensor, scale: float, offset: float,
                      min_value: int, value_range: int) -> torch.Tensor:
    """ x -> round(clip(x*scale + offset, min_value, min_value + value_range - 1)) (Appendix D.1.1). """
    lo = float(min_value)
    hi = float(min_value + value_range - 1)
    return torch.round((x * scale + offset).clamp(lo, hi))


def _integerize_reflect(x: torch.Tensor, min_value: int, value_range: int) -> torch.Tensor:
    """
    x -> round(x), reflected (not clipped) into [min_value, min_value + value_range - 1]
    (Appendix D.2.1: "rounded to the nearest integer and reflected into the
    bounded range"). Reflection avoids the mass pile-up at the boundary that
    clipping would cause for a mixture with real tails.
    """
    lo = float(min_value)
    hi = float(min_value + value_range - 1)
    span = hi - lo
    period = 2.0 * span
    y = torch.remainder(torch.round(x) - lo, period)
    y = torch.where(y > span, period - y, y)
    return lo + y


def _folded_gaussian_mixture_probs_1d(means: torch.Tensor, stds: torch.Tensor, weights: torch.Tensor,
                                       min_value: int, value_range: int, n_periods: int = 6) -> torch.Tensor:
    """
    Exact (up to Gaussian-tail truncation) pmf of round(y) reflected into
    [min_value, min_value+value_range-1] for a 1-D Gaussian-mixture y, i.e.
    the analytic marginal for _integerize_reflect applied to a scalar
    Gaussian mixture (means/stds/weights: (k,) tensors, one component each).

    round() assigns each integer m the continuum mass over [m-0.5, m+0.5);
    reflection then folds every integer m (not just those in range) into the
    box via the same "mod period, fold at span" map _integerize_reflect uses
    on continuous inputs. So p(j) for j in [0, value_range) is the sum, over
    every integer m with reflect(m) == j, of the mixture's mass in
    [m-0.5, m+0.5). Since the mixture is Gaussian its tails decay
    super-exponentially, so summing `n_periods` reflected copies each side of
    the box (n_periods=6 -> +/-6 full folds) is enough for float32 precision
    for any of this module's parameter scales.
    """
    lo = float(min_value)
    span = float(value_range - 1)
    period = 2.0 * span
    j = torch.arange(value_range, dtype=torch.float32, device=means.device)  # (S,)

    # every integer m (relative to lo) that reflects onto j, within n_periods folds each side
    fold = torch.arange(-n_periods, n_periods + 1, dtype=torch.float32, device=means.device)  # (F,)
    m_same = fold[:, None] * period + j[None, :]          # unreflected copy: reflect(m_same) == j
    m_mirr = fold[:, None] * period - j[None, :]           # mirrored copy:    reflect(m_mirr) == j
    m_all = torch.cat([m_same, m_mirr], dim=0) + lo         # (2F, S), back to absolute coordinates

    # mass each component puts in [m-0.5, m+0.5) for every candidate integer m
    z_hi = (m_all[None, :, :] + 0.5 - means[:, None, None]) / stds[:, None, None]  # (k, 2F, S)
    z_lo = (m_all[None, :, :] - 0.5 - means[:, None, None]) / stds[:, None, None]
    normal = torch.distributions.Normal(0.0, 1.0)
    comp_mass = (normal.cdf(z_hi) - normal.cdf(z_lo)).sum(dim=1)  # (k, S), summed over folds

    return (weights[:, None] * comp_mass).sum(dim=0)  # (S,)


class DiscreteEightGaussians:
    """
    Appendix D.1.1 "target distribution": an 8-component Gaussian mixture
    arranged evenly on a circle of radius `radius`, isotropic noise variance
    `noise`, one component chosen uniformly at random per sample, then
    integerized via _integerize_clip with the paper's default parameters
    (scale=30.0, offset=80.0, min_value=0, value_range=196), giving values
    in {0,...,195}^2.
    """
    def __init__(self, radius: float = 2.0, noise: float = 0.1,
                 scale: float = 30.0, offset: float = 80.0,
                 min_value: int = 0, value_range: int = 196, device=None):
        self.dim = 2
        self.S = value_range
        self.radius = radius
        self.noise = noise
        self.scale = scale
        self.offset = offset
        self.min_value = min_value
        self.value_range = value_range
        self.device = device

    def sample(self, N: int) -> torch.Tensor:
        k = torch.randint(0, 8, (N,), device=self.device)
        theta = k.float() * (2.0 * math.pi / 8.0)
        centers = self.radius * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)
        x = centers + math.sqrt(self.noise) * torch.randn(N, 2, device=self.device)
        return _integerize_clip(x, self.scale, self.offset, self.min_value, self.value_range)


class DiscreteTwoMoons:
    """
    Appendix D.1.1 "source distribution": the standard two-moons dataset
    (matching sklearn.datasets.make_moons's construction, reimplemented
    here directly to avoid an extra dependency), noise=0.1, shifted to be
    approximately centered at the origin by subtracting (0.5, 0.25), then
    integerized via _integerize_clip with the paper's default parameters
    (scale=30.0, offset=80.0, min_value=0, value_range=196), giving values
    in {0,...,195}^2.
    """
    def __init__(self, noise: float = 0.1, scale: float = 30.0, offset: float = 80.0,
                 min_value: int = 0, value_range: int = 196, device=None):
        self.dim = 2
        self.S = value_range
        self.noise = noise
        self.scale = scale
        self.offset = offset
        self.min_value = min_value
        self.value_range = value_range
        self.device = device

    def sample(self, N: int) -> torch.Tensor:
        n_out = N // 2
        n_in = N - n_out

        t_out = torch.linspace(0.0, math.pi, n_out, device=self.device)
        outer = torch.stack([torch.cos(t_out), torch.sin(t_out)], dim=-1)

        t_in = torch.linspace(0.0, math.pi, n_in, device=self.device)
        inner = torch.stack([1.0 - torch.cos(t_in), 1.0 - torch.sin(t_in) - 0.5], dim=-1)

        x = torch.cat([outer, inner], dim=0)
        # sklearn.datasets.make_moons applies `noise` directly as a standard
        # deviation (X += generator.normal(scale=noise, ...)), unlike the
        # 8-Gaussians component noise above (which the paper specifies as a
        # variance) -- do not sqrt() it here.
        x = x + self.noise * torch.randn(N, 2, device=self.device)
        x = x - torch.tensor([0.5, 0.25], device=self.device)  # re-center near the origin
        return _integerize_clip(x, self.scale, self.offset, self.min_value, self.value_range)


class LowRankGaussianMixture:
    """
    Appendix D.2.1: a k-component Gaussian mixture in a low-dimensional
    latent space R^r (r fixed, e.g. r=3), projected into the d-dimensional
    ambient space via a fixed random projection plus isotropic noise, then
    integerized by rounding and reflecting into [0, S-1]. Intrinsic
    complexity (r, k) is held fixed as `dim` (d) is varied, which is the
    paper's mechanism for studying how sample quality/denoiser error scale
    with ambient dimension (see scaling_dimension.py).

    Mixture and projection parameters are sampled once at construction
    (fixed per instance); every .sample(N) call draws from that same fixed
    mixture, matching "we hold these parameters constant as we scale in d".

    One implementation choice not pinned down precisely by the paper text:
    since the latent means are drawn as N(0, sigma^2 I_r) (not shifted) and
    the ambient projection P z + eps is then also ~0-mean, "shifted to lie
    near the center of the integer range" is applied here as an additive
    offset of S/2 on the final ambient-space output, before rounding.
    """
    def __init__(self, dim: int, r: int = 3, k: int = 5,
                 mean_scale: float = 20.0, cov_scale: float = 10.0, min_eigenvalue: float = 0.1,
                 projection_scale: float = 1.0, noise_scale: float = 1.0,
                 min_value: int = 0, value_range: int = 256, device=None, generator=None):
        self.dim = dim
        self.r = r
        self.k = k
        self.S = value_range
        self.min_value = min_value
        self.value_range = value_range
        self.noise_scale = noise_scale
        self.device = device

        g = generator

        # component means: N(0, sigma^2 I_r), sigma = mean_scale / sqrt(r)
        sigma = mean_scale / math.sqrt(r)
        self.means = sigma * torch.randn(k, r, device=device, generator=g)

        # component covariances: eigenvalues ~ Exponential(cov_scale), clamped
        # below min_eigenvalue, conjugated by a random orthogonal matrix
        eigvals = -cov_scale * torch.log(torch.rand(k, r, device=device, generator=g).clamp(min=1e-12))
        eigvals = eigvals.clamp(min=min_eigenvalue)
        covs = torch.zeros(k, r, r, device=device)
        for i in range(k):
            Q, _ = torch.linalg.qr(torch.randn(r, r, device=device, generator=g))
            covs[i] = Q @ torch.diag(eigvals[i]) @ Q.T
        self.chol = torch.linalg.cholesky(covs)  # (k, r, r)

        # mixture weights: Dirichlet(1,...,1)
        self.weights = torch.distributions.Dirichlet(torch.ones(k, device=device)).sample()

        # random projection R^r -> R^d, entries scaled by projection_scale/sqrt(r)
        self.P = (projection_scale / math.sqrt(r)) * torch.randn(dim, r, device=device, generator=g)

        # ambient-space centering offset (see docstring)
        self.center_offset = value_range / 2.0

    def sample(self, N: int) -> torch.Tensor:
        comp = torch.multinomial(self.weights, N, replacement=True)  # (N,)
        eps_latent = torch.randn(N, self.r, 1, device=self.device)
        z = self.means[comp] + (self.chol[comp] @ eps_latent).squeeze(-1)  # (N, r)
        eps_ambient = self.noise_scale * torch.randn(N, self.dim, device=self.device)
        y = z @ self.P.T + eps_ambient + self.center_offset  # (N, dim)
        return _integerize_reflect(y, self.min_value, self.value_range)

    def probs(self) -> torch.Tensor:
        """
        Exact 1-D marginal pmf (dim=1 only): P z + eps is itself a k-component
        Gaussian mixture (a linear map / independent-noise sum of Gaussians is
        Gaussian), with component i's scalar mean/var given by projecting
        means[i]/covs[i] through the (1, r) row P and adding noise_scale**2;
        _folded_gaussian_mixture_probs_1d then gives the exact pmf of that
        mixture's round-and-reflect integerization.
        """
        assert self.dim == 1, "probs() is only implemented for dim=1"
        p_row = self.P[0]                                          # (r,)
        means_1d = (self.means @ p_row)                            # (k,) = P @ means[i] per component
        covs = self.chol @ self.chol.transpose(-1, -2)              # (k, r, r)
        vars_1d = torch.einsum('i,kij,j->k', p_row, covs, p_row) + self.noise_scale ** 2
        stds_1d = vars_1d.clamp(min=1e-12).sqrt()
        return _folded_gaussian_mixture_probs_1d(
            means_1d + self.center_offset, stds_1d, self.weights,
            self.min_value, self.value_range)


class CorrelatedBetaBinomial:
    """Exchangeable, non-Gaussian count target in arbitrary dimension.

    For each observation, draw a shared latent probability

        p ~ Beta(alpha, beta),    X_j | p ~ Binomial(num_trials, p).

    Conditional independence given ``p`` creates positive dependence between
    every pair of coordinates. Marginally, each coordinate is beta-binomial,
    hence skewed and overdispersed relative to a binomial distribution. By
    default samples are returned in a centered native-integer frame; adding
    ``center`` maps them exactly back to the bounded support
    ``{0, ..., num_trials}^d`` (pass ``centered=False`` to get that native,
    uncentered frame directly -- e.g. for the N_0-only forward processes in
    compare_noise_types.py).
    """

    def __init__(self, dim: int, num_trials: int = 64, alpha: float = 2.0,
                 beta: float = 5.0, centered: bool = True, device=None):
        self.dim = dim
        self.num_trials = num_trials
        self.alpha = alpha
        self.beta = beta
        self.centered = centered
        self.device = device
        self.S = num_trials + 1
        # An integer shift preserves the lattice while centering the native-Z
        # target as closely as possible to its analytic marginal mean.
        self.center = int(round(num_trials * alpha / (alpha + beta)))

    def sample(self, N: int) -> torch.Tensor:
        concentration = torch.tensor([self.alpha, self.beta], device=self.device)
        p = torch.distributions.Dirichlet(concentration).sample((N,))[:, :1]
        probs = p.expand(N, self.dim)
        counts = torch.distributions.Binomial(
            total_count=float(self.num_trials), probs=probs).sample()
        return counts - self.center if self.centered else counts

    def probs(self) -> torch.Tensor:
        """
        Exact 1-D marginal pmf (dim=1 only), in the native uncentered frame
        {0,...,num_trials} regardless of `self.centered`: the standard
        Beta-Binomial pmf P(X=k) = C(n,k) * B(k+alpha, n-k+beta) / B(alpha,beta).
        """
        assert self.dim == 1, "probs() is only implemented for dim=1"
        n = float(self.num_trials)
        k = torch.arange(self.num_trials + 1, dtype=torch.float32, device=self.device)
        alpha = torch.tensor(self.alpha, dtype=torch.float32, device=self.device)
        beta_ = torch.tensor(self.beta, dtype=torch.float32, device=self.device)
        log_binom = (torch.lgamma(torch.tensor(n + 1.0, device=self.device))
                     - torch.lgamma(k + 1) - torch.lgamma(n - k + 1))
        log_beta_num = torch.lgamma(k + alpha) + torch.lgamma(n - k + beta_) - torch.lgamma(n + alpha + beta_)
        log_beta_den = torch.lgamma(alpha) + torch.lgamma(beta_) - torch.lgamma(alpha + beta_)
        return torch.exp(log_binom + log_beta_num - log_beta_den)


if __name__ == "__main__":
    torch.manual_seed(0)
    eight_g = DiscreteEightGaussians()
    two_moons = DiscreteTwoMoons()
    lr_gmm = LowRankGaussianMixture(dim=16)
    beta_binomial = CorrelatedBetaBinomial(dim=16)

    for name, dist in [("DiscreteEightGaussians", eight_g),
                        ("DiscreteTwoMoons", two_moons),
                        ("LowRankGaussianMixture(dim=16)", lr_gmm),
                        ("CorrelatedBetaBinomial(dim=16)", beta_binomial)]:
        x = dist.sample(2000)
        print(f"{name:35s} shape={tuple(x.shape)}  min={x.min().item():.0f}  max={x.max().item():.0f}  "
              f"mean={x.float().mean().item():.2f}")
