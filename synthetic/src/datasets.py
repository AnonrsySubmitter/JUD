from cmath import pi
import torch
import torch.nn.functional as F
from scipy.special import beta

class UnivariateDiscreteDistribution:
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.dim    = 1

    def probs(self):
        return torch.exp(self.log_probs())

    def log_likelihood(self, x):
        assert(x.shape[1] == self.dim)
        return self.log_probs()[x].sum(dim=-1)

    def sample(self, N):
        return torch.multinomial(self.probs().T, N, replacement=True).T


class Poisson(UnivariateDiscreteDistribution):
    def __init__(self, lam, S, device):
        super().__init__(device)
        self.lam    = torch.tensor([lam], dtype=torch.float32, device=device) # Poisson parameter
        self.S      = S
        assert(len(self.lam) == 1)

    def log_probs(self):
        # define un-normalized log-probabilities
        x_all = torch.arange(self.S, dtype=torch.float32, device=self.device)[:,None]
        log_probs_unnormalized = -self.lam + x_all * torch.log(self.lam) - torch.lgamma(x_all + 1)
        return F.log_softmax(log_probs_unnormalized, dim=0)


class PoissonMixture(UnivariateDiscreteDistribution):
    """ Define bimodal mixture of Poisson distributions with support [0,...,S-1] """ 
    def __init__(self, device):
        super().__init__(device)
        self.w      = torch.tensor([0.1, 0.9], dtype=torch.float32, device=device) # mixture weights
        self.lam    = torch.tensor([1.0, 100], dtype=torch.float32, device=device) # Poisson parameters
        self.S      = 140
        assert(len(self.w) == len(self.lam))

    def log_probs_component(self):
        # define un-normalized log-probabilities
        x_all = torch.arange(self.S, dtype=torch.float32, device=self.device)[:,None]
        log_probs_unnormalized = -self.lam + x_all * torch.log(self.lam) - torch.lgamma(x_all + 1)
        # normalize log-probabilities
        return F.log_softmax(log_probs_unnormalized, dim=0)

    def probs_component(self):
        return torch.exp(self.log_probs_component())

    def log_probs(self):
        log_probs_component_evals = self.log_probs_component()
        return torch.logsumexp(log_probs_component_evals + torch.log(self.w), dim=1)

    def sample(self, N):
        # sample mixture components
        components = torch.multinomial(self.w, num_samples=N, replacement=True)
        # compute poisson probabilities for each component
        poisson_probs_component = self.probs_component()
        # sample each component and extra samples to avoid re-sampling
        samples_component =  torch.multinomial(poisson_probs_component.T, N, replacement=True).T
        return samples_component[torch.arange(N), components].unsqueeze(-1)


class ZeroInflatedPoisson(UnivariateDiscreteDistribution):
    """ Define zero-inflated Poisson distribution with parameter lambda > 0 and support [0,...,S-1] """ 
    def __init__(self, device):
        super().__init__(device)
        self.pi0    = torch.tensor([0.7], dtype=torch.float32, device=device)      # zero-inflation parameter
        self.lam    = torch.tensor([5.0], dtype=torch.float32, device=device)      # rate parameter
        self.S      = 50

    def log_probs(self):
        x = torch.arange(self.S, device=self.device).float()[:, None]
        log_p = (
            torch.log1p(-self.pi0)
            - self.lam
            + x * torch.log(self.lam)
            - torch.lgamma(x + 1)
        )
        # stable computation for zero mass
        log_p[0] = torch.logaddexp(
            torch.log(self.pi0),
            torch.log1p(-self.pi0) - self.lam
        )
        return F.log_softmax(log_p, dim=0)


class NegativeBinomialMixture(UnivariateDiscreteDistribution):
    """ Define bimodal mixture of Negative Binomial distributions with support [0,...,S-1] 
    """ 
    def __init__(self, device):
        super().__init__(device)
        self.dim    = 1                                              # dimension
        self.w      = torch.tensor([0.8, 0.2], dtype=torch.float32, device=device)  # mixture weights
        self.r      = torch.tensor([1.0, 10.0], dtype=torch.float32, device=device) # Negative Binomial r parameters
        self.p      = torch.tensor([0.9, 0.1], dtype=torch.float32, device=device)  # Negative Binomial p parameters
        self.S      = 150
        assert(len(self.w) == len(self.r) == len(self.p))

    def negative_binomial_log_probs(self, r, p):
        """
        Negative Binomial PMF

        k : tensor of non-negative counts
        r : shape parameter (>0)
        p : success probability in (0,1)
        """
        x_all = torch.arange(self.S, dtype=torch.float32, device=self.device)[:,None]
        r = torch.as_tensor(r, dtype=torch.float64, device=self.device)
        p = torch.as_tensor(p, dtype=torch.float64, device=self.device)
        log_pmf = (
            torch.lgamma(x_all + r)
            - torch.lgamma(x_all + 1)
            - torch.lgamma(r)
            + x_all * torch.log1p(-p)
            + r * torch.log(p)
        )
        return log_pmf

    def log_probs_component(self):
        # define un-normalized log-probabilities
        log_probs_unnormalized = []
        for i in range(len(self.w)):
            log_probs_unnormalized.append(self.negative_binomial_log_probs(self.r[i], self.p[i]))
        log_probs_unnormalized = torch.cat(log_probs_unnormalized, dim=1)
        # normalize log-probabilities
        return F.log_softmax(log_probs_unnormalized, dim=0)

    def probs_component(self):
        return torch.exp(self.log_probs_component())

    def log_probs(self):
        log_probs_component_evals = self.log_probs_component()
        return torch.logsumexp(log_probs_component_evals + torch.log(self.w), dim=1)

    def sample(self, N):
        # sample mixture components
        components = torch.multinomial(self.w, num_samples=N, replacement=True)
        # sample from corresponding Negative Binomial distributions
        negbinom_probs_component = self.probs_component()
        # sample each component and extra samples to avoid re-sampling
        samples_component =  torch.multinomial(negbinom_probs_component.T, N, replacement=True).T
        return samples_component[torch.arange(N), components].unsqueeze(-1)


class NegativeBinomialMixtureBalanced(NegativeBinomialMixture):
    """ Same bimodal Negative Binomial mixture as NegativeBinomialMixture, but
    with weights rebalanced so the spike (r=1, p=0.9) and the bulk (r=10,
    p=0.1) reach roughly comparable peak PMF heights on a shared linear axis
    (spike peak slightly taller). The spike concentrates almost all of its
    mass in a single bin (peak density ~0.9) while the bulk spreads its mass
    over ~90 bins (peak density ~0.014), so making both visible requires
    w_spike << w_bulk despite the spike otherwise dominating visually under
    equal weights.
    """
    def __init__(self, device):
        super().__init__(device)
        self.w = torch.tensor([0.03, 0.97], dtype=torch.float32)


class BetaNegativeBinomial(UnivariateDiscreteDistribution):
    """ Define Beta-Negative-Binomial distribution using numerical integration
    """ 
    def __init__(self, device):
        super().__init__(device)
        self.dim    = 1        # dimension
        self.S      = 100
        self.r      = 5        # number of failures of negative binomial
        self.a      = 1.5      # Beta distribution parameters
        self.b      = 1.5      # Beta distribution parameters
        self.n_points = 1000 # number of integration points
        self.probs_eval = self.probs()
    
    def probs(self):
        """ Evaluate PDF/PMF of Beta-Negative-Binomial distribution
        Compute the PDF/PMF of the Beta-Negative-Binomial distribution
        using numerical integration.        
        Returns:
            pmf: tensor of PMF values at k
        """
        x_all = torch.arange(self.S, dtype=torch.float32, device=self.device)[:,None]
        r = torch.tensor(self.r, device=self.device)

        # Integration points
        p = torch.linspace(0.0, 1.0, self.n_points, device=self.device)[1:-1]  # avoid 0 and 1
        dp = p[1] - p[0]

        # Beta PDF: p^(a-1) * (1-p)^(b-1) / B(a,b)
        log_beta_const = torch.lgamma(torch.tensor(self.a, device=self.device)) + torch.lgamma(torch.tensor(self.b, device=self.device)) - torch.lgamma(torch.tensor(self.a+self.b, device=self.device))
        beta_pdf = torch.exp((self.a-1)*torch.log(p) + (self.b-1)*torch.log(1-p) - log_beta_const)

        pmf = []
        for ki in x_all:
            # NB PMF: comb(k+r-1, k) * p^r * (1-p)^k
            
            log_nb_coeff = torch.lgamma(ki + r) - torch.lgamma(r) - torch.lgamma(ki + 1)
            nb_pmf = torch.exp(log_nb_coeff + r*torch.log(p) + ki*torch.log(1-p))
            
            # Integrate NB * Beta over p
            pmf_val = torch.sum(nb_pmf * beta_pdf) * dp
            pmf.append(pmf_val)
        
        return torch.tensor(pmf, device=self.device)[:,None]

    def log_probs(self):
        if not hasattr(self, 'probs_eval'):
            self.probs_eval = self.probs()
        return torch.log(self.probs_eval)

    def sample(self, N):
        if not hasattr(self, 'probs_eval'):
            self.probs_eval = self.probs()
        return torch.multinomial(self.probs_eval.T, N, replacement=True).T


class ZipfDistribution(UnivariateDiscreteDistribution):
    def __init__(self, device):
        super().__init__(device)
        self.alpha = torch.tensor([1.7], dtype=torch.float32, device=device)     # Zipf parameter
        self.zeta_alpha = torch.special.zeta(                     # Riemann zeta function ζ(alpha)
            torch.tensor(self.alpha, dtype=torch.float64, device=device),
            torch.tensor(1.0, dtype=torch.float64, device=device)
        )
        self.S = 50                                           # support

    def log_probs(self):
        # define un-normalized log-probabilities
        x_all = torch.arange(1, self.S+1, dtype=torch.float32, device=self.device)[:,None]
        log_probs_unnormalized = - self.alpha * torch.log(x_all) - torch.log(self.zeta_alpha)
        # normalize log-probabilities
        return F.log_softmax(log_probs_unnormalized, dim=0)


class YuleSimonDistribution(UnivariateDiscreteDistribution):
    """ Define Yule-Simon distribution with parameter \rho > 0 and support [1,...,S] """ 
    def __init__(self, device):
        super().__init__(device)
        self.device = device
        self.rho    = torch.tensor([2.0], dtype=torch.float32, device=device)     # Yule-Simon parameter
        self.S      = 50                                           # support

    def log_probs(self):
        # define un-normalized log-probabilities at all x
        x_all = torch.arange(1, self.S+1, dtype=torch.float32, device=self.device)[:,None]
        #probs_unnormalized = self.rho * beta(x_all, self.rho + 1)
        logprobs_unnormalized = torch.log(self.rho) + torch.lgamma(x_all) + torch.lgamma(self.rho + 1) - torch.lgamma(x_all + self.rho + 1)
        # normalize log-probabilities
        return F.log_softmax(logprobs_unnormalized, dim=0)


if __name__=='__main__':

    # generate samples
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sample and compute probabilities
    dataset = PoissonMixture(device)
    samples = dataset.sample(10000)
    probs_overall = dataset.probs()
    print(probs_overall.shape)

    import matplotlib.pyplot as plt
    plt.hist(samples.numpy(), bins=range(0,dataset.S), density=True)
    plt.plot(probs_overall.numpy(), 'r-')
    plt.show()
