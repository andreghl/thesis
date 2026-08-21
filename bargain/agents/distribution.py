import torch as th
from stable_baselines3.common.distributions import Distribution
from torch.distributions import Bernoulli, Dirichlet
import torch.nn.functional as f
import torch.nn as nn
import torch

class CustomDistribution(Distribution):

    def log_prob_from_params(self, *args, **kwargs) -> tuple[th.Tensor, th.Tensor]:
        pass

    def actions_from_params(self, *args, **kwargs) -> th.Tensor:
        pass

    def __init__(self, n_vehicles: int, eps: float = 1e-6):
        super().__init__()
        self.n_vehicles = n_vehicles
        self.action_dim = 2 * n_vehicles + 1
        self.eps = eps

        self.dist_coalition = None
        self.dist_payoff = None
        self.dist_response = None

        self.mask0 = None
        self.mask1 = None
        self.mask2 = None

    def proba_distribution_net(self, latent_dim: int):
        return nn.Linear(latent_dim, self.action_dim)

    def proba_distribution(self, logits: torch.Tensor, role: torch.Tensor):
        n = self.n_vehicles
        coalition_logits = logits[:, :n]
        payoff_logits = logits[:, n:2 * n]
        response_logits = logits[:, 2 * n:2 * n + 1]

        self.dist_coalition = Bernoulli(logits=coalition_logits)
        self.dist_payoff = Dirichlet(f.softplus(payoff_logits) + self.eps)
        self.dist_response = Bernoulli(logits=response_logits)

        self.mask0 = (role == 0).float()
        self.mask1 = (role == 1).float()
        self.mask2 = (role == 2).float()
        return self

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        n = self.n_vehicles
        a_coalition = actions[:, :n]
        a_payoff = actions[:, n:2 * n]
        a_response = actions[:, 2 * n:2 * n + 1]

        lp_coalition = self.dist_coalition.log_prob(a_coalition).sum(dim=-1)
        lp_response = self.dist_response.log_prob(a_response).sum(dim=-1)

        # Dirichlet log_prob blows up (NaN/-inf) on non-simplex input, e.g. rows
        # where payoff wasn't the active head and the buffer stored zeros/garbage.
        # Compute it, then zero out exactly those entries before combining.

        lp_payoff = self.dist_payoff.log_prob(a_payoff)
        lp_payoff = torch.nan_to_num(lp_payoff, nan=0.0, neginf=0.0, posinf=0.0)

        return (self.mask0 * lp_coalition
                + self.mask1 * lp_payoff
                + self.mask2 * lp_response)

    def entropy(self) -> torch.Tensor:
        ent_coalition = self.dist_coalition.entropy().sum(dim=-1)
        ent_payoff = self.dist_payoff.entropy()
        ent_response = self.dist_response.entropy().sum(dim=-1)
        return (self.mask0 * ent_coalition
                + self.mask1 * ent_payoff
                + self.mask2 * ent_response)

    def sample(self) -> torch.Tensor:
        n = self.n_vehicles
        s_coalition = self.dist_coalition.sample()   # (batch, n)
        s_payoff = self.dist_payoff.sample()          # (batch, n)
        s_response = self.dist_response.sample()      # (batch, 1)

        default_payoff = torch.full(
            (s_coalition.shape[0], n),
            1.0 / n,
            device=s_coalition.device,
            dtype=s_coalition.dtype)

        out = torch.zeros(s_coalition.shape[0], self.action_dim, device=s_coalition.device)
        m0, m1, m2 = self.mask0.unsqueeze(-1), self.mask1.unsqueeze(-1), self.mask2.unsqueeze(-1)
        out[:, :n] = m0 * s_coalition
        out[:, n:2 * n] = m1 * s_payoff + (1 - m1) * default_payoff
        out[:, 2 * n:2 * n + 1] = m2 * s_response

        print(out)
        return out

    def mode(self) -> torch.Tensor:
        n = self.n_vehicles
        m_coalition = (self.dist_coalition.probs > 0.5).float()
        m_payoff = self.dist_payoff.concentration / self.dist_payoff.concentration.sum(dim=-1, keepdim=True)
        m_response = (self.dist_response.probs > 0.5).float()

        default_payoff = torch.full(
            (m_coalition.shape[0], n),
            1.0 / n,
            device = m_coalition.device,
            dtype = m_coalition.dtype)

        out = torch.zeros(m_coalition.shape[0], self.action_dim, device=m_coalition.device)
        m0, m1, m2 = self.mask0.unsqueeze(-1), self.mask1.unsqueeze(-1), self.mask2.unsqueeze(-1)
        out[:, :n] = m0 * m_coalition
        out[:, n:2 * n] = m1 * m_payoff + (1 - m1) * default_payoff
        out[:, 2 * n:2 * n + 1] = m2 * m_response
        return out