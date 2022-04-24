# Greg Attra
# 04.23.22

'''
SAC Networks
'''

from turtle import forward
from torch import device, nn
import torch
from torch.distributions.normal import Normal
from lib.networks.nn import init_weights


class CriticNetwork(nn.Module):
    def __init__(self, obs_size: int, action_size: int, dim_size: int, lr: float) -> None:
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.network = nn.Sequential(
            nn.Linear(obs_size + action_size, dim_size),
            nn.ReLU(),
            nn.Linear(dim_size, dim_size),
            nn.ReLU(),
            nn.Linear(dim_size, 1)
        )

        # self.network.apply(init_weights)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.network(x)


class ValueNetwork(nn.Module):
    def __init__(self, obs_size: int, dim_size: int, lr: float):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.network = nn.Sequential(
            nn.Linear(obs_size, dim_size),
            nn.ReLU(),
            nn.Linear(dim_size, dim_size),
            nn.ReLU(),
            nn.Linear(dim_size, 1)
        )
        
        # self.network.apply(init_weights)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return self.network(x)


class ActorNetwork(nn.Module):
    def __init__(self, obs_size: int, action_size: int, dim_size: int, action_scale: float, lr: float):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.network = nn.Sequential(
            nn.Linear(obs_size, dim_size),
            nn.ReLU(),
            nn.Linear(dim_size, dim_size),
            nn.ReLU(),
            nn.Linear(dim_size, action_size * 2)
        )
        
        # self.network.apply(init_weights)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.action_scale = action_scale
        self.action_size = action_size

    def forward(self, x):
        action_dists = self.network(x).view(-1, self.action_size, 2)
        mu = action_dists[:, :, 0]
        sigma = action_dists[:, :, 1]
        sigma = torch.clamp(sigma, min=1e-6, max=1.)
        
        return mu, sigma

    def sample(self, x, reparam: bool = False):
        mu, sigma = self.forward(x)
        probs = Normal(mu, sigma)

        if reparam:
            actions = probs.rsample()
        else:
            actions = probs.sample()
        
        action = torch.tanh(actions)
        log_probs = probs.log_prob(action)

        return actions, log_probs
