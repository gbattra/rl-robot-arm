# Greg Attra
# 04.11.22

"""
Wrapper classes / utils for nn's
"""

from torch import nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, obs_size: int, action_size: int, dim_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, dim_size),
            nn.ReLU(),
            nn.Linear(dim_size, dim_size),
            nn.ReLU(),
            nn.Linear(dim_size, action_size)
        )

    def forward(self, x):
        return self.network(x)
