# Greg Attra
# 04.23.22

'''
DQN Network
'''

from torch import nn
from lib.networks.nn import init_weights


class Dqn(nn.Module):
    def __init__(self, obs_size: int, action_size: int, dim_size: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_size, dim_size),
            nn.ReLU(),
            # nn.Linear(dim_size, dim_size),
            # nn.ReLU(),
            nn.Linear(dim_size, action_size)
        )

        # self.network.apply(init_weights)

    def forward(self, x):
        return self.network(x)