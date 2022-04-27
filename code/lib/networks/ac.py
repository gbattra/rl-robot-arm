# Greg Attra
# 04.24.22

'''
Actor-Critic network'''

from torch import nn
import torch
from lib.networks.nn import init_weights


class ActorCriticNetwork(nn.Module):
    def __init__(
            self,
            lr: float,
            obs_size: int,
            n_actions: int,
            n_joints: int,
            dim_size: int):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = n_actions * n_joints
        self.n_actions = n_actions
        self.n_joints = n_joints
        self.dim_size = dim_size

        self.network = nn.Sequential(
            nn.Linear(obs_size, dim_size),
            nn.ReLU(),
            nn.Linear(dim_size, dim_size),
            nn.ReLU()
        )
        self.value_layer = nn.Linear(dim_size, self.n_joints)
        self.policy_layer = nn.Linear(dim_size, self.action_size)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.network(x)
        value = self.value_layer(x)
        policy = torch.softmax(self.policy_layer(x).view(-1, self.n_joints, self.n_actions),dim=-1)

        return value, policy
