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
            dim_size: int,
            save_path: str):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = n_actions * n_joints
        self.n_actions = n_actions
        self.n_joints = n_joints
        self.dim_size = dim_size
        self.save_path = save_path

        self.network = nn.Sequential(
            nn.Linear(obs_size, dim_size),
            nn.ReLU(),
            nn.Linear(dim_size, dim_size),
            nn.ReLU()
        )
        self.network.apply(init_weights)

        self.value_layer = nn.Linear(dim_size, self.n_joints)
        self.value_layer.weight.data.fill_(0.0)
        self.value_layer.bias.data.fill_(0.0)

        self.policy_layer = nn.Linear(dim_size, self.action_size)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        x = self.network(x)
        value = self.value_layer(x)
        policy = torch.softmax(self.policy_layer(x).view(-1, self.n_joints, self.n_actions),dim=-1)

        return value, policy
