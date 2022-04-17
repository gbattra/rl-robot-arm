# Greg Attra
# 04.11.22

"""
Buffer functions. Some code inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

from collections import deque, namedtuple
import random
from typing import List, Tuple

import torch


class ReplayBuffer:
    def __init__(
            self,
            size: int,
            state_size: int,
            action_size: int,
            n_envs: int) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.index = 0
        self.buffers_filled = False
        self.size = size
        self.state_size = state_size
        self.action_size = action_size
        self.n_envs = n_envs

        self.states_buffer = torch.zeros((n_envs, size, state_size)).to(self.device)
        self.actions_buffer = torch.zeros((n_envs, size, action_size)).long().to(self.device)
        self.next_states_buffer = torch.zeros((n_envs, size, state_size)).to(self.device)
        self.rewards_buffer = torch.zeros((n_envs, size, 1)).to(self.device)
        self.dones_buffer = torch.zeros((n_envs, size, 1)).bool().to(self.device)

    def add(self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rwds: torch.Tensor,
            dones: torch.Tensor) -> None:
        self.states_buffer[:, self.index, :] = states[:]
        self.actions_buffer[:, self.index, :] = actions[:]
        self.next_states_buffer[:, self.index, :] = next_states[:]
        self.rewards_buffer[:, self.index, :] = rwds[:]
        self.dones_buffer[:, self.index, :] = dones[:]
        
        self.index += 1
        if self.index >= self.size - 1:
            self.index = 0
            self.buffers_filled = True

    def sample(self, batch_size: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_step_idx = self.size if self.buffers_filled else self.index
        env_idxs = torch.randint(0, self.n_envs, (batch_size, 1))
        step_idxs = torch.randint(0, max_step_idx, (batch_size, 1))

        sample_states = self.states_buffer[env_idxs, step_idxs, :].squeeze(-2)
        sample_actions = self.actions_buffer[env_idxs, step_idxs, :].squeeze(-2)
        sample_next_states = self.next_states_buffer[env_idxs, step_idxs, :].squeeze(-2)
        sample_rwds = self.rewards_buffer[env_idxs, step_idxs, :].squeeze(-2)
        sample_dones = self.dones_buffer[env_idxs, step_idxs, :].squeeze(-2)

        return sample_states, sample_actions, sample_next_states, sample_rwds, sample_dones
