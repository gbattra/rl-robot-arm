# Greg Attra
# 04.11.22

"""
Buffer functions. Some code inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

from enum import Enum
from typing import Tuple

import torch


class BufferType(Enum):
    STANDARD = 0
    WINNING = 1
    HER = 2


class ReplayBuffer:
    def __init__(
            self,
            size: int,
            state_size: int,
            action_size: int,
            n_envs: int) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.sample_index = 0
        self.sample_buffers_filled = False
        self.size = size
        self.state_size = state_size
        self.action_size = action_size
        self.n_envs = n_envs

        self.states_buffer = torch.zeros((size, state_size)).to(self.device)
        self.actions_buffer = torch.zeros((size, action_size)).long().to(self.device)
        self.next_states_buffer = torch.zeros((size, state_size)).to(self.device)
        self.rewards_buffer = torch.zeros((size, 1)).to(self.device)
        self.dones_buffer = torch.zeros((size, 1)).bool().to(self.device)

    def add(self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rwds: torch.Tensor,
            dones: torch.Tensor) -> None:
        n_samples = states.shape[0]
        if self.sample_index + n_samples >= self.size - 1:
            n_samples = self.size - self.sample_index
        self.states_buffer[self.sample_index:self.sample_index+n_samples, :] = states.view(-1, states.shape[-1])[:n_samples,:]
        self.actions_buffer[self.sample_index:self.sample_index+n_samples, :] = actions.view(-1, actions.shape[-1])[:n_samples,:]
        self.next_states_buffer[self.sample_index:self.sample_index+n_samples, :] = next_states.view(-1, next_states.shape[-1])[:n_samples,:]
        self.rewards_buffer[self.sample_index:self.sample_index+n_samples, :] = rwds.view(-1, rwds.shape[-1])[:n_samples,:]
        self.dones_buffer[self.sample_index:self.sample_index+n_samples, :] = dones.view(-1, dones.shape[-1])[:n_samples,:]
        
        self.sample_index += n_samples
        if self.sample_index >= self.size - 1:
            self.sample_index = 0
            self.sample_buffers_filled = True

    def sample(self, batch_size: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_step_idx = self.size if self.sample_buffers_filled else self.sample_index
        step_idxs = torch.randint(0, max_step_idx, (batch_size, 1), device=self.device).squeeze(-1)

        sample_states = self.states_buffer[step_idxs]
        sample_actions = self.actions_buffer[step_idxs]
        sample_next_states = self.next_states_buffer[step_idxs]
        sample_rwds = self.rewards_buffer[step_idxs]
        sample_dones = self.dones_buffer[step_idxs]

        return sample_states, sample_actions, sample_next_states, sample_rwds, sample_dones
