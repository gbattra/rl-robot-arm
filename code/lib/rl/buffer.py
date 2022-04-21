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

        self.sample_index = 0
        self.winning_index = 0
        self.sample_buffers_filled = False
        self.winning_buffers_filled = False
        self.size = size
        self.state_size = state_size
        self.action_size = action_size
        self.n_envs = n_envs

        self.states_buffer = torch.zeros((size, state_size)).to(self.device)
        self.actions_buffer = torch.zeros((size, action_size)).long().to(self.device)
        self.next_states_buffer = torch.zeros((size, state_size)).to(self.device)
        self.rewards_buffer = torch.zeros((size, 1)).to(self.device)
        self.dones_buffer = torch.zeros((size, 1)).bool().to(self.device)

        self.winning_states_buffer = torch.zeros((size, state_size)).to(self.device)
        self.winning_actions_buffer = torch.zeros((size, action_size)).long().to(self.device)
        self.winning_next_states_buffer = torch.zeros((size, state_size)).to(self.device)
        self.winning_rewards_buffer = torch.zeros((size, 1)).to(self.device)
        self.winning_dones_buffer = torch.zeros((size, 1)).bool().to(self.device)

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

        self._add_winning(states, actions, next_states, rwds, dones)

    def _add_winning(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rwds: torch.Tensor,
            dones: torch.Tensor) -> None:
        _winning = (rwds > 0).squeeze(-1)
        sample_size = states[_winning].shape[0]
        if sample_size == 0:
            return
        if self.winning_index + sample_size >= self.size:
            sample_size = self.size - self.winning_index - 1
        self.winning_states_buffer[self.winning_index:self.winning_index + sample_size, :] = states[_winning][:sample_size, :]
        self.winning_actions_buffer[self.winning_index:self.winning_index + sample_size, :] = actions[_winning][:sample_size, :]
        self.winning_next_states_buffer[self.winning_index:self.winning_index + sample_size, :] = next_states[_winning][:sample_size, :]
        self.winning_rewards_buffer[self.winning_index:self.winning_index + sample_size, :] = rwds[_winning][:sample_size, :]
        self.winning_dones_buffer[self.winning_index:self.winning_index + sample_size, :] = dones[_winning][:sample_size, :]

        self.winning_index += sample_size
        if self.winning_index >= self.size - 1:
            self.winning_index = 0
            self.winning_buffers_filled = True

    def _sample_winnings(self, batch_size: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_step_idx = self.size if self.winning_buffers_filled else self.winning_index
        step_idxs = torch.randint(0, max_step_idx, (batch_size, 1)).squeeze(-1)

        winning_states = self.winning_states_buffer[step_idxs, :]
        winning_actions = self.winning_actions_buffer[step_idxs, :]
        winning_next_states = self.winning_next_states_buffer[step_idxs, :]
        winning_rwds = self.winning_rewards_buffer[step_idxs, :]
        winning_dones = self.winning_dones_buffer[step_idxs, :]

        return winning_states, winning_actions, winning_next_states, winning_rwds, winning_dones

    def sample(self, batch_size: int, winning=False) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if winning:
            return self._sample_winnings(batch_size)

        max_step_idx = self.size if self.sample_buffers_filled else self.sample_index
        step_idxs = torch.randint(0, max_step_idx, (batch_size, 1)).squeeze(-1)

        sample_states = self.states_buffer[step_idxs, :]
        sample_actions = self.actions_buffer[step_idxs, :]
        sample_next_states = self.next_states_buffer[step_idxs, :]
        sample_rwds = self.rewards_buffer[step_idxs, :]
        sample_dones = self.dones_buffer[step_idxs, :]

        return sample_states, sample_actions, sample_next_states, sample_rwds, sample_dones
