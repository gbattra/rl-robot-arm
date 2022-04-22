# Greg Attra
# 04.22.22

"""
Buffer functions. Some code inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

from typing import Tuple

import torch

from lib.buffers.buffer import ReplayBuffer


class WinBuffer(ReplayBuffer):
    def __init__(
            self,
            size: int,
            state_size: int,
            action_size: int,
            n_envs: int,
            pct_winning: float) -> None:
        super().__init__(size, state_size, action_size, n_envs)
        self.pct_winning = pct_winning
        self.winning_index = 0
        self.winning_buffers_filled = False

        self.winning_states_buffer = torch.zeros((size, state_size), device=self.device)
        self.winning_actions_buffer = torch.zeros((size, action_size), device=self.device).long()
        self.winning_next_states_buffer = torch.zeros((size, state_size), device=self.device)
        self.winning_rewards_buffer = torch.zeros((size, 1), device=self.device)
        self.winning_dones_buffer = torch.zeros((size, 1), device=self.device).bool()

    def add(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rwds: torch.Tensor,
            dones: torch.Tensor) -> None:
        super().add(states, actions, next_states, rwds, dones)

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

    def sample(self, batch_size: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_step_idx = self.size if self.winning_buffers_filled else self.winning_index

        win_batch_size = int(batch_size * self.pct_winning)
        if win_batch_size > self.winning_index and not self.winning_buffers_filled:
            win_batch_size = self.winning_index

        sample_batch_size = int(batch_size - win_batch_size)
        sample_states, sample_actions, sample_next_states, sample_rwds, sample_dones = super().sample(sample_batch_size)

        if win_batch_size == 0:
            return sample_states, sample_actions, sample_next_states, sample_rwds, sample_dones
        
        step_idxs = torch.randint(0, max_step_idx, (win_batch_size, 1), device=self.device).squeeze(-1)
        winning_states = self.winning_states_buffer[step_idxs]
        winning_actions = self.winning_actions_buffer[step_idxs]
        winning_next_states = self.winning_next_states_buffer[step_idxs]
        winning_rwds = self.winning_rewards_buffer[step_idxs]
        winning_dones = self.winning_dones_buffer[step_idxs]

        states = torch.vstack([sample_states, winning_states])
        next_states = torch.vstack([sample_next_states, winning_next_states])
        actions = torch.vstack([sample_actions, winning_actions])
        rewards = torch.vstack([sample_rwds, winning_rwds])
        dones = torch.vstack([sample_dones, winning_dones])

        return states, actions, next_states, rewards, dones
