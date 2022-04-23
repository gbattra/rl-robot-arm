# Greg Attra
# 04.23.22

'''
Buffer for HER
'''

import torch
from lib.buffers.buffer import ReplayBuffer


class HerBuffer(ReplayBuffer):
    def __init__(
            self,
            size: int,
            state_size: int,
            action_size: int,
            n_envs: int,
            n_steps: int) -> None:
        super().__init__(size, state_size, action_size, n_envs)
        self.n_steps = n_steps

        self.traj_states = torch.zeros((self.n_envs, self.n_steps, self.state_size), device=self.device)
        self.traj_actions = torch.zeros((self.n_envs, self.n_steps, self.action_size), device=self.device).long()
        self.traj_next_states = torch.zeros((self.n_envs, self.n_steps, self.state_size), device=self.device)
        self.traj_rwds = torch.zeros((self.n_envs, self.n_steps, 1), device=self.device)
        self.traj_dones = torch.zeros((self.n_envs, self.n_steps, 1), device=self.device).bool()

        self.env_traj_steps = torch.zeros((self.n_envs), device=self.device).long()
        self.env_idxs = torch.arange(self.n_envs, device=self.device).long()

    def add(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rwds: torch.Tensor,
            dones: torch.Tensor) -> None:
        self.traj_states[self.env_idxs, self.env_traj_steps, :] = states[:]
        self.traj_actions[self.env_idxs, self.env_traj_steps, :] = actions[:]
        self.traj_next_states[self.env_idxs, self.env_traj_steps, :] = next_states[:]
        self.traj_rwds[self.env_idxs, self.env_traj_steps, :] = rwds[:]
        self.traj_dones[self.env_idxs, self.env_traj_steps, :] = dones[:]

        self.env_traj_steps[:] += 1

        self._flush_dones(dones)

    def _flush_dones(self, dones: torch.Tensor) -> None:
        flush_envs = self.env_idxs[dones[:, 0] + (self.env_traj_steps >= self.n_steps)]
        for env_idx in flush_envs:
            traj_states = self.traj_states[env_idx, :self.env_traj_steps[env_idx]]
            traj_actions = self.traj_actions[env_idx, :self.env_traj_steps[env_idx]]
            traj_next_states = self.traj_next_states[env_idx, :self.env_traj_steps[env_idx]]
            traj_rwds = self.traj_rwds[env_idx, :self.env_traj_steps[env_idx]]
            traj_dones = self.traj_dones[env_idx, :self.env_traj_steps[env_idx]]
            
            goal_state = traj_next_states[-1][-6:-3]
            traj_dones[-1] = True
            traj_rwds[-1] = 1.0

            traj_states[:, -3:] = goal_state[:]
            traj_next_states[:, -3:] = goal_state[:]

            super().add(traj_states, traj_actions, traj_next_states, traj_rwds, traj_dones)
        
        self.env_traj_steps[flush_envs] = 0
