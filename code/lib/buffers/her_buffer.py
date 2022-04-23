# Greg Attra
# 04.23.22

'''
Buffer for HER
'''

from typing import Tuple
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
        super().add(states, actions, next_states, rwds, dones)
        self.traj_states[self.env_idxs, self.env_traj_steps, :] = states[:]
        self.traj_actions[self.env_idxs, self.env_traj_steps, :] = actions[:]
        self.traj_next_states[self.env_idxs, self.env_traj_steps, :] = next_states[:]
        self.traj_rwds[self.env_idxs, self.env_traj_steps, :] = rwds[:]
        self.traj_dones[self.env_idxs, self.env_traj_steps, :] = dones[:]

        self.env_traj_steps[:] += 1

        self._flush_dones(dones)

    def _flush_dones(self, dones: torch.Tensor) -> None:
        results = flush_dones(
            dones,
            self.env_idxs,
            self.env_traj_steps,
            self.traj_states,
            self.traj_actions,
            self.traj_next_states,
            self.traj_rwds,
            self.traj_dones,
            self.state_size,
            self.action_size,
            self.n_steps,
            self.device)
        all_traj_states, all_traj_actions, all_traj_next_states, all_traj_dones, all_traj_rwds = results

        super().add(all_traj_states, all_traj_actions, all_traj_next_states, all_traj_dones, all_traj_rwds)
        

@torch.jit.script
def flush_dones(
            dones,
            env_idxs,
            env_traj_steps,
            traj_states,
            traj_actions,
            traj_next_states,
            traj_rwds,
            traj_dones,
            state_size: int,
            action_size: int,
            n_steps: int,
            device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        flush_envs = env_idxs[dones[:, 0] + (env_traj_steps >= n_steps)]
        total_steps = int(env_traj_steps[flush_envs].sum().item())
        
        all_traj_states = torch.zeros((total_steps, state_size), device=device)
        all_traj_next_states = torch.zeros((total_steps, state_size), device=device)
        all_traj_actions = torch.zeros((total_steps, action_size), device=device, dtype=torch.long)
        all_traj_dones = torch.zeros((total_steps, 1), device=device, dtype=torch.bool)
        all_traj_rwds = torch.zeros((total_steps, 1), device=device)

        traj_idx = 0
        for env_idx in flush_envs:
            traj_size = env_traj_steps[env_idx]
            next_traj_idx = traj_idx + traj_size

            env_traj_states = traj_states[env_idx, :traj_size]
            env_traj_actions = traj_actions[env_idx, :traj_size]
            env_traj_next_states = traj_next_states[env_idx, :traj_size]
            env_traj_rwds = traj_rwds[env_idx, :traj_size]
            env_traj_dones = traj_dones[env_idx, :traj_size]

            goal_state = env_traj_next_states[-1][-6:-3]
            env_traj_dones[-1] = True
            env_traj_rwds[-1] = 1.0

            env_traj_states[:, -3:] = goal_state[:]
            env_traj_next_states[:, -3:] = goal_state[:]

            all_traj_states[traj_idx:next_traj_idx, :] = env_traj_states[:]
            all_traj_next_states[traj_idx:next_traj_idx, :] = env_traj_next_states[:]
            all_traj_actions[traj_idx:next_traj_idx, :] = env_traj_actions[:]
            all_traj_dones[traj_idx:next_traj_idx, :] = env_traj_dones[:]
            all_traj_rwds[traj_idx:next_traj_idx, :] = env_traj_rwds[:]

            traj_idx = next_traj_idx

        env_traj_steps[flush_envs] = 0

        return all_traj_states, all_traj_actions, all_traj_next_states, all_traj_dones, all_traj_rwds
