# Greg Attra
# 04.17.22

'''
Class representing an agent for action selection and learning
'''

from abc import abstractmethod
from typing import Dict

import torch
from lib.envs.env import Env


class Agent:
    @abstractmethod
    def act(self, state: torch.Tensor, t: int) -> torch.Tensor:
        '''
        Choose actions based on state
        '''
        pass

    @abstractmethod
    def remember(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            s_primes: torch.Tensor,
            rwds: torch.Tensor,
            dones: torch.Tensor) -> None:
        '''
        Store a transition in the replay buffer
        '''
        pass

    @abstractmethod
    def optimize(self, timestep: int) -> torch.Tensor:
        '''
        Update the DQN based on experience
        '''
        pass

    @abstractmethod
    def save_checkpoint(self, filename: str) -> None:
        '''
        Save model states to dir
        '''
        pass

    def step(
            self,
            env: Env,
            timestep: int) -> Dict:
        s = env.compute_observations()
        a = self.act(s, timestep)
        s_prime, r, done, _ = env.step(a)

        self.remember(s, a, s_prime, r, done)

        loss = self.optimize(timestep)

        # reset envs which have finished task
        env._reset_dones(torch.arange(env.n_envs, device=self.device)[done[:, 0]])

        return s, a, s_prime, r, done, {'loss': loss}
