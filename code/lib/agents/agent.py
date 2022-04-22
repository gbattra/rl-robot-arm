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
    def train(
            self,
            env: Env,
            n_epochs: int,
            n_episodes: int,
            n_steps: int) -> Dict:
        pass
