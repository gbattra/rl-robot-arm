# Greg Attra
# 04.17.22

'''
Class representing an agent for action selection and learning
'''

from abc import abstractmethod
from typing import Callable, Dict

import torch
from tqdm import trange
from lib.envs.env import Env


class Agent:
    def __init__(self, save_path: str) -> None:
        self.save_path = save_path

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

    def train(
            self,
            env: Env,
            n_epochs: int,
            n_episodes: int,
            n_steps: int,
            analytics: Callable) -> Dict:
        gt = 0
        for p in trange(n_epochs, desc="Epoch", leave=False):
            for e in trange(n_episodes, desc="Episode", leave=False):
                env.reset()
                for t in trange(n_steps, desc="Step", leave=False):
                    s = env.compute_observations()
                    a = self.act(s, gt)
                    s_prime, r, done, _ = env.step(a)

                    self.remember(s, a, s_prime, r, done)

                    loss = self.optimize(gt)
                    analytics(r, done, loss, p, e, t)

                    # reset envs which have finished task
                    env._reset_dones(torch.arange(env.n_envs, device=self.device)[done[:, 0]])

                    gt += 1
            torch.save(self.policy_net.state_dict(), self.save_path)
