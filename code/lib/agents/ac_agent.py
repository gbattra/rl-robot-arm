# Greg Attra
# 04.23.22

'''
SAC Agent
'''

from typing import Callable
import torch

from lib.agents.agent import Agent
from lib.buffers.buffer import ReplayBuffer
from lib.networks.ac import ActorCriticNetwork


class ActorCriticAgent(Agent):
    def __init__(
            self,
            buffer: ReplayBuffer,
            n_actions: int,
            n_joints: int,
            obs_size: int,
            network_dim_size: int,
            batch_size: int,
            action_scale: float,
            lr: float,
            gamma: float,
            epsilon: Callable,
            target_update_freq: int,
            save_path: str
        ) -> None:
        super().__init__(save_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.buffer = buffer
        self.lr = lr
        self.action_scale = action_scale
        self.gamma = gamma
        self.epsilon = epsilon
        self.obs_size = obs_size
        self.n_joints = n_joints
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.actor_critic = ActorCriticNetwork(lr, obs_size, n_actions, n_joints, network_dim_size, save_path).to(self.device)

    def save_checkpoint(self) -> None:
        torch.save(self.policy_net.state_dict(), self.save_path)

    def act(self, state: torch.Tensor, t: int) -> torch.Tensor:
        with torch.no_grad():
            # if torch.rand(1) < self.epsilon(t):
            #     actions = torch.rand((state.shape[0], self.n_joints), device=self.device) \
            #         * torch.randint(-1, 2, (state.shape[0], self.n_joints), device=self.device)
            # else:
            _, policy = self.actor_critic(state)
            action_probs = torch.distributions.Categorical(policy)
            actions = action_probs.sample()
        
        return actions

    def remember(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            s_primes: torch.Tensor,
            rwds: torch.Tensor,
            dones: torch.Tensor) -> None:
        self.buffer.add(states, actions, s_primes, rwds, dones)

    def optimize(self, timestep: int) -> torch.Tensor:
        return 0