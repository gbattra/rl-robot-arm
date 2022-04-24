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
            alpha: float,
            lr: float,
            gamma: float,
            target_update_freq: int,
            save_path: str
        ) -> None:
        super().__init__(save_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.buffer = buffer
        self.alpha = alpha
        self.lr = lr
        self.action_scale = action_scale
        self.gamma = gamma
        self.obs_size = obs_size
        self.n_joints = n_joints
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.actor_critic = ActorCriticNetwork(lr, obs_size, n_actions, n_joints, network_dim_size, save_path).to(self.device)
        self.target_actor_critic = ActorCriticNetwork(lr, obs_size, n_actions, n_joints, network_dim_size, save_path).to(self.device)
        self.target_actor_critic.load_state_dict(self.actor_critic.state_dict())

    def save_checkpoint(self) -> None:
        torch.save(self.actor_critic.state_dict(), self.save_path)

    def act(self, state: torch.Tensor, t: int) -> torch.Tensor:
        with torch.no_grad():
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
        if self.buffer.sample_index < self.batch_size and not self.buffer.sample_buffers_filled:
            return torch.tensor(0, device=self.device)

        samples = self.buffer.sample(self.batch_size)
        states, actions, next_states, rewards, dones = samples

        state_values, action_probs = self.actor_critic(states)
        next_state_values, _ = self.target_actor_critic(next_states)
        action_dist = torch.distributions.Categorical(action_probs)
        action_log_probs = action_dist.log_prob(actions)

        target_values = rewards + (self.gamma * next_state_values * ~dones)
        td_error = target_values - state_values
        actor_loss = (-action_log_probs * td_error) * self.alpha
        critic_loss = td_error ** 2

        total_loss = (1./self.batch_size) * (actor_loss + critic_loss).sum()

        self.actor_critic.optimizer.zero_grad()
        total_loss.backward()
        self.actor_critic.optimizer.step()

        if timestep % self.target_update_freq == 0:
            self.target_actor_critic.load_state_dict(self.actor_critic.state_dict())

        return total_loss
