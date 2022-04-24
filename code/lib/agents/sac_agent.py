# Greg Attra
# 04.23.22

'''
SAC Agent
'''

from typing import Callable
import torch

from lib.agents.agent import Agent
from lib.buffers.buffer import ReplayBuffer
from lib.networks.sac import ActorNetwork, CriticNetwork, ValueNetwork


class SacAgent(Agent):
    def __init__(
            self,
            buffer: ReplayBuffer,
            obs_size: int,
            action_size: int,
            network_dim_size: int,
            batch_size: int,
            actor_lr: float,
            critic_lr: float,
            value_lr: float,
            gamma: float,
            epsilon: Callable,
            target_update_freq: int,
            save_path: str
        ) -> None:
        super().__init__(save_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.buffer = buffer
        self.actor_net = ActorNetwork(obs_size, action_size, network_dim_size, actor_lr)
        self.critic_1_net = CriticNetwork(obs_size, action_size, network_dim_size, critic_lr)
        self.critic_2_net = CriticNetwork(obs_size, action_size, network_dim_size, critic_lr)
        self.b_value_net = ValueNetwork(obs_size, network_dim_size, value_lr)
        self.t_value_net = ValueNetwork(obs_size, network_dim_size, value_lr)
        self.gamma = gamma
        self.epsilon = epsilon
        self.obs_size = obs_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.target_update_freq: target_update_freq
        
        self.t_value_net.load_state_dict(self.b_value_net.state_dict())

    def act(self, state: torch.Tensor, t: int) -> torch.Tensor:
        with torch.no_grad():
            if torch.rand(1) < self.epsilon(t):
                actions, _ = self.actor_net.sample(state, noise=True)
            else:
                actions, _ = self.actor_net.sample(state)
        
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
            return 0

        samples = self.buffer.sample(self.batch_size)
        states, actions, next_states, rewards, dones = samples

        # optimize value network
        value_actions, value_log_probs = self.actor_net.sample(states)
        critic_states = torch.cat((states, value_actions), dim=1)
        critic_1_values = self.critic_1_net(critic_states)
        critic_2_values = self.critic_2_net(critic_states)
        critic_values = torch.min(critic_1_values, critic_2_values)

        state_values = self.b_value_net(states)
        next_state_values = self.t_value_net(next_states)
        next_state_values[dones] = 0.0

        self.b_value_net.optimizer.zero_grad()
        target_state_values = critic_values - value_log_probs
        value_loss = self.b_value_net.loss_fn(state_values, target_state_values)
        value_loss.backward()
        self.b_value_net.optimizer.step()

        # optimize actor network
        actor_actions, actor_log_probs = self.actor_net.sample(states, noise=True)
        critic_states = torch.cat((states, actor_actions), dim=1)
        critic_1_values = self.critic_1_net(critic_states)
        critic_2_values = self.critic_2_net(critic_states)
        critic_values = torch.min(critic_1_values, critic_2_values)

        self.actor_net.optimizer.zero_grad()
        actor_loss = torch.mean(actor_log_probs - critic_values)
        actor_loss.backward()
        self.actor_net.optimizer.step()

        # optimize critic network
        self.critic_1_net.optimizer.zero_grad()
        self.critic_2_net.optimizer.zero_grad()
        target_critic_value = rewards + (self.gamma * next_state_values)

        critic_old_states = torch.cat((states, actions), dim=1)
        critic_1_old_values = self.critic_1_net(critic_old_states)
        critic_2_old_values = self.critic_2_net(critic_old_states)

        critic_1_loss = self.critic_1_net.loss_fn(critic_1_old_values, target_critic_value)
        critic_2_loss = self.critic_2_net.loss_fn(critic_2_old_values, target_critic_value)
        critic_loss = critic_1_loss + critic_2_loss
        critic_loss.backward()

        self.critic_1_net.optimizer.step()
        self.critic_2_net.optimizer.step()

        if timestep % self.target_update_freq:
            self.t_value_net.load_state_dict(self.b_value_net.state_dict()) 

