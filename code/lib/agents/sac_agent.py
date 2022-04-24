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
            action_scale: float,
            alpha: float,
            actor_lr: float,
            critic_lr: float,
            gamma: float,
            epsilon: Callable,
            target_update_freq: int,
            save_path: str
        ) -> None:
        super().__init__(save_path)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.buffer = buffer
        self.alpha = alpha
        self.action_scale = action_scale
        self.gamma = gamma
        self.epsilon = epsilon
        self.obs_size = obs_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.actor = ActorNetwork(obs_size, action_size, network_dim_size, action_scale, actor_lr).to(self.device)
        self.behavior_critic = CriticNetwork(obs_size, action_size, network_dim_size, critic_lr).to(self.device)
        self.target_critic = CriticNetwork(obs_size, action_size, network_dim_size, critic_lr).to(self.device)
        # self.critic_2_net = CriticNetwork(obs_size, action_size, network_dim_size, critic_lr).to(self.device)
        # self.b_value_net = ValueNetwork(obs_size, network_dim_size, value_lr).to(self.device)
        # self.t_value_net = ValueNetwork(obs_size, network_dim_size, value_lr).to(self.device)
        
        self.target_critic.load_state_dict(self.behavior_critic.state_dict())

    def save_checkpoint(self) -> None:
        torch.save(self.policy_net.state_dict(), self.save_path)

    def act(self, state: torch.Tensor, t: int) -> torch.Tensor:
        with torch.no_grad():
            if torch.rand(1) < self.epsilon(t):
                actions = torch.rand((state.shape[0], self.action_size), device=self.device) \
                    * torch.randint(-1, 2, (state.shape[0], self.action_size), device=self.device)
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

        # critic loss
        next_actions, next_log_probs = self.actor.sample(next_states)
        critic_next_states = torch.cat((next_states, next_actions), dim=1)
        next_state_values = self.target_critic(critic_next_states)
        next_state_values[dones] = 0.0
        entropies = self.alpha * next_log_probs
        target_values = rewards + (self.gamma * (next_state_values - entropies))

        critic_states = torch.cat((states, actions), dim=1)
        critic_values = self.behavior_critic(critic_states)

        critic_loss = self.behavior_critic.loss_fn(critic_values, target_values)
        self.behavior_critic.optimizer.zero_grad()
        critic_loss.backward()
        self.behavior_critic.optimizer.step()

        # actor loss
        reparam_actions, reparam_log_probs = self.actor.sample(states, reparam=True)
        reparam_critic_states = torch.cat((states, reparam_actions), dim=1)
        reparam_critic_values = self.critic_net(reparam_critic_states)
        reparam_entropies = self.alpha * reparam_log_probs
        
        actor_loss = (1./self.batch_size) * (reparam_critic_values - reparam_entropies).sum(dim=1)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        if timestep % self.target_update_freq:
            self.target_critic.load_state_dict(self.behavior_critic.state_dict())

    # def optimize(self, timestep: int) -> torch.Tensor:
    #     if self.buffer.sample_index < self.batch_size and not self.buffer.sample_buffers_filled:
    #         return 0

    #     samples = self.buffer.sample(self.batch_size)
    #     states, actions, next_states, rewards, dones = samples

    #     # optimize value network
    #     policy_actions, policy_log_probs = self.actor_net.sample(states)
    #     critic_states = torch.cat((states, policy_actions), dim=1)
    #     critic_1_values = self.critic_1_net(critic_states)
    #     critic_2_values = self.critic_2_net(critic_states)
    #     critic_values = torch.min(critic_1_values, critic_2_values)

    #     state_values = self.b_value_net(states)
    #     next_state_values = self.t_value_net(next_states)
    #     next_state_values[dones] = 0.0
    #     target_state_values = critic_values - policy_log_probs.sum(1, keepdim=True)

    #     value_loss = self.b_value_net.loss_fn(state_values, target_state_values)

    #     self.b_value_net.optimizer.zero_grad()
    #     value_loss.backward()
    #     self.b_value_net.optimizer.step()

    #     # optimize actor network
    #     actor_actions, actor_log_probs = self.actor_net.sample(states, noise=True)
    #     actor_critic_states = torch.cat((states, actor_actions), dim=1)
    #     actor_critic_1_values = self.critic_1_net(actor_critic_states)
    #     actor_critic_2_values = self.critic_2_net(actor_critic_states)
    #     actor_critic_values = torch.min(actor_critic_1_values, actor_critic_2_values)

    #     actor_loss = torch.mean(actor_log_probs.sum(1, keepdim=True) - actor_critic_values)

    #     self.actor_net.optimizer.zero_grad()
    #     actor_loss.backward()
    #     self.actor_net.optimizer.step()

    #     # optimize critic network
    #     target_critic_value = (rewards - actor_log_probs.detach().sum(1, keepdim=True)) + (self.gamma * next_state_values)

    #     old_critic_states = torch.cat((states, actions), dim=1)
    #     old_critic_1_values = self.critic_1_net(old_critic_states)
    #     old_critic_2_values = self.critic_2_net(old_critic_states)
    #     critic_1_loss = self.critic_1_net.loss_fn(old_critic_1_values, target_critic_value)
    #     critic_2_loss = self.critic_2_net.loss_fn(old_critic_2_values, target_critic_value)

    #     critic_loss = critic_1_loss + critic_2_loss

    #     self.critic_1_net.optimizer.zero_grad()
    #     self.critic_2_net.optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_1_net.optimizer.step()
    #     self.critic_2_net.optimizer.step()

    #     # sync target net weigths
    #     if timestep % self.target_update_freq:
    #         self.t_value_net.load_state_dict(self.b_value_net.state_dict())
