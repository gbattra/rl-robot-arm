# Greg Attra
# 04.22.2022

'''
DQN Agent
'''


from typing import Callable, Dict

import torch
from torch import nn
from tqdm import trange
from lib.agents.agent import Agent
from lib.buffers.buffer import ReplayBuffer
from lib.envs.env import Env


class DQNAgent(Agent):
    def __init__(
            self,
            agent_id: int,
            n_dofs: int,
            n_dof_actions: int,
            buffer: ReplayBuffer,
            policy_net: nn.Module,
            target_net: nn.Module,
            loss_fn: Callable,
            optimizer: torch.optim.Optimizer,
            epsilon: Callable,
            gamma: float,
            batch_size: int,
            target_update_freq: int,
            save_path: str) -> None:
        super().__init__(save_path)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent_id = agent_id
        self.n_dofs = n_dofs
        self.n_dof_actions = n_dof_actions
        self.buffer = buffer
        self.policy_net = policy_net
        self.target_net = target_net
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

    def save_checkpoint(self) -> None:
        torch.save(self.policy_net.state_dict(), self.save_path)

    def remember(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            rwds: torch.Tensor,
            dones: torch.Tensor) -> None:
        self.buffer.add(states, actions, next_states, rwds, dones)

    def act(self, state: torch.Tensor, t: int) -> torch.Tensor:
        with torch.no_grad():
            a_vals: torch.Tensor = self.policy_net(state)
            # reshape output to correspond to joints: [N x 21] -> [N x n_joints x n_joint_actions]
            joint_a_vals: torch.Tensor = a_vals.view(
                (-1,) + (self.n_dofs, self.n_dof_actions)
            ).to(self.device)
            randoms = torch.rand(state.shape[0], device=self.device) < self.epsilon(t)
            # get random action indices in shape: [N x n_joints]
            random_actions = torch.randint(
                0, self.n_dof_actions, (joint_a_vals.shape[0], joint_a_vals.shape[1]), device=self.device
            )

            # get max a_vals per joint: [N x n_joints]
            policy_actions = joint_a_vals.max(-1)[1]
            policy_actions[randoms] = random_actions[randoms]
        return policy_actions

    def optimize(self, timestep: int) -> float:
        n_joint_actions = self.n_dof_actions
        n_joints = self.n_dofs

        if self.buffer.sample_index < self.batch_size and not self.buffer.sample_buffers_filled:
            return 0

        samples = self.buffer.sample(self.batch_size)
        states, actions, next_states, rewards, dones = samples

        target_action_values = (
            self.target_net(next_states).view((-1,) + (n_joints, n_joint_actions)).max(-1)[0]
        )
        q_targets = rewards + (self.gamma * target_action_values * ~dones)
        q_est = (
            self.policy_net(states)
            .view((-1,) + (n_joints, n_joint_actions))
            .gather(-1, actions.unsqueeze(-1))
        )

        loss = (1./self.batch_size) * self.loss_fn(q_est, q_targets.unsqueeze(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if timestep % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss