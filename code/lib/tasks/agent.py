# Greg Attra
# 04.17.22

'''
Class representing an agent for action selection and learning
'''

from abc import abstractmethod
from typing import Callable

import torch
from torch import nn
from lib.rl.buffer import ReplayBuffer, Transition


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


class ApproachBoxAgent(Agent):
    def __init__(
            self,
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
            her: bool = False) -> None:
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.her = her
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

    def remember(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            s_primes: torch.Tensor,
            rwds: torch.Tensor,
            dones: torch.Tensor) -> None:
        for i in range(states.shape[0]):
            self.buffer.add(Transition(states[i], actions[i], s_primes[i], rwds[i], dones[i]))
            if dones[i]:
                self.buffer.add_done(Transition(states[i], actions[i], s_primes[i], rwds[i], dones[i]), i)

    def act(self, state: torch.Tensor, t: int) -> torch.Tensor:
        with torch.no_grad():
            a_vals: torch.Tensor = self.policy_net(state).to(self.device)
            # reshape output to correspond to joints: [N x 21] -> [N x n_joints x n_joint_actions]
            joint_a_vals: torch.Tensor = a_vals.view(
                (-1,) + (self.n_dofs, self.n_dof_actions)
            ).to(self.device)
            randoms = torch.rand(state.shape[0]) < self.epsilon(t)
            # get random action indices in shape: [N x n_joints]
            random_actions = torch.randint(
                0, self.n_dof_actions, (joint_a_vals.shape[0], joint_a_vals.shape[1])
            ).to(self.device)

            # get max a_vals per joint: [N x n_joints]
            policy_actions = joint_a_vals.max(-1)[1].to(self.device)
            policy_actions[randoms] = random_actions[randoms]
        return policy_actions

    def optimize(self, timestep: int) -> torch.Tensor:
        n_joint_actions = self.n_dof_actions
        n_joints = self.n_dofs

        if self.buffer.sample_buffer_size >= self.batch_size:
            sample = self.buffer.sample(self.batch_size)
        else:
            sample = self.buffer.sample(self.buffer.sample_buffer_size)

        sample_batch = Transition(*zip(*sample))

        states = torch.stack(sample_batch.state)
        next_states = torch.stack(sample_batch.next_state)
        actions = torch.stack(sample_batch.action)
        rewards = torch.stack(sample_batch.reward)
        dones = torch.stack(sample_batch.done)

        if not self.her:
            if self.buffer.dones_buffer_size >= self.batch_size:
                dones_sample = self.buffer.sample_dones(self.batch_size)
            elif self.buffer.dones_buffer_size == 0:
                dones_sample = sample.copy()
            else:
                dones_sample = self.buffer.sample_dones(self.buffer.dones_buffer_size)

            dones_batch = Transition(*zip(*dones_sample))

            done_states = torch.stack(dones_batch.state)
            dones_next_states = torch.stack(dones_batch.next_state)
            dones_actions = torch.stack(dones_batch.action)
            dones_rewards = torch.stack(dones_batch.reward)
            dones_dones = torch.stack(dones_batch.done)

            states = torch.vstack([states, done_states])
            next_states = torch.vstack([next_states, dones_next_states])
            actions = torch.vstack([actions, dones_actions])
            rewards = torch.vstack([rewards, dones_rewards])
            dones = torch.vstack([dones, dones_dones])

        if self.her:
            # set all state rewards to 0 and dones to False
            rewards_her = torch.ones_like(rewards).to(self.device)
            dones_her = torch.ones_like(dones).bool().to(self.device)
            states_her = states.clone().to(self.device)
            next_states_her = next_states.clone().to(self.device)
            actions_her = actions.clone().to(self.device)

            # sample non-terminal state and set dones
            # sample_target_idx = torch.randint(0, states.shape[0], (1,1)).to(device).item()
            # rewards_her[sample_target_idx] = 1.
            # dones_her[sample_target_idx] = True

            # set all other state rewards to 0
            target_hand_pos = next_states[:, -6:-3]
            states_her[:, -3:] = target_hand_pos[:,:]
            next_states_her[:, -3:] = target_hand_pos[:,:]

            states = torch.vstack([states, states_her])
            next_states = torch.vstack([next_states, next_states_her])
            actions = torch.vstack([actions, actions_her])
            rewards = torch.vstack([rewards, rewards_her])
            dones = torch.vstack([dones, dones_her])


        target_action_values = (
            self.target_net(next_states).view((-1,) + (n_joints, n_joint_actions)).max(-1)[0]
        )
        q_targets = rewards + (self.gamma * target_action_values * ~dones)
        q_est = (
            self.policy_net(states)
            .view((-1,) + (n_joints, n_joint_actions))
            .gather(-1, actions.unsqueeze(-1))
        )

        loss = self.loss_fn(q_est, q_targets.unsqueeze(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if timestep % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss
