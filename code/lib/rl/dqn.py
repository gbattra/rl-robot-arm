# Greg Attra
# 04.11.22

'''
Algorithm for dqn learning
'''

from re import S
from tqdm import trange
from typing import Callable, Dict, Tuple

import torch

import numpy as np

from torch import nn, Tensor
from lib.rl.buffer import ReplayBuffer, Transition


def generate_dqn_policy(
        q_net: nn.Module,
        epsilon: Callable[[int], float]) -> Callable[[Tensor, int], Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def select_action(X: Tensor, t: int):
        with torch.no_grad():
            a_vals = q_net(X.to(device).float()).cpu().numpy()
            if np.random.random() < epsilon(t):
                return np.random.choice(a_vals.shape[1])
            return np.random.choice(np.flatnonzero(a_vals == a_vals.max()))

    return select_action


def dqn(
        reset_task: Callable[[], Tensor],
        step_task: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor, Dict]],
        policy_net: nn.Module,
        target_net: nn.Module,
        train: Callable[[nn.Module, nn.Module, ReplayBuffer], None],
        epsilon: Callable[[int], float],
        analytics: Callable[[Tensor, Tensor, int, int, int], None],
        buffer: ReplayBuffer,
        target_update_freq: int,
        n_epochs: int,
        n_episodes: int,
        n_steps: int) -> Dict:
    policy = generate_dqn_policy(policy_net, epsilon)

    target_net.load_state_dict(policy_net.state_dict())

    for p in trange(n_epochs, desc='Epoch', leave=False):
        for e in trange(n_episodes, desc='Episode', leave=False):
            s = reset_task()
            for t in trange(n_steps, desc='Step', leave=False):
                a = policy(s, t)
                s_prime, r, done, _ = step_task(a)
                [buffer.add(Transition(ts, ta, tsp, tr, td) for ts, ta, tsp, tr, td in zip(s, a, s_prime, r, done))]

                analytics(r, done, p, e, s)

                train(policy_net, target_net, buffer)

                if t % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())
    return {}


def train_dqn(
        policy_net: nn.Module,
        target_net: nn.Module,
        buffer: ReplayBuffer,
        loss_fn: Callable,
        optimizer: torch.optim.Optimizer,
        gamma: float,
        batch_size: int) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if len(buffer) < batch_size:
        return

    sample = buffer.sample(batch_size)
    batch = Transition(*zip(*sample))

    states = torch.from_numpy(np.array(batch.state)).float().to(device)
    next_states = torch.from_numpy(np.array(batch.next_state)).float().to(device)
    actions = torch.from_numpy(np.array(batch.action)).unsqueeze(1).long().to(device)
    rewards = torch.from_numpy(np.array(batch.reward)).unsqueeze(1).float().to(device)
    dones = torch.from_numpy(np.array(batch.done)).unsqueeze(1).float().to(device)

    action_values = target_net(next_states).max(1)[0].unsqueeze(1)
    q_targets = rewards + (gamma * action_values * (1. - dones))
    q_est = policy_net(states).gather(1, actions)

    loss = loss_fn(q_est, q_targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
