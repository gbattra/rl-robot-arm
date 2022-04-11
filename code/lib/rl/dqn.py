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
        optimizer: Callable[[nn.Module, nn.Module, ReplayBuffer], None],
        epsilon: Callable[[int], float],
        analytics: Callable[[Tensor, Tensor, int, int, int], None],
        buffer: ReplayBuffer,
        target_update_freq: int,
        n_epochs: int,
        n_episodes: int,
        n_steps: int) -> None:
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

                optimizer(policy_net, target_net, buffer)

                if t % target_update_freq == 0:
                    target_net.load_state_dict(policy_net.state_dict())
