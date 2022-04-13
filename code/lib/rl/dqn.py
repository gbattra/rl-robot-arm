# Greg Attra
# 04.11.22

"""
Algorithm for dqn learning
"""

from tqdm import trange
from typing import Callable, Dict, Tuple

import torch

import numpy as np

from torch import nn, Tensor
from lib.rl.buffer import ReplayBuffer, Transition


def dqn(
    reset_task: Callable[[], Tensor],
    step_task: Callable[[Tensor], Tuple[Tensor, Tensor, Tensor, Dict]],
    policy: Callable[[Tensor, int], Tensor],
    buffer: ReplayBuffer,
    optimize: Callable[[ReplayBuffer, int], None],
    analytics: Callable[[Tensor, Tensor, int, int, int], None],
    n_epochs: int,
    n_episodes: int,
    n_steps: int,
) -> Dict:
    for p in trange(n_epochs, desc="Epoch", leave=False):
        for e in trange(n_episodes, desc="Episode", leave=False):
            s = reset_task()
            for t in trange(n_steps, desc="Step", leave=False):
                a = policy(s, t)
                s_prime, r, done, _ = step_task(a)

                for i in range(s.shape[0]):
                    buffer.add(Transition(s[i], a[i], s_prime[i], r[i], done[i]))

                analytics(r, done, p, e, s)

                optimize(buffer, t)
    return {}
