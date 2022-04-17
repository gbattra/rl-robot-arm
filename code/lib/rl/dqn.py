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
from lib.tasks.agent import Agent
from lib.tasks.env import Env


def dqn(
    env: Env,
    agent: Agent,
    buffer: ReplayBuffer,
    optimize: Callable[[ReplayBuffer, int], float],
    analytics: Callable[[Tensor, Tensor, float, int, int], None],
    n_epochs: int,
    n_episodes: int,
    n_steps: int
) -> Dict:
    for p in trange(n_epochs, desc="Epoch", leave=False):
        global_timestep = 0
        for e in trange(n_episodes, desc="Episode", leave=False):
            env.reset(None)
            for t in trange(n_steps, desc="Step", leave=False):
                s = env.compute_observations()
                a = agent.act(s, global_timestep)
                s_prime, r, done, _ = env.step(a)

                for i in range(s.shape[0]):
                    buffer.add(Transition(s[i], a[i], s_prime[i], r[i], done[i]), i)
                    if done[i]:
                        buffer.add_done(Transition(s[i], a[i], s_prime[i], r[i], done[i]), i)

                loss = optimize(buffer, t)
                analytics(r, done, loss, p, e, t)

                # reset envs which have finished task
                env.reset(done)

                global_timestep += 1
    return {}
