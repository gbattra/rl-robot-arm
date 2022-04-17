# Greg Attra
# 04.11.22

"""
Buffer functions. Some code inspired by https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

from collections import deque, namedtuple
import random
from typing import List


Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "done")
)


class ReplayBuffer:
    def __init__(self, size: int) -> None:
        self.sample_buffer_size = 0
        self.dones_buffer_size = 0
        self.buffer = deque([], maxlen=size)
        self.dones = deque([], maxlen=size)

    def add(self, transition: Transition) -> None:
        self.sample_buffer_size += 1
        self.buffer.append(transition)

    def add_done(self, transition: Transition, _: int) -> None:
        self.dones_buffer_size += 1
        self.dones.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def sample_dones(self, batch_size: int) -> List[Transition]:
        return random.sample(self.dones, batch_size)

    def __len__(self):
        return len(self.buffer)


class HerReplayBuffer(ReplayBuffer):
    def __init__(self, size: int, n_envs: int) -> None:
        super().__init__(size)
        self.n_envs = n_envs
        self.trajectories: List[List[Transition]] = [[]] * n_envs

    def add(self, transition: Transition, i: int) -> None:
        super().add(transition)
        self.trajectories[i].append(transition)
        if transition.done:
            self.trajectories[i] = []

    def flush_trajectory(self, i: int) -> None:
        trajectory = self.trajectories[i]
        if len(trajectory) == 0:
            return
        terminal_transition = trajectory[-1]
        terminal_transition.done[:] = True
        # use hand position as terminal goal
        terminal_goal = terminal_transition.next_state[-6:-3]
        for transition in trajectory:
            state_her = transition.state.clone()
            state_her[-3:] = terminal_goal[:]
            
            next_state_her = transition.next_state.clone()
            next_state_her[-3:] = terminal_goal[:]

            her_transition = Transition(
                state_her,
                transition.action.clone(),
                next_state_her,
                transition.reward.clone(),
                transition.done.clone()
            )
            self.buffer.append(her_transition)
        self.trajectories[i] = []

    def flush_trajectories(self) -> None:
        for i in range(self.n_envs):
            self.flush_trajectory(i)
