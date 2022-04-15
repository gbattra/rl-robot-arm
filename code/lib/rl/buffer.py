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

    def add_done(self, transition: Transition) -> None:
        self.dones_buffer_size += 1
        self.dones.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def sample_dones(self, batch_size: int) -> List[Transition]:
        return random.sample(self.dones, batch_size)

    def __len__(self):
        return len(self.buffer)
