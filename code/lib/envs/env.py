# Greg Attra
# 04.26.22

'''
Base Env class
'''

from abc import abstractmethod

import torch


class Env:
    @abstractmethod
    def compute_observations(self) -> torch.Tensor:
        ...

    @abstractmethod
    def step(self, actions: torch.Tensor) -> None:
        ...
