# Greg Attra
# 04.26.22

'''
Base Env class
'''

from abc import abstractmethod
from typing import Dict, Tuple

import torch


class Env:
    @abstractmethod
    def compute_observations(self) -> torch.Tensor:
        ...

    @abstractmethod
    def step(self, actions: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        ...
