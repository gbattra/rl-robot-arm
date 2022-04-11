# Greg Attra
# 04.11.22

'''
Sim protocol for typing
'''

from dataclasses import dataclass
from typing import Protocol
from isaacgym import gymapi

@dataclass
class Sim(Protocol):
    sim: gymapi.Sim
