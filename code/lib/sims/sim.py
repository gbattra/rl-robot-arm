# Greg Attra
# 04.11.22

'''
Sim protocol for typing
'''

from abc import abstractmethod
from dataclasses import dataclass
from isaacgym import gymapi

from lib.sims.arm_and_box_sim import ArmAndBoxSim

@dataclass
class Sim:
    sim: gymapi
