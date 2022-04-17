# Greg Attra
# 04.11.22

'''
Sim protocol for typing
'''

from dataclasses import dataclass
from isaacgym import gymapi

@dataclass
class Sim:
    sim: gymapi
