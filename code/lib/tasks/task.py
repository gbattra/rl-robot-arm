# Greg Attra
# 04.11.22


'''
Protocol for running tasks
'''

from dataclasses import dataclass
from typing import Any, Callable, Protocol
from isaacgym import gymapi
from lib.sims.sim import Sim


@dataclass
class Task(Protocol):
    current_states: Callable[[Sim, gymapi.Gym], Any]
    choose_actions: Callable
    step_sim: Callable[[Sim, Any, gymapi.Gym], Any]
