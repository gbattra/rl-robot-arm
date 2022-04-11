# Greg Attra
# 04.11.22

'''
Task for training arm to approach target position
'''

from dataclasses import dataclass
from typing import Any
from isaacgym import gymapi
from lib.sims.arm_and_box_sim import ArmAndBoxSim
from lib.tasks.task import Task


@dataclass
class ApproachTask(Task):
    pass


def current_state(sim: ArmAndBoxSim, gym: gymapi.Gym):
    '''
    Compute current state for approach task
    '''
    pass


def choose_action(states: Any):
    '''
    Choose an action given the current states
    '''
    pass


def step_sim(sim: ArmAndBoxSim, actions: Any, gym: gymapi.Gym):
    '''
    Step the sim by taking the chosen actions
    '''
    pass
