# Greg Attra
# 04.11.22

'''
Task for training arm to approach target position
'''

from dataclasses import dataclass

import torch
from isaacgym import gymapi, gymtorch, torch_utils
from lib.sims.arm_and_box_sim import ArmAndBoxSim, step_sim
from lib.sims.sim import Sim
from lib.tasks.task import Task
import numpy as np


@dataclass
class ApproachTask(Task):
    sim: ArmAndBoxSim
    dof_targets: torch.Tensor


def initialize_task(sim: ArmAndBoxSim, gym: gymapi.Gym) -> ApproachTask:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return ApproachTask(
        sim=sim,
        dof_targets=torch.zeros(
            (len(sim.arm_handles), len(sim.parts.arm.dof_props)),
            dtype=torch.float,
            device=device)
    )


def step_actions(task: ApproachTask, actions: torch.Tensor, gym: gymapi.Gym):
    '''
    Step the sim by taking the chosen actions
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    targets = task.dof_targets + actions
    targets = torch_utils.tensor_clamp(
        targets,
        torch_utils.to_torch(task.sim.parts.arm.dof_props['lower'], device=device),
        torch_utils.to_torch(task.sim.parts.arm.dof_props['lower'], device=device))
    gym.set_dof_position_target_tensor(task.sim.sim, gymtorch.unwrap_tensor(task.dof_targets))

    step_sim(task.sim, gym)

    return None, None, None, None
