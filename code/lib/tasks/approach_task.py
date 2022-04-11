# Greg Attra
# 04.11.22

'''
Task for training arm to approach target position
'''

from enum import IntEnum
from isaacgym import gymapi, gymtorch, torch_utils
from dataclasses import dataclass

from lib.sims.arm_and_box_sim import ArmAndBoxSim, step_sim
from lib.sims.sim import Sim
from lib.tasks.task import Task
import numpy as np
import torch
from torch import nn


class ApproachTaskActions(IntEnum):
    FWD = 1
    NEUT = 0
    REV = -1


@dataclass
class ApproachTask(Task):
    sim: ArmAndBoxSim
    actions: torch.Tensor
    dof_targets: torch.Tensor


def initialize_approach_task(sim: ArmAndBoxSim, gym: gymapi.Gym) -> ApproachTask:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    actions: torch.Tensor = torch.tensor([
        ApproachTaskActions.FWD.value,
        ApproachTaskActions.NEUT.value,
        ApproachTaskActions.REV.value
    ])
    return ApproachTask(
        sim=sim,
        actions=actions,
        dof_targets=torch.zeros(
            (len(sim.arm_handles), len(sim.parts.arm.dof_props)),
            dtype=torch.float,
            device=device)
    )


def reset_approach_task(task: ApproachTask, gym: gymapi.Gym) -> torch.Tensor:
    pass


def step_approach_task(task: ApproachTask, actions: torch.Tensor, gym: gymapi.Gym):
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


def approach_task_network(task: ApproachTask) -> nn.Sequential:
    n_dofs = task.sim.parts.arm.n_dofs
    n_actions = task.actions.size

    curent_dof_pos_size = n_dofs
    current_dof_vel_size = n_dofs,
    target_dof_pos_size = n_dofs
    box_pos_size = 3

    input_size = (curent_dof_pos_size + current_dof_vel_size + target_dof_pos_size + box_pos_size)
    output_size = (n_dofs ** n_actions)

    return nn.Sequential(
        nn.Linear(input_size, 1000),
        nn.ReLU(),
        # nn.Linear(1000, 1000),
        # nn.ReLU(),
        nn.Linear(1000, output_size)
    )
