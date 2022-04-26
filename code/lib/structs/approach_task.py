# Greg Attra
# 04.11.22

"""
Task for training arm to approach target position
"""

from lib.structs.arm_and_box_sim import ArmAndBoxSim
from enum import Enum, IntEnum
from dataclasses import dataclass
import torch


class ApproachTaskActions(IntEnum):
    REV = -1.0
    NEUT = 0.0
    FWD = 1.0


class ActionMode(Enum):
    DOF_TARGET = 'target'
    DOF_POSITION = 'position'


@dataclass
class ApproachTaskConfig:
    action_scale: float
    distance_threshold: float
    gripper_offset_z: float
    episode_length: int
    randomize: bool
    action_mode: ActionMode


@dataclass
class ApproachTask:
    env_steps: torch.Tensor
    sim: ArmAndBoxSim
    action_scale: float
    observation_size: int
    action_size: int
    dof_targets: torch.Tensor
    distance_threshold: float
    gripper_offest_z: float
    episode_length: int
    randomize: bool
