# Greg Attra
# 04.11.22

"""
Task for training arm to approach target position
"""

from enum import IntEnum
from dataclasses import dataclass

from lib.cfg.arm_and_box_sim import ArmAndBoxSim
import torch


class ApproachTaskActions(IntEnum):
    REV = -1.0
    NEUT = 0.0
    FWD = 1.0


@dataclass
class ApproachTaskConfig:
    action_scale: float
    distance_threshold: float
    gripper_offset_z: float
    max_episode_steps: int


@dataclass
class ApproachTask:
    env_steps: torch.Tensor
    max_episode_steps: int
    sim: ArmAndBoxSim
    action_scale: float
    observation_size: int
    action_size: int
    dof_targets: torch.Tensor
    distance_threshold: float
    gripper_offest_z: float
