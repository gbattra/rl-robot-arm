# Greg Attra
# 04.11.22

"""
Task for training arm to approach target position
"""

from enum import IntEnum
from isaacgym import gymapi
from dataclasses import dataclass

from lib.sims.arm_and_box_sim import ArmAndBoxSim
import torch
from torch import nn


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


def initialize_approach_task(
    config: ApproachTaskConfig, sim: ArmAndBoxSim, gym: gymapi.Gym
) -> ApproachTask:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_dofs = sim.parts.arm.n_dofs
    n_actions = len(ApproachTaskActions)

    curent_dof_pos_size = n_dofs
    current_dof_vel_size = n_dofs
    target_dof_pos_size = n_dofs
    hand_pos_size = 3
    box_pos_size = 3

    obs_size = (
        curent_dof_pos_size \
        # + current_dof_vel_size \
        + target_dof_pos_size \
        + hand_pos_size + box_pos_size
    )
    action_size = n_actions * n_dofs

    env_steps = torch.zeros(sim.n_envs).to(device)

    return ApproachTask(
        gripper_offest_z=config.gripper_offset_z,
        env_steps=env_steps,
        max_episode_steps=config.max_episode_steps,
        sim=sim,
        action_scale=config.action_scale,
        observation_size=obs_size,
        action_size=action_size,
        dof_targets=torch.zeros(
            (len(sim.arm_handles), len(sim.parts.arm.dof_props)),
            dtype=torch.float,
            device=device,
        ),
        distance_threshold=config.distance_threshold,
    )
