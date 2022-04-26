# Greg Attra
# 04.10.22

"""
Env with a robot arm and a box
"""

from isaacgym import gymapi
from dataclasses import dataclass
from typing import Any, List, Optional
from lib.structs.sim import Sim

import torch


@dataclass
class AssetConfig:
    asset_root: str
    asset_file: str
    asset_options: gymapi.AssetOptions


@dataclass
class ArmConfig:
    hand_link: str
    left_finger_link: str
    right_finger_link: str
    asset_config: AssetConfig
    stiffness: float
    damping: float
    start_pose: gymapi.Transform


@dataclass
class ViewerConfig:
    headless: bool
    pos: gymapi.Vec3
    look_at: gymapi.Vec3


@dataclass
class BoxConfig:
    height: float
    width: float
    depth: float
    friction: float
    start_pose: gymapi.Transform
    asset_options: gymapi.AssetOptions


@dataclass
class ArmAndBoxSimConfig:
    n_envs: int
    env_spacing: float
    n_envs_per_row: int
    arm_config: ArmConfig
    compute_device: int
    graphics_device: int
    physics_engine: gymapi.SimType
    sim_params: gymapi.SimParams
    plane_params: gymapi.PlaneParams
    viewer_config: ViewerConfig
    box_config: BoxConfig
    n_actors_per_env: int


@dataclass
class Box:
    asset: gymapi.Asset


@dataclass
class Arm:
    asset: gymapi.Asset
    dof_props: Any
    n_dofs: int
    lower_limits: torch.Tensor
    upper_limits: torch.Tensor
    name: str


@dataclass
class ArmAndBoxSimParts:
    arm: Arm
    box: Box


@dataclass
class ArmAndBoxSim(Sim):
    sim: gymapi.Sim
    viewer: Optional[gymapi.Viewer]
    parts: ArmAndBoxSimParts
    n_envs: int
    env_ptrs: List
    arm_handles: List
    box_handles: List
    dof_states: torch.Tensor
    dof_positions: torch.Tensor
    dof_velocities: torch.Tensor
    box_poses: torch.Tensor
    hand_poses: torch.Tensor
    left_finger_poses: torch.Tensor
    right_finger_poses: torch.Tensor
    rb_states: torch.Tensor
    root_states: torch.Tensor
    _root_states: torch.Tensor
    init_root: torch.Tensor
