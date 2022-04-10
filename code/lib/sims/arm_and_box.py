# Greg Attra
# 04.10.22

'''
Env with a robot arm and a box
'''

from dataclasses import dataclass
from typing import List, Optional, Tuple

from attr import field
import torch
from isaacgym import gymapi


@dataclass
class AssetConfig:
    asset_root: str
    asset_file: str
    asset_options: gymapi.AssetOptions


@dataclass
class ArmConfig:
    asset_config: AssetConfig
    stiffness: float
    damping: float
    start_pose: gymapi.Transform


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


@dataclass
class Arm:
    asset: gymapi.Asset
    dof_props: gymapi.GymDofProperties
    name: str


@dataclass
class ArmAndBoxSimParts:
    arm: Optional[Arm] = None


@dataclass
class ArmAndBoxSim:
    sim: gymapi.Sim
    parts: ArmAndBoxSimParts
    env_ptrs: List = field(default=list())
    arm_handles: List = field(default=list())
    box_handles: List = field(default=list())


def load_asset(
        sim: ArmAndBoxSim,
        asset_config: AssetConfig,
        gym: gymapi.Gym) -> gymapi.Asset:
    asset = gym.load_asset(
        sim.sim,
        asset_config.asset_root,
        asset_config.asset_file,
        asset_config.asset_options)
    return asset


def create_env(
        config: ArmAndBoxSimConfig,
        sim: ArmAndBoxSim,
        gym: gymapi.Gym) -> None:
    assert sim.parts.arm is not None, 'sim.parts.arm is None'

    env_idx = len(ArmAndBoxSim)
    env_lower = gymapi.Vec3(
        -config.env_spacing,
        -config.env_spacing,
        0)
    env_upper = gymapi.Vec3(
        config.env_spacing,
        config.env_spacing,
        config.env_spacing)
    env_ptr = gym.create_env(
        sim.sim,
        env_lower,
        env_upper,
        config.n_envs_per_row)
    
    # add arm actor
    arm_handle = gym.create_actor(
        env=env_ptr,
        asset=sim.parts.arm.asset,
        pose=config.arm_start_pose,
        name='arm',
        group=env_idx,
        filter=1,
        segmentationId=0)
    gym.set_actor_dof_properties(env_ptr, arm_handle, sim.parts.arm.dof_props)
    gym.enable_actor_dof_force_sensors(env_ptr, arm_handle)

    # add box

    # register handles with sim
    sim.env_ptrs.append(env_ptr)
    sim.arm_handles.append(arm_handle)


def build_parts(config: ArmAndBoxSimConfig, gym: gymapi.Gym) -> ArmAndBoxSimParts:
    # add objects / assets
    parts: ArmAndBoxSimParts = ArmAndBoxSimParts()
    
    # load arm asset
    arm_asset = load_asset(config.arm_config.asset_config)
    dof_props = gym.get_asset_dof_properties(arm_asset)
    arm: Arm = Arm(arm_asset, dof_props, 'arm')
    parts.arm = arm

    return parts


def initialize_sim(config: ArmAndBoxSimConfig, gym: gymapi.Gym) -> ArmAndBoxSim:
    sim = gym.create_sim(
        config.compute_device,
        config.graphics_device,
        config.physics_engine,
        config.sim_params
    )

    parts: ArmAndBoxSimParts = build_parts(config, gym)
    arm_and_box_sim: ArmAndBoxSim = ArmAndBoxSim(sim, parts)

    for i in config.n_envs:
        create_env(config, arm_and_box_sim, gym)

    return arm_and_box_sim

