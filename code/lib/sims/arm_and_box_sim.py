# Greg Attra
# 04.10.22

'''
Env with a robot arm and a box
'''

from dataclasses import dataclass
from typing import Any, List, Optional

from attr import field
from isaacgym import gymapi
import numpy as np

from lib.sims.sim import Sim


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
class ViewerConfig:
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


@dataclass
class Box:
    asset: gymapi.Asset


@dataclass
class Arm:
    asset: gymapi.Asset
    dof_props: Any
    name: str


@dataclass
class ArmAndBoxSimParts:
    arm: Arm
    box: Box


@dataclass
class ArmAndBoxSim(Sim):
    sim: gymapi.Sim
    viewer: gymapi.Viewer
    parts: ArmAndBoxSimParts
    env_ptrs: List
    arm_handles: List
    box_handles: List


def load_asset(
        asset_config: AssetConfig,
        sim: gymapi.Sim,
        gym: gymapi.Gym) -> gymapi.Asset:
    asset = gym.load_asset(
        sim,
        asset_config.asset_root,
        asset_config.asset_file,
        asset_config.asset_options)
    return asset


def create_env(
        config: ArmAndBoxSimConfig,
        sim: ArmAndBoxSim,
        gym: gymapi.Gym) -> None:
    assert sim.parts.arm is not None, 'sim.parts.arm is None'

    env_idx = len(sim.env_ptrs)
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
        pose=config.arm_config.start_pose,
        name='arm',
        group=env_idx,
        filter=1,
        segmentationId=0)

    # get joint limits and ranges for arm
    arm_dof_props = gym.get_asset_dof_properties(sim.parts.arm.asset)
    arm_conf = 0.5 * (arm_dof_props['upper'] + arm_dof_props['lower'])

    # set default DOF states
    default_dof_state = np.zeros(len(arm_dof_props), gymapi.DofState.dtype)
    default_dof_state["pos"][:7] = arm_conf[:7]
    
    gym.set_actor_dof_states(env_ptr, arm_handle, default_dof_state, gymapi.STATE_ALL)
    gym.set_actor_dof_properties(env_ptr, arm_handle, sim.parts.arm.dof_props)
    gym.enable_actor_dof_force_sensors(env_ptr, arm_handle)

    # add box
    box_handle = gym.create_actor(
        env=env_ptr,
        asset=sim.parts.box.asset,
        pose=config.box_config.start_pose,
        name='box',
        group=env_idx,
        filter=1,
        segmentationId=0)
    
    # add friction to box
    box_props = gym.get_actor_rigid_shape_properties(env_ptr, box_handle)
    box_props[0].friction = config.box_config.friction
    box_props[0].rolling_friction = config.box_config.friction
    box_props[0].torsion_friction = config.box_config.friction
    gym.set_actor_rigid_shape_properties(env_ptr, box_handle, box_props)

    # set color of box
    gym.set_rigid_body_color(env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(.8, .2, .2))

    # register handles with sim
    sim.env_ptrs.append(env_ptr)
    sim.arm_handles.append(arm_handle)
    sim.box_handles.append(box_handle)


def build_parts(
        config: ArmAndBoxSimConfig,
        sim: gymapi.Sim,
        gym: gymapi.Gym) -> ArmAndBoxSimParts:
    # load arm asset
    arm_asset = load_asset(config.arm_config.asset_config, sim, gym)

    dof_props = gym.get_asset_dof_properties(arm_asset)
    dof_props['stiffness'][:].fill(config.arm_config.stiffness)
    dof_props['damping'][:].fill(config.arm_config.damping)

    arm: Arm = Arm(arm_asset, dof_props, 'arm')

    # load box asset
    box_asset = gym.create_box(
        sim,
        config.box_config.width,
        config.box_config.height,
        config.box_config.depth,
        config.box_config.asset_options)
    box: Box = Box(box_asset)

    parts: ArmAndBoxSimParts = ArmAndBoxSimParts(arm, box)

    return parts


def initialize_sim(config: ArmAndBoxSimConfig, gym: gymapi.Gym) -> ArmAndBoxSim:
    sim = gym.create_sim(
        config.compute_device,
        config.graphics_device,
        config.physics_engine,
        config.sim_params
    )

    # add the ground
    gym.add_ground(sim, config.plane_params)

    # load assets and build models
    parts: ArmAndBoxSimParts = build_parts(config, sim, gym)

    # setup viewer
    viewer: gymapi.Viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    
    arm_and_box_sim: ArmAndBoxSim = ArmAndBoxSim(sim, viewer, parts, [], [], [])

    for i in range(config.n_envs):
        create_env(config, arm_and_box_sim, gym)

    # look at middle of scene
    gym.viewer_camera_look_at(
        viewer,
        arm_and_box_sim.env_ptrs[config.n_envs // 2 + config.n_envs_per_row // 2],
        config.viewer_config.pos,
        config.viewer_config.look_at)

    return arm_and_box_sim


def start_sim(sim: ArmAndBoxSim, gym: gymapi.Gym) -> None:
    '''
    Start the sim / pre-loop setup
    '''
    gym.prepare_sim(sim.sim)


def destroy_sim(sim: ArmAndBoxSim, gym: gymapi.Gym) -> None:
    gym.destroy_viewer(sim.viewer)
    gym.destroy_sim(sim.sim)


def step_sim(sim: ArmAndBoxSim, gym: gymapi.Gym) -> None:
    # pre-physics
    gym.refresh_rigid_body_state_tensor(sim.sim)
    gym.refresh_dof_state_tensor(sim.sim)
    gym.refresh_jacobian_tensors(sim.sim)
    gym.refresh_mass_matrix_tensors(sim.sim)

    # physics step
    gym.simulate(sim.sim)
    gym.fetch_results(sim.sim, True)

    # render step
    gym.step_graphics(sim.sim)
    gym.draw_viewer(sim.viewer, sim.sim, render_collision=False)
    gym.sync_frame_time(sim.sim)
