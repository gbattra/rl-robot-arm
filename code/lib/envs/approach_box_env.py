# Greg Attra
# 04.22.2022


import math
from typing import Dict, Optional, Tuple
from isaacgym import gymapi, gymtorch, torch_utils
import torch
import numpy as np
from lib.structs.arm_and_box_sim import ArmAndBoxSimConfig, AssetConfig

from lib.structs.approach_task import ApproachTaskActions, ApproachTaskConfig


def load_asset(
    asset_config: AssetConfig, sim: gymapi.Sim, gym: gymapi.Gym
) -> gymapi.Asset:
    asset = gym.load_asset(
        sim,
        asset_config.asset_root,
        asset_config.asset_file,
        asset_config.asset_options,
    )
    return asset


class ApproachEnv:
    def __init__(
            self,
            sim_config: ArmAndBoxSimConfig,
            task_config: ApproachTaskConfig,
            gym: gymapi.Gym) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gym = gym

        self.sim: gymapi.Sim = self.gym.create_sim(
            sim_config.compute_device,
            sim_config.graphics_device,
            sim_config.physics_engine,
            sim_config.sim_params
        )

        self.n_envs = sim_config.n_envs

        self.gym.add_ground(self.sim, sim_config.plane_params)

        self.arm_asset = load_asset(sim_config.arm_config.asset_config, self.sim, self.gym)

        dof_props = self.gym.get_asset_dof_properties(self.arm_asset)
        dof_props["stiffness"][:].fill(sim_config.arm_config.stiffness)
        dof_props["damping"][:].fill(sim_config.arm_config.damping)

        self.arm_n_dofs = len(dof_props)
        self.arm_upper_limits = torch_utils.to_torch(dof_props["upper"], device=self.device)
        self.arm_lower_limits = torch_utils.to_torch(dof_props["lower"], device=self.device)
        
        # load box asset
        self.box_asset = self.gym.create_box(
            self.sim,
            sim_config.box_config.width,
            sim_config.box_config.height,
            sim_config.box_config.depth,
            sim_config.box_config.asset_options,
        )

        self.env_ptrs = []
        self.arm_handles = []
        self.box_handles = []
        for i in range(sim_config.n_envs):
            env_idx = i
            env_lower = gymapi.Vec3(-sim_config.env_spacing, -sim_config.env_spacing, 0)
            env_upper = gymapi.Vec3(sim_config.env_spacing, sim_config.env_spacing, sim_config.env_spacing)
            env_ptr = self.gym.create_env(self.sim, env_lower, env_upper, sim_config.n_envs_per_row)

            # add arm actor
            arm_handle = self.gym.create_actor(
                env=env_ptr,
                asset=self.arm_asset,
                pose=sim_config.arm_config.start_pose,
                name="arm",
                group=env_idx,
                filter=1,
                segmentationId=0,
            )

            # get joint limits and ranges for arm
            arm_conf = 0.5 * (dof_props["upper"] + dof_props["lower"])

            # set default DOF states
            default_dof_state = np.zeros(self.arm_n_dofs, gymapi.DofState.dtype)
            default_dof_state["pos"][:7] = arm_conf[:7]

            self.gym.set_actor_dof_states(env_ptr, arm_handle, default_dof_state, gymapi.STATE_ALL)
            self.gym.set_actor_dof_properties(env_ptr, arm_handle, dof_props)
            self.gym.enable_actor_dof_force_sensors(env_ptr, arm_handle)

            # add box
            box_handle = self.gym.create_actor(
                env=env_ptr,
                asset=self.box_asset,
                pose=sim_config.box_config.start_pose,
                name="box",
                group=env_idx,
                filter=0,
                segmentationId=0,
            )

            # add friction to box
            box_props = self.gym.get_actor_rigid_shape_properties(env_ptr, box_handle)
            box_props[0].friction = sim_config.box_config.friction
            box_props[0].rolling_friction = sim_config.box_config.friction
            box_props[0].torsion_friction = sim_config.box_config.friction
            self.gym.set_actor_rigid_shape_properties(env_ptr, box_handle, box_props)

            # set color of box
            self.gym.set_rigid_body_color(
                env_ptr, box_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.2, 0.2)
            )

            # register handles with sim
            self.env_ptrs.append(env_ptr)
            self.arm_handles.append(arm_handle)
            self.box_handles.append(box_handle)

        self.viewer: Optional[gymapi.Viewer] = None
        self.headless = sim_config.viewer_config.headless
        if not self.headless:
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            # look at middle of scene
            self.gym.viewer_camera_look_at(
                self.viewer,
                self.env_ptrs[sim_config.n_envs // 2 + sim_config.n_envs_per_row // 2],
                sim_config.viewer_config.pos,
                sim_config.viewer_config.look_at,
            )

        self.gym.prepare_sim(self.sim)

        # get dof state buffer
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(_dof_states)
        self.dof_positions = self.dof_states.view(sim_config.n_envs, -1, 2)[..., 0]
        self.dof_velocities = self.dof_states.view(sim_config.n_envs, -1, 2)[..., 1]

        # get root states
        _root_states = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(_root_states).view(sim_config.n_envs, -1, 13)

        # get hand poses
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states).view(sim_config.n_envs, -1, 13)

        self.hand_idx: int = self.gym.find_actor_rigid_body_handle(
            self.env_ptrs[0], self.arm_handles[0], sim_config.arm_config.hand_link
        )
        self.hand_poses: torch.Tensor = self.rb_states[:, self.hand_idx]

        self.left_finger_idx: int = self.gym.find_actor_rigid_body_handle(
            self.env_ptrs[0], self.arm_handles[0], sim_config.arm_config.left_finger_link
        )
        self.left_finger_poses: torch.Tensor = self.rb_states[:, self.left_finger_idx]


        self.right_finger_idx: int = self.gym.find_actor_rigid_body_handle(
            self.env_ptrs[0], self.arm_handles[0], sim_config.arm_config.right_finger_link
        )
        self.right_finger_poses: torch.Tensor = self.rb_states[:, self.right_finger_idx]

        self.box_poses: torch.Tensor = self.rb_states[:, -1]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.gripper_offset_z = task_config.gripper_offset_z

        # task info
        self.max_episode_steps = task_config.max_episode_steps
        self.action_scale = task_config.action_scale
        self.observation_size = self.arm_n_dofs + 3 + 3
        self.action_size = len(ApproachTaskActions) * self.arm_n_dofs
        self.distance_threshold = task_config.distance_threshold

        # task state
        self.env_current_steps = torch.zeros(self.n_envs).to(self.device)
        self.dof_targets = torch.zeros((self.n_envs, self.arm_n_dofs), device=self.device)
        self.init_root = self.root_states.clone()

        self.state_buf = torch.zeros((self.n_envs, self.observation_size)).to(self.device)
        self.rwd_buf = torch.zeros((self.n_envs, 1)).to(self.device)
        self.dones_buf = torch.zeros((self.n_envs, 1)).to(self.device)

    def reset_done(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.compute_observations(), self.compute_dones()

    def reset(self) -> torch.Tensor:
        reset_envs = torch.arange(self.n_envs).to(self.device)

        # set default DOF states
        conf_signs = (torch.randint(0, 2, (self.n_envs, self.arm_n_dofs)).to(self.device) * 2.) - 1.
        arm_confs: torch.Tensor = torch.rand((self.n_envs, self.arm_n_dofs), device=self.device)
        arm_confs = torch_utils.tensor_clamp(
            arm_confs * (2 * math.pi) * conf_signs,
            self.arm_lower_limits,
            self.arm_upper_limits,
        )
        self.dof_positions[reset_envs, :] = arm_confs[reset_envs, :]
        self.dof_velocities[reset_envs, :] = .0
        self.dof_targets[reset_envs, :] = arm_confs[reset_envs, :]

        rands = torch.rand((self.n_envs, 3)).to(self.device)
        signs = (torch.randint(0, 2, (self.n_envs, 3)).to(self.device) * 2.) - 1.
        box_poses = (torch.ones((self.n_envs, 3)).to(self.device) * 0.25 + (rands * 0.5)) * signs
        
        # box_poses[..., 0] = .5
        # box_poses[..., 1] = .5
        # box_poses[..., 2] = .05
        box_poses[..., 2] = torch.abs(box_poses[..., 2])

        root_states = self.init_root.clone().to(self.device)
        root_states[reset_envs, 1, :3] = box_poses[reset_envs, :]
        
        self.env_current_steps[reset_envs] = 0

        all_actor_indices = torch.arange(2 * self.n_envs, dtype=torch.int32) \
            .to(self.device).view(self.n_envs, 2)
        actor_indices = all_actor_indices[reset_envs, 1]
        
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(root_states),
            gymtorch.unwrap_tensor(actor_indices),
            len(actor_indices))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))

        self.compute_observations()

    def compute_observations(self) -> torch.Tensor:
        state: torch.Tensor = torch.cat(
            (
                # self.dof_positions,
                # self.dof_velocities,
                self.dof_targets,
                self.hand_poses[:, 0:3],
                self.box_poses[:, 0:3],
            ),
            axis=1,
        )

        self.state_buf[:,:-6] = self.dof_positions[:, :]
        self.state_buf[:,-6:-3] = self.hand_poses[:, :3]
        self.state_buf[:, -3:] = self.box_poses[:, :3]

        return self.state_buf

    def compute_dones(self) -> torch.Tensor:
        # distances: torch.Tensor = torch.norm(
        #     self.left_finger_poses[:, 0:3] - self.box_poses[:, 0:3], p=2, dim=-1
        # ).to(self.device)
        # dones: torch.Tensor = distances.le(self.distance_threshold).to(self.device)
        # return dones.unsqueeze(-1)
        dones = (self.env_timesteps >= self.max_timestep).to(self.device).unsqueeze(-1)
        return dones

    def compute_rewards(self):
        self.rwd_buf[:, :] = \
            compute_rewards(
                    self.left_finger_poses,
                    self.right_finger_poses,
                    self.box_poses,
                    self.hand_poses,
                    self.env_current_steps,
                    self.max_episode_steps,
                    self.n_envs,
                    self.device)


    def step(self, actions: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step the sim by taking the chosen actions
        """
        self.env_current_steps += 1
        actions = (actions - 1.0) * self.action_scale
        targets = self.dof_targets + actions
        targets = torch_utils.tensor_clamp(
            targets,
            self.arm_lower_limits,
            self.arm_upper_limits,
        )

        self.dof_targets = targets

        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.dof_targets)
        )

        # self.dof_positions[:, :] = targets[:,:]
        # self.dof_velocities[:, :] = .0
        # self.dof_targets[:, :] = targets[:, :]
        # self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))

        self.tick()

        self.compute_observations()
        self.compute_rewards()

        return self.state_buf, self.rwd_buf, self.dones_buf, {}

    def tick(self) -> None:
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # physics step
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # render step
        if not self.headless:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, render_collision=False)
            self.gym.sync_frame_time(self.sim)

    def destroy(self) -> None:
        # self.gym.destroy_viewer(self.viewer)
        # for env_ptr in self.env_ptrs:
        #     self.gym.destroy_env(env_ptr)
        self.gym.destroy_sim(self.sim)


@torch.jit.script
def compute_rewards(
    left_finger_poses,
    right_finger_poses,
    box_poses,
    hand_poses,
    env_timesteps,
    max_timestep: int,
    n_envs: int,
    device: str
):
    lf_distances: torch.Tensor = torch.norm(
        left_finger_poses[:, 0:3] - box_poses[:, 0:3], p=2, dim=-1
    )
    lf_distances: torch.Tensor = torch.norm(
        right_finger_poses[:, 0:3] - box_poses[:, 0:3], p=2, dim=-1
    )
    h_targets = box_poses.clone()
    h_targets[:, 2] += .125
    h_distances: torch.Tensor = torch.norm(
        hand_poses[:, 0:3] - h_targets[:, 0:3], p=2, dim=-1
    )
    rwds: torch.Tensor = torch.ones((n_envs, 1)).to(device) * -0.01
    lf_close: torch.Tensor = lf_distances.le(0.2)
    lf_closer = lf_distances.le(0.1)
    lf_closest = lf_distances.le(0.05)
    lf_on_target = lf_distances.le(0.01)

    rf_close: torch.Tensor = lf_distances.le(0.2)
    rf_closer = lf_distances.le(0.1)
    rf_closest = lf_distances.le(0.05)
    rf_on_target = lf_distances.le(0.01)

    h_close: torch.Tensor = h_distances.le(0.2)
    h_closer = h_distances.le(0.1)
    h_closest = h_distances.le(0.075)
    h_on_target = h_distances.le(0.045)

    rwds[lf_close * rf_close * h_close, :] = .1
    rwds[lf_closer * rf_closer * h_closer, :] = .5
    rwds[lf_closest * rf_closest * h_closest, :] = 1.
    rwds[lf_on_target * rf_on_target * h_on_target, :] = 2.

    # box_lifted = ((box_poses[:, 2] > 0.05) * lf_closer * rf_closer)
    # box_z_rwd = torch.zeros_like(rwds).to(device)
    # box_z_rwd[box_lifted, :] = ((1. - (1. - torch.clamp(box_poses[box_lifted, 2], 0, 1.))) * 10.).unsqueeze(-1)
    # rwds[:, :] += box_z_rwd
    return rwds
