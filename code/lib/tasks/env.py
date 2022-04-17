# Greg Attra
# 04.11.22


"""
Protocol for running tasks
"""

from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from isaacgym import gymapi, gymtorch, torch_utils

import torch
from lib.tasks.task import ApproachTask


@dataclass
class Task:
    pass


class Env:
    @abstractmethod
    def compute_observations(self) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_rewards(self) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_dones(self) -> torch.Tensor:
        pass

    @abstractmethod
    def reset(self, dones: Optional[torch.Tensor]) -> None:
        pass

    @abstractmethod
    def step(self, actions: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        pass


class ApproachBoxEnv(Env):
    def __init__(
            self,
            task: ApproachTask,
            gym: gymapi.Gym) -> None:
        super().__init__()
        self.gym = gym

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        sim = task.sim
        self.sim = sim.sim
        self.viewer = sim.viewer
        self.n_envs = sim.n_envs
        self.env_ptrs = sim.env_ptrs
        self.arm_handles = sim.arm_handles
        self.box_handles = sim.box_handles
        self.dof_states = sim.dof_states
        self.dof_positions = sim.dof_positions
        self.dof_velocities = sim.dof_velocities
        self.box_poses = sim.box_poses
        self.hand_poses = sim.hand_poses
        self.rb_states = sim.rb_states
        self.root_states = sim.root_states
        self.init_root = sim.init_root
        self.gripper_offset_z = task.gripper_offest_z

        # arm info
        self.arm_n_dofs = sim.parts.arm.n_dofs
        self.arm_lower_limits = sim.parts.arm.lower_limits
        self.arm_upper_limits = sim.parts.arm.upper_limits

        # task info
        self.max_episode_steps = task.max_episode_steps
        self.action_scale = task.action_scale
        self.observation_size = task.observation_size
        self.action_size = task.action_size
        self.distance_threshold = task.distance_threshold

        # task state
        self.env_current_steps = task.env_steps
        self.dof_targets = task.dof_targets

    def compute_observations(self) -> torch.Tensor:
        state: torch.Tensor = torch.cat(
            (
                self.dof_positions,
                # self.dof_velocities,
                self.dof_targets,
                self.hand_poses[:, 0:3],
                self.box_poses[:, 0:3],
            ),
            axis=1,
        )
        return state

    def compute_rewards(self) -> torch.Tensor:
        target_pos = self.box_poses[:, 0:3]
        target_pos[:, 2] += self.gripper_offset_z
        distances: torch.Tensor = torch.norm(
            self.hand_poses[:, 0:3] - self.box_poses[:, 0:3], p=2, dim=-1
        ).to(self.device)
        winning: torch.Tensor = distances.le(self.distance_threshold).to(self.device)
        rwds: torch.Tensor = torch.ones_like(winning).float().to(self.device)
        rwds[~winning] = 0
        return rwds.unsqueeze(-1)

    def compute_dones(self) -> torch.Tensor:
        distances: torch.Tensor = torch.norm(
            self.hand_poses[:, 0:3] - self.box_poses[:, 0:3], p=2, dim=-1
        ).to(self.device)
        dones: torch.Tensor = distances.le(self.distance_threshold).to(self.device)
        return dones.unsqueeze(-1)
        # if self.env_current_steps[0] >= self.max_episode_steps:
        #     return torch.ones((self.n_envs, 1)).bool().to(self.device)
        # return torch.zeros((self.n_envs, 1)).bool().to(self.device)

    def reset(self, dones: Optional[torch.Tensor]) -> None:
        reset_envs = torch.arange(self.n_envs).to(self.device)
        if dones is not None:
            timeout_envs = self.env_current_steps > self.max_episode_steps
            reset_envs = reset_envs[dones.squeeze(-1) + timeout_envs]

        if reset_envs.shape[0] == 0:
            return

        # set default DOF states
        arm_confs: torch.Tensor = torch.rand(
            (len(reset_envs), self.arm_n_dofs), device=self.device
        )
        arm_confs = torch_utils.tensor_clamp(
            arm_confs,
            self.arm_lower_limits,
            self.arm_upper_limits,
        )
        self.dof_positions[reset_envs, :] = arm_confs[:, :]
        self.dof_velocities[reset_envs, :] = .0
        self.dof_targets[reset_envs, :] = arm_confs[:]

        rands = torch.rand((self.n_envs, 3)).to(self.device)
        signs = (torch.randint(0, 2, (self.n_envs, 3)).to(self.device) * 2.) - 1.
        box_poses = (torch.ones((self.n_envs, 3)).to(self.device) * 0.25 + (rands * 0.5)) * signs
        
        box_poses[..., 0] = .5
        box_poses[..., 1] = .5
        box_poses[..., 2] = .0

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

        return

    def step(self, actions: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Step the sim by taking the chosen actions
        """
        self.env_current_steps[:] += 1
        actions = (actions.float() - 1.0) * self.action_scale
        targets = self.dof_targets + actions
        targets = torch_utils.tensor_clamp(
            targets,
            self.arm_lower_limits,
            self.arm_upper_limits,
        )

        self.dof_targets[:,:] = targets[:,:]

        self.gym.set_dof_position_target_tensor(
            self.sim, gymtorch.unwrap_tensor(self.dof_targets)
        )

        self.tick()

        observations = self.compute_observations()
        rewards = self.compute_rewards()
        dones = self.compute_dones()

        return observations, rewards, dones, {}

    def tick(self) -> None:
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        # physics step
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        # render step
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, render_collision=False)
        self.gym.sync_frame_time(self.sim)

    def destroy(self) -> None:
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
