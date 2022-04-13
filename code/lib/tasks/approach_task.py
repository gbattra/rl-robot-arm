# Greg Attra
# 04.11.22

"""
Task for training arm to approach target position
"""

from enum import IntEnum
from typing import Callable, Dict, Optional, Tuple
from isaacgym import gymapi, gymtorch, torch_utils
from dataclasses import dataclass
from lib.rl.buffer import ReplayBuffer, Transition

from lib.sims.arm_and_box_sim import ArmAndBoxSim, step_sim
from lib.sims.sim import Sim
from lib.tasks.task import Task
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class ApproachTaskActions(IntEnum):
    REV = -1.0
    NEUT = 0.0
    FWD = 1.0


@dataclass
class ApproachTaskConfig:
    action_scale: float
    distance_threshold: float


@dataclass
class ApproachTask(Task):
    sim: ArmAndBoxSim
    action_scale: float
    observation_size: int
    action_size: int
    dof_targets: torch.Tensor
    distance_threshold: float


def initialize_approach_task(
    config: ApproachTaskConfig, sim: ArmAndBoxSim, gym: gymapi.Gym
) -> ApproachTask:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_dofs = sim.parts.arm.n_dofs
    n_actions = len(ApproachTaskActions)

    curent_dof_pos_size = n_dofs
    current_dof_vel_size = n_dofs
    target_dof_pos_size = n_dofs
    box_pos_size = 3

    obs_size = (
        curent_dof_pos_size + current_dof_vel_size + target_dof_pos_size + box_pos_size
    )
    action_size = n_actions * n_dofs

    return ApproachTask(
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


def compute_approach_task_observations(
    task: ApproachTask, gym: gymapi.Gym
) -> torch.Tensor:
    state: torch.Tensor = torch.cat(
        (
            task.sim.dof_positions,
            task.sim.dof_velocities,
            task.dof_targets,
            task.sim.box_poses[:, 0:3],
        ),
        axis=1,
    )
    return state


def compute_approach_task_rewards(
    task: ApproachTask, observations: torch.Tensor, gym: gymapi.Gym
) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    distances: torch.Tensor = torch.norm(
        task.sim.hand_poses[:, 0:3] - task.sim.box_poses[:, 0:3], p=2, dim=-1
    ).to(device)
    dones: torch.Tensor = distances.le(task.distance_threshold).to(device)
    rwds: torch.Tensor = torch.ones(dones.shape).to(device) * dones
    return rwds.unsqueeze(-1)


def compute_approach_task_dones(
    task: ApproachTask, observations: torch.Tensor, gym: gymapi.Gym
) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    distances: torch.Tensor = torch.norm(
        task.sim.hand_poses[:, 0:3] - task.sim.box_poses[:, 0:3], p=2, dim=-1
    ).to(device)
    dones: torch.Tensor = distances.le(task.distance_threshold).to(device)
    return dones.unsqueeze(-1)


def reset_approach_task(
    task: ApproachTask, gym: gymapi.Gym, dones: Optional[torch.Tensor]
) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    reset_envs = torch.arange(task.sim.n_envs).to(device)
    if dones is not None:
        done_envs = dones.nonzero(as_tuple=False).unsqueeze(-1)
        reset_envs = reset_envs[done_envs].to(device)

    # set default DOF states
    arm_confs: torch.Tensor = torch.rand(
        (len(reset_envs), task.sim.parts.arm.n_dofs), device=device
    )
    arm_confs = torch_utils.tensor_clamp(
        arm_confs,
        torch_utils.to_torch(task.sim.parts.arm.dof_props["lower"], device=device),
        torch_utils.to_torch(task.sim.parts.arm.dof_props["upper"], device=device),
    )
    # default_dof_state = torch.zeros(2, task.sim.parts.arm.n_dofs).to(device)
    # default_dof_state[0, :7] = arm_conf[:7]
    task.sim.dof_positions[reset_envs, :] = torch.zeros_like(
        task.sim.dof_positions[reset_envs], device=device
    )
    task.sim.dof_positions[reset_envs, :7] = arm_confs[:, :7]
    task.sim.dof_velocities[reset_envs, :] = torch.zeros_like(
        task.sim.dof_velocities[reset_envs], device=device
    )
    task.dof_targets[reset_envs, :] = arm_confs[:]

    reset_envs = reset_envs.to(dtype=torch.int32).to(device)
    # gym.set_dof_state_tensor_indexed(
    #     task.sim.sim,
    #     gymtorch.unwrap_tensor(task.sim.dof_states),
    #     gymtorch.unwrap_tensor(reset_envs),
    #     len(reset_envs),
    # )
    gym.set_dof_state_tensor(task.sim.sim, gymtorch.unwrap_tensor(task.sim.dof_states))

    return compute_approach_task_observations(task, gym)


def step_approach_task(
    task: ApproachTask, actions: torch.Tensor, gym: gymapi.Gym
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Step the sim by taking the chosen actions
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actions = (actions.float() - 1.0) * task.action_scale
    targets = task.dof_targets + actions
    targets = torch_utils.tensor_clamp(
        targets,
        torch_utils.to_torch(task.sim.parts.arm.dof_props["lower"], device=device),
        torch_utils.to_torch(task.sim.parts.arm.dof_props["upper"], device=device),
    )

    task.dof_targets = targets
    print(task.dof_targets[0])

    gym.set_dof_position_target_tensor(
        task.sim.sim, gymtorch.unwrap_tensor(task.dof_targets)
    )

    step_sim(task.sim, gym)

    observations = compute_approach_task_observations(task, gym)
    rewards = compute_approach_task_rewards(task, observations, gym)
    dones = compute_approach_task_dones(task, observations, gym)

    return observations, rewards, dones, {}


def approach_task_network(task: ApproachTask) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(task.observation_size, 1000),
        nn.ReLU(),
        # nn.Linear(1000, 1000),
        # nn.ReLU(),
        nn.Linear(1000, task.action_size),
    )


def approach_task_dqn_policy(
    task: ApproachTask, q_net: nn.Module, epsilon: Callable[[int], float]
) -> Callable[[torch.Tensor, int], torch.Tensor]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_joint_actions = len(ApproachTaskActions)
    n_joints = task.sim.parts.arm.n_dofs

    def select_action(X: torch.Tensor, t: int) -> torch.Tensor:
        with torch.no_grad():
            a_vals: torch.Tensor = q_net(X).to(device)
            # reshape output to correspond to joints: [N x 21] -> [N x n_joints x n_joint_actions]
            joint_a_vals: torch.Tensor = a_vals.view(
                (-1,) + (n_joints, n_joint_actions)
            ).to(device)
            if torch.rand(1).item() < epsilon(t):
                # get random action indices in shape: [N x n_joints]
                joint_actions = torch.randint(
                    0, n_joint_actions, (joint_a_vals.shape[0], joint_a_vals.shape[1])
                ).to(device)
            else:
                # get max a_vals per joint: [N x n_joints]
                joint_actions = joint_a_vals.max(-1)[1].to(device)

        return joint_actions

    return select_action


def approach_task_optimize_dqn(
    task: ApproachTask,
    buffer: ReplayBuffer,
    timestep: int,
    policy_net: nn.Module,
    target_net: nn.Module,
    loss_fn: Callable,
    optimizer: torch.optim.Optimizer,
    gamma: float,
    batch_size: int,
    target_update_freq: int,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_joint_actions = len(ApproachTaskActions)
    n_joints = task.sim.parts.arm.n_dofs

    if len(buffer) < batch_size:
        return

    sample = buffer.sample(batch_size)
    batch = Transition(*zip(*sample))

    states = torch.stack(batch.state)
    next_states = torch.stack(batch.next_state)
    actions = torch.stack(batch.action)
    rewards = torch.stack(batch.reward)
    dones = torch.stack(batch.done)

    target_action_values = (
        target_net(next_states).view((-1,) + (n_joints, n_joint_actions)).max(-1)[0]
    )
    q_targets = rewards + (gamma * target_action_values * dones)
    q_est = (
        policy_net(states)
        .view((-1,) + (n_joints, n_joint_actions))
        .gather(-1, actions.unsqueeze(-1))
    )

    loss = loss_fn(q_est, q_targets.unsqueeze(-1))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if timestep % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
