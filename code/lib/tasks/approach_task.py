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
    max_episode_steps: int


@dataclass
class ApproachTask(Task):
    env_steps: torch.Tensor
    max_episode_steps: int
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
    hand_pos_size = 3
    box_pos_size = 3

    obs_size = (
        curent_dof_pos_size \
        # + current_dof_vel_size \
        # + target_dof_pos_size \
        + hand_pos_size + box_pos_size
    )
    action_size = n_actions * n_dofs

    env_steps = torch.zeros(sim.n_envs).to(device)

    return ApproachTask(
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


def compute_approach_task_observations(
    task: ApproachTask, gym: gymapi.Gym
) -> torch.Tensor:
    state: torch.Tensor = torch.cat(
        (
            # task.sim.dof_positions,
            # task.sim.dof_velocities,
            task.dof_targets,
            task.sim.hand_poses[:, 0:3],
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
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    reset_envs = torch.arange(task.sim.n_envs).to(device)
    if dones is not None:
        timeout_envs = task.env_steps > task.max_episode_steps
        reset_envs = reset_envs[dones.squeeze(-1) + timeout_envs]

    if reset_envs.shape[0] == 0:
        return

    # set default DOF states
    arm_confs: torch.Tensor = torch.rand(
        (len(reset_envs), task.sim.parts.arm.n_dofs), device=device
    )
    arm_confs = torch_utils.tensor_clamp(
        arm_confs,
        task.sim.parts.arm.lower_limits,
        task.sim.parts.arm.upper_limits,
    )
    task.sim.dof_positions[reset_envs, :] = arm_confs[:, :]
    task.sim.dof_velocities[reset_envs, :] = .0
    task.dof_targets[reset_envs, :] = arm_confs[:]

    rands = torch.rand((task.sim.n_envs, 3)).to(device)
    signs = (torch.randint(0, 2, (task.sim.n_envs, 3)).to(device) * 2.) - 1.
    box_poses = (torch.ones((task.sim.n_envs, 3)).to(device) * 0.25 + (rands * 0.5)) * signs
    box_poses[..., 2] = .0

    root_states = task.sim.init_root.clone().to(device)
    root_states[reset_envs, 1, :3] = box_poses[reset_envs, :]
    
    task.env_steps[reset_envs] = 0

    all_actor_indices = torch.arange(2 * task.sim.n_envs, dtype=torch.int32).to(device).view(task.sim.n_envs, 2)
    actor_indices = all_actor_indices[reset_envs, 1]
    
    gym.set_actor_root_state_tensor_indexed(
        task.sim.sim,
        gymtorch.unwrap_tensor(root_states),
        gymtorch.unwrap_tensor(actor_indices),
        len(actor_indices))
    gym.set_dof_state_tensor(task.sim.sim, gymtorch.unwrap_tensor(task.sim.dof_states))

    return


def step_approach_task(
    task: ApproachTask, actions: torch.Tensor, gym: gymapi.Gym
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
    """
    Step the sim by taking the chosen actions
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    task.env_steps[:] += 1
    actions = (actions.float() - 1.0) * task.action_scale
    targets = task.dof_targets + actions
    targets = torch_utils.tensor_clamp(
        targets,
        task.sim.parts.arm.lower_limits,
        task.sim.parts.arm.upper_limits,
    )

    task.dof_targets = targets

    gym.set_dof_position_target_tensor(
        task.sim.sim, gymtorch.unwrap_tensor(task.dof_targets)
    )

    step_sim(task.sim, gym)

    observations = compute_approach_task_observations(task, gym)
    rewards = compute_approach_task_rewards(task, observations, gym)
    dones = compute_approach_task_dones(task, observations, gym)

    return observations, rewards, dones, {}


def approach_task_network(task: ApproachTask) -> nn.Sequential:
    dim_size = 64
    return nn.Sequential(
        nn.Linear(task.observation_size, dim_size),
        nn.ReLU(),
        nn.Linear(dim_size, dim_size),
        nn.ReLU(),
        nn.Linear(dim_size, task.action_size),
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
            randoms = torch.rand(X.shape[0]) < epsilon(t)
            # get random action indices in shape: [N x n_joints]
            random_actions = torch.randint(
                0, n_joint_actions, (joint_a_vals.shape[0], joint_a_vals.shape[1])
            ).to(device)

            # get max a_vals per joint: [N x n_joints]
            policy_actions = joint_a_vals.max(-1)[1].to(device)
            policy_actions[randoms] = random_actions[randoms]
        return policy_actions

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
    her: bool = False
) -> float:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_joint_actions = len(ApproachTaskActions)
    n_joints = task.sim.parts.arm.n_dofs

    if buffer.sample_buffer_size >= batch_size:
        sample = buffer.sample(batch_size)
    else:
        sample = buffer.sample(buffer.sample_buffer_size)

    sample_batch = Transition(*zip(*sample))

    states = torch.stack(sample_batch.state)
    next_states = torch.stack(sample_batch.next_state)
    actions = torch.stack(sample_batch.action)
    rewards = torch.stack(sample_batch.reward)
    dones = torch.stack(sample_batch.done)

    if not her:
        if buffer.dones_buffer_size >= batch_size:
            dones_sample = buffer.sample_dones(batch_size)
        elif buffer.dones_buffer_size == 0:
            dones_sample = sample.copy()
        else:
            dones_sample = buffer.sample_dones(buffer.dones_buffer_size)

        dones_batch = Transition(*zip(*dones_sample))

        done_states = torch.stack(dones_batch.state)
        dones_next_states = torch.stack(dones_batch.next_state)
        dones_actions = torch.stack(dones_batch.action)
        dones_rewards = torch.stack(dones_batch.reward)
        dones_dones = torch.stack(dones_batch.done)

        states = torch.vstack([states, done_states])
        next_states = torch.vstack([next_states, dones_next_states])
        actions = torch.vstack([actions, dones_actions])
        rewards = torch.vstack([rewards, dones_rewards])
        dones = torch.vstack([dones, dones_dones])

    if her:
        # set all state rewards to 0 and dones to False
        rewards_her = torch.ones_like(rewards).to(device)
        dones_her = torch.ones_like(dones).bool().to(device)
        states_her = states.clone().to(device)
        next_states_her = next_states.clone().to(device)
        actions_her = actions.clone().to(device)

        # sample non-terminal state and set dones
        # sample_target_idx = torch.randint(0, states.shape[0], (1,1)).to(device).item()
        # rewards_her[sample_target_idx] = 1.
        # dones_her[sample_target_idx] = True

        # set all other state rewards to 0
        target_hand_pos = next_states[:, -6:-3]
        states_her[:, -3:] = target_hand_pos[:,:]
        next_states_her[:, -3:] = target_hand_pos[:,:]

        states = torch.vstack([states, states_her])
        next_states = torch.vstack([next_states, next_states_her])
        actions = torch.vstack([actions, actions_her])
        rewards = torch.vstack([rewards, rewards_her])
        dones = torch.vstack([dones, dones_her])


    target_action_values = (
        target_net(next_states).view((-1,) + (n_joints, n_joint_actions)).max(-1)[0]
    )
    q_targets = rewards + (gamma * target_action_values * ~dones)
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

    return loss
