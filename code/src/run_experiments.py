# Greg Attra
# 04.24.22

'''
Run a bunch of experiments
'''


import math
from isaacgym import gymapi, gymutil
from typing import Callable

import torch
from lib.agents.ac_agent import ActorCriticAgent
from lib.agents.agent import Agent
from lib.analytics.analytics import Analytics, initialize_analytics
from lib.analytics.plotting import plot_learning
from lib.buffers.her_buffer import HerBuffer
from lib.buffers.win_buffer import WinBuffer
from lib.envs.approach_env import ApproachEnvContinuous, ApproachEnvDiscrete
from lib.buffers.buffer import BufferType, ReplayBuffer
from lib.runner import Runner
from lib.structs.algorithm import Algorithm
from lib.structs.arm_and_box_sim import (
    ArmAndBoxSimConfig,
    ArmConfig,
    AssetConfig,
    BoxConfig,
    ViewerConfig,
)

from lib.structs.approach_task import (
    ActionMode,
    ApproachTaskActions,
    ApproachTaskConfig,
)
from lib.structs.experiment import Experiment
from lib.networks.dqn import Dqn
from torch import nn
from lib.agents.dqn_agent import DQNAgent

GAMMA: float = .99
LEARNING_RATE: float = 0.001
ACTOR_ALPHA: float = 0.2

EPS_START: float = 1.0
EPS_END: float = 0.05
EPS_DECAY: float = 0.9999

REPLAY_BUFFER_SIZE: int = 1000000
TARGET_UPDATE_FREQ: int = 100

N_ENVS: int = 1
N_EPOCHS: int = 2
N_EPISODES: int = 5
N_STEPS: int = 10


PLOT_FREQ: int = N_STEPS
SAVE_FREQ: int = N_STEPS * N_EPISODES

def run_experiment(
        env: ApproachEnvContinuous,
        experiment: Experiment,
        debug: bool):

    env.randomize = experiment.randomize
    env.action_scale = experiment.action_scale
    env.distance_threshold = experiment.dist_thresh
    env.action_mode = experiment.action_mode

    if experiment.buffer_type == BufferType.STANDARD:
        buffer: ReplayBuffer = ReplayBuffer(experiment.replay_buffer_size, env.observation_size, env.arm_n_dofs, env.n_envs)
    elif experiment.buffer_type == BufferType.HER:
        buffer: ReplayBuffer = HerBuffer(
            experiment.replay_buffer_size,
            env.observation_size,
            env.arm_n_dofs,
            env.n_envs,
            experiment.n_timesteps,
            torch.long)
    else:
        buffer: ReplayBuffer = WinBuffer(experiment.replay_buffer_size, env.observation_size, env.arm_n_dofs, env.n_envs, 0.25)

    if experiment.algo_name == Algorithm.AC:
        agent: Agent = ActorCriticAgent(
            buffer=buffer,
            obs_size=env.observation_size,
            n_actions=len(ApproachTaskActions),
            n_joints=env.arm_n_dofs,
            network_dim_size=experiment.dim_size,
            batch_size=experiment.batch_size,
            action_scale=experiment.action_scale,
            alpha=1e-3,
            lr=experiment.lr,
            gamma=GAMMA,
            target_update_freq=experiment.target_update_freq
        )
    else:
        epsilon: Callable[[int], float] = lambda t: max(
            EPS_END, EPS_START * (experiment.eps_decay**t)
        )
        
        policy_net: nn.Module = Dqn(env.observation_size, env.action_size, experiment.dim_size).to(env.device)
        target_net: nn.Module = Dqn(env.observation_size, env.action_size, experiment.dim_size).to(env.device)
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=experiment.lr)
        loss_fn = nn.MSELoss()

        agent: Agent = DQNAgent(
        agent_id=experiment.agent_id,
        n_dofs=env.arm_n_dofs,
        n_dof_actions=len(ApproachTaskActions),
        buffer=buffer,
        policy_net=policy_net,
        target_net=target_net,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epsilon=epsilon,
        gamma=experiment.gamma,
        batch_size=experiment.batch_size,
        target_update_freq=experiment.target_update_freq
    )

    analytics: Analytics = initialize_analytics(
        experiment=experiment,
        plot_freq=PLOT_FREQ,
        save_freq=SAVE_FREQ,
        debug=debug
    )

    runner: Runner = Runner(
        env=env,
        agent=agent,
        analytics=analytics,
        experiment=experiment)
    runner.run()


def main():

    custom_parameters = [
        {"name": "--headless", "type": bool, "default": False},
        {"name": "--debug", "type": bool, "default": False}
    ]

    args = gymutil.parse_arguments(custom_parameters=custom_parameters)
    sim_params: gymapi.SimParams = gymapi.SimParams()
    sim_params.use_gpu_pipeline = True
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
    sim_params.physx.num_threads = args.num_threads

    # arm asset configs
    arm_asset_options: gymapi.AssetOptions = gymapi.AssetOptions()
    arm_asset_options.fix_base_link = True
    arm_asset_options.disable_gravity = True
    arm_asset_options.armature = 0.01
    arm_asset_options.flip_visual_attachments = True
    arm_asset_options.collapse_fixed_joints = True
    arm_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

    # box assset configs
    box_asset_options: gymapi.AssetOptions = gymapi.AssetOptions()
    box_asset_options.density = 4.0
    box_asset_options.disable_gravity = True
    box_asset_options.fix_base_link = True

    # plane config
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
                
    gym: gymapi.Gym = gymapi.acquire_gym()

    sim_config: ArmAndBoxSimConfig = ArmAndBoxSimConfig(
        n_envs=N_ENVS,
        env_spacing=1.5,
        n_envs_per_row=int(math.sqrt(N_ENVS)),
        n_actors_per_env=2,
        arm_config=ArmConfig(
            hand_link="panda_link7",
            left_finger_link="panda_leftfinger",
            right_finger_link="panda_rightfinger",
            asset_config=AssetConfig(
                asset_root="assets",
                asset_file="urdf/franka_description/robots/franka_panda.urdf",
                asset_options=arm_asset_options,
            ),
            stiffness=8000,
            damping=4000,
            start_pose=gymapi.Transform(),
        ),
        box_config=BoxConfig(
            width=0.075,
            height=0.075,
            depth=0.075,
            friction=0.1,
            start_pose=gymapi.Transform(p=gymapi.Vec3(0.5, 0.5, 0.5)),
            asset_options=box_asset_options,
        ),
        compute_device=args.compute_device_id,
        graphics_device=args.graphics_device_id,
        physics_engine=gymapi.SIM_PHYSX,
        sim_params=sim_params,
        plane_params=plane_params,
        viewer_config=ViewerConfig(
            headless=args.headless, pos=gymapi.Vec3(3, 2, 2), look_at=gymapi.Vec3(-3, -2, -2)
        ),
    )

    action_scale = .1
    dist_thresh = 0.2

    task_config: ApproachTaskConfig = ApproachTaskConfig(
        action_scale=action_scale,
        gripper_offset_z=0,
        distance_threshold=dist_thresh,
        episode_length=N_STEPS,
        randomize=False,
        action_mode=ActionMode.DOF_TARGET
    )

    env = ApproachEnvDiscrete(sim_config, task_config, gym)

    dim = 64*2*2
    batch_size = N_ENVS
    for algo in [Algorithm.DQN, Algorithm.AC]:
        agent_id = 0
        for buffer_type in [BufferType.WINNING, BufferType.HER, BufferType.STANDARD]:
            for randomize in [False, True]:
                experiment = Experiment(
                    algo_name=algo.value,
                    n_epochs=N_EPOCHS,
                    n_episodes=N_EPISODES,
                    n_timesteps=N_STEPS,
                    dim_size=dim,
                    agent_id=agent_id,
                    n_envs=N_ENVS,
                    batch_size=batch_size,
                    lr=0.0001,
                    buffer_type=buffer_type,
                    eps_decay=EPS_DECAY,
                    randomize=randomize,
                    gamma=GAMMA,
                    action_scale=action_scale,
                    dist_thresh=dist_thresh,
                    target_update_freq=TARGET_UPDATE_FREQ,
                    replay_buffer_size=REPLAY_BUFFER_SIZE,
                    action_mode=ActionMode.DOF_POSITION
                )
                run_experiment(
                    env=env,
                    experiment=experiment,
                    debug=args.debug)
                agent_id += 1

    env.destroy()

if __name__ == "__main__":
    main()
