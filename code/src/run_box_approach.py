# Greg Attra
# 04.20.22

'''
Box Approach env adhering to Elegant RL env for multiprocessing
'''


import isaacgym
from isaacgym import gymapi, gymutil

from elegantrl.train.config import Arguments
from elegantrl.agents.AgentDQN import AgentDQN
from elegantrl.train.run import train_and_evaluate_mp, train_and_evaluate
from elegantrl.envs.IsaacGym import IsaacVecEnv

import torch

from lib.tasks.approach.env import ApproachEnv


from typing import Callable

from lib.analytics.plot_learning import Analytics, initialize_analytics, plot_learning, save_analytics
from lib.rl.buffer import ReplayBuffer
from lib.rl.nn import DQN
from lib.sims.arm_and_box_sim import (
    ArmAndBoxSim,
    ArmAndBoxSimConfig,
    ArmConfig,
    AssetConfig,
    BoxConfig,
    ViewerConfig,
    destroy_sim,
    initialize_sim,
)

from torch import Tensor, nn
from lib.sims.sim import Sim
from lib.tasks.agent import DQNAgent
from lib.tasks.env import ApproachBoxEnv
from lib.tasks.task import (
    ApproachTask,
    ApproachTaskActions,
    ApproachTaskConfig,
    initialize_approach_task,
)

GAMMA: float = 0.99
LEARNING_RATE: float = 0.001

EPS_START: float = 1.0
EPS_END: float = 0.05
EPS_DECAY: float = 0.9999

REPLAY_BUFFER_SIZE: int = 10000000
TARGET_UPDATE_FREQ: int = 10000
BATCH_SIZE: int = 1000
DIM_SIZE: int = 500

N_EPOCHS: int = 1
N_EPISODES: int = 2
N_STEPS: int = 200

PLOT_FREQ: int = 100
SAVE_FREQ: int = 99

def run_experiment(
        sim_config: ArmAndBoxSimConfig,
        task_config: ApproachTaskConfig,
        dim: int,
        two_layers: bool,
        agent_id: int,
        n_envs: int):
    gym: gymapi.Gym = gymapi.acquire_gym()
    
    epsilon: Callable[[int], float] = lambda t: max(
        EPS_END, EPS_START * (EPS_DECAY**t)
    )
                
    env = ApproachEnv(sim_config, task_config, gym)

    policy_net: nn.Module = DQN(env.observation_size, env.action_size, dim, two_layers).to(env.device)
    target_net: nn.Module = DQN(env.observation_size, env.action_size, dim, two_layers).to(env.device)
    buffer: ReplayBuffer = ReplayBuffer(REPLAY_BUFFER_SIZE, env.observation_size, env.arm_n_dofs, env.n_envs)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    agent: DQNAgent = DQNAgent(
        agent_id=agent_id,
        n_dofs=env.arm_n_dofs,
        n_dof_actions=len(ApproachTaskActions),
        buffer=buffer,
        policy_net=policy_net,
        target_net=target_net,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epsilon=epsilon,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ
    )

    analytics: Analytics = initialize_analytics(
        agent_id=agent_id,
        n_epochs=N_EPOCHS,
        n_episodes=N_EPISODES,
        n_timesteps=N_STEPS,
        n_envs=n_envs,
        plot_freq=PLOT_FREQ,
        save_freq=SAVE_FREQ,
        lr=LEARNING_RATE,
        ep_length=N_STEPS,
        dim_size=dim,
        action_scale=.1,
        dist_thresh=.1,
        two_layers=two_layers,
        debug=True,
    )

    try:
        agent.train(
            env,
            N_EPOCHS,
            N_EPISODES,
            N_STEPS,
            lambda r, d, l, p, e, t: plot_learning(analytics, r,d,l,p,e,t))
        save_analytics(analytics)
        env.destroy()
    except KeyboardInterrupt:
        print("Exitting..")

def main():

    args = gymutil.parse_arguments()
    sim_params: gymapi.SimParams = gymapi.SimParams()
    sim_params.use_gpu_pipeline = True
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

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

    agent_id = 0
    for dim in [250, 512]:
        for n_envs in [1000, 2000]:
            for two_layers in [True, False]:
                agent_id += 1
                sim_config: ArmAndBoxSimConfig = ArmAndBoxSimConfig(
                    n_envs=n_envs,
                    env_spacing=1.5,
                    n_envs_per_row=10,
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
                        headless=True, pos=gymapi.Vec3(3, 2, 2), look_at=gymapi.Vec3(-3, -2, -2)
                    ),
                )

                task_config: ApproachTaskConfig = ApproachTaskConfig(
                    action_scale=0.05, gripper_offset_z=0.1, distance_threshold=0.1, max_episode_steps=200
                )

                run_experiment(sim_config, task_config, dim, two_layers, agent_id, n_envs)


if __name__ == "__main__":
    main()
