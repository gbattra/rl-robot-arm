# Greg Attra
# 04.20.22

'''
Box Approach env adhering to Elegant RL env for multiprocessing
'''


from isaacgym import gymapi, gymutil

import torch

from typing import Callable

from lib.analytics.plot_learning import Analytics, initialize_analytics, plot_learning, save_analytics
from lib.buffers.win_buffer import WinBuffer
from lib.envs.approach_box_env import ApproachEnv
from lib.buffers.buffer import BufferType, ReplayBuffer
from lib.networks.nn import Dqn
from lib.structs.arm_and_box_sim import (
    ArmAndBoxSimConfig,
    ArmConfig,
    AssetConfig,
    BoxConfig,
    ViewerConfig,
)

from torch import nn
from lib.agents.dqn_agent import DQNAgent
from lib.structs.approach_task import (
    ApproachTaskActions,
    ApproachTaskConfig,
)

GAMMA: float = 0.99
LEARNING_RATE: float = 0.001

EPS_START: float = 1.0
EPS_END: float = 0.05
EPS_DECAY: float = 0.9999

REPLAY_BUFFER_SIZE: int = 1000000
TARGET_UPDATE_FREQ: int = 10000
BATCH_SIZE: int = 250
DIM_SIZE: int = 500
N_ENVS: int = 100

N_EPOCHS: int = 3
N_EPISODES: int = 100
N_STEPS: int = 450

PLOT_FREQ: int = 100
SAVE_FREQ: int = 99

def run_experiment(
        env: ApproachEnv,
        dim: int,
        two_layers: bool,
        agent_id: int,
        n_envs: int,
        batch_size: int,
        buffer_type: BufferType):
    
    epsilon: Callable[[int], float] = lambda t: max(
        EPS_END, EPS_START * (EPS_DECAY**t)
    )

    policy_net: nn.Module = Dqn(env.observation_size, env.action_size, dim, two_layers).to(env.device)
    target_net: nn.Module = Dqn(env.observation_size, env.action_size, dim, two_layers).to(env.device)

    if buffer_type == BufferType.STANDARD:
        buffer: ReplayBuffer = ReplayBuffer(REPLAY_BUFFER_SIZE, env.observation_size, env.arm_n_dofs, env.n_envs)
    else:
        buffer: ReplayBuffer = WinBuffer(REPLAY_BUFFER_SIZE, env.observation_size, env.arm_n_dofs, env.n_envs, 0.25)

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
        action_scale=env.action_scale,
        dist_thresh=env.distance_threshold,
        two_layers=two_layers,
        batch_size=batch_size,
        buffer_type=buffer_type,
        debug=True,
    )

    agent.train(
        env,
        N_EPOCHS,
        N_EPISODES,
        N_STEPS,
        lambda r, d, l, p, e, t: plot_learning(analytics, r,d,l,p,e,t))
    save_analytics(analytics)


def main():

    custom_parameters = [
        {"name": "--headless", "type": bool, "default": False}
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
        n_envs_per_row=10,
        n_actors_per_env=2,
        arm_config=ArmConfig(
            hand_link="panda_link7",
            left_finger_link="panda_leftfinger",
            right_finger_link="panda_rightfinger",
            # hand_link='lbr_iiwa_link_7',
            # left_finger_link='lbr_iiwa_link_7',
            # right_finger_link='lbr_iiwa_link_7',
            asset_config=AssetConfig(
                asset_root="assets",
                # asset_file='urdf/kuka_iiwa/model.urdf',
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

    task_config: ApproachTaskConfig = ApproachTaskConfig(
        action_scale=0.1, gripper_offset_z=0, distance_threshold=.1
    )

    env = ApproachEnv(sim_config, task_config, gym)

    agent_id = 0
    for dim in [250, 512]:
        for batch_size in [250, 500]:
            for buffer_type in [BufferType.STANDARD, BufferType.WINNING]:
                agent_id += 1
                run_experiment(env, dim, False, agent_id, N_ENVS, batch_size, buffer_type)

    env.destroy()

if __name__ == "__main__":
    main()
