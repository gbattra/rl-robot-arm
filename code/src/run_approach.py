# Greg Attra
# 04.10.22

'''
Executable for running the approach task
'''

from isaacgym import gymapi, gymutil

from typing import Callable
from lib.rl.buffer import ReplayBuffer
from lib.rl.dqn import dqn, train_dqn
from lib.rl.nn import NeuralNetwork, approach_network
from lib.sims.arm_and_box_sim import ArmAndBoxSim, ArmAndBoxSimConfig, ArmConfig, AssetConfig, BoxConfig, ViewerConfig, destroy_sim, initialize_sim

import numpy as np
import torch

from torch import Tensor, nn
from lib.tasks.approach_task import ApproachTask, ApproachTaskActions, initialize_approach_task, reset_approach_task, step_approach_task
from lib.tasks.task import Task


GAMMA: float = .99
LEARNING_RATE: float = 0.001

EPS_START: float = 1.
EPS_END: float = 0.05
EPS_DECAY: float = 0.999

REPLAY_BUFFER_SIZE: int = 100000
TARGET_UPDATE_FREQ: int = 10000
BATCH_SIZE: int = 150

N_EPOCHS: int = 100
N_EPISODES: int = 1000
N_STEPS: int = 10000


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    args = gymutil.parse_arguments()
    
    sim_params: gymapi.SimParams = gymapi.SimParams()
    sim_params.use_gpu_pipeline = True
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(.0, .0, -9.8)

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
    box_asset_options.density = 4.

    # plane config
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    
    config: ArmAndBoxSimConfig = ArmAndBoxSimConfig(
        n_envs=10,
        env_spacing=1.5,
        n_envs_per_row=5,
        arm_config=ArmConfig(
            asset_config=AssetConfig(
                asset_root='assets',
                asset_file='urdf/franka_description/robots/franka_panda.urdf',
                asset_options=arm_asset_options
            ),
            stiffness=8000,
            damping=4000,
            start_pose=gymapi.Transform()
        ),
        box_config=BoxConfig(
            width=0.075,
            height=0.075,
            depth=0.075,
            friction=0.1,
            start_pose=gymapi.Transform(
                p=gymapi.Vec3(0, .75, .75)
            ),
            asset_options=box_asset_options
        ),
        compute_device=args.compute_device_id,
        graphics_device=args.graphics_device_id,
        physics_engine=gymapi.SIM_PHYSX,
        sim_params=sim_params,
        plane_params=plane_params,
        viewer_config=ViewerConfig(
            pos=gymapi.Vec3(3, 2, 2),
            look_at=gymapi.Vec3(-3, -2, -2)
        )
    )

    gym: gymapi.Gym = gymapi.acquire_gym()
    sim: ArmAndBoxSim = initialize_sim(config, gym)
    task: ApproachTask = initialize_approach_task(sim, gym)

    policy_net: nn.Module = NeuralNetwork(approach_network())
    target_net: nn.Module = NeuralNetwork(approach_network())
    buffer: ReplayBuffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
    
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    train: Callable[[nn.Module, nn.Module, ReplayBuffer], None] = \
        lambda p_net, t_net, buff: train_dqn(
            policy_net=p_net,
            target_net=t_net,
            buffer=buff,
            loss_fn=loss_fn,
            optimizer=optimizer,
            gamma=GAMMA,
            batch_size=BATCH_SIZE
        )

    epsilon = lambda t: max(EPS_END, EPS_START * (EPS_DECAY ** t))

    results = dqn(
        reset_task=lambda: reset_approach_task(task, gym),
        step_task=lambda actions: step_approach_task(task, actions, gym),
        policy_net=policy_net,
        target_net=target_net,
        buffer=buffer,
        train=train,
        epsilon=epsilon,
        analytics=lambda r, d, p, e, t: None,
        n_epochs=N_EPOCHS,
        n_episodes=N_EPISODES,
        n_steps=N_STEPS
    )

    try:
        pass
    except KeyboardInterrupt:
        print('Exitting..')

    destroy_sim(sim, gym)


if __name__ == '__main__':
    main()
