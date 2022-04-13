# Greg Attra
# 04.10.22
"""
Executable for running the approach task
"""

from isaacgym import gymapi, gymutil

from typing import Callable
from lib.rl.buffer import ReplayBuffer
from lib.rl.dqn import dqn
from lib.rl.nn import NeuralNetwork
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

import torch

from torch import Tensor, nn
from lib.tasks.approach_task import (
    ApproachTask,
    ApproachTaskActions,
    ApproachTaskConfig,
    approach_task_dqn_policy,
    approach_task_optimize_dqn,
    approach_task_network,
    initialize_approach_task,
    reset_approach_task,
    step_approach_task,
)

GAMMA: float = 0.99
LEARNING_RATE: float = 0.001

EPS_START: float = 1.0
EPS_END: float = 0.05
EPS_DECAY: float = 0.999

REPLAY_BUFFER_SIZE: int = 100000
TARGET_UPDATE_FREQ: int = 10000
BATCH_SIZE: int = 150

N_EPOCHS: int = 100
N_EPISODES: int = 1000
N_STEPS: int = 10000


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # plane config
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)

    sim_config: ArmAndBoxSimConfig = ArmAndBoxSimConfig(
        n_envs=100,
        env_spacing=1.5,
        n_envs_per_row=10,
        n_actors_per_env=2,
        arm_config=ArmConfig(
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
            start_pose=gymapi.Transform(p=gymapi.Vec3(0, 0.75, 0.75)),
            asset_options=box_asset_options,
        ),
        compute_device=args.compute_device_id,
        graphics_device=args.graphics_device_id,
        physics_engine=gymapi.SIM_PHYSX,
        sim_params=sim_params,
        plane_params=plane_params,
        viewer_config=ViewerConfig(
            headless=False, pos=gymapi.Vec3(3, 2, 2), look_at=gymapi.Vec3(-3, -2, -2)
        ),
    )

    task_config: ApproachTaskConfig = ApproachTaskConfig(action_scale=0.1)

    gym: gymapi.Gym = gymapi.acquire_gym()
    sim: ArmAndBoxSim = initialize_sim(sim_config, gym)
    task: ApproachTask = initialize_approach_task(task_config, sim, gym)

    policy_net: nn.Module = NeuralNetwork(approach_task_network(task)).to(device)
    target_net: nn.Module = NeuralNetwork(approach_task_network(task)).to(device)
    buffer: ReplayBuffer = ReplayBuffer(REPLAY_BUFFER_SIZE)

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()

    optimize: Callable[
        [nn.Module, nn.Module, ReplayBuffer], None
    ] = lambda buff, t: approach_task_optimize_dqn(
        task=task,
        buffer=buff,
        timestep=t,
        policy_net=policy_net,
        target_net=target_net,
        loss_fn=loss_fn,
        optimizer=optimizer,
        gamma=GAMMA,
        batch_size=BATCH_SIZE,
        target_update_freq=TARGET_UPDATE_FREQ,
    )

    epsilon: Callable[[int], float] = lambda t: max(
        EPS_END, EPS_START * (EPS_DECAY**t)
    )

    policy: Callable[[Tensor, int], Tensor] = approach_task_dqn_policy(
        task, policy_net, epsilon
    )

    results = dqn(
        reset_task=lambda: reset_approach_task(task, gym),
        step_task=lambda actions: step_approach_task(task, actions, gym),
        policy=policy,
        buffer=buffer,
        optimize=optimize,
        analytics=lambda r, d, p, e, t: None,
        n_epochs=N_EPOCHS,
        n_episodes=N_EPISODES,
        n_steps=N_STEPS,
    )

    try:
        pass
    except KeyboardInterrupt:
        print("Exitting..")

    destroy_sim(sim, gym)


if __name__ == "__main__":
    main()
