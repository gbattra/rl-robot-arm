# Greg Attra
# 04.26.22

'''
Evaluate / watch a trained model perform
'''
import json
from isaacgym import gymapi, gymutil
from lib.agents.ac_agent import ActorCriticPlayer
from lib.analytics.visualize import experiment_from_config
from lib.envs.approach_env import ApproachEnvDiscrete
from lib.player import Player
from lib.structs.algorithm import Algorithm
from lib.structs.arm_and_box_sim import (
    ArmAndBoxSimConfig,
    ArmConfig,
    AssetConfig,
    BoxConfig,
    ViewerConfig,
)

from lib.structs.approach_task import (
    ApproachTaskActions,
    ApproachTaskConfig,
)
from lib.networks.dqn import Dqn
from torch import nn
from lib.agents.dqn_agent import DQNPlayer

N_EPS: int = 1000
N_STEPS: int = 100

def main():
    custom_parameters = [
        {"name": "--dir", "type": str},
        {"name": "--algo", "type": str},
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
        n_envs=1,
        env_spacing=1.5,
        n_envs_per_row=1,
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
            headless=False, pos=gymapi.Vec3(3, 2, 2), look_at=gymapi.Vec3(-3, -2, -2)
        ),
    )

    config_path = f'{args.dir}/config.json'
    f = open(config_path)
    config = json.load(f)
    
    experiment = experiment_from_config(config)

    task_config: ApproachTaskConfig = ApproachTaskConfig(
        action_scale=experiment.action_scale,
        gripper_offset_z=0,
        distance_threshold=experiment.dist_thresh,
        episode_length=experiment.n_timesteps,
        randomize=experiment.randomize,
        action_mode=experiment.action_mode
    )

    env = ApproachEnvDiscrete(sim_config, task_config, gym)

    if Algorithm(args.algo) == Algorithm.DQN:
        policy_net: nn.Module = Dqn(env.observation_size, env.action_size, experiment.dim_size).to(env.device)
        agent_player = DQNPlayer(env.arm_n_dofs, len(ApproachTaskActions), policy_net)
    else:
        agent_player = ActorCriticPlayer(
            len(ApproachTaskActions),
            env.arm_n_dofs,
            env.observation_size,
            experiment.dim_size,
            experiment.action_scale)
            
    agent_player.load(f'{args.dir}/models/final.pth')
    player = Player(agent_player, env, N_EPS, N_STEPS)
    player.play()


if __name__ == '__main__':
    main()
