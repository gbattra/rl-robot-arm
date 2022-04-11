# Greg Attra
# 04.10.22

'''
Executable for running the approach task
'''

from lib.sims.arm_and_box import ArmAndBoxSim, ArmAndBoxSimConfig, ArmConfig, AssetConfig, ViewerConfig, destroy_sim, initialize_sim, start_sim, step_sim
from isaacgym import gymapi, gymutil
import numpy as np


def main():
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
    arm_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_VEL

    # plane config
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    
    config: ArmAndBoxSimConfig = ArmAndBoxSimConfig(
        n_envs=1,
        env_spacing=10.,
        n_envs_per_row=1,
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

    start_sim(sim, gym)
    while True:
        try:
            step_sim(sim, gym)
        except KeyboardInterrupt:
            print('Exitting..')
            break

    destroy_sim(sim, gym)


if __name__ == '__main__':
    main()
