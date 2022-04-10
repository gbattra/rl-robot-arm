# Greg Attra
# 04.10.22

'''
Executable for running the approach task
'''

from lib.sims.arm_and_box import ArmAndBoxSimConfig, ArmConfig, AssetConfig
from lib.tasks.approach import run_approach
from isaacgym import gymapi, gymutil


def main():
    args = gymutil.parse_arguments()
    sim_params: gymapi.SimParams = gymapi.SimParams()
    sim_params.use_gpu_pipeline = True
    config: ArmAndBoxSimConfig = ArmAndBoxSimConfig(
        n_envs=1,
        env_spacing=10.,
        n_envs_per_row=1,
        arm_config=ArmConfig(
            asset_config=AssetConfig(
                asset_root='',
                asset_file='',
                asset_options=gymapi.AssetOptions()
            ),
            stiffness=8000,
            damping=4000,
            start_pose=gymapi.Transform()
        ),
        compute_device=args.compute_device_id,
        graphics_device=args.graphics_device_id,
        physics_engine=gymapi.SIM_PHYSX,
        sim_params=sim_params
    )

    run_approach(config)


if __name__ == '__main__':
    main()
