# Greg Attra
# 04.10.22

'''
Executable for running the approach task
'''

from lib.sims.arm_and_box import ArmAndBoxSimConfig, ArmConfig, AssetConfig
from lib.tasks.approach import run_approach
from isaacgym import gymapi


config: ArmAndBoxSimConfig(
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
    compute_device=0,
    graphics_device=0,
    physics_engine=gymapi.SIM_PHYSX,
    sim_params=gymapi.SimParams()
)


def main():
    run_approach(config)


if __name__ == '__main__':
    main()
