# Greg Attra
# 04.10.22

'''
Task for moving the end-effector to the target pose
'''

from distutils.command.config import config
import isaacgym.gymapi as gymapi
from lib.sims.arm_and_box import ArmAndBoxSim, ArmAndBoxSimConfig, ArmConfig, AssetConfig, initialize_sim, start_sim, step_sim


def run_approach(config: ArmAndBoxSimConfig) -> None:
    gym = gymapi.acquire_gym()

    sim: ArmAndBoxSim = initialize_sim(config, gym)

    start_sim(sim, gym)
    while True:
        step_sim(sim, gym)

