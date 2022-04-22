# Greg Attra
# 04.20.22


'''
Approach box environment
'''

import math
from typing import Dict, Optional, Tuple
from gym import Env
from isaacgym import gymapi, gymtorch, torch_utils
import torch
import numpy as np
from lib.structs.arm_and_box_sim import ArmAndBoxSimConfig, AssetConfig

from lib.structs.approach_task import ApproachTaskActions, ApproachTaskConfig

from rl_games.common import vecenv


def load_asset(
    asset_config: AssetConfig, sim: gymapi.Sim, gym: gymapi.Gym
) -> gymapi.Asset:
    asset = gym.load_asset(
        sim,
        asset_config.asset_root,
        asset_config.asset_file,
        asset_config.asset_options,
    )
    return asset


class RLGPUEnv(vecenv.IVecEnv):
    def __init__(self, env: Env):
        self.env = env

    def step(self, action):
        obs, rwds, dones, misc = self.env.step(action)
        return {'states': obs}, rwds, dones, misc

    def reset(self):
        return {
            'states': self.env.reset()
        }
    
    def reset_done(self):
        return {
            'states': self.env.reset_done()
        }

    def get_number_of_agents(self):
        return self.env.n_env

    def get_env_info(self):
        info = {}
        info['action_space'] = self.env.action_space
        info['observation_space'] = self.env.observation_space

        if self.env.num_states > 0:
            info['state_space'] = self.env.state_space
            print(info['action_space'], info['observation_space'], info['state_space'])
        else:
            print(info['action_space'], info['observation_space'])

        return info
