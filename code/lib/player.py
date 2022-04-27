# Greg Attra
# 04.26.22

'''
Player which runs a trained agent in an environment
'''

import torch
from tqdm import trange
from lib.agents.agent import Agent
from lib.envs.approach_env import ApproachEnv
from lib.envs.env import Env


class Player:
    def __init__(
            self,
            agent: Agent,
            env: ApproachEnv,
            n_steps: int) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = agent
        self.env = env
        self.n_steps = n_steps

    def play(self) -> None:
        self.env.reset()
        for t in trange(self.n_steps, desc='Step'):
            s = self.env.compute_observations()
            a = self.agent.act(s, t)
            _, _, done, _ = self.env.step(a)

            # reset envs which have finished task
            self.env._reset_dones(torch.arange(self.env.n_envs, device=self.device)[done[:, 0]])
