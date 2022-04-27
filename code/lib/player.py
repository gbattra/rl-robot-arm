# Greg Attra
# 04.26.22

'''
Player which runs a trained agent in an environment
'''

from tqdm import trange
from lib.agents.agent import Agent, AgentMode
from lib.envs.env import Env


class Player:
    def __init__(
            self,
            agent: Agent,
            env: Env,
            n_steps: int) -> None:
        self.agent = agent
        self.env = env
        self.n_steps = n_steps

    def play(self) -> None:
        for t in trange(self.n_steps, desc='Step'):
            s = self.env.compute_observations()
            a = self.agent.act(s, t)

            self.env.step(a)
