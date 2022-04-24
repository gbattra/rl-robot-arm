# Greg Attra
# 04.24.22

'''
Wrapper class for running experiment and saving stuff
'''

from typing import Callable, Dict

from tqdm import trange
from elegantrl.envs.isaac_tasks.base.vec_task import Env
from lib.agents.agent import Agent
from lib.structs.experiment import Experiment


class Runner:
    def __init__(self) -> None:
        pass
    
    def run(
            self,
            experiment: Experiment,
            env: Env,
            agent: Agent,
            analytics: Callable) -> Dict:
        gt = 0
        for p in trange(experiment.n_epochs, desc="Epoch", leave=False):
            for e in trange(experiment.n_episodes, desc="Episode", leave=False):
                env.reset()
                for t in trange(experiment.n_timesteps, desc="Step", leave=False):
                    _, _, _, r, done, info = agent.step(env, gt)

                    analytics(r, done, info['loss'], p, e, t)

                    gt += 1
            agent.save_checkpoint(
                f'experiments/{experiment.algo_name}/{experiment.agent_id}/models/{experiment.algo_name}_{experiment.agent_id}_{p}.pth')
