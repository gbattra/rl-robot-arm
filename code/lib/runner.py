# Greg Attra
# 04.24.22

'''
Wrapper class for running experiment and saving stuff
'''

import json
import os
from typing import Callable, Dict
from attr import asdict

from tqdm import trange
from elegantrl.envs.isaac_tasks.base.vec_task import Env
from lib.agents.agent import Agent
from lib.analytics.analytics import Analytics
from lib.analytics.plotting import plot_learning, save_data, save_plot
from lib.structs.experiment import Experiment


class Runner:
    def __init__(
            self,
            name: str,
            experiment: Experiment,
            analytics: Analytics,
            env: Env,
            agent: Agent) -> None:
        self.experiment = experiment
        self.analytics = analytics
        self.env = env
        self.agent = agent
        self.name = name

        self.root = f'experiments/{name}/{experiment.algo_name}/{experiment.agent_id}'
        self.analytics_dir = f'{self.root}/analytics'
        self.figs_dir = f'{self.analytics_dir}/figs'
        self.data_dir = f'{self.analytics_dir}/data'
        self.models_dir = f'{self.root}/models'

        os.makedirs(self.root, exist_ok=True)
        os.makedirs(self.analytics_dir, exist_ok=True)
        os.makedirs(self.figs_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)

        with open(f'{self.root}/config.json', 'w') as f:
            json.dump(experiment.to_dict(), f, indent=4, sort_keys=True)
        
    def run(self) -> Dict:
        gt = 0
        for p in trange(self.experiment.n_epochs, desc="Epoch", leave=False):
            for e in trange(self.experiment.n_episodes, desc="Episode", leave=False):
                self.env.reset()
                for t in trange(self.experiment.n_timesteps, desc="Step", leave=False):
                    _, _, _, r, done, info = self.agent.step(self.env, gt)

                    plot_learning(self.analytics, r, done, info['loss'], p, e, t)

                    gt += 1

            self.agent.save_checkpoint(
                f'{self.models_dir}/{p}.pth')
            save_plot(self.analytics, f'{self.figs_dir}/{p}.png')

        self.agent.save_checkpoint(
            f'{self.models_dir}/final.pth')
        save_plot(self.analytics, f'{self.figs_dir}/final.png')
        save_data(self.analytics, f'{self.data_dir}')
            
