# Greg Attra
# 04.24.22

'''
Wrapper class for running experiment and saving stuff
'''

import os
from typing import Callable, Dict

from tqdm import trange
from elegantrl.envs.isaac_tasks.base.vec_task import Env
from lib.agents.agent import Agent
from lib.analytics.analytics import Analytics
from lib.analytics.plotting import plot_learning, save_data, save_plot
from lib.structs.experiment import Experiment


class Runner:
    def __init__(
            self,
            experiment: Experiment,
            analytics: Analytics,
            env: Env,
            agent: Agent) -> None:
        self.experiment = experiment
        self.analytics = analytics
        self.env = env
        self.agent = agent

        self.root = f'experiments/{experiment.algo_name}/{experiment.agent_id}'
        self.analytics_dir = f'{self.root}/analytics'
        self.figs_dir = f'{self.analytics_dir}/figs'
        self.data_dir = f'{self.analytics_dir}/data'
        self.models_dir = f'{self.root}/models'

        os.makedirs(self.root)
        os.makedirs(self.analytics_dir)
        os.makedirs(self.figs_dir)
        os.makedirs(self.data_dir)
        os.makedirs(self.models_dir)

    
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
                f'{self.models_dir}/{self.experiment.algo_name}_{self.experiment.agent_id}_{p}.pth')
            save_plot(self.anaytics, f'{self.figs_dir}/debug/{p}.png')

        self.agent.save_checkpoint(
            f'{self.models_dir}/{self.experiment.algo_name}_{self.experiment.agent_id}_final.pth')
        save_plot(self.anaytics, f'{self.figs_dir}/debug/final.png')
        save_data(self.anaytics, f'{self.data_dir}')
            
