# Greg Attra
# 04.26.22

'''
Code to visualize the data
'''

import json
import os
from typing import Callable, Dict, List
from lib.buffers.buffer import BufferType
from lib.structs.approach_task import ActionMode

from lib.structs.experiment import Experiment

import matplotlib.pyplot as plt
import numpy as np

from lib.structs.plot_config import PlotConfig


def experiment_from_config(config: Dict) -> Experiment:
    return Experiment(
        algo_name=config.algo_name,
        gamma=config.gamma,
        dim_size=config.dim_size,
        agent_id=config.agent_id,
        n_envs=config.n_envs,
        n_epochs=config.n_epochs,
        n_episodes=config.n_episodes,
        n_timesteps=config.n_timesteps,
        batch_size=config.batch_size,
        lr=config.lr,
        buffer_type=BufferType(config.buffer_type),
        eps_decay=config.eps_decay,
        randomize=config.randomize,
        action_scale=config.action_scale,
        dist_thresh=config.dist_thresh,
        target_update_freq=config.target_update_freq,
        replay_buffer_size=config.replay_buffer_size,
        action_mode=ActionMode(config.action_mode)
    )


def load_data(experiment_filter: Callable[[Experiment], bool], datadirs: List[str]) -> np.ndarray:
    total_data = []
    for datadir in datadirs:
        for resultsdir in os.listdir(datadir):
            root = f'{datadir}/{resultsdir}'

            # load config from path
            config_path = f'{root}/config.json'
            f = open(config_path)
            config = json.load(f)
            
            experiment = experiment_from_config(config)

            # check that experiment passes any of the filters
            if not experiment_filter(experiment):
                return np.array([])

            # load numpy data from file
            data = np.load(f'{root}/analytics/data/total_rewards.py')
            total_data.append(data)
    
    return total_data


def plot_results(plot_config: PlotConfig):
    # setup plot
    plt.figure()
    plt.clf()

    # collect data
    for plot_component in plot_config.components:
        data = load_data(plot_component.filter_func, plot_component.datadirs)
        plt.plot(data, label=plot_component.label, color=plot_component.color)

    plt.xlabel(plot_config.xaxis)
    plt.ylabel(plot_config.yaxis)
    plt.legend()

    plt.show()
