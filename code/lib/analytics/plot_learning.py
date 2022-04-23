# Greg Attra
# 04.14.22

'''
Functions for plotting learning
'''

from dataclasses import dataclass
from time import time
import torch
import matplotlib.pyplot as plt
import numpy as np

from lib.buffers.buffer import BufferType
from lib.structs.experiment import Experiment


@dataclass
class Analytics:
    experiment: Experiment
    plot_freq: int
    save_freq: int
    env_timesteps: torch.Tensor
    epoch_rewards: torch.Tensor
    epoch_episodes: torch.Tensor
    epoch_episode_lengths: torch.Tensor
    debug: bool


def initialize_analytics(
        experiment: Experiment,
        plot_freq: int,
        save_freq: int,
        debug: bool = False) -> Analytics:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env_timesteps = torch.zeros((experiment.n_envs, 1)).to(device)
    epoch_rewards = torch.zeros((1, experiment.n_episodes * experiment.n_epochs)).to(device)
    epoch_episodes = torch.zeros((1, experiment.n_episodes * experiment.n_epochs)).to(device)
    epoch_episode_lengths = torch.zeros((1, experiment.n_episodes * experiment.n_epochs)).to(device)

    analytics = Analytics(
        experiment=experiment,
        env_timesteps=env_timesteps,
        epoch_rewards=epoch_rewards,
        epoch_episodes=epoch_episodes,
        epoch_episode_lengths=epoch_episode_lengths,
        plot_freq=plot_freq,
        save_freq=save_freq,
        debug=debug
    )
    return analytics


def plot_learning(
        analytics: Analytics,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        loss: torch.Tensor,
        epoch: int,
        episode: int,
        timestep: int) -> None:
    cur_ep = (epoch * analytics.experiment.n_episodes) + episode
    cur_step = (cur_ep * analytics.experiment.n_timesteps) + timestep
    analytics.env_timesteps[:] += 1
    analytics.epoch_rewards[0, cur_ep] += rewards.sum().item()
    analytics.epoch_episodes[0, cur_ep] += dones.long().sum().item()

    if not analytics.debug:
        return

    if cur_ep > 1 and cur_step % analytics.save_freq == 0:
        save_analytics(analytics, 'debug')

    # plt.plot(epoch_episodes[e, :episode] / analytics.env_timesteps.shape[0])
    # plt.plot(analytics.epoch_episode_lengths[:epoch].mean().detach().numpy(), label='Epoch Avg Episode Length')

    if not cur_step % analytics.plot_freq == 0:
        return
    
    plt.figure(1, figsize=(9, 8))
    plt.clf()

    epoch_rewards = analytics.epoch_rewards.detach().cpu().numpy()
    epoch_episodes = analytics.epoch_episodes.detach().cpu().numpy()

    plt.plot(epoch_episodes[0, :cur_ep] / analytics.env_timesteps.shape[0], label=f'Episode Reward')
    # plt.plot(epoch_rewards[0, :cur_ep] / analytics.env_timesteps.shape[0], label=f'Episode Reward')

    desc = str(analytics.experiment)
    plt.xlabel(f'Episode \n {desc}')
    plt.ylabel('Reward')
    plt.title(f'DQN Agent {analytics.experiment.agent_id}')

    plt.legend()

    plt.pause(0.1)
    plt.show(block=False)

    if episode == analytics.experiment.n_episodes - 1 and timestep == analytics.experiment.n_timesteps - 1:
        plt.savefig(f'figs/debug/dqn_{time()}.png')


def save_analytics(analytics: Analytics, root: str) -> None:
    plt.figure(2, figsize=(9, 8))
    plt.clf()

    epoch_rewards = analytics.epoch_rewards.detach().cpu().numpy()
    plt.plot(epoch_rewards[0] / analytics.env_timesteps.shape[0], label=f'Episode Rewards')
    desc = str(analytics.experiment)
    plt.xlabel(f'Episode \n {desc}')
    plt.ylabel('Reward')
    plt.title(f'DQN Agent {analytics.experiment.agent_id}')

    plt.legend()

    plt.savefig(f'figs/{root}/DQN_{analytics.experiment.agent_id}_{time()}.png')
