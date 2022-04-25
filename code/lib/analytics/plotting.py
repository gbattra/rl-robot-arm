# Greg Attra
# 04.24.22

'''
Plot stuff
'''

from time import time
import numpy as np
import torch
import matplotlib.pyplot as plt

from lib.analytics.analytics import Analytics


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
    analytics.epoch_rewards[0, cur_ep] += rewards.detach().sum().item()
    analytics.epoch_episodes[0, cur_ep] += dones.detach().long().sum().item()
    analytics.epoch_losses[0, cur_ep] += loss.detach().item()

    if not analytics.debug:
        return

    # plt.plot(epoch_episodes[e, :episode] / analytics.env_timesteps.shape[0])
    # plt.plot(analytics.epoch_episode_lengths[:epoch].mean().detach().numpy(), label='Epoch Avg Episode Length')

    if not cur_step % analytics.plot_freq == 0:
        return
    
    plt.figure(1, figsize=(9, 10))
    plt.clf()

    epoch_rewards = analytics.epoch_rewards.detach().cpu().numpy()
    epoch_episodes = analytics.epoch_episodes.detach().cpu().numpy()
    epoch_losses = analytics.epoch_losses.detach().cpu().numpy()

    plt.plot(epoch_episodes[0, :cur_ep] / analytics.env_timesteps.shape[0], label=f'Episode Reward')
    plt.plot(epoch_losses[0, :cur_ep] / analytics.env_timesteps.shape[0], linestyle='dotted', label=f'Episode Losses')
    # plt.plot(epoch_rewards[0, :cur_ep] / analytics.env_timesteps.shape[0], label=f'Episode Reward')

    desc = str(analytics.experiment)
    plt.xlabel(f'Episode \n {desc}')
    plt.ylabel('Reward')
    plt.title(str(analytics))

    plt.legend()

    plt.pause(0.1)
    plt.show(block=False)


def save_plot(analytics: Analytics, filename: str) -> None:
    # plt.figure(2, figsize=(9, 10))
    plt.clf()

    epoch_rewards = analytics.epoch_rewards.detach().cpu().numpy()
    epoch_losses = analytics.epoch_losses.detach().cpu().numpy()
    plt.plot(epoch_rewards[0] / analytics.env_timesteps.shape[0], label=f'Episode Rewards')
    plt.plot(epoch_losses[0] / analytics.env_timesteps.shape[0], linestyle='dotted', label=f'Episode Losses')
    plt.xlabel(f'Episode')
    plt.ylabel('Reward')
    plt.title(str(analytics))

    plt.legend()

    plt.savefig(filename)


def save_data(analytics: Analytics, filepath: str):
    total_rewards = analytics.epoch_rewards.cpu().numpy()
    np.save(f'{filepath}/total_rewards', total_rewards)
