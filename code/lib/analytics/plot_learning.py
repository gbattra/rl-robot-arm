# Greg Attra
# 04.14.22

'''
Functions for plotting learning
'''

from dataclasses import dataclass
import torch
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class Analytics:
    analytics_freq: int
    env_timesteps: torch.Tensor
    epoch_rewards: torch.Tensor
    epoch_episodes: torch.Tensor
    epoch_episode_lengths: torch.Tensor


def initialize_analytics(
        n_epochs: int,
        n_episodes: int,
        n_timesteps: int,
        analytics_freq: int,
        n_envs: int) -> Analytics:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env_timesteps = torch.zeros((n_envs, 1)).to(device)
    epoch_rewards = torch.zeros((n_epochs, n_episodes)).to(device)
    epoch_episodes = torch.zeros((n_epochs, n_episodes)).to(device)
    epoch_episode_lengths = torch.zeros((n_epochs, n_episodes)).to(device)

    analytics = Analytics(
        analytics_freq=analytics_freq,
        env_timesteps=env_timesteps,
        epoch_rewards=epoch_rewards,
        epoch_episodes=epoch_episodes,
        epoch_episode_lengths=epoch_episode_lengths
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
    analytics.env_timesteps[:] += 1
    analytics.epoch_rewards[epoch, episode] = analytics.epoch_rewards[epoch, episode] + rewards.sum().item()
    analytics.epoch_episodes[epoch, episode] = analytics.epoch_episodes[epoch, episode] + dones.long().sum().item()

    # env_episode_lengths = analytics.env_timesteps[dones]
    # analytics.epoch_episode_lengths[epoch] += env_episode_lengths.sum().item()

    analytics.env_timesteps[dones] = 0

    if timestep % analytics.analytics_freq != 0:
        return

    plt.figure(1)
    plt.clf()

    # epoch_rewards = analytics.epoch_rewards.detach().cpu().numpy()
    epoch_episodes = analytics.epoch_episodes.detach().cpu().numpy()
    for e in range(epoch+1):
        # plt.plot(epoch_rewards[e, :episode] / analytics.env_timesteps.shape[0], label=f'Epoch {e} Reward')
        plt.plot(epoch_episodes[e, :episode] / analytics.env_timesteps.shape[0])
    # plt.plot(analytics.epoch_episode_lengths[:epoch].mean().detach().numpy(), label='Epoch Avg Episode Length')

    # if len(d_t) >= 100:
    #         means = d_t.unfold(0, 100, 1).mean(1).view(-1)
    #         means = torch.cat((torch.zeros(99), means))
    #         plt.plot(means.numpy(), label=label)
    plt.pause(0.1)
    # plt.legend()
    plt.show(block=False)
