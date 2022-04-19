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


@dataclass
class Analytics:
    agent_id: int
    n_epochs: int
    n_episodes: int
    n_timesteps: int
    plot_freq: int
    save_freq: int
    env_timesteps: torch.Tensor
    epoch_rewards: torch.Tensor
    epoch_episodes: torch.Tensor
    epoch_episode_lengths: torch.Tensor
    lr: float
    ep_length: int
    dim_size: float
    action_scale: float
    dist_thresh: float
    debug: bool


def initialize_analytics(
        agent_id: int,
        n_epochs: int,
        n_episodes: int,
        n_timesteps: int,
        n_envs: int,
        plot_freq: int,
        save_freq: int,
        lr: float,
        ep_length: int,
        dim_size: float,
        action_scale: float,
        dist_thresh: float,
        debug: bool = False) -> Analytics:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env_timesteps = torch.zeros((n_envs, 1)).to(device)
    epoch_rewards = torch.zeros((1, n_episodes * n_epochs)).to(device)
    epoch_episodes = torch.zeros((1, n_episodes * n_epochs)).to(device)
    epoch_episode_lengths = torch.zeros((1, n_episodes * n_epochs)).to(device)

    analytics = Analytics(
        n_epochs=n_epochs,
        n_episodes=n_episodes,
        n_timesteps=n_timesteps,
        plot_freq=plot_freq,
        save_freq=save_freq,
        env_timesteps=env_timesteps,
        epoch_rewards=epoch_rewards,
        epoch_episodes=epoch_episodes,
        epoch_episode_lengths=epoch_episode_lengths,
        lr=lr,
        ep_length=ep_length,
        dim_size=dim_size,
        action_scale=action_scale,
        dist_thresh=dist_thresh,
        agent_id=agent_id,
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
    gt = (epoch * analytics.n_episodes) + episode
    analytics.env_timesteps[:] += 1
    analytics.epoch_rewards[0, gt] += rewards.sum().item()
    analytics.epoch_episodes[0, gt] += dones.long().sum().item()

    analytics.env_timesteps[dones] = 0

    if timestep % analytics.plot_freq != 0:
        return

    plt.figure(1)
    plt.clf()

    epoch_rewards = analytics.epoch_rewards.detach().cpu().numpy()
    epoch_episodes = analytics.epoch_episodes.detach().cpu().numpy()
    plt.plot(epoch_rewards[0, :gt] / analytics.env_timesteps.shape[0], label=f'Episode Reward')
    # plt.plot(epoch_episodes[e, :episode] / analytics.env_timesteps.shape[0])
    # plt.plot(analytics.epoch_episode_lengths[:epoch].mean().detach().numpy(), label='Epoch Avg Episode Length')

    if analytics.debug:
        plt.pause(0.1)
        plt.show(block=False)

    if episode == analytics.n_episodes - 1 and timestep == analytics.n_timesteps - 1:
        plt.savefig(f'figs/debug/dqn_{time()}.png')


def save_analytics(analytics: Analytics) -> None:
    plt.figure(figsize=(9, 7))
    plt.clf()

    epoch_rewards = analytics.epoch_rewards.detach().cpu().numpy()
    plt.plot(epoch_rewards[0] / analytics.env_timesteps.shape[0], label=f'Episode Rewards')
    desc = f'LR: {analytics.lr} | Dim Size: {analytics.dim_size} | Action Scale: {analytics.action_scale} | Dist. Thresh.: {analytics.dist_thresh} | Epsd. Length: {analytics.ep_length}'
    plt.xlabel(f'Episode \n {desc}')
    plt.ylabel('Reward')
    plt.title(f'DQN Agent {analytics.agent_id}')

    plt.legend()

    plt.savefig(f'figs/results/DQN_{analytics.agent_id}_{time()}.png')
