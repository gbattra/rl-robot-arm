# Greg Attra
# 04.14.22

'''
Functions for plotting learning
'''

from dataclasses import dataclass
from lib.structs.experiment import Experiment
import torch


@dataclass
class Analytics:
    experiment: Experiment
    plot_freq: int
    save_freq: int
    env_timesteps: torch.Tensor
    epoch_rewards: torch.Tensor
    epoch_episodes: torch.Tensor
    epoch_losses: torch.Tensor
    epoch_episode_lengths: torch.Tensor
    debug: bool

    def __str__(self) -> str:
        return f'{self.experiment.algo_name} Agent {self.experiment.agent_id}'


def initialize_analytics(
        experiment: Experiment,
        plot_freq: int,
        save_freq: int,
        debug: bool = False) -> Analytics:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env_timesteps = torch.zeros((experiment.n_envs, 1)).to(device)
    epoch_rewards = torch.zeros((1, experiment.n_episodes * experiment.n_epochs)).to(device)
    epoch_episodes = torch.zeros((1, experiment.n_episodes * experiment.n_epochs)).to(device)
    epoch_losses = torch.zeros((1, experiment.n_episodes * experiment.n_epochs)).to(device)
    epoch_episode_lengths = torch.zeros((1, experiment.n_episodes * experiment.n_epochs)).to(device)

    analytics = Analytics(
        experiment=experiment,
        env_timesteps=env_timesteps,
        epoch_rewards=epoch_rewards,
        epoch_episodes=epoch_episodes,
        epoch_losses=epoch_losses,
        epoch_episode_lengths=epoch_episode_lengths,
        plot_freq=plot_freq,
        save_freq=save_freq,
        debug=debug
    )
    return analytics
