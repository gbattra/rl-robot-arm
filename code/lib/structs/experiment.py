# Greg Attra
# 04.23.22

'''
Object for holding experiments
'''

from dataclasses import dataclass
from typing import Dict

from cv2 import Algorithm

from lib.buffers.buffer import BufferType
from lib.structs.approach_task import ActionMode


@dataclass
class Experiment:
    algo: Algorithm
    gamma: float
    dim_size: int
    agent_id: int
    n_envs: int
    n_epochs: int
    n_episodes: int
    n_timesteps: int
    batch_size: int
    lr: float
    buffer_type: BufferType
    eps_decay: float
    randomize: bool
    action_scale: float
    dist_thresh: float
    target_update_freq: float
    replay_buffer_size: int
    action_mode: ActionMode

    def __str__(self) -> str:
        return f'LR: {self.lr} | Dim Size: {self.dim_size} | Action Scale: {self.action_scale} | Dist. Thresh.: {self.dist_thresh} '\
            + f'| Epsd. Length: {self.n_timesteps} | N Envs: {self.n_envs} \n ' \
            + f' | Batch Size: {self.batch_size} | {self.buffer_type} | Eps Decay: {self.eps_decay} | {self.action_mode} \n' \
            + f' | Randomize: {self.randomize}'

    def to_dict(self) -> Dict:
        return {
            'algo_name': self.algo.value,
            'gamma': self.gamma,
            'dim_size': self.dim_size,
            'agent_id': self.agent_id,
            'n_envs': self.n_envs,
            'n_epochs': self.n_epochs,
            'n_episodes': self.n_episodes,
            'n_timesteps': self.n_timesteps,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'buffer_type': self.buffer_type.value,
            'eps_decay': self.eps_decay,
            'randomize': self.randomize,
            'action_scale': self.action_scale,
            'dist_thresh': self.dist_thresh,
            'target_update_freq': self.target_update_freq,
            'replay_buffer_size': self.replay_buffer_size,
            'action_mode': self.action_mode.value
        }
