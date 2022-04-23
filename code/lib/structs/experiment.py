# Greg Attra
# 04.23.22

'''
Object for holding experiments
'''

from dataclasses import dataclass

from lib.buffers.buffer import BufferType


@dataclass
class Experiment:
    gamma: float
    dim_size: int
    two_layers: bool
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

    def __str__(self) -> str:
        return f'LR: {self.lr} | Dim Size: {self.dim_size} | Action Scale: {self.action_scale} | Dist. Thresh.: {self.dist_thresh} '\
            + f'| Epsd. Length: {self.n_timesteps} | N Envs: {self.n_envs} \n | Two Layers: {self.two_layers}' \
            + f' | Batch Size: {self.batch_size} | {self.buffer_type} | Eps Decay: {self.eps_decay}'
