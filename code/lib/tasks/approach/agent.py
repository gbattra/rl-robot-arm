# Greg Attra
# 04.20.22

'''
Agent implementation for ElegantRL
'''

import torch

from elegantrl.agents.AgentBase import AgentBase


class DQNAgent(AgentBase):
    def __init__(self, net_dim: int, state_dim: int, action_dim: int, gpu_id=0, args=None):
        super().__init__(net_dim, state_dim, action_dim, gpu_id, args)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent_id = args.agent_id
        self.her = args.her
        self.n_dofs = args.n_dofs
        self.n_dof_actions = args.n_dof_actions
        self.buffer = args.buffer
        self.policy_net = args.policy_net
        self.target_net = args.target_net
        self.loss_fn = args.loss_fn
        self.optimizer = args.optimizer
        self.epsilon = args.epsilon
        self.gamma = args.gamma
        self.batch_size = args.batch_size
        self.target_update_freq = args.target_update_freq

    def explore_vec_env(self, env, target_step) -> list:
        return super().explore_vec_env(env, target_step)