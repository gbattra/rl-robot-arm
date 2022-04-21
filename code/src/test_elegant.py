from elegantrl.envs.IsaacGym import IsaacVecEnv, IsaacOneEnv
import isaacgym
import torch  # import torch after import IsaacGym modules

from elegantrl.agents.AgentPPO import AgentPPO
from elegantrl.train.run import train_and_evaluate_mp
from elegantrl.train.config import Arguments

def main():
    env_func = IsaacVecEnv
    env_args = {
        'env_num': 4096,
        'env_name': 'Ant',
        'max_step': 1000,
        'state_dim': 60,
        'action_dim': 8,
        'if_discrete': False,
        'target_return': 14000.0,
        'headless': True,
        'device_id': 0,  # set by worker
        'if_print': False,  # if_print=False in default
    }

    args = Arguments(agent=AgentPPO, env_func=env_func, env_args=env_args)

    '''set one env for evaluator'''
    args.eval_env_func = IsaacOneEnv
    args.eval_env_args = args.env_args.copy()
    args.eval_env_args['env_num'] = 1

    '''set other hyper-parameters'''
    args.net_dim = 2 ** 9
    args.batch_size = args.net_dim * 4
    args.target_step = args.max_step
    args.repeat_times = 2 ** 4

    args.save_gap = 2 ** 9
    args.eval_gap = 2 ** 8
    args.eval_times1 = 2 ** 0
    args.eval_times2 = 2 ** 2

    args.worker_num = 1
    args.learner_gpus = 0
    train_and_evaluate_mp(args)

if __name__ == '__main__':
    main()