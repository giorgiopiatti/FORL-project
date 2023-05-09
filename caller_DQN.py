import os
import torch
import argparse
import numpy as np
from copy import deepcopy
from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer, OffpolicyTrainer
from tianshou.policy import BasePolicy, DQNPolicy, RandomPolicy


from enviroment.briscola_gym.briscola import BriscolaEnv
from tianshou.env import PettingZooEnv

import shortuuid
from multi_agents_rl.buffer import MultiAgentVectorReplayBuffer

from multi_agents_rl.collector import MultiAgentCollector
from multi_agents_rl.mapolicy import MultiAgentPolicyManager
from multi_agents_rl.observation_wrapper import VectorEnvNormObs

def env_func():
    return PettingZooEnv(BriscolaEnv(use_role_ids=True, normalize_reward=False))


def get_agent(args, is_fixed=False):
    net = Net(args.state_shape, args.action_shape,
              hidden_sizes=args.hidden_sizes, device=args.device
              ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    if is_fixed:
        optim = torch.optim.SGD(net.parameters(), lr=0)
    agent = DQNPolicy(
        net, optim, args.gamma, args.n_step,
        target_update_freq=args.target_update_freq)
    return agent


def get_random_agent(args):
    agent = RandomPolicy(
        observation_space=args.state_shape,
        action_space=args.action_shape,
    )
    return agent



def selfplay(args):  # always train first agent, start from random policy
    train_envs = SubprocVectorEnv([env_func for _ in range(args.num_parallel_env)])
    test_envs = SubprocVectorEnv([env_func for _ in range(args.num_parallel_env)])
    
    train_envs = VectorEnvNormObs(train_envs)
    test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
    test_envs.set_obs_rms(train_envs.get_obs_rms())
    
    # seed

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Env
    briscola = BriscolaEnv(use_role_ids=True, normalize_reward=False)
    env = PettingZooEnv(briscola)
    args.state_shape = briscola.observation_space_shape
    args.action_shape = briscola.action_space_shape

    # initialize agents and ma-policy
    id_agent_learning = 0
    agent_caller = get_agent(args)
    agents = [agent_caller,  get_random_agent(args),
              get_random_agent(args), get_random_agent(args), get_random_agent(args)]
    # {'caller': 0, 'callee': 1, 'good_1': 2, 'good_2': 3, 'good_3': 4}
    LERNING_AGENTS_ID = ['caller']
    policy = MultiAgentPolicyManager(agents, env, LERNING_AGENTS_ID)

     # collector
    train_collector = MultiAgentCollector(
        policy, train_envs, LERNING_AGENTS_ID,
        MultiAgentVectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True)
    test_collector = MultiAgentCollector(
        policy, test_envs, LERNING_AGENTS_ID, exploration_noise=True)

    # Log
    args.algo_name = "DQN_caller_vs_random"
    log_name = os.path.join(
        args.algo_name, f'{str(args.seed)}_{shortuuid.uuid()[:8]}')
    log_path = os.path.join(args.logdir, log_name)
    os.makedirs(log_path, exist_ok=True)
    print(f'Saving results in path: {log_path}')
    print('-----------------------------------')
    if args.logger == "wandb":
        logger = WandbLogger(
            train_interval=1,
            update_interval=1,
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        model_save_path = os.path.join(log_path, f'caller_policy_best_epoch.pth')
        torch.save(policy.policies['caller'].state_dict(), model_save_path)

    def train_fn(epoch, env_step):
        if epoch <= 50:
            policy.policies['caller'].set_eps(args.eps_train)
        else:
            policy.policies['caller'].set_eps(0.7)

    def test_fn(epoch, env_step):
        policy.policies['caller'].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, id_agent_learning]

    trainer = OffpolicyTrainer(policy,
                               train_collector,
                               test_collector,
                               max_epoch=args.epoch,
                               step_per_epoch=args.step_per_epoch,
                               step_per_collect=0, # NOTE this is keep but ignore by our collector
                               episode_per_collect=args.episode_per_collect,
                               episode_per_test=args.test_num,
                               batch_size=args.batch_size,
                               save_best_fn=save_best_fn,
                               logger=logger,
                               update_per_step=args.update_per_step,
                               reward_metric=reward_metric,
                               test_fn=test_fn,
                               train_fn=train_fn,
                               test_in_train=False)
    trainer.run()

    model_save_path = os.path.join(log_path, 'caller_policy_last_epoch.pth')
    torch.save(policy.policies['caller'].state_dict(), model_save_path)

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=10)
    NUM_GAMES = 15000
    parser.add_argument('--buffer-size', type=int, default=8*NUM_GAMES)
    parser.add_argument('--step-per-epoch', type=int, default=5*8*NUM_GAMES)
    parser.add_argument('--episode-per-collect', type=int, default=NUM_GAMES)
    parser.add_argument('--update-per-step', type=float,
                        default=0.01)  # NOTE 8*1500 gradient steps
    parser.add_argument('--batch-size', type=int, default=512)
    

    parser.add_argument('--num-parallel-env', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.0)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument(
        '--gamma', type=float, default=1.0, help='a smaller gamma favors earlier win'
    )
    parser.add_argument('--n-step', type=int, default=1)
    parser.add_argument('--target-update-freq', type=int, default=1.0)
    parser.add_argument('--hidden-sizes', type=int, nargs='*', default=[64,64])
   
   
    parser.add_argument('--logdir', type=str,
                        default='log/')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="FORL_briscola")
    parser.add_argument("--resume-id", type=str, default=None)
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


# train the agent and watch its performance in a match!
args = get_args()
selfplay(args)
