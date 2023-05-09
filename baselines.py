import os
import torch
import argparse
import numpy as np
from copy import deepcopy
from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.env import SubprocVectorEnv, DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer, OffpolicyTrainer
from tianshou.policy import BasePolicy, DQNPolicy
from agents.heuristic_agent import HeuristicAgent


from enviroment.briscola_gym.briscola import BriscolaEnv
from tianshou.env import PettingZooEnv

import shortuuid
from multi_agents_rl.buffer import MultiAgentVectorReplayBuffer

from multi_agents_rl.collector import MultiAgentCollector
from multi_agents_rl.mapolicy import MultiAgentPolicyManager
from agents.random import RandomPolicy


def env_func():
    return PettingZooEnv(BriscolaEnv(use_role_ids=True, normalize_reward=False, save_raw_state=True, heuristic_ids=['callee', 'good_1', 'good_2', 'good_3']))


def get_random_agent(args):
    agent = RandomPolicy(
        device='cpu',
        observation_space=args.state_shape,
        action_space=args.action_shape,
    )
    return agent


def get_heuristic_agent(args):
    agent = HeuristicAgent(
        device='cpu',
        observation_space=args.state_shape,
        action_space=args.action_shape,
    )
    return agent


def selfplay(args):  # always train first agent, start from random policy
    train_envs = SubprocVectorEnv(
        [env_func for _ in range(args.num_parallel_env)])
    test_envs = SubprocVectorEnv(
        [env_func for _ in range(args.num_parallel_env)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Env
    briscola = BriscolaEnv(use_role_ids=True, normalize_reward=False,
                           save_raw_state=True,  heuristic_ids=['callee', 'good_1', 'good_2', 'good_3'])
    env = PettingZooEnv(briscola)
    args.state_shape = briscola.observation_space_shape
    args.action_shape = briscola.action_space_shape

    # initialize agents and ma-policy
    id_agent_learning = 0
    agents_heuristic = [get_random_agent(args),  get_heuristic_agent(args),
                        get_random_agent(args), get_random_agent(args), get_random_agent(args)]
    # {'caller': 0, 'callee': 1, 'good_1': 2, 'good_2': 3, 'good_3': 4}
    LERNING_AGENTS_ID = []
    policy = MultiAgentPolicyManager(agents_heuristic, env, LERNING_AGENTS_ID)

    collector = MultiAgentCollector(
        policy, DummyVectorEnv([lambda: PettingZooEnv(BriscolaEnv(use_role_ids=True, normalize_reward=False, save_raw_state=True, heuristic_ids=['callee'], render_mode='terminal'))]), LERNING_AGENTS_ID, exploration_noise=False)

    collector.collect(n_episode=1, render=0.1)

    test_collector = MultiAgentCollector(
        policy, test_envs, LERNING_AGENTS_ID, exploration_noise=True)

    res = test_collector.collect(n_episode=args.test_num)
    print('Callee heuristic', res['rew'],  res['rew_std'])

    agents_random = [get_random_agent(args),  get_random_agent(args),
                     get_random_agent(args), get_random_agent(args), get_random_agent(args)]
    # {'caller': 0, 'callee': 1, 'good_1': 2, 'good_2': 3, 'good_3': 4}
    LERNING_AGENTS_ID = []
    policy = MultiAgentPolicyManager(agents_random, env, LERNING_AGENTS_ID)

    test_collector = MultiAgentCollector(
        policy, test_envs, LERNING_AGENTS_ID, exploration_noise=True)

    res = test_collector.collect(n_episode=args.test_num)
    print('All Random', res['rew'], res['rew_std'])

    agents_good = [get_random_agent(args),  get_random_agent(args),
                   get_heuristic_agent(args), get_heuristic_agent(args), get_heuristic_agent(args)]
    # {'caller': 0, 'callee': 1, 'good_1': 2, 'good_2': 3, 'good_3': 4}
    LERNING_AGENTS_ID = []
    policy = MultiAgentPolicyManager(agents_good, env, LERNING_AGENTS_ID)

    test_collector = MultiAgentCollector(
        policy, test_envs, LERNING_AGENTS_ID, exploration_noise=True)

    res = test_collector.collect(n_episode=args.test_num)
    print('Good Random', res['rew'], res['rew_std'])


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-parallel-env', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=42)
    return parser


def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]


# train the agent and watch its performance in a match!
args = get_args()
selfplay(args)
