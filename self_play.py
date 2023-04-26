import os
import torch
import argparse
import numpy as np
from copy import deepcopy
from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import BasicLogger
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import BasePolicy, DQNPolicy, RandomPolicy, \
    MultiAgentPolicyManager


from enviroment.briscola_gym.briscola import BriscolaEnv
from tianshou.env import PettingZooEnv 

def env_func():
    return PettingZooEnv(BriscolaEnv())

def watch_selfplay(args, agent):
    test_envs = DummyVectorEnv([env_func for _ in range(args.test_num)])
    agent.set_eps(args.eps_test)
    policy = MultiAgentPolicyManager([agent, *[deepcopy(agent)]*4], env_func())
    policy.eval()
    collector = Collector(policy, test_envs)
    result = collector.collect(n_episode=2)
    rews = result["rews"]
    print(f"Final reward: {rews[:, 0].mean()}")


def selfplay(args, num_generation=2): # always train first agent, start from random policy
    train_envs = DummyVectorEnv([env_func for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([env_func for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # train_envs.seed(args.seed)
    # test_envs.seed(args.seed)

    # model
    env = PettingZooEnv(BriscolaEnv())
    args.state_shape = env.observation_space
    args.action_shape = env.action_space

    net = Net(args.state_shape, args.action_shape,
            hidden_sizes=args.hidden_sizes, device=args.device
            ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    agent_learn = DQNPolicy(
        net, optim, args.gamma, args.n_step,
        target_update_freq=args.target_update_freq)

    net_fixed = Net(args.state_shape, args.action_shape,
            hidden_sizes=args.hidden_sizes, device=args.device
            ).to(args.device)
    optim_fixed = torch.optim.SGD(net_fixed.parameters(), lr=0)
    agent_fixed = DQNPolicy(
        net_fixed, optim_fixed, args.gamma, args.n_step,
        target_update_freq=args.target_update_freq)


    # initialize agents and ma-policy
    agents = [agent_learn, agent_fixed, agent_fixed, agent_fixed, agent_fixed]
    policy = MultiAgentPolicyManager(agents, env)

    # collector
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    # policy.set_eps(1)
    train_collector.collect(n_step=args.batch_size * args.training_num)
    # log
    log_path = os.path.join(args.logdir, 'briscola5', 'dqn')
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = BasicLogger(writer)

    def save_fn(policy):
        pass

    def stop_fn(mean_rewards):
        return mean_rewards >= 0.1

    def train_fn(epoch, env_step):
        for i in range(5):
            policy.policies[f'player_{i}'].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        for i in range(5):
            policy.policies[f'player_{i}'].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, 0]

    # trainer
    for i_gen in range(num_generation):
        result = offpolicy_trainer(
            policy, train_collector, test_collector, args.epoch,
            args.step_per_epoch, args.step_per_collect, args.test_num,
            args.batch_size, train_fn=train_fn, test_fn=test_fn,
            stop_fn=stop_fn, save_fn=save_fn, update_per_step=args.update_per_step,
            logger=logger, test_in_train=False, reward_metric=reward_metric)
        
        for i in range(1,5):
            policy.policies[f'player_{i}'].load_state_dict(policy.policies['player_0'].state_dict()) #Copy current agent

        print('==={} Generations Evolved==='.format(i_gen+1))
    
    model_save_path = os.path.join(args.logdir, 'briscola5', 'dqn', 'policy_selfplay.pth')
    torch.save(policy.policies['player_0'].state_dict(), model_save_path)

    return result, policy.policies['player_0']



def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument(
        '--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win'
    )
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=10)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
    )
    parser.add_argument('--training-num', type=int, default=16)
    parser.add_argument('--test-num', type=int, default=16)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument(
        '--win-rate',
        type=int,
        default=60,
        help='the expected winning rate: Optimal policy can get 0.7'
    )
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='no training, '
        'watch the play of pre-trained models'
    )
    parser.add_argument(
        '--agent-id',
        type=int,
        default=2,
        help='the learned agent plays as the'
        ' agent_id-th player. Choices are 1 and 2.'
    )
    parser.add_argument(
        '--resume-path',
        type=str,
        default='',
        help='the path of agent pth file '
        'for resuming from a pre-trained agent'
    )
    parser.add_argument(
        '--opponent-path',
        type=str,
        default='',
        help='the path of opponent agent pth file '
        'for resuming from a pre-trained agent'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]




# train the agent and watch its performance in a match!
args = get_args()
result, agent = selfplay(args)
watch_selfplay(args, agent)