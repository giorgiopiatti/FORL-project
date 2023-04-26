import os
import torch
import argparse
import numpy as np
from copy import deepcopy
from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer, OffpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import BasePolicy, DQNPolicy, RandomPolicy, \
    MultiAgentPolicyManager


from enviroment.briscola_gym.briscola import BriscolaEnv
from tianshou.env import PettingZooEnv 

import shortuuid

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


def get_agent(args, is_fixed=False):
    net =  None
    optim = None

    if is_fixed:
         optim = None
    agent = None
    return agent

def selfplay(args): # always train first agent, start from random policy
    train_envs = DummyVectorEnv([env_func for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([env_func for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Env
    briscola = BriscolaEnv()
    env = PettingZooEnv(briscola)
    args.state_shape = briscola.observation_space_shape
    args.action_shape = briscola.action_space_shape

    # initialize agents and ma-policy
    id_agent_learning = 0
    agents = [get_agent(args) , get_agent(args, is_fixed=True) , get_agent(args, is_fixed=True), get_agent(args, is_fixed=True), get_agent(args, is_fixed=True)]
    policy = MultiAgentPolicyManager(agents, env)

    # collector
    train_collector = Collector(
        policy, train_envs,
        VectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    
    # Log
    args.algo_name = "dqn"
    log_name = os.path.join(args.algo_name, str(args.seed), "_", shortuuid.uuid()[:8])
    log_path = os.path.join(args.logdir, log_name)
    
    if args.logger == "wandb":
        logger = WandbLogger(
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
        pass
    
    def train_fn(epoch, env_step):
        for i in range(5):
            policy.policies[f'player_{i}'].set_eps(args.eps_train)

    def test_fn(epoch, env_step):
        for i in range(5):
            policy.policies[f'player_{i}'].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, id_agent_learning]

    # trainer
    for i_gen in range(args.num_generation):

        trainer = OffpolicyTrainer(policy, train_collector, test_collector, args.epoch,
            args.step_per_epoch, args.step_per_collect, episode_per_test=args.test_num,
            batch_size=args.batch_size, train_fn=train_fn, test_fn=test_fn, #stop_fn=stop_fn
            save_best_fn=save_best_fn, update_per_step=args.update_per_step,
            logger=logger, test_in_train=False, reward_metric=reward_metric)
        
        previous_best = None
        for epoch, epoch_stat, info in trainer:
            if previous_best is not None:
                if previous_best*1.05 < epoch_stat['test_reward']:
                    break
            previous_best = info['best_reward']

            # Rotate learning agent position
            old_id_agent_learning = id_agent_learning
            id_agent_learning = (id_agent_learning + 1) % 5
            policy.replace_policy(agents[old_id_agent_learning], env.agents[id_agent_learning])
            policy.replace_policy(agents[id_agent_learning], env.agents[old_id_agent_learning])
           
        
        for i in range(0,5):
            if i != id_agent_learning:
                policy.policies[f'player_{i}'].load_state_dict(policy.policies[f'player_{id_agent_learning}'].state_dict()) #Copy current agent

        print('==={} Generations Evolved==='.format(i_gen+1))
    
    model_save_path = os.path.join(args.logdir, 'briscola5', 'dqn', 'policy_selfplay.pth')
    torch.save(policy.policies['player_0'].state_dict(), model_save_path)

    return result, policy.policies['player_0']



def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    P = 64
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument(
        '--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win'
    )
    parser.add_argument('--n-step', type=int, default=5)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=P)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
    )
    parser.add_argument('--training-num', type=int, default=P)
    parser.add_argument('--num-generation', type=int, default=200)
    parser.add_argument('--test-num', type=int, default=400)
    parser.add_argument('--logdir', type=str, default='/cluster/scratch/piattigi/FORL_briscola/')
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
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="briscola")
    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]




# train the agent and watch its performance in a match!
args = get_args()
result, agent = selfplay(args)
watch_selfplay(args, agent)