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
from tianshou.utils.net.discrete import NoisyLinear
from tianshou.trainer import offpolicy_trainer, OffpolicyTrainer
from tianshou.data import Collector, VectorReplayBuffer, PrioritizedVectorReplayBuffer
from tianshou.policy import BasePolicy, RainbowPolicy, RandomPolicy, \
    MultiAgentPolicyManager

from torch import nn
from enviroment.briscola_gym.briscola import BriscolaEnv
from tianshou.env import PettingZooEnv

import shortuuid


def env_func():
    return PettingZooEnv(BriscolaEnv(use_role_ids=True, normalize_reward=False))


# def watch_selfplay(args, agent):
#     test_envs = DummyVectorEnv([env_func for _ in range(args.test_num)])
#     agent.set_eps(args.eps_test)
#     policy = MultiAgentPolicyManager([agent, *[deepcopy(agent)]*4], env_func())
#     policy.eval()
#     collector = Collector(policy, test_envs)
#     result = collector.collect(n_episode=2)
#     rews = 120*result["rews"]
#     print(f"Final reward: {rews[:, 0].mean()}")


def get_agent(args, is_fixed=False):
    def noisy_linear(x, y):
        if not args.no_noisy:
            return nn.Linear(x,y)
        else:
            return NoisyLinear(x, y, args.noisy_std)

    net = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
        softmax=True,
        num_atoms=args.num_atoms,
        dueling_param=None if args.no_dueling else ({
            "linear_layer": noisy_linear
        }, {
            "linear_layer": noisy_linear
        }
        )
    ).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    if is_fixed:
        optim = torch.optim.SGD(net.parameters(), lr=0)
    agent = RainbowPolicy(
        net, optim, 
        args.gamma,
        args.num_atoms,
        args.v_min,
        args.v_max,
        args.n_step,
        target_update_freq=args.target_update_freq).to(args.device)
    return agent


def get_random_agent(args):
    agent = RandomPolicy(
        observation_space=args.state_shape,
        action_space=args.action_shape,
    )
    return agent


def selfplay(args):  # always train first agent, start from random policy
    train_envs = DummyVectorEnv([env_func for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([env_func for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # Env
    briscola = BriscolaEnv(use_role_ids=True)
    env = PettingZooEnv(briscola)
    args.state_shape = briscola.observation_space_shape
    args.action_shape = briscola.action_space_shape

    # initialize agents and ma-policy
    id_agent_learning = 0
    agent_caller = get_agent(args)
    agent_callee = get_agent(args)
    agents = [agent_caller, agent_callee,
              get_random_agent(args), get_random_agent(args), get_random_agent(args)]
    # {'caller': 0, 'callee': 1, 'good_1': 2, 'good_2': 3, 'good_3': 4}
    policy = MultiAgentPolicyManager(agents, env)

    if args.no_priority:
        buffer = VectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs)
        )
    else:
        buffer = PrioritizedVectorReplayBuffer(
            args.buffer_size,
            buffer_num=len(train_envs),
            alpha=args.alpha,
            beta=args.beta,
            weight_norm=not args.no_weight_norm
        )
    # collector
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs, exploration_noise=True)
    
    # Log
    args.algo_name = "dqn_caller_callee_rainbow"
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
        model_save_path = os.path.join(log_path, f'policy_best_epoch.pth')
        torch.save(policy.policies['caller'].state_dict(), model_save_path)

    def train_fn(epoch, env_step):
        # nature DQN setting, linear decay in the first 1M steps
        if env_step <= 1e6:
            eps = args.eps_train - env_step / 1e6 * \
                (args.eps_train - args.eps_train_final)
        else:
            eps = args.eps_train_final
        policy.policies['caller'].set_eps(eps)
        policy.policies['callee'].set_eps(eps)
        if env_step % 1000 == 0:
            logger.write("train/env_step", env_step, {"train/eps": eps})
        if not args.no_priority:
            if env_step <= args.beta_anneal_step:
                beta = args.beta - env_step / args.beta_anneal_step * \
                    (args.beta - args.beta_final)
            else:
                beta = args.beta_final
            buffer.set_beta(beta)
            if env_step % 1000 == 0:
                logger.write("train/env_step", env_step, {"train/beta": beta})

    def test_fn(epoch, env_step):
        policy.policies['caller'].set_eps(args.eps_test)
        policy.policies['callee'].set_eps(args.eps_test)

    def reward_metric(rews):
        return rews[:, id_agent_learning]

    # test train_collector and start filling replay buffer
    train_collector.collect(n_step=args.batch_size * args.training_num)
    trainer = OffpolicyTrainer(policy, train_collector, test_collector, args.epoch,
                               args.step_per_epoch, args.step_per_collect, episode_per_test=args.test_num,
                               batch_size=args.batch_size, train_fn=train_fn, test_fn=test_fn,
                               save_best_fn=save_best_fn, update_per_step=args.update_per_step,
                               logger=logger, test_in_train=False, reward_metric=reward_metric)
    trainer.run()

    model_save_path = os.path.join(log_path, 'policy_last_epoch.pth')
    torch.save(policy.policies['caller'].state_dict(), model_save_path)

    return result, policy.policies['caller']


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    P = 10
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.005)
    parser.add_argument('--eps-train', type=float, default=1.0)
    parser.add_argument("--eps-train-final", type=float, default=0.05)
    parser.add_argument('--buffer-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.0000625)
    parser.add_argument(
        '--gamma', type=float, default=1.0, help='a smaller gamma favors earlier win'
    )
    parser.add_argument("--num-atoms", type=int, default=51)
    parser.add_argument("--v-min", type=float, default=0.)
    parser.add_argument("--v-max", type=float, default=120.)
    parser.add_argument("--noisy-std", type=float, default=0.1)
    parser.add_argument("--no-dueling", action="store_true", default=False)
    parser.add_argument("--no-noisy", action="store_true", default=False)
    parser.add_argument("--no-priority", action="store_true", default=False)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.4)
    parser.add_argument("--beta-final", type=float, default=1.)
    parser.add_argument("--beta-anneal-step", type=int, default=5000000)
    parser.add_argument("--no-weight-norm", action="store_true", default=False)
    parser.add_argument("--n-step", type=int, default=8)
    parser.add_argument("--target-update-freq", type=int, default=0) # NOTE no target
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--step-per-epoch", type=int, default=100000)
    parser.add_argument("--step-per-collect", type=int, default=10)
    parser.add_argument("--update-per-step", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--training-num", type=int, default=10)
    parser.add_argument('--test-num', type=int, default=400)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
    )
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
result, agent = selfplay(args)
