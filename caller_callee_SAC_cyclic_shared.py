import os
import torch
import argparse
import numpy as np
from copy import deepcopy
from typing import Optional, Tuple
from torch.utils.tensorboard import SummaryWriter

from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.env import SubprocVectorEnv
from tianshou.utils.net.common import Net
from tianshou.trainer import offpolicy_trainer, OffpolicyTrainer
from tianshou.data import ReplayBuffer
from tianshou.policy import BasePolicy, DQNPolicy

from tianshou.policy import DiscreteSACPolicy
from enviroment.briscola_gym.briscola import BriscolaEnv
from tianshou.env import PettingZooEnv

import shortuuid
from agents.pg import PGPolicy
from agents.random import RandomPolicy
from agents.ac import Actor, Critic
from multi_agents_rl.buffer import MultiAgentVectorReplayBuffer
from multi_agents_rl.collector import MultiAgentCollector
from multi_agents_rl.mapolicy import MultiAgentPolicyManager


def env_func():
    return PettingZooEnv(BriscolaEnv(use_role_ids=True, normalize_reward=False))


def get_2_agent_shared(args, is_fixed=False):
    net = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
              device=args.device)
    actor = Actor(net, args.action_shape, softmax_output=False,
                  device=args.device).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    net_c1 = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
                 device=args.device)
    critic1 = Critic(net_c1, last_size=args.action_shape,
                     device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    net_c2 = Net(args.state_shape, hidden_sizes=args.hidden_sizes,
                 device=args.device)
    critic2 = Critic(net_c2, last_size=args.action_shape,
                     device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)
    if args.auto_alpha:
        target_entropy = 0.98 * np.log(np.prod(args.action_shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=args.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    policy = DiscreteSACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        args.tau,
        args.gamma,
        args.alpha,
        estimation_step=args.n_step,
        reward_normalization=args.rew_norm
    ).to(args.device)


    policy2 = DiscreteSACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        args.tau,
        args.gamma,
        args.alpha,
        estimation_step=args.n_step,
        reward_normalization=args.rew_norm
    ).to(args.device)

    return policy, policy2


def get_random_agent(args):
    agent = RandomPolicy(
        observation_space=args.state_shape,
        action_space=args.action_shape,
        device=args.device
    )
    return agent


def selfplay(args):  # always train first agent, start from random policy
    train_envs = SubprocVectorEnv([env_func for _ in range(args.num_parallel_env)])
    test_envs = SubprocVectorEnv([env_func for _ in range(args.num_parallel_env)])
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
    agent_caller, agent_callee = get_2_agent_shared(args)
    agents = [agent_caller, agent_callee,
              get_random_agent(args), get_random_agent(args), get_random_agent(args)]
    # {'caller': 0, 'callee': 1, 'good_1': 2, 'good_2': 3, 'good_3': 4}


    # Log
    args.algo_name = "SAC_caller_callee_cyclic_vs_random"
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


  
    LERNING_AGENTS_ID = ['caller', 'callee']
    policy = MultiAgentPolicyManager(agents, env, LERNING_AGENTS_ID)

    # collector
    train_collector = MultiAgentCollector(
        policy, train_envs, LERNING_AGENTS_ID,
        MultiAgentVectorReplayBuffer(args.buffer_size, len(train_envs)),
        exploration_noise=False)
    test_collector = MultiAgentCollector(
        policy, test_envs, LERNING_AGENTS_ID, exploration_noise=False)


    def save_best_fn(policy):
        model_save_path = os.path.join(log_path, f'caller_policy_best_epoch.pth')
        torch.save(policy.policies['caller'].state_dict(), model_save_path)
        model_save_path = os.path.join(log_path, f'callee_policy_best_epoch.pth')
        torch.save(policy.policies['callee'].state_dict(), model_save_path)

    trainer = OffpolicyTrainer(policy,
                            train_collector,
                            test_collector,
                            max_epoch=args.epoch*args.num_generations,
                            step_per_epoch=args.step_per_epoch,
                            step_per_collect=0,  # NOTE this is keep but ignore by our collector
                            episode_per_collect=args.episode_per_collect,
                            episode_per_test=args.test_num,
                            batch_size=args.batch_size,
                            save_best_fn=save_best_fn,
                            logger=logger,
                            update_per_step=args.update_per_step,
                            test_in_train=False)

    policy.learning_policies = 'caller'
    for epoch, epoch_stat, info in trainer:
        if  epoch % args.epoch == 0:
            policy.learning_policies = 'callee' if (policy.learning_policies == 'caller') else 'caller'

    
    model_save_path = os.path.join(log_path, 'caller_policy_last_epoch.pth')
    torch.save(policy.policies['caller'].state_dict(), model_save_path)
    model_save_path = os.path.join(log_path, 'callee_policy_last_epoch.pth')
    torch.save(policy.policies['callee'].state_dict(), model_save_path)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--buffer-size', type=int, default=2*8*1000)
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--alpha-lr', type=float, default=3e-4)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.02)
    parser.add_argument('--auto-alpha', action="store_true", default=False)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--num-generations', type=int, default=10)
    parser.add_argument('--step-per-epoch', type=int, default=2*8*1000)
    parser.add_argument('--episode-per-collect', type=int, default=1000)
    parser.add_argument('--update-per-step', type=float,
                        default=1.0)  # NOTE before 0.1
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--hidden-sizes', type=int,
                        nargs='*', default=[128, 128, 128, 128])
    parser.add_argument('--num-parallel-env', type=int, default=16)
    parser.add_argument('--rew-norm', action="store_true", default=False)
    parser.add_argument('--n-step', type=int, default=1)

    parser.add_argument('--num-generation', type=int, default=200)
    parser.add_argument('--test-num', type=int, default=400)
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
