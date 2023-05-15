# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from agents.heuristic_agent import HeuristicAgent
from environment.briscola_communication.actions import BriscolaCommsAction


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="FORL_Briscola",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=16*5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=16,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=8, #NOTE finite game
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=1.0,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--num-test-games", type=int, default=1000,
        help="")
    parser.add_argument("--freq-eval-test", type=int, default=1000,
        help="")

    parser.add_argument("--briscola-train-mode", type=str, default="solo")
    parser.add_argument("--briscola-roles-train", type=str, default="caller")
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--briscola-callee-heuristic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--briscola-caller-heuristic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)

    parser.add_argument("--briscola-communicate", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--briscola-communicate-truth-only", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--briscola-communicate-second-phase", type=int, default=1)

    parser.add_argument('--logdir', type=str,
                        default='log/')
    args = parser.parse_args()

    if args.briscola_communicate:
        args.num_steps =  args.num_steps * 2

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    assert args.briscola_train_mode in ["solo", "bad_multiple_networks", "bad_single_network"]
    if args.briscola_train_mode == 'solo':
        assert args.num_generations == 1
        assert args.briscola_roles_train in ["caller", "callee"]
        assert not (args.briscola_caller_heuristic and args.briscola_roles_train == 'caller')
        assert not (args.briscola_callee_heuristic and args.briscola_roles_train == 'callee')
    if args.briscola_train_mode in ['bad_multiple_networks', 'bad_single_network']:
        assert args.num_generations >= 1
        args.briscola_roles_train = ['caller', 'callee']
        assert not args.briscola_caller_heuristic
        assert not args.briscola_callee_heuristic
    # fmt: on
    return args


def make_env(seed, role_training, briscola_agents, verbose=True, deterministic_eval=False):
    def thunk():
        if args.briscola_communicate:
            env = BriscolaEnv(normalize_reward=False, render_mode='terminal_env' if verbose else None,
                              role=role_training,  agents=briscola_agents, deterministic_eval=deterministic_eval, device='cpu',
                              communication_say_truth=briscola_communicate_truth)
        else:
            env = BriscolaEnv(normalize_reward=False, render_mode='terminal_env' if verbose else None,
                              role=role_training,  agents=briscola_agents, deterministic_eval=deterministic_eval, device='cpu')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        args.observation_shape = env.observation_shape
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(args.observation_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(args.observation_shape, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action_mask, action=None, deterministic=False):
        logits = self.actor(x)
        logits[~action_mask] = -torch.inf
        probs = Categorical(logits=logits)
        if action is None and not deterministic:
            action = probs.sample()
        if action is None and deterministic:
            action = logits.argmax(axis=-1)
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def evaluate(save=False):
    env = gym.vector.SyncVectorEnv(
        [make_env(args.seed+(args.num_generations*args.num_envs)+i, role_now_training, briscola_agents, deterministic_eval=True)
         for i in range(1)]
    )
    data, _ = env.reset()
    next_obs, next_mask = torch.tensor(data['observation'], device=device,  dtype=torch.float), torch.tensor(
        data['action_mask'], dtype=torch.bool, device=device)

    count_truth_comm = 0
    for step in range(0, args.num_steps):
        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, logprob, _, value = agent.get_action_and_value(
                next_obs, next_mask, deterministic=True)
            if args.briscola_communicate and (action >= 40).all():
                #Action is communicating
                count_truth_comm += (action <
                                     (40+BriscolaCommsAction.NUM_MESSAGES)).sum()

        # TRY NOT TO MODIFY: execute the game and log data.
        data, reward, done, _, info = env.step(action.cpu().numpy())
        next_obs, next_mask = torch.tensor(data['observation'], device=device,  dtype=torch.float), torch.tensor(
            data['action_mask'], dtype=torch.bool, device=device)
    metric = None


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    log_path = os.path.join(args.logdir, run_name)
    os.makedirs(log_path, exist_ok=True)

    if args.briscola_communicate:
        from environment.briscola_communication.briscola import BriscolaEnv
        briscola_communicate_truth = True  # Start with true
    else:
        from environment.briscola_base.briscola import BriscolaEnv

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    dummy_envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i, 'caller', {})
         for i in range(args.num_envs)]
    )
    assert isinstance(dummy_envs.single_action_space,
                      gym.spaces.Discrete), "only discrete action space is supported"

    # env setup
    if args.briscola_train_mode == 'solo':
        role_now_training = args.briscola_roles_train
        args.num_generations = 1
        briscola_agents = {'caller': 'random', 'callee': 'random',  'good_1': 'random',
                           'good_2': 'random', 'good_3': 'random'}
        if args.briscola_caller_heuristic:
            briscola_agents['caller'] = HeuristicAgent()
        if args.briscola_callee_heuristic:
            briscola_agents['callee'] = HeuristicAgent()
        agent = Agent(dummy_envs).to(device)

    elif args.briscola_train_mode == 'bad_multiple_networks':
        args.num_generations = args.num_generations * \
            len(args.briscola_roles_train)
        role_now_training = 'caller'
        briscola_agents = {'caller': 'random', 'callee': 'random',  'good_1': 'random',
                           'good_2': 'random', 'good_3': 'random'}

        agent_caller = Agent(dummy_envs)
        agent_callee = Agent(dummy_envs)
        agent_caller.load_state_dict(torch.load(
            './models/caller.pth'))
        agent_callee.load_state_dict(torch.load(
            './models/callee.pth'))
        agent_caller.eval()
        agent_callee.eval()
        agent = agent_caller
        briscola_agents['caller'] = agent_caller
        briscola_agents['callee'] = agent_callee

    elif args.briscola_train_mode == 'bad_single_network':
        args.num_generations = args.num_generations * \
            len(args.briscola_roles_train)
        role_now_training = 'caller'
        briscola_agents = {'caller': 'random', 'callee': 'random',  'good_1': 'random',
                           'good_2': 'random', 'good_3': 'random'}

        agent = Agent(dummy_envs).to(device)
        agent_old = Agent(dummy_envs).to('cpu')
        agent.eval()
        agent_old.eval()
        briscola_agents['caller'] = agent

    # if args.briscola_communicate and (args.briscola_train_mode == 'bad_single_network' or args.briscola_train_mode == 'bad_multiple_networks'):
    #     briscola_agents['callee'] = 'random_truth'  # Need to learn messages

    for i in range(args.num_test_games):
        evaluate()
