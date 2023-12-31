# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
from distutils.util import strtobool
import copy
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from environment.briscola_communication.actions import BriscolaCommsAction
import pandas as pd

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
    parser.add_argument("--total-timesteps", type=int, default=8*10000000,
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
    parser.add_argument("--freq-save-model", type=int, default=100000, help= "")
    parser.add_argument("--save-old-model-freq", type=int, default=500,
        help="")
    parser.add_argument("--num-old-models-to-save", type=int, default=2, help="")

    parser.add_argument("--briscola-communicate", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--briscola-communicate-truth", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)

    parser.add_argument("--sample-batch-env", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument('--logdir', type=str,
                        default='log/')
    parser.add_argument('--resume-path', type=str, default=None)
    args = parser.parse_args()

    if args.briscola_communicate:
        args.num_steps =  args.num_steps * 2
        args.total_timesteps = args.total_timesteps*2

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    # fmt: on
    return args


ENV_DEVICE = 'cpu'


def make_env(seed, role_training, briscola_agents, verbose=False, deterministic_eval=False):
    def thunk():
        if args.briscola_communicate:
            env = BriscolaEnv(normalize_reward=False, render_mode='terminal_env' if verbose else None,
                              role=role_training,  agents=briscola_agents, deterministic_eval=deterministic_eval, device=ENV_DEVICE,
                              communication_say_truth=args.briscola_communicate_truth)
        else:
            env = BriscolaEnv(normalize_reward=False, render_mode='terminal_env' if verbose else None,
                              role=role_training,  agents=briscola_agents, deterministic_eval=deterministic_eval, device=ENV_DEVICE)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        args.observation_shape = env.observation_shape
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(
                probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(logits.device)
            logits = torch.where(self.masks, logits,
                                 torch.tensor(-1e8).to(logits.device))
            super(CategoricalMasked, self).__init__(
                probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p,
                              torch.tensor(0.0).to(self.masks.device))
        return -p_log_p.sum(-1)


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(args.observation_shape, args.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dim, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(args.observation_shape, args.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dim, args.hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(args.hidden_dim,
                       envs.single_action_space.n), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action_mask, action=None, deterministic=False):
        logits = self.actor(x)
        probs = CategoricalMasked(logits=logits, masks=action_mask)
        if action is None and not deterministic:
            action = probs.sample()
        if action is None and deterministic:
            logits[~action_mask] = -torch.inf
            action = logits.argmax(axis=-1)
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def save_model(step='last'):
    model_save_path = os.path.join(
        log_path, f'policy_{step}_{global_step}.pth')
    torch.save(
        {
            'global_step': global_step,
            'model_state_dict': agent.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'lr': optimizer.param_groups[0]["lr"]
        },
        model_save_path
    )
    if args.track:
        artifact = wandb.Artifact(
            f'policy_{step}_{global_step}', type='model')
        artifact.add_file(model_save_path)
        run.log_artifact(artifact)


weights_adversary_elo = None
elo_adversary = 1000  # init


def evaluate_elo():
    global elo_adversary
    global weights_adversary_elo
    adversary_agent = Agent(dummy_envs).to(ENV_DEVICE)
    adversary_agent.load_state_dict(weights_adversary_elo)
    adversary_agent.eval()
    agent_cpu = Agent(dummy_envs).to(ENV_DEVICE)
    agent_cpu.load_state_dict(agent.state_dict())
    agent_cpu.eval()

    # 1 Game
    config = {'callee': agent_cpu,  'good_1': adversary_agent,
              'good_2': adversary_agent, 'good_3': adversary_agent}
    env = gym.vector.SyncVectorEnv(
        [make_env(args.seed+(args.num_envs)+i, 'caller', config, deterministic_eval=True)
         for i in range(args.num_test_games)]
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
    reward_bad = reward.mean()

    # 2 Game
    config = {'callee': adversary_agent,  'caller': adversary_agent,
              'good_2': agent_cpu, 'good_3': agent_cpu}
    env = gym.vector.SyncVectorEnv(
        [make_env(args.seed+(args.num_envs)+i, 'good_1', config, deterministic_eval=True)
         for i in range(args.num_test_games)]
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
    reward_good = reward.mean()

    expected_score = 60
    mean_reward = (reward_bad+reward_good)/2

    elo_adversary += 10*(mean_reward-expected_score)
    writer.add_scalar(f"test/ELO", elo_adversary, global_step)
    # Save
    weights_adversary_elo = copy.deepcopy(agent.state_dict())


best = {'model_bad_vs_random': 0,
        'model_good_vs_random': 0, 'model_vs_model': 0}

def log_coms(info, name, roles):
    for r in roles:
        m = pd.concat([a[f'stats_comms_{r}'] for a in info]).mean()
        for (colname, colval) in m.items():
            writer.add_scalar(
                f"test/{name}/{r}/{colname}", colval, global_step)

    m = np.array([a[f'stats_truth'] for a in info]).mean()
    writer.add_scalar(f"test/{name}/truth_ratio", m, global_step)

def evaluate(save=False):
    random_model = 'random'
    agent_cpu = Agent(dummy_envs).to(ENV_DEVICE)
    agent_cpu.load_state_dict(agent.state_dict())
    agent_cpu.eval()
    settings = [
        {'name': 'model_bad_vs_random', 'agents': {'callee': agent_cpu,  'good_1': random_model,
                                                   'good_2': random_model, 'good_3': random_model}, 'model': 'caller'},
        {'name': 'model_good_vs_random', 'agents': {'caller': random_model, 'callee': random_model,  'good_1': agent_cpu,
                                                    'good_2': agent_cpu, 'good_3': agent_cpu}, 'model': 'good_1'},
        {'name': 'model_vs_model', 'agents': {'caller': agent_cpu, 'callee': agent_cpu,  'good_1': agent_cpu,
                                              'good_2': agent_cpu, 'good_3': agent_cpu}, 'model': 'callee'}
    ]
    for s in settings:
        name = s['name']
        env = gym.vector.SyncVectorEnv(
            [make_env(args.seed+(args.num_envs)+i, s['model'], s['agents'], deterministic_eval=True)
             for i in range(args.num_test_games)]
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

        if args.briscola_communicate:
            if name == 'model_bad_vs_random':
                log_coms(info['final_info'], name, ['callee', 'caller'])
            elif name == 'model_good_vs_random':
                log_coms(info['final_info'], name, ['good'])
            elif name == 'model_vs_model':
                log_coms(info['final_info'], name, [
                         'callee', 'caller', 'good'])

        writer.add_scalar(f"test/reward_{name}_mean",
                          reward.mean(), global_step)
        writer.add_scalar(f"test/reward_{name}_std",
                          reward.std(), global_step)
        metric = reward.mean()
        if metric > best[name] and save:
            best[name] = metric


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

    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True
        )

        wandb.define_metric(
            f"test/reward_model_bad_vs_random_mean", summary="max")
        wandb.define_metric(
            f"test/reward_model_good_vs_random_mean", summary="max")
        wandb.define_metric(f"test/reward_model_vs_model_mean", summary="max")
        wandb.define_metric("test/ELO", summary="max")

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

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
    old_agents = []

    def pick_random_agents():
        role = np.random.choice(['caller', 'callee', 'good'], size=1, p=[
                                0.25, 0.25, 0.5])[0]  # was uniform

        if role == 'good':
            role = np.random.choice(['good_1', 'good_2', 'good_3'], size=1)[0]

        def sample_model():
            if len(old_agents) > 0:
                m = np.random.choice(
                    ['old_agent', 'random', 'heuristic'], size=1, p=[0.6, 0.1, 0.3])[0]  # 0.6 0.3 0.1
            else:
                m = np.random.choice(
                    ['random', 'heuristic'], size=1, p=[0.5, 0.5])[0]
            if m == 'old_agent':
                w = old_agents[np.random.choice(len(old_agents), size=1)[0]]
                m = Agent(dummy_envs).to(ENV_DEVICE)
                m.eval()
                m.load_state_dict(w)
            elif m == 'heuristic':
                if args.briscola_communicate:
                    from agents.heuristic_agent_comm import HeuristicAgent
                    m = HeuristicAgent()
                else:
                    from agents.heuristic_agent import HeuristicAgent
                    m = HeuristicAgent()
            return m

        agents_env = {'caller': sample_model(), 'callee': sample_model(),  'good_1': sample_model(),
                      'good_2': sample_model(), 'good_3': sample_model()}
        del agents_env[role]
        return role, agents_env

    agent = Agent(dummy_envs).to(device)
    agent.eval()

    optimizer = optim.Adam(
        agent.parameters(), lr=args.learning_rate, eps=1e-5)

    id_log_model_training = 0
    global_step = 0
    start_time = time.time()
    start_step = 0
    start_update = 1

    if args.resume_path is not None:
        saved = torch.load(args.resume_path, map_location='cpu')
        agent.load_state_dict(saved['model_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        global_step = saved['global_step']
        start_step = global_step
        # +1 because we want to start from the next update
        start_update = (global_step // args.batch_size) + 1
        print(f"Loaded model from {args.resume_path}")


    role_now_training, briscola_agents = pick_random_agents()
    # Seed is incremented at each generationspick_random_agents
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i + args.num_envs, role_now_training, briscola_agents)
            for i in range(args.num_envs)]
    )

    num_updates = args.total_timesteps // args.batch_size

    evaluate()
    weights_adversary_elo = copy.deepcopy(agent.state_dict())
    for update in range(start_update, num_updates + 1):

        # Save old models after each  args.save_old_model_freq  up to args.num_old_models_to_save
        if update % args.save_old_model_freq == 0:
            if len(old_agents) == args.num_old_models_to_save:
                old_agents.pop(0)
            old_agents.append(copy.deepcopy(agent.state_dict()))

        # Set sampled enviroment
        if args.sample_batch_env:
            role_now_training, briscola_agents = pick_random_agents()
        for i in range(args.num_envs):
            if not args.sample_batch_env:
                role_now_training, briscola_agents = pick_random_agents()
            envs.envs[i].agents = briscola_agents
            envs.envs[i].role = role_now_training

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs,
                           args.observation_shape)).to(device)
        mask = torch.zeros((args.num_steps, args.num_envs,
                            envs.single_action_space.n), dtype=torch.bool).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) +
                              envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        data, _ = envs.reset()
        next_obs, next_mask = torch.tensor(data['observation'], device=device,  dtype=torch.float), torch.tensor(
            data['action_mask'], dtype=torch.bool, device=device)
        next_done = torch.zeros(args.num_envs).to(device)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            mask[step] = next_mask
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, next_mask)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            data, reward, done, _, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_mask, next_done = torch.tensor(data['observation'], device=device, dtype=torch.float), torch.tensor(
                data['action_mask'], dtype=torch.bool, device=device), torch.tensor(done, dtype=torch.float, device=device)

            if 'final_info' in info.keys():
                for item in info['final_info']:
                    if "episode" in item.keys():
                        writer.add_scalar(
                            f"charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar(
                            f"charts/episodic_length", item["episode"]["l"], global_step)
                        break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(
                1, -1)  # TODO should we mask something here?
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * \
                    nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * \
                    args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1, args.observation_shape))
        b_mask = mask.reshape((-1, envs.single_action_space.n))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_mask[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                    torch.clamp(ratio, 1 - args.clip_coef,
                                1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(
                        v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break
        agent.eval()
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(f"charts/learning_rate",
                          optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar(f"losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar(f"losses/policy_loss",
                          pg_loss.item(), global_step)
        writer.add_scalar(f"losses/entropy",
                          entropy_loss.item(), global_step)
        writer.add_scalar(f"losses/old_approx_kl",
                          old_approx_kl.item(), global_step)
        writer.add_scalar(f"losses/approx_kl",
                          approx_kl.item(), global_step)
        writer.add_scalar(f"losses/clipfrac",
                          np.mean(clipfracs), global_step)
        writer.add_scalar(f"losses/explained_variance",
                          explained_var, global_step)
        print("SPS:", int((global_step-start_step) / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int((global_step - start_step) /
                                            (time.time() - start_time)), global_step)

        if update % args.freq_eval_test == 0:
            evaluate(save=True)
            evaluate_elo()
        if update % args.freq_save_model == 0:
            save_model('train')
    # End generation

    if num_updates % args.freq_eval_test > 0:
        evaluate(save=True)
        evaluate_elo()

    save_model()
    envs.close()
    writer.close()
