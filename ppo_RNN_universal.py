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
    parser.add_argument("--save-old-model-freq", type=int, default=500,
        help="")
    parser.add_argument("--num-old-models-to-save", type=int, default=2, help="")

    parser.add_argument("--briscola-communicate", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--briscola-communicate-truth-only", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--briscola-communicate-second-phase", type=int, default=8*5000000)

    parser.add_argument("--rnn-out-size", type=int, default=128)

    parser.add_argument('--logdir', type=str,
                        default='log/')
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
                              communication_say_truth=briscola_communicate_truth)
        else:
            env = BriscolaEnv(1, args.rnn_out_size, normalize_reward=False, render_mode='terminal_env' if verbose else None,
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
    def __init__(self, env):
        super().__init__()
        self.lstm = nn.LSTM(
            np.prod(env.previous_round_shape), args.rnn_out_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.prod(env.current_round_shape) +
                       args.rnn_out_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(
                nn.Linear(64, env.num_actions), std=0.01)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.prod(env.current_round_shape) +
                       args.rnn_out_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1)
        )

        self.offset_round = env.current_round_shape[-1]

    def get_states(self, x, lstm_state, done):
        x_features_round = x[:, :, :self.offset_round]
        x_previous_round = x[:, :,  self.offset_round:]

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        x_previous_round = x_previous_round.reshape(
            (-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))

        out_lstm = []
        for xr, d in zip(x_previous_round, done):
            o, lstm_state = self.lstm(
                xr.unsqueeze(0),  ((1.0 - d).view(1, -1, 1) * lstm_state[0], (1.0 - d).view(1, -1, 1) * lstm_state[1]))
            out_lstm += [o]

        out_lstm = torch.flatten(torch.cat(out_lstm), 0, 1)
        new_hidden = torch.cat([x_features_round.squeeze(1),
                                out_lstm], dim=1)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, action_mask, lstm_state, done, action=None,  deterministic=False):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        action_mask = action_mask.squeeze()
        logits = self.actor(hidden)
        probs = CategoricalMasked(logits=logits, masks=action_mask)
        if action is None and not deterministic:
            action = probs.sample()
        if action is None and deterministic:
            if len(action_mask.shape) == 1:
                action_mask = action_mask.unsqueeze(0)
            logits[~action_mask] = -torch.inf
            action = logits.argmax(axis=-1)
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden), lstm_state


def save_model(step='last', name=''):
    model_save_path = os.path.join(log_path, f'{name}_policy_{step}.pth')
    torch.save(agent.state_dict(), model_save_path)
    if args.track:
        artifact = wandb.Artifact(f'{name}_policy_{step}', type='model')
        artifact.add_file(model_save_path)
        run.log_artifact(artifact)


best = {'model_bad_vs_random': 0,
        'model_good_vs_random': 0, 'model_vs_model': 0}


def evaluate(save=False):
    random_model = 'random'
    agent_cpu = Agent(dummy_env).to(ENV_DEVICE)
    agent_cpu.load_state_dict(agent.state_dict())
    agent_cpu.eval()
    settings = [
        {'name': 'model_bad_vs_random', 'agents': {'callee': random_model,  'good_1': random_model,
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
        next_lstm_state = (
            torch.zeros(agent.lstm.num_layers, args.num_test_games,
                        agent.lstm.hidden_size).to(device),
            torch.zeros(agent.lstm.num_layers, args.num_test_games,
                        agent.lstm.hidden_size).to(device),
        )  # hidden and cell states
        next_done = torch.zeros(args.num_test_games).to(device)

        count_truth_comm = 0
        for step in range(0, args.num_steps):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                    next_obs, next_mask, next_lstm_state, next_done, deterministic=True)
                if args.briscola_communicate and (action >= 40).all():
                    # Action is communicating
                    count_truth_comm += (action <
                                         (40+BriscolaCommsAction.NUM_MESSAGES)).sum()

            # TRY NOT TO MODIFY: execute the game and log data.
            data, reward, done, _, info = env.step(action.cpu().numpy())
            next_obs, next_mask, next_done = torch.tensor(data['observation'], device=device,  dtype=torch.float), torch.tensor(
                data['action_mask'], dtype=torch.bool, device=device), torch.tensor(done, dtype=torch.float, device=device)
        metric = None

        if args.briscola_communicate:
            writer.add_scalar(f"test/truth_ratio_{name}_{s['model']}",
                              count_truth_comm/(8.0*args.num_test_games), global_step)

        writer.add_scalar(f"test/reward_{name}_mean",
                          reward.mean(), global_step)
        writer.add_scalar(f"test/reward_{name}_std",
                          reward.std(), global_step)
        metric = reward.mean()
        if metric > best[name] and save:
            save_model(step='best', name=name)
            best[name] = metric


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    log_path = os.path.join(args.logdir, run_name)
    os.makedirs(log_path, exist_ok=True)

    if args.briscola_communicate:
        # from environment.briscola_communication.briscola import BriscolaEnv
        raise Exception()
        briscola_communicate_truth = True  # Start with true
    else:
        from environment.briscola_base.briscola_rnn import BriscolaEnv

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

    dummy_env = make_env(args.seed, 'caller', {})()

    # env setup
    old_agents = []

    def pick_random_agents():
        #role = np.random.choice(['caller', 'callee', 'good'], size=1)[0]
        role = 'caller'

        if role == 'good':
            role = np.random.choice(['good_1', 'good_2', 'good_3'], size=1)[0]

        def sample_model():
            if len(old_agents) > 0:
                m = np.random.choice(
                    ['old_agent', 'random', 'heuristic'], size=1, p=[0.6, 0.3, 0.1])[0]
            else:
                m = np.random.choice(
                    ['random', 'heuristic'], size=1, p=[0.5, 0.5])[0]
            if m == 'old_agent':
                w = old_agents[np.random.choice(len(old_agents), size=1)[0]]
                m = Agent(dummy_env).to(ENV_DEVICE)
                m.eval()
                m.load_state_dict(w)
            elif m == 'heuristic':
                if args.briscola_communicate:
                    raise Exception('Not implemented!')
                    # m = HeuristicAgent()
                else:
                    m = HeuristicAgent()
            return m

        agents_env = {'caller': sample_model(), 'callee': sample_model(),  'good_1': sample_model(),
                      'good_2': sample_model(), 'good_3': sample_model()}
        del agents_env[role]
        return role, agents_env

    agent = Agent(dummy_env).to(device)
    agent.eval()

    id_log_model_training = 0
    global_step = 0
    start_time = time.time()

    role_now_training, briscola_agents = pick_random_agents()
    # Seed is incremented at each generationspick_random_agents
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.seed + i + args.num_envs, role_now_training, briscola_agents)
            for i in range(args.num_envs)]
    )

    optimizer = optim.Adam(
        agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs)
                      + args.observation_shape).to(device)
    mask = torch.zeros((args.num_steps, args.num_envs, 1,
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
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs,
                    agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs,
                    agent.lstm.hidden_size).to(device),
    )  # hidden and cell states

    num_updates = args.total_timesteps // args.batch_size

    evaluate()
    for update in range(1, num_updates + 1):

        # Save old models after each  args.save_old_model_freq  up to args.num_old_models_to_save
        if update % args.save_old_model_freq == 0:
            if len(old_agents) == args.num_old_models_to_save:
                old_agents.pop(0)
            old_agents.append(agent.state_dict())

        # Set sampled enviroment
        role_now_training, briscola_agents = pick_random_agents()
        for i in range(args.num_envs):
            envs.envs[i].agents = briscola_agents
            envs.envs[i].role = role_now_training

        initial_lstm_state = (
            next_lstm_state[0].clone(), next_lstm_state[1].clone())
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
                action, logprob, _, value, next_lstm_state = agent.get_action_and_value(
                    next_obs, next_mask, next_lstm_state, next_done)
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
            next_value = agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
            ).reshape(1, -1)
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
        b_obs = obs.reshape((-1,) + args.observation_shape)
        b_mask = mask.reshape((-1, 1, envs.single_action_space.n))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        agent.train()
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(
            args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                # be really careful about the index
                mb_inds = flatinds[:, mbenvinds].ravel()

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds], b_mask[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds],
                     initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
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
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
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
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
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
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step /
                                            (time.time() - start_time)), global_step)

        if update % args.freq_eval_test == 0:
            evaluate(save=True)
    # End generation

    if num_updates % args.freq_eval_test > 0:
        evaluate(save=True)

    save_model()
    envs.close()
    writer.close()
