# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/dqn/#dqnpy
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agents.heuristic_agent import HeuristicAgent
from utils.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

from environment.briscola_base.briscola import BriscolaEnv
import stable_baselines3 as sb3


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
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")

    # Algorithm specific arguments
    parser.add_argument("--total-timesteps", type=int, default=8*5000000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=8*1000000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=1.0,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    parser.add_argument("--num-test-games", type=int, default=1000,
        help="")
    parser.add_argument("--freq-eval-test", type=int, default=80000,
        help="")
    
    # fmt: on
    #assert args.num_envs == 1, "vectorized envs are not supported at the moment"
    parser.add_argument("--briscola-train-mode", type=str, default="solo")
    parser.add_argument("--briscola-roles-train", type=str, default="caller")
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--briscola-callee-heuristic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--briscola-caller-heuristic", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument('--logdir', type=str,
                        default='log/')
    args = parser.parse_args()

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

    return args


def make_env(seed, role_training, briscola_agents, verbose=False, deterministic_eval=False):
    def thunk():
        env = BriscolaEnv(normalize_reward=False, render_mode='terminal_env' if verbose else None,
                          role=role_training,  agents=briscola_agents, deterministic_eval=deterministic_eval, device='cpu')
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        args.observation_shape = env.observation_shape
        return env

    return thunk

def save_model(step='last'):
    for name, agent_env in briscola_agents.items():
        if isinstance(agent_env, nn.Module) and name != role_now_training:
            model_save_path = os.path.join(log_path, f'{name}_policy_{step}_during_{role_now_training}.pth')
            torch.save(agent_env.state_dict(), model_save_path)
            if args.track:
                artifact = wandb.Artifact(f'{name}_policy_best_during_{role_now_training}', type='model')
                artifact.add_file(model_save_path)
                run.log_artifact(artifact)
    model_save_path = os.path.join(log_path, f'{role_now_training}_policy_{step}_during_{role_now_training}.pth')
    torch.save(q_network.state_dict(), model_save_path)
    if args.track:
        artifact = wandb.Artifact(f'{role_now_training}_policy_{step}_during_{role_now_training}', type='model')
        artifact.add_file(model_save_path)
        run.log_artifact(artifact)

best_bad = 0
best_good = 0
def evaluate(save=False):
    env = gym.vector.SyncVectorEnv(
        [make_env(args.seed+(args.num_generations*args.num_envs)+i, role_now_training, briscola_agents, deterministic_eval=True)
         for i in range(args.num_test_games)]
    )
    data, _ = env.reset()
    next_obs, next_mask = torch.tensor(data['observation'], device=device,  dtype=torch.float), torch.tensor(
        data['action_mask'], dtype=torch.bool, device=device)

    for step in range(0, 8):
        # ALGO LOGIC: action logic
        with torch.no_grad():
            q_values = q_network(next_obs, next_mask)
            action = torch.argmax(q_values, dim=1).cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        data, reward, done, _, info = env.step(action)
        next_obs, next_mask = torch.tensor(data['observation'], device=device,  dtype=torch.float), torch.tensor(
            data['action_mask'], dtype=torch.bool, device=device)
    metric = None
    if role_now_training in ['caller', 'callee']:
        writer.add_scalar("test/reward_bad_team_mean", reward.mean(), global_step)
        writer.add_scalar("test/reward_bad_team_std", reward.std(), global_step)
        writer.add_scalar("test/reward_good_team_mean", (120.0 - reward).mean(), global_step)
        writer.add_scalar("test/reward_good_team_std", (120.0 - reward).std(), global_step)
        metric = reward.mean()
        global best_good
        if metric > best_good and save:
            save_model(step='best')
            best_good = metric
    else:
        writer.add_scalar("test/reward_good_team_mean", reward.mean(), global_step)
        writer.add_scalar("test/reward_good_team_std", reward.std(), global_step)
        writer.add_scalar("test/reward_bad_team_mean", (120.0 - reward).mean(), global_step)
        writer.add_scalar("test/reward_bad_team_std", (120.0 - reward).std(), global_step)
        metric = (120.0 - reward).mean()
        global best_bad
        if metric > best_bad and save:
            save_model(step='best')
            best_bad = metric
    return metric

# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(args.observation_shape, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, env.single_action_space.n),
        )

    def forward(self, x, mask):
        logits =  self.network(x)
        logits[~mask] = -torch.inf
        return logits


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = parse_args()
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    log_path = os.path.join(args.logdir, run_name)
    os.makedirs(log_path, exist_ok=True)
    if args.track:
        import wandb

        run = wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        wandb.define_metric("test/reward_good_team_mean", summary="max")
        wandb.define_metric("test/reward_bad_team_mean", summary="max")

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
    if args.briscola_train_mode == 'solo':
        role_now_training = args.briscola_roles_train
        args.num_generations = 1
        briscola_agents = {'caller': 'random', 'callee': 'random',  'good_1': 'random',
                           'good_2': 'random', 'good_3': 'random'}
        if args.briscola_caller_heuristic:
            briscola_agents['caller'] = HeuristicAgent()
        if args.briscola_callee_heuristic:
            briscola_agents['callee'] = HeuristicAgent()
        
        q_network = QNetwork(dummy_envs).to(device)
        target_network = QNetwork(dummy_envs).to(device)
        target_network.load_state_dict(q_network.state_dict())

    # elif args.briscola_train_mode == 'bad_multiple_networks':
    #     args.num_generations = args.num_generations*len(args.briscola_roles_train)
    #     role_now_training = 'caller'
    #     briscola_agents = {'caller': 'random', 'callee': 'random',  'good_1': 'random',
    #                        'good_2': 'random', 'good_3': 'random'}

    #     agent_caller = Agent(dummy_envs).to(device)
    #     agent_callee = Agent(dummy_envs)
    #     agent_caller.eval()
    #     agent_callee.eval()
    #     agent = agent_caller
    #     briscola_agents['caller'] = agent_caller

    # elif args.briscola_train_mode == 'bad_single_network':
    #     args.num_generations = args.num_generations*len(args.briscola_roles_train)
    #     role_now_training = 'caller'
    #     briscola_agents = {'caller': 'random', 'callee': 'random',  'good_1': 'random',
    #                        'good_2': 'random', 'good_3': 'random'}

    #     agent = Agent(dummy_envs).to(device)
    #     agent_old = Agent(dummy_envs).to('cpu')
    #     agent.eval()
    #     agent_old.eval()
    #     briscola_agents['caller'] = agent

    id_log_model_training = 0
    global_step = 0
    start_time = time.time()
    for ngen in range(args.num_generations):
        # if args.briscola_train_mode != 'solo':
        #     role_now_training = args.briscola_roles_train[ngen%len(args.briscola_roles_train)]
        # print(f'Now training {role_now_training}')
        # if ngen > 0:
        #     id_log_model_training = ngen%len(args.briscola_roles_train)
        #     if ngen == 1 and args.briscola_train_mode == 'bad_multiple_networks':
        #         briscola_agents['callee'] = agent_callee
        #     if args.briscola_train_mode == 'bad_single_network':
        #         if ngen == 1:
        #             briscola_agents['caller'] = agent_old
        #             briscola_agents['callee'] = agent_old
        #         agent_old.load_state_dict(agent.state_dict())
        #         agent_old.eval()
        #     elif args.briscola_train_mode == 'bad_multiple_networks':
        #         if role_now_training == 'caller':
        #             agent = agent_caller
        #             agent.to(device)
        #             agent_callee.to('cpu')
        #         if role_now_training == 'callee':
        #             agent = agent_callee
        #             agent.to(device)
        #             agent_caller.to('cpu')

        # Seed is incremented at each generations
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.seed + i + args.num_envs*ngen, role_now_training, briscola_agents)
             for i in range(args.num_envs)]
        )

        
        optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
       

        rb = ReplayBuffer(
            args.buffer_size,
            envs.single_observation_space['observation'],
            envs.single_observation_space['action_mask'],
            envs.single_action_space,
            device,
            handle_timeout_termination=False,
        )
        start_time = time.time()

        # TRY NOT TO MODIFY: start the game
        obs_env, _ = envs.reset(seed=args.seed)
        obs = obs_env['observation']
        mask = obs_env['action_mask']
        for global_step in range(args.total_timesteps):
            # ALGO LOGIC: put action logic here
            epsilon = linear_schedule(
                args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            if random.random() < epsilon:
                matrix = np.arange(envs.single_action_space.n).reshape(
                    envs.single_action_space.n, 1)
                matrix = np.tile(matrix, (1, mask.shape[0])).transpose()
                actions = matrix[np.array(mask, dtype=np.bool_)] # TODO Check here
                actions = np.apply_along_axis(
                    lambda x: np.random.choice(x, size=(1,)), 0, actions)
            else:
                q_values = q_network(torch.tensor(obs, dtype=torch.float, device=device), torch.tensor(mask, dtype=torch.bool, device=device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            # TRY NOT TO MODIFY: execute the game and log data.
            tmp, rewards, terminated, truncated, infos = envs.step(actions)
            next_obs = tmp['observation']
            next_mask = tmp['action_mask']

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            if "final_info" in infos:
                for info in infos["final_info"]:
                    # Skip the envs that are not done
                    if "episode" not in info:
                        continue
                    writer.add_scalar(f"charts/{role_now_training}/episodic_return",
                                    info["episode"]["r"], global_step)
                    writer.add_scalar(f"charts/{role_now_training}/episodic_length",
                                    info["episode"]["l"], global_step)
                    writer.add_scalar(f"charts/{role_now_training}/epsilon", epsilon, global_step)

            # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
            real_next_obs = next_obs.copy()
            real_next_mask = next_mask.copy()
            for idx, d in enumerate(truncated):
                if d:
                    # TODO check
                    real_next_mask[idx] = infos["final_observation"]['observation'][idx]
                    # TODO check
                    real_next_mask[idx] = infos["final_observation"]['action_mask'][idx]
            rb.add(obs, mask, real_next_obs, real_next_mask,
                actions, rewards, terminated, infos)

            # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
            obs = next_obs
            mask = next_mask

            # ALGO LOGIC: training.
            if global_step > args.learning_starts:
                if global_step % args.train_frequency == 0:
                    data = rb.sample(args.batch_size)
                    with torch.no_grad():
                        target_max, _ = target_network(data.next_observations, data.next_actions_mask).max(
                            dim=1)  # NOTE we should mask here?
                        td_target = data.rewards.flatten() + args.gamma * target_max * \
                            (1 - data.dones.flatten())
                    old_val = q_network(data.observations, data.actions_mask).gather(
                        1, data.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    if global_step % 100 == 0:
                        writer.add_scalar(f"losses/{role_now_training}/td_loss", loss, global_step)
                        writer.add_scalar(f"losses/{role_now_training}/q_values",
                                        old_val.mean().item(), global_step)
                        print("SPS:", int(global_step / (time.time() - start_time)))
                        writer.add_scalar(
                            "charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    # optimize the model
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # update target network
                if global_step % args.target_network_frequency == 0:
                    for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                        target_network_param.data.copy_(
                            args.tau * q_network_param.data +
                            (1.0 - args.tau) * target_network_param.data
                        )
                if global_step % args.freq_eval_test == 0:
                    evaluate(save=True)
        if global_step % args.freq_eval_test > 0:
            evaluate(save=True)

    envs.close()
    writer.close()
