from distributed import Client
import csv
from dask import delayed
import pandas as pd
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
from agents.elo_agents import AgentRNN, AgentNN

from environment.briscola_communication.actions import BriscolaCommsAction




SEED = 42
NUM_TEST_GAMES = 1000
DEVICE = 'cpu'


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--briscola-communicate", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True)
    parser.add_argument("--path-pool", type=str)
    parser.add_argument("--file-pool", type=str)
    parser.add_argument("--path-scores", type=str)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--briscola-communicate-truth", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True)
    parser.add_argument("--split", type=int)

    args = parser.parse_args()
    return args


def make_env(arch, seed, role_training, briscola_agents):
    def thunk():
        if args.briscola_communicate:
            env = BriscolaEnv(arch, role=role_training,
                          agents=briscola_agents, device=DEVICE, communication_say_truth=args.briscola_communicate_truth)
        else:
            env = BriscolaEnv(arch, role=role_training,
                          agents=briscola_agents, device=DEVICE)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.seed(seed)
        return env

    return thunk


def run_game(config, teamA='caller'):
    modelA = config[teamA]
    arch = 'rnn' if isinstance(modelA, AgentRNN) else 'nn'
    env = gym.vector.SyncVectorEnv(
        [make_env(arch, SEED+16+i, teamA, config)
         for i in range(NUM_TEST_GAMES)]
    )
    data, info = env.reset()
    next_obs, next_mask = torch.tensor(data['observation'],  dtype=torch.float, device=DEVICE), torch.tensor(
        data['action_mask'], dtype=torch.bool, device=DEVICE)

    if isinstance(modelA, AgentRNN):
        next_lstm_state = (
            torch.zeros(modelA.lstm.num_layers, NUM_TEST_GAMES,
                        modelA.lstm.hidden_size, device=DEVICE),
            torch.zeros(modelA.lstm.num_layers, NUM_TEST_GAMES,
                        modelA.lstm.hidden_size, device=DEVICE),
        )  # hidden and cell states
        next_done = torch.zeros(NUM_TEST_GAMES, device=DEVICE)

    for _ in range(0, NUM_STEPS):
        # ALGO LOGIC: action logic
        with torch.no_grad():
            if isinstance(modelA, AgentRNN):
                action, logprob, _, value, next_lstm_state = modelA.get_action_and_value(
                    next_obs, next_mask, next_lstm_state, next_done, deterministic=True)
                action = action.cpu().numpy()
            elif isinstance(modelA, AgentNN):
                action, logprob, _, value = modelA.get_action_and_value(
                    next_obs, next_mask, deterministic=True)
                action = action.cpu().numpy()
            elif isinstance(modelA, str) and modelA == 'random':
                state = info['raw_state']
                action = list(map(lambda x: x.actions[np.random.choice(
                    len(x.actions), size=1)[0]].to_action_id(), state))
            elif isinstance(modelA, HeuristicAgent):
                state = info['raw_state']
                action = list(
                    map(lambda x: modelA.get_heuristic_action(x, [a.to_action_id() for a in x.actions]).to_action_id(), state))

        # TRY NOT TO MODIFY: execute the game and log data.
        data, reward, done, _, info = env.step(action)
        if isinstance(modelA, AgentRNN):
            next_done = torch.tensor(done, dtype=torch.float, device=DEVICE)
        next_obs, next_mask = torch.tensor(data['observation'],  dtype=torch.float, device=DEVICE), torch.tensor(
            data['action_mask'], dtype=torch.bool, device=DEVICE),

    stats = None
    if args.briscola_communicate:
        s1 = pd.concat([a[f'stats_comms_caller'] for a in info['final_info']]).mean().add_prefix('caller/')
        s2 = pd.concat([a[f'stats_comms_callee'] for a in info['final_info']]).mean().add_prefix('callee/')
        s3 = pd.concat([a[f'stats_comms_good'] for a in info['final_info']]).mean().add_prefix('good/')
        stats = pd.concat([s1, s2, s3]).to_frame().transpose()

    rewardA = reward.mean()
    rewardB = 120.0-rewardA
    return rewardA, rewardB, stats



def get_model(name):
    if type(name) == int:
        entry = df.iloc[name]
        path = f"{args.path_pool}/{entry['file_name']}"
        if entry['arch'] == 'rnn':
            model = AgentRNN(rnn_out_size=entry['rnn_out_size'], hidden_dim=entry['hidden_dim'], num_actions=NUM_ACTIONS,
                             current_round_shape=CURRENT_ROUND_SHAPE, previous_round_shape=PREVIOUS_ROUND_SHAPE)
        else:
            model = AgentNN(
                hidden_dim=entry['hidden_dim'], num_actions=NUM_ACTIONS, observation_shape=OBSERVATION_SHAPE)
        model.load_state_dict(torch.load(
            path,  map_location=DEVICE)['model_state_dict'])
        model.to(DEVICE)
    elif name == 'heuristic':
        model = HeuristicAgent()
    elif name == 'random':
        model = 'random'
    return model


def run(modelA, modelB):
    a = get_model(modelA)
    b = get_model(modelB)
    config = {'caller': a, 'callee': a,
              'good_1': b, 'good_2': b, 'good_3': b}
    rewardA, rewardB, stats = run_game(config, teamA='caller')

    aName = modelA if type(modelA) == str else f"{df.iloc[modelA]['name']}"
    bName = modelB if type(modelB) == str else f"{df.iloc[modelB]['name']}"
    aStep = 0 if type(modelA) == str else df.iloc[modelA]['step']
    bStep = 0 if type(modelB) == str else df.iloc[modelB]['step']

    res = pd.DataFrame({'model_bad_id':  modelA, 'model_good_id': modelB,  'model_bad_name': aName, 'model_bad_step': aStep,
                       'model_good_name': bName, 'model_good_step': bStep, 'reward_bad': rewardA, 'reward_good': rewardB}, index=[0])

    if args.briscola_communicate:
        res = pd.concat([res, stats], axis=1)

    return res


if __name__ == '__main__':
    args = parse_args()
    if args.briscola_communicate:
        NUM_STEPS = 16
        CURRENT_ROUND_SHAPE = (1, 184)
        NUM_ACTIONS = 50
        PREVIOUS_ROUND_SHAPE = (1, 111)
        OBSERVATION_SHAPE = 224 
        from agents.heuristic_agent_comm import HeuristicAgent
        from environment.briscola_communication.briscola_ELO import BriscolaEnv
    else:
        # NOCOM
        NUM_STEPS = 8
        CURRENT_ROUND_SHAPE = (1, 159)
        NUM_ACTIONS = 40
        PREVIOUS_ROUND_SHAPE = (1, 86)
        OBSERVATION_SHAPE = 199
        from agents.heuristic_agent import HeuristicAgent
        from environment.briscola_base.briscola_ELO import BriscolaEnv

    NUM_SPLITS = 8
    
    df = pd.read_csv(args.file_pool)
    start = 0
    end = len(df)
    pairs = [(i, j) for i in range(start, end) for j in range(start, end)]
    for i in range(end):
        pairs.append((i, 'random'))
        pairs.append((i, 'heuristic'))
        pairs.append(('random', i))
        pairs.append(('heuristic', i))
    pairs.append(('random', 'heuristic'))
    pairs.append(('heuristic', 'random'))
    pairs.append(('random', 'random'))
    pairs.append(('heuristic', 'heuristic'))

    if args.split == NUM_SPLITS -1:
        pairs = pairs[args.split*len(pairs)//NUM_SPLITS:] # last split
    else:
        pairs = pairs[args.split*len(pairs)//NUM_SPLITS: (args.split+1)*len(pairs)//NUM_SPLITS]

    print(len(pairs))
    N = args.num_workers
    CSV_PATH_SCORES = f'{args.path_scores[:-4]}_{args.split}.csv'
    client = Client(n_workers=N//2, threads_per_worker=2)

    s = time.time()
    for i in range(0, len(pairs), N):
        df_tmp = [delayed(run)(*pairs[j])
                  for j in range(i, min(i+N, len(pairs)))]
        res = delayed(pd.concat)(df_tmp, ignore_index=True)
        resC = res.compute()

        resC.to_csv(CSV_PATH_SCORES, index=False, quoting=csv.QUOTE_ALL,
                    mode='a', header=not os.path.exists(CSV_PATH_SCORES))
        print('Saved i=', i+N)
        print('s/run=', (time.time()-s)/(i+N))
