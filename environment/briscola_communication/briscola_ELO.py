
from copy import deepcopy
import numpy as np
import functools

import torch
from agents.elo_agents import AgentNN, AgentRNN
from agents.heuristic_agent_comm import HeuristicAgent
from environment.briscola_communication.card import NULLCARD_VECTOR, Card
from typing import List, Tuple
from environment.briscola_communication.player_state import BriscolaPlayerState
from environment.briscola_communication.actions import (
    BriscolaAction,
    BriscolaCommsAction,
    PlayCardAction,
)
from gymnasium.utils import seeding
from environment.briscola_communication import Game

import gymnasium as gym
from gymnasium.spaces import Discrete
from gymnasium import spaces
from environment.briscola_communication.utils import DICT_SUIT_TO_INT, Roles
import torch.nn as nn

from environment.briscola_communication.utils import (
    CARD_POINTS,
    CARD_RANK_WITHIN_SUIT_INDEX,
)


def one_hot(a, shape):
    b = np.zeros(shape)
    for el in a:
        b[el] = 1
    return b


def wins(card1, card2, index1, index2, briscola_suit):
    winner_card = card1
    winner_index = index1
    if winner_card.suit == briscola_suit:
        if (
            CARD_RANK_WITHIN_SUIT_INDEX[winner_card.rank]
            < CARD_RANK_WITHIN_SUIT_INDEX[card2.rank]
            and card2.suit == briscola_suit
        ):
            winner_card = card2
            winner_index = index2

    else:
        if card2.suit == briscola_suit or (
            CARD_RANK_WITHIN_SUIT_INDEX[card2.rank]
            > CARD_RANK_WITHIN_SUIT_INDEX[winner_card.rank]
            and card2.suit == winner_card.suit
        ):
            winner_card = card2
            winner_index = index2
    return winner_card, winner_index


def winning(trace_round, briscola_suit):
    if trace_round == []:
        return None, None
    else:
        winning_player_obj, winning_action = trace_round[0]
        winning_player = winning_player_obj.player_id
        winning_card = winning_action.card
        for i, (player, action) in enumerate(trace_round):
            player_id = player.player_id
            current_card = action.card

            if i >= 1:
                winning_card, winning_player = wins(
                    winning_card, current_card, winning_player, player_id, briscola_suit
                )

        return winning_card, winning_player


class BriscolaEnv(gym.Env):
    ''' Briscola Environment

        State Encoding
        shape = [N, 40]

        Convention: if use_role_ids = False, we play as player 0.

    '''

    def __init__(self, arch, role='caller',
                 agents={'callee': 'random',  'good_1': 'random',
                         'good_2': 'random', 'good_3': 'random'},
                 device='cpu',
                communication_say_truth=False):

        self.name = 'briscola_5'
        self.game = Game(communication_say_truth=communication_say_truth)
        self.use_role_ids = True
        self.role = role
        self.agents = agents
        self.device = device
        self.num_actions = 40 + 10
        self.action_space = spaces.Discrete(40)
        self.arch = arch

        # NN
        self.nn_raw_observation_space = spaces.Dict({
                "caller_id": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
                "role": spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
                "player_id": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
                "briscola_suit": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8),
                "belief": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
                "current_hand": spaces.Box(low=0, high=1, shape=(40,), dtype=np.int8),
                "played_cards": spaces.Box(low=0, high=1, shape=(40,), dtype=np.int8),
                "trace_round": spaces.Box(low=0, high=1, shape=(40,), dtype=np.int8),
                "winning_card": spaces.Box(low=0, high=1, shape=(40,), dtype=np.int8),
                "winning_player": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
                "round_points": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
                "point_players": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
                "position": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
                "is_last": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
                "comms_round": spaces.Box(
                    low=-1, high=1, shape=(5 * 5,), dtype=np.int8
                ),
        })

        # RNN
        self.rnn_raw_observation_space_current_round = spaces.Dict({
            "caller_id": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
            "role": spaces.Box(low=0, high=1, shape=(3,), dtype=np.int8),
            "player_id": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
            "briscola_suit": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8),
            "belief": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
            "current_hand": spaces.Box(low=0, high=1, shape=(40,), dtype=np.int8),
            "trace_round": spaces.Box(low=0, high=1, shape=(40,), dtype=np.int8),
            "winning_card": spaces.Box(low=0, high=1, shape=(40,), dtype=np.int8),
            "winning_player": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
            "round_points": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
            "point_players": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
            "position": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
            "is_last": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
            "comms_round": spaces.Box(
                low=-1, high=1, shape=(5 * 5,), dtype=np.int8
            ),
        })

        self.rnn_raw_observation_space_previous_round = spaces.Dict({
                "played_cards": spaces.Box(low=0, high=1, shape=(40,), dtype=np.int8),
                "winning_card": spaces.Box(low=0, high=1, shape=(40,), dtype=np.int8),
                "winning_player": spaces.Box(low=0, high=1, shape=(5,), dtype=np.int8),
                "round_points": spaces.Box(low=0, high=1, shape=(1,), dtype=np.int8),
                "comms_round": spaces.Box(
                    low=-1, high=1, shape=(5 * 5,), dtype=np.int8
                ),
            })

        if self.arch == 'nn':
            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(low=0, high=1, shape=(spaces.flatdim(self.nn_raw_observation_space),), dtype=np.int8),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(self.num_actions,), dtype=np.int8
                    )
                }
            )

            self.observation_shape = spaces.flatdim(
                self.nn_raw_observation_space)
        elif self.arch == 'rnn':
            self.current_round_shape = (1, spaces.flatdim(
                self.rnn_raw_observation_space_current_round))

            self.previous_round_shape = (1, spaces.flatdim(
                self.rnn_raw_observation_space_previous_round))

            dim = spaces.flatdim(self.rnn_raw_observation_space_previous_round) + \
                spaces.flatdim(self.rnn_raw_observation_space_current_round)

            self.observation_space = spaces.Dict(
                {
                    "observation": spaces.Box(low=0, high=1, shape=(1, dim,), dtype=np.int8),
                    "action_mask": spaces.Box(
                        low=0, high=1, shape=(1, self.num_actions,), dtype=np.int8
                    )
                }
            )
            self.observation_shape = (1, dim)
        else:
            raise ValueError(f'Invalid arch: {arch}')

    def _play_until_is_me(self, state, player_id):
        while player_id != self._player_id and not self.game.is_over():
            current_role = self._int_to_name(player_id)
            if isinstance(self.agents[current_role], AgentNN):
                x = self._get_obs(player_id, arch='nn')
                with torch.no_grad():
                    action_t, _, _, _ = self.agents[current_role].get_action_and_value(
                        torch.tensor(x['observation'],
                                     dtype=torch.float, device=self.device),
                        torch.tensor(x['action_mask'],
                                     dtype=torch.bool, device=self.device),
                        deterministic=True)
                action = self._decode_action(action_t.cpu().numpy().item())
            elif isinstance(self.agents[current_role], AgentRNN):
                lstm_state = self.lstm_states[current_role]
                done = torch.zeros((1,)).to(self.device)
                x = self._get_obs(player_id, arch='rnn')
                with torch.no_grad():
                    action_t, _, _, _, next_lstm_state = self.agents[current_role].get_action_and_value(
                        torch.tensor(x['observation'],
                                     dtype=torch.float, device=self.device).unsqueeze(0),
                        torch.tensor(x['action_mask'],
                                     dtype=torch.bool, device=self.device).unsqueeze(0),
                        lstm_state, done,
                        deterministic=True)
                action = self._decode_action(action_t.cpu().numpy().item())
                self.lstm_states[current_role] = next_lstm_state
            elif isinstance(self.agents[current_role], str) and self.agents[current_role] == 'random':
                action = state.actions[self.np_random.choice(
                    len(state.actions), size=1)[0]]
            elif isinstance(self.agents[current_role], HeuristicAgent):
                action = self.agents[current_role].get_heuristic_action(
                    state, [a.to_action_id() for a in state.actions])
            else:
                raise ValueError(f'self.agents[{current_role}] is invalid')

            state, player_id = self.game.step(action)

        return state

    def reset(self, seed=None, options=None):
        state, player_id = self.game.init_game()
        self.lstm_states = {
            name: (
                torch.zeros(1, 1, agent.rnn_out_size).to(self.device),
                torch.zeros(1, 1, agent.rnn_out_size).to(self.device),
            ) for (name, agent) in self.agents.items() if isinstance(agent, AgentRNN)
        }

        self._construct_int_name_mappings(
            self.game.caller_id, self.game.callee_id)
        self._player_id = self._name_to_int(self.role)
        # Play until ist my first turn to play a card
        state = self._play_until_is_me(state, player_id)
        observation = self._get_obs(self._player_id)
        return observation, {'raw_state': state}

    def _get_obs(self, player_id, arch=None):
        state = self.game.get_state(player_id)
        action_mask = np.zeros(self.num_actions, "int8")
        for a in state.actions:
            action_mask[a.to_action_id()] = 1

        if arch is None:
            arch = self.arch

        if arch == 'nn':
            return dict({"observation": self.nn_extract_state(state), "action_mask": action_mask})
        elif arch == 'rnn':
            return dict({"observation": np.expand_dims(self.rnn_extract_state(state), axis=0), "action_mask": np.expand_dims(action_mask, axis=0)})
        else:
            raise ValueError(f'arch {arch} is invalid')

    def step(self, action):
        action = self._decode_action(action)
        next_state, next_player_id = self.game.step(action)
        # Play until is my turn
        next_state = self._play_until_is_me(next_state, next_player_id)
        info = {'raw_state': next_state}
        terminated = False
        reward = np.array([0.0])
        if self.game.is_over():
            terminated = True
            reward = self.game.judger.judge_payoffs(
                self.game.caller_id, self.game.callee_id
            )
            reward = reward[self._player_id]
            info['stats_comms_caller'] = self.game.stats_comms.get_vector(
                Roles.CALLER)
            info['stats_comms_callee'] = self.game.stats_comms.get_vector(
                Roles.CALLEE)
            info['stats_comms_good'] = self.game.stats_comms.get_vector(
                Roles.GOOD_PLAYER)
            info['stats_truth'] = self.game.stats_comms.get_truth()
            
        observation = self._get_obs(self._player_id)
        return observation, reward, terminated, False, info

    def _decode_action(self, action_id) -> BriscolaAction:
        ''' Action id -> the action in the game. Must be implemented in the child class.

        Args:
            action_id (int): the id of the action

        Returns:
            action : the action that will be passed to the game engine.
        '''
        if action_id < 40:
            return PlayCardAction.from_action_id(action_id)
        else:
            return BriscolaCommsAction.from_action_id(action_id)

    def _construct_int_name_mappings(self, caller_id, callee_id):
        ids = [0, 1, 2, 3, 4]

        ids.remove(callee_id)
        ids.remove(caller_id)

        self.int_to_roles_map = {
            caller_id: 'caller',
            callee_id: 'callee',
            ids[0]: 'good_1',
            ids[1]: 'good_2',
            ids[2]: 'good_3'
        }
        self.role_to_int_map = {
            'caller': caller_id,
            'callee': callee_id,
            'good_1': ids[0],
            'good_2': ids[1],
            'good_3': ids[2]
        }

    def _int_to_name(self, ind):
        if self.use_role_ids:
            return self.int_to_roles_map[ind]
        return self.agents[ind]

    def _name_to_int(self, name):
        if self.use_role_ids:
            return self.role_to_int_map[name]
        return self.agents.index(name)

    def seed(self, seed):
        self.np_random, seed = seeding.np_random(seed)
        self.game.seed(seed=seed)

    def nn_extract_state(self, state: BriscolaPlayerState):
        briscola_suit = state.called_card.suit
        wc, wp = winning(state.trace_round, briscola_suit)
        flattened_trace = [item for round in state.trace for item in round]

        points_in_round = 0
        for i, (prev_player, prev_card) in enumerate(state.trace_round):
            points_in_round += CARD_POINTS[prev_card.card.rank]

        comms = np.zeros((5 * 5,))
        for player, comm in state.comms_round:
            comms[player.player_id * 5 + comm.message.value] = (
                1 if comm.positive else -1
            )

        encoding = dict(
            caller_id=one_hot([state.caller_id], shape=5),
            role=one_hot([state.role.value - 1], shape=3),
            player_id=one_hot([state.player_id], shape=5),
            briscola_suit=one_hot([DICT_SUIT_TO_INT[briscola_suit]], shape=4),
            belief=one_hot([], shape=5)
            if state.called_card_player == -1
            else one_hot([state.called_card_player], shape=5),
            current_hand=one_hot(
                [card.to_num() for card in state.current_hand], shape=40
            ),
            played_cards=one_hot(
                [card.to_num() for _, card in flattened_trace], shape=40
            ),
            trace_round=one_hot(
                [card.to_num() for _, card in state.trace_round], shape=40
            ),
            winning_card=one_hot([], shape=40)
            if wc is None
            else one_hot([wc.to_num()], shape=40),
            winning_player=one_hot([], shape=5)
            if wp is None
            else one_hot([wp], shape=5),
            round_points=points_in_round / 120,
            point_players=np.array(state.points) / 120,
            position=one_hot([state.position_in_round], shape=5),
            is_last=len(state.trace_round) == 4,
            comms_round=comms,
        )
        # return encoding
        return spaces.utils.flatten(self.nn_raw_observation_space, encoding)

    def rnn_extract_state(self, state: BriscolaPlayerState):
        ''' Encode state:
            last dim is card encoding = 40
        Args:
            state (dict): dict of original state
        '''
        briscola_suit = state.called_card.suit
        wc, wp = winning(state.trace_round, briscola_suit)
        flattened_trace = [item for round in state.trace for item in round]

        points_in_round = 0
        for i, (prev_player, prev_card) in enumerate(state.trace_round):
            points_in_round += CARD_POINTS[prev_card.card.rank]

        comms = np.zeros((5 * 5,))
        for player, comm in state.comms_round:
            comms[player.player_id * 5 + comm.message.value] = (
                1 if comm.positive else -1
            )

        encoding_current_round = dict(
            caller_id=one_hot([state.caller_id], shape=5),
            role=one_hot([state.role.value - 1], shape=3),
            player_id=one_hot([state.player_id], shape=5),
            briscola_suit=one_hot([DICT_SUIT_TO_INT[briscola_suit]], shape=4),
            belief=one_hot([], shape=5)
            if state.called_card_player == -1
            else one_hot([state.called_card_player], shape=5),
            current_hand=one_hot(
                [card.to_num() for card in state.current_hand], shape=40
            ),
            trace_round=one_hot(
                [card.to_num() for _, card in state.trace_round], shape=40
            ),
            winning_card=one_hot([], shape=40)
            if wc is None
            else one_hot([wc.to_num()], shape=40),
            winning_player=one_hot([], shape=5)
            if wp is None
            else one_hot([wp], shape=5),
            round_points=points_in_round / 120,
            point_players=np.array(state.points) / 120,
            position=one_hot([state.position_in_round], shape=5),
            is_last=len(state.trace_round) == 4,
            comms_round=comms,
        )

        if len(state.trace) >= 1:
            comms = np.zeros((5 * 5,))
            for player, comm in state.trace_comms[-1]:
                comms[player.player_id * 5 + comm.message.value] = (
                    1 if comm.positive else -1
                )

            last_round = state.trace[-1]
            wc, wp = winning(last_round, briscola_suit)

            points_in_round = 0
            for _, prev_card in last_round:
                points_in_round += CARD_POINTS[prev_card.card.rank]

            encoding_previous_round = dict(
                played_cards=one_hot(
                    [card.card.to_num() for _, card in last_round], shape=40
                ),
                winning_card=one_hot([wc.to_num()], shape=40),
                winning_player=one_hot([wp], shape=5),
                round_points=points_in_round / 120,
                comms_round=comms,
            )
        else:
            encoding_previous_round = dict(
                played_cards=one_hot([], shape=40),
                winning_card=one_hot([], shape=40),
                winning_player=one_hot([], shape=5),
                round_points=0,
                comms_round=one_hot([], shape=25)
            )

        # return encoding
        a = spaces.utils.flatten(self.rnn_raw_observation_space_current_round, encoding_current_round)
        b = spaces.utils.flatten(self.rnn_raw_observation_space_previous_round, encoding_previous_round)
        return np.concatenate([a, b])
