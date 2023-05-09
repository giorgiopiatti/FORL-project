
from copy import deepcopy
import numpy as np
import functools

from enviroment.briscola_gym.card import NULLCARD_VECTOR, Card
from typing import List, Tuple
from enviroment.briscola_gym.player_state import BriscolaPlayerState
from enviroment.briscola_gym.actions import BriscolaAction, PlayCardAction

from enviroment.briscola_gym import Game

import gymnasium as gym
from gymnasium.spaces import Discrete
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from gymnasium import spaces


def env(render_mode=None):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = BriscolaEnv(render_mode=internal_render_mode)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


class BriscolaEnv(AECEnv):
    ''' Briscola Environment

        State Encoding
        shape = [N, 40]

    '''

    def __init__(self, render_mode=None, use_role_ids=False, normalize_reward=True, save_raw_state=False, heuristic_ids=[]):

        self.name = 'briscola_5'
        self.game = Game()
        self.screen = None
        self.normalize_reward = normalize_reward
        self.use_role_ids = use_role_ids
        self.save_raw_state = save_raw_state
        self.heuristic_ids = heuristic_ids
        if not hasattr(self, "agents"):
            if self.use_role_ids:
                self.agents = ['caller', 'callee',
                               'good_1', 'good_2', 'good_3']
            else:
                self.agents = [f"player_{i}" for i in range(5)]

        self.possible_agents = self.agents
        self.num_actions = 40
        self.action_spaces = spaces.Dict(
            {agent: spaces.Discrete(40) for agent in self.agents}
        )

        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        # (value, seed, points)
        card_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([
                                11, 4, 11]), shape=(3,), dtype=np.int8)
        # (0,0,0) is used to pad, we encode 1-4 the suits, and 1-11 the rank

        player_id_space = spaces.Box(low=1, high=5, shape=(1,), dtype=np.int8)

        played_card_space = spaces.Dict(
            {'player': player_id_space, 'card': card_space})  # 'turn': player_id_space,
        round_space = spaces.Tuple([played_card_space]*5)
        # 0 is used for padding

        self.observation_spaces = spaces.Dict(
            {agent:
                spaces.Dict(
                    {
                        "observation": spaces.Dict({
                            'caller_id': player_id_space,
                            'points': spaces.Box(low=0, high=120, shape=(5,), dtype=np.int8),
                            'role': spaces.Box(low=1, high=3, shape=(1,), dtype=np.int8),
                            'player_id': player_id_space,
                            'called_card': card_space,
                            'current_hand': spaces.Tuple([card_space]*8),
                            # 'other_hands': spaces.Tuple([card_space]*32),
                            # 'trace': spaces.Tuple([round_space]*8),
                            'trace_round': round_space

                        }),
                        "action_mask": spaces.Box(
                            low=0, high=1, shape=(self.num_actions,), dtype=np.int8
                        ),
                    }
                )
                for agent in self.agents
             }
        )

        self.render_mode = render_mode

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    @property
    def observation_space_shape(self):
        return spaces.flatdim(self.observation_spaces[self._int_to_name(0)]['observation'])

    @property
    def action_space_shape(self):
        return spaces.flatdim(self.action_spaces[self._int_to_name(0)])

    @property
    def _original_observation_space(self):
        return self.observation_spaces[self._int_to_name(0)]['observation']

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return
        if self.render_mode == 'terminal':
            player_id = self._name_to_int(self.agent_selection)
            print(
                f'Player Turn {self.agent_selection}, Called {self.game.called_card}, Current Hand {self.game.players[player_id].current_hand}, Points {self.game.judger.points} Current Round {self.game.round.trace}')

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pass

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed=seed)
        state, player_id = self.game.init_game()
        self._construct_int_name_mappings(
            self.game.caller_id, self.game.callee_id)
        self.agent_selection = self._int_to_name(player_id)
        self.rewards = self._convert_to_dict(
            [0 for _ in range(self.num_agents)])
        self._cumulative_rewards = self._convert_to_dict(
            [0 for _ in range(self.num_agents)]
        )
        self.terminations = self._convert_to_dict(
            [False for _ in range(self.num_agents)]
        )
        self.truncations = self._convert_to_dict(
            [False for _ in range(self.num_agents)]
        )
        self.infos = self._convert_to_dict(
            [{"legal_moves": []} for _ in range(self.num_agents)]
        )
        self.next_legal_moves = list(sorted(self._get_legal_actions(state)))
        self._last_obs = self.observe(self.agent_selection)

        return self._last_obs

    def observe(self, agent):
        state = self.game.get_state(self._name_to_int(agent))
        if self.save_raw_state and agent in self.heuristic_ids:
            self.infos[agent]['raw_state'] = deepcopy(state)

        if self.game.is_over():
            self.infos[agent]['final_state'] = {a: self._extract_state(
                self.game.get_state(self._name_to_int(a))) for a in self.agents}

        legal_moves = self._get_legal_actions(state)
        action_mask = np.zeros(self.num_actions, "int8")
        for i in legal_moves:
            action_mask[i] = 1

        return dict({"observation": self._extract_state(state), "action_mask": action_mask})

    def step(self, action):
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)

        action = self._decode_action(action)
        next_state, next_player_id = self.game.step(action)
        next_player = self._int_to_name(next_player_id)
        self.agent_selection = next_player

        self._last_obs = self.observe(self.agent_selection)

        if self.game.is_over():
            self.rewards = self._convert_to_dict(
                self._scale_rewards(self._get_payoffs())
            )
            self.next_legal_moves = []
            self.terminations = self._convert_to_dict(
                [True for _ in range(self.num_agents)]
            )
            self.truncations = self._convert_to_dict(
                [False for _ in range(self.num_agents)]
            )
        else:
            self.next_legal_moves = self._get_legal_actions(
                self.game.get_state(next_player_id))

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = next_player
        self._accumulate_rewards()
        self._deads_step_first()

        # if self.render_mode == "human":
        #      self.render()

    def _extract_state(self, state: BriscolaPlayerState):
        ''' Encode state:
            last dim is card encoding = 40
        Args:
            state (dict): dict of original state
        '''

        encoding = dict(
            caller_id=np.array([state.caller_id+1]),
            points=state.points,
            role=np.array([state.role.value]),
            player_id=np.array([state.player_id+1]),
            called_card=state.called_card.vector(),
            current_hand=_pad_cards(_cards2array(state.current_hand), 8),
            #other_hands=_pad_cards(_cards2array(state.other_hands), 32),
            # trace=_trace2array(state.trace),
            trace_round=_round2array(state.trace_round)
        )
        # return state
        return spaces.utils.flatten(self._original_observation_space, encoding)

    def _get_payoffs(self):
        ''' Get the payoffs of players. Must be implemented in the child class.

        Returns:
            payoffs (list): a list of payoffs for each player
        '''
        return self.game.judger.judge_payoffs(self.game.caller_id, self.game.callee_id)

    def _decode_action(self, action_id) -> BriscolaAction:
        ''' Action id -> the action in the game. Must be implemented in the child class.

        Args:
            action_id (int): the id of the action

        Returns:
            action : the action that will be passed to the game engine.
        '''
        if action_id < 40:
            return PlayCardAction.from_action_id(action_id)

    def _get_legal_actions(self, state: BriscolaPlayerState):
        ''' Get all legal actions for current state

        Returns:
            legal_actions (list): a list of legal actions' id
        '''
        legal_actions = state.actions
        legal_actions = dict(
            {a.to_action_id(): None for a in legal_actions})
        return legal_actions

    def _scale_rewards(self, reward):
        if self.normalize_reward:
            return reward / 120.0
        return reward

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

    def _convert_to_dict(self, list_of_list):
        return {self._int_to_name(i):  list_of_list[i] for i in range(self.num_agents)}

    def seed(self, seed):
        self.game.seed(seed)


def _cards2array(cards: List[Card]):
    matrix = []
    for card in cards:
        matrix.append(card.vector())
    return matrix


def _trace2array(trace: List[List[Tuple[int, Card]]]):
    enc = []
    for r in trace:
        enc.append(_round2array(r))
    if len(enc) < 8:
        for _ in range(8 - len(enc)):
            enc.append(_pad_round([], 5))
    return enc


def _round2array(round: List[Tuple[int, Card]]):
    enc = []
    for i, move in enumerate(round):
        # turn=np.array([i+1]),
        enc.append(dict(player=np.array(
            [move[0].player_id+1]), card=move[1].card.vector()))
    return _pad_round(enc, 5)


def _pad_round(lst: list, max_len: int):
    if len(lst) < max_len:
        for _ in range(max_len - len(lst)):
            # turn=np.array([0]),
            lst.append(dict(player=np.array([0]), card=NULLCARD_VECTOR))
    return lst


def _pad_cards(lst: list, max_len: int):
    if len(lst) < max_len:
        for _ in range(max_len - len(lst)):
            lst.append(NULLCARD_VECTOR)
    return lst
