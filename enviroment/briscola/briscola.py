from collections import OrderedDict
import numpy as np

from rlcard.envs import Env
from enviroment.briscola.card import Card
from typing import List, Tuple
from enviroment.briscola.player_state import BriscolaPlayerState
from enviroment.briscola.actions import BriscolaAction, PlayCardAction

from enviroment.briscola import Game


class BriscolaEnv(Env):
    ''' Briscola Environment

        State Encoding
        shape = [N, 40]

    '''

    def __init__(self, config):

        self.name = 'briscola_5'
        self.game = Game()
        super().__init__(config)

        self.state_shape = [[49*41] for _ in range(self.num_players)]
        self.action_shape = [None for _ in range(self.num_players)]

    def set_agents(self, agents):
        '''
        Set the agents that will interact with the environment.
        This function must be called before `run`.

        Args:
            agents (list): List of Agent classes
        '''
        self.agents = agents  # TODO change in role base check reset

    def _extract_state(self, state: BriscolaPlayerState):
        ''' Encode state:
            last dim is card encoding = 40
        Args:
            state (dict): dict of original state
        '''

        trace = _trace2array(state.trace)  # shape=[8,5,41]
        trace_round = _round2array(state.trace_round)  # shape=[5,41]
        current_hand = _cards2array(state.current_hand)  # shape=[41]
        other_hands = _cards2array(state.other_hands)    # shape=[41]

        called_card = _cards2array([state.called_card])  # shape[41]

        info = [
            state.caller_id,
            # state.caller_points_bet,
            *state.points,
            state.role.value,
            state.player_id]
        padded_40_info = np.pad(
            info, (0, 41 - len(info)), 'constant')

        obs = np.concatenate((np.expand_dims(padded_40_info, axis=0),
                              np.expand_dims(called_card, axis=0),
                              np.expand_dims(current_hand, axis=0),
                              np.expand_dims(other_hands, axis=0),
                              trace.reshape(
                                  [trace.shape[0]*trace.shape[1], 41]),
                              trace_round
                              ))
        obs = obs.flatten('F')

        extracted_state = OrderedDict(
            {'obs': obs, 'legal_actions': self._get_legal_actions(state)})
        extracted_state['action_record'] = self.action_recorder
        extracted_state['raw_legal_actions'] = [a for a in state.actions]
        extracted_state['raw_debug_state'] = state

        # assert obs.shape == self.state_shape
        return extracted_state

    def get_payoffs(self):
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
        legal_actions = OrderedDict(
            {a.to_action_id(): None for a in legal_actions})
        return legal_actions

    # def get_action_feature(self, action):
    # TODO think about this method!
    #     ''' We can have action features

    #     Returns:
    #         (numpy.array): The action features
    #     '''
    #     return _cards2array([self._decode_action(action).card])


def _cards2array(cards: List[Card]):
    """
    Encoding of cards in a vector of dim 4*10
        result[i] == 1 <==> card i is present in card
    """
    matrix = np.zeros(
        [len(Card.valid_suit), len(Card.valid_rank)], dtype=np.int8)
    for card in cards:
        matrix[card.get_index_suit(), card.get_index_rank()] = 1

    matrix = matrix.flatten('F')
    matrix = np.append(matrix, [0])
    return matrix


def _trace2array(trace: List[List[Tuple[int, Card]]]):
    enc = np.zeros(
        [8, 5, len(Card.valid_suit)*len(Card.valid_rank) + 1], dtype=np.int8)
    for i in range(len(trace)):
        enc[i] = _round2array(trace[i])

    return enc


def _round2array(round: List[Tuple[int, Card]]):
    enc = np.zeros(
        [5, len(Card.valid_suit)*len(Card.valid_rank) + 1], dtype=np.int8)
    for i, move in enumerate(round):
        enc[move[0].player_id] = _cards2array([move[1].card])
        enc[move[0].player_id, enc.shape[1]-1] = i

    return enc
