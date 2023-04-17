from collections import OrderedDict
import numpy as np

from rlcard.envs import Env
from briscola.card import Card
from typing import List, Tuple
from briscola.player import BriscolaPlayerState
from briscola.actions import BriscolaAction

from briscola import Game


class BriscolaEnv(Env):
    ''' Briscola Environment

        State Encoding
        shape = [N, 40]

    '''

    def __init__(self, config):

        self.name = 'briscola_5'
        self.game = Game()
        super().__init__(config)

    def _extract_state(self, state: BriscolaPlayerState):
        ''' Encode state:
            last dim is card encoding = 40
        Args:
            state (dict): dict of original state
        '''

        trace = _trace2array(state.trace)  # shape=[8,5,5,40]
        trace_round = _round2array(state.trace_round)  # shape=[5,40]
        current_hand = _cards2array(state.current_hand)  # shape=[40]
        others_hand = _cards2array(state.others_hand)    # shape=[40]

        called_card = _cards2array([state.called_card]),  # shape[40]

        info = [
            state.caller_id,
            state.caller_points_bet,
            *state.points,
            state.role,
            state.player_id]
        padded_40_info = np.pad(info, (0, 40 - len(info)), 'constant')

        obs = np.concatenate((padded_40_info,
                              called_card,
                              current_hand,
                              others_hand,
                              trace.reshape(
                                  [trace.shape[0]*trace.shape[1]*trace.shape[2], 40]),
                              trace_round
                              ))

        extracted_state = OrderedDict(
            {'obs': obs, 'legal_actions': self._get_legal_actions(state)})
        extracted_state['action_record'] = self.action_recorder
        return extracted_state

    def get_payoffs(self):
        ''' Get the payoffs of players. Must be implemented in the child class.

        Returns:
            payoffs (list): a list of payoffs for each player
        '''
        return self.game.judger.judge_payoffs(self.game.caller_id, self.game.calle_id)

    def _decode_action(self, action_id) -> BriscolaAction:
        ''' Action id -> the action in the game. Must be implemented in the child class.

        Args:
            action_id (int): the id of the action

        Returns:
            action : the action that will be passed to the game engine.
        '''
        return BriscolaAction.from_action_id(action_id)

    def _get_legal_actions(self, state: BriscolaPlayerState):
        ''' Get all legal actions for current state

        Returns:
            legal_actions (list): a list of legal actions' id
        '''
        legal_actions = state.actions
        legal_actions = [a.to_action_id() for a in legal_actions]
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

    return matrix.flatten('F')


def _trace2array(trace: List[List[Tuple(int, Card)]]):
    enc = np.zeros(
        [8, 5, len(Card.valid_suit)*len(Card.valid_rank)], dtype=np.int8)
    for i in range(len(trace)):
        enc[i] = _round2array(trace[i])

    return enc


def _round2array(round: List[Tuple(int, Card)]):
    enc = np.zeros(
        [5, len(Card.valid_suit)*len(Card.valid_rank)], dtype=np.int8)
    for move in round:
        enc[move[0]] = _cards2array([move[1]])

    return enc
