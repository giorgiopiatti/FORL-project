''' Implement Briscola Game class
'''
import functools
from heapq import merge
import numpy as np
from briscola.dealer import BriscolaDealer

from briscola.utils import briscola_sort_card, cards2str
from briscola.player import BriscolaPlayer
from briscola.round import BriscolaRound
from briscola.judger import BriscolaJudger
from briscola.card import Card
from typing import List, Tuple


class BriscolaPublicState:
    caller_id: int
    caller_points_bet: int
    called_card: Card

    """
    List of round traces, inner list has dimension 5
    """
    trace: List[List[Tuple(int, Card)]]
    trace_round: List[Tuple(int, Card)]
    points: List[int]

    def __init__(self, caller_id, caller_points_bet, called_card) -> None:
        self.caller_points_bet = caller_points_bet
        self.caller_id = caller_id
        self.called_card = called_card
        self.trace = []
        self.trace_round = []
        self.points = []

    def update_state_on_round_end(self, points):
        self.trace.append(self.trace_round)
        self.trace_round = []
        self.points = points

    def update_state_on_round_step(self, round: BriscolaRound):
        self.trace_round = round.trace


class BriscolaGame:
    ''' Provide game APIs for env to run BRISCOLA and get corresponding state
    information.
    '''

    def __init__(self, allow_step_back=False):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = 5

    def init_game(self):
        ''' Initialize players and state.

        Returns:
            dict: first state in one game
            int: current player's id
        '''
        # initialize public variables
        self.winner_id = None
        self.history = []

        # initialize players
        self.players = [BriscolaPlayer(num, self.np_random)
                        for num in range(self.num_players)]

        # Distribute cards
        self.dealer = BriscolaDealer(self.np_random)
        self.dealer.shuffle()
        self.dealer.deal_cards(self.players)

        # Initial bet
        self.dummy_bet()
        self.calle_id = self.get_calle_id()

        self.round = BriscolaRound(self.caller_id, self.briscola_suit)
        self.round.initiate(self.players)

        # initialize judger
        self.judger = BriscolaJudger(self.players,
                                     self.caller_id,
                                     self.called_card,
                                     self.caller_points_bet)

        # get state of first player
        player_id = self.round.current_player
        self.state = self.get_state(player_id)

        # Public state visible to everyone
        self.public = BriscolaPublicState(
            self.caller_id, self.caller_points_bet, self.called_card)

        return self.state, player_id

    def dummy_bet(self):
        self.caller_id = self.np_random.choice(5, 1)[0]

        self.briscola_suit
        self.called_card
        self.caller_points_bet

    def get_calle_id(self):
        for player in self.players:
            current_hand = player.current_hand
            for c in current_hand:
                if self.called_card == c:
                    return player.player_id

        raise Exception("Called card not found!")

    def step(self, action):
        ''' Perform one draw of the game

        Args:
            action (str): specific action of briscola. Eg: '33344'

        Returns:
            dict: next player's state
            int: next player's id
        '''
        if self.allow_step_back:
            # TODO: don't record game.round, game.players, game.judger if allow_step_back not set
            pass

        player = self.players[self.round.current_player]
        self.round.proceed_round(player, action)
        self.round.update_current_player()
        self.public.update_state_on_round_step(self.round)

        # get next state
        if self.round.round_ended:
            winner, points = self.round.end_round()
            self.round = BriscolaRound(winner, self.briscola_suit)
            self.judger.points[winner] += points
            self.public.update_state_on_round_end(self.judger.points)

        state = self.get_state(self.round.current_player)
        self.state = state

        return state, self.round.get_current_player()

    def step_back(self):
        ''' Return to the previous state of the game

        Returns:
            (bool): True if the game steps back successfully
        '''
        if not self.trace:
            return False
        return False

    def get_state(self, player_id):
        ''' Return player's state

        Args:
            player_id (int): player id

        Returns:
            (dict): The state of the player
        '''
        player = self.players[player_id]
        others_hands = self._get_others_current_hand(player)

        state = player.get_state(self.public, others_hands)

        return state

    @staticmethod
    def get_num_actions():
        ''' Return the total number of abstract acitons

        Returns:
            int: the total number of abstract actions of briscola
        '''
        return 40

    def get_player_id(self):
        ''' Return current player's id

        Returns:
            int: current player's id
        '''
        return self.round.current_player

    def get_num_players(self):
        ''' Return the number of players in briscola

        Returns:
            int: the number of players in briscola
        '''
        return self.num_players

    def is_over(self):
        ''' Judge whether a game is over

        Returns:
            Bool: True(over) / False(not over)
        '''
        if self.winner_id is None:
            return False
        return True

    def _get_others_current_hand(self, player) -> List[Card]:
        other_cards = []
        for i in range(1, 5):
            p = self.players[(player.player_id+i) % len(self.players)]
            other_cards.append(p.current_hand)

        others_hand = merge(
            other_cards, key=functools.cmp_to_key(briscola_sort_card))

        return others_hand
