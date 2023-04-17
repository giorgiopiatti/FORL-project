''' Implement Briscola Player class
'''
import functools

from briscola.utils import cards2str, briscola_sort_card
from typing import List, Tuple
from briscola.card import Card
from briscola.game import BriscolaPublicState
from briscola.actions import BriscolaAction, PlayCardAction


class BriscolaPlayerState:
    # Public Information
    caller_id: int
    caller_points_bet: int
    called_card: Card

    """
    List of round traces, inner list has dimension 5
    """
    trace: List[List[Tuple(int, Card)]]
    trace_round: List[Tuple(int, Card)]
    points: List[int]

    # Player private Information
    role: str
    player_id: int
    current_hand: List[str]
    others_hand: List[str]
    actions: List[BriscolaAction]

    def __init__(self, public: BriscolaPublicState,
                 role: str, player_id: int,  current_hand: List[str],
                 others_hand: List[str], actions: List[str]) -> None:
        self.caller_id = public.caller_id
        self.caller_points_bet = public.caller_points_bet
        self.called_card = public.called_card
        self.trace = public.trace
        self.trace_round = public.trace_round
        self.points = public.points

        # Private info
        self.role = role
        self.player_id = player_id
        self.current_hand = current_hand
        self.others_hand = others_hand
        self.actions = actions


class BriscolaPlayer:
    ''' Player can store cards in the player's hand and the role,
    determine the actions can be made according to the rules,
    and can perfrom corresponding action
    '''

    def __init__(self, player_id, np_random):
        ''' Give the player an id in one game

        Args:
            player_id (int): the player_id of a player

        Notes:
            1. role: A player's temporary role in one game
            2. played_cards: The cards played in one round
            3. hand: Initial cards
            4. _current_hand: The rest of the cards after playing some of them
        '''
        self.np_random = np_random
        self.player_id = player_id
        self.initial_hand: List[Card] = None
        self._current_hand: List[Card] = []
        self.role = ''
        self.played_cards = None

        # record cards removed from self._current_hand for each play()
        # and restore cards back to self._current_hand when play_back()
        self._recorded_played_cards = []

    @property
    def current_hand(self):
        return self._current_hand

    def set_current_hand(self, value):
        self._current_hand = value

    def get_state(self, public, others_hands):
        state = BriscolaPlayerState(
            public=public,
            role=self.role,
            player_id=self.player_id,
            current_hand=self._current_hand,
            others_hands=others_hands,
            actions=self.available_actions()
        )
        return state

    def available_actions(self):
        ''' Get the actions can be made based on the rules
        '''
        return [PlayCardAction(c) for c in self._current_hand]

    def play(self, action):
        ''' Perfrom action
        Args:
            action (string): specific action
        '''
        self.played_cards = action
        for i, remain_card in enumerate(self._current_hand):
            if action == remain_card:
                self._current_hand.remove(self._current_hand[i])
                break
        self._recorded_played_cards.append(action)
        return self

    def play_back(self):
        ''' Restore recorded cards back to self._current_hand
        '''
        removed_cards = self._recorded_played_cards.pop()
        self._current_hand.extend(removed_cards)
        self._current_hand.sort(key=functools.cmp_to_key(briscola_sort_card))
