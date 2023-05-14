''' Implement Briscola Player class
'''
import functools

from environment.briscola_communication.utils import cards2str
from typing import List, Tuple
from environment.briscola_communication.card import Card
from environment.briscola_communication.actions import BriscolaAction, PlayCardAction
from environment.briscola_communication.player_state import BriscolaPlayerState
from environment.briscola_communication.utils import Roles


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
        self.role = Roles.GOOD_PLAYER
        self.played_cards = None

        # record cards removed from self._current_hand for each play()
        # and restore cards back to self._current_hand when play_back()
        self._recorded_played_cards = []

    @property
    def current_hand(self) -> List[Card]:
        return self._current_hand

    def set_current_hand(self, value):
        self._current_hand = value

    def get_state(self, public, other_hands, available_actions, available_actions_all_mess=None):
        state = BriscolaPlayerState(
            public=public,
            role=self.role,
            player_id=self.player_id,
            current_hand=self._current_hand,
            other_hands=other_hands,
            actions=available_actions,
            actions_all_coms=available_actions_all_mess
        )
        return state

    def available_actions_card(self):
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
            if action.card == remain_card:
                self._current_hand.remove(self._current_hand[i])
                self._recorded_played_cards.append(action.card)
                return self
        raise Exception(f"Invalid action for current state for player {self} with role {self.role}")

    def __repr__(self) -> str:
        return f'Player {self.player_id}'
