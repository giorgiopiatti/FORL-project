""" Implement Briscola Game class
"""
import functools
import pandas as pd
from heapq import merge
import numpy as np
from environment.briscola_communication.actions import BriscolaCommsAction, Messages
from environment.briscola_communication.dealer import BriscolaDealer
from environment.briscola_communication.public_state import BriscolaPublicState

from environment.briscola_communication.utils import (
    CARD_RANK_WITHIN_SUIT,
    Roles,
    DEBUG_ENV,
)
from environment.briscola_communication.player import BriscolaPlayer
from environment.briscola_communication.round import BriscolaRound
from environment.briscola_communication.judger import BriscolaJudger
from environment.briscola_communication.card import Card
from typing import List, Tuple


class ComStats:

    def __init__(self) -> None:
        all_actions = [
            BriscolaCommsAction(truth=t, message=m)
            for m in Messages
            for t in [True, False]
        ]
        self.actions_caller = {a.get_name(): 0 for a in all_actions}
        self.actions_callee = {a.get_name(): 0 for a in all_actions}
        self.actions_good = {a.get_name(): 0 for a in all_actions}
        self.count_true_caller = 0
        self.count_true_callee = 0
        self.count_true_good = 0
        self.num_truth = 0
        self.keys = self.actions_callee.keys()

    def get_vector(self, role):
        denom = 8
        if role == Roles.CALLER:
            actions = self.actions_caller
            truth = self.count_true_caller
        elif role == Roles.CALLEE:
            actions = self.actions_callee
            truth = self.count_true_callee
        elif role == Roles.GOOD_PLAYER:
            actions = self.actions_good
            truth = self.count_true_good
            denom = 8*3

        return pd.DataFrame({'truth_ratio': truth/denom, **{k: actions[k]/denom for k in self.keys}}, index=[0])

    def update(self, player: BriscolaPlayer, action: BriscolaCommsAction):
        if action.truth:
            self.num_truth += 1
        if player.role == Roles.CALLER and action.truth:
            self.count_true_caller += 1
        if player.role == Roles.CALLEE and action.truth:
            self.count_true_callee += 1
        if player.role == Roles.GOOD_PLAYER and action.truth:
            self.count_true_good += 1

        if player.role == Roles.CALLER:
            self.actions_caller[action.get_name()] += 1
        elif player.role == Roles.CALLEE:
            self.actions_callee[action.get_name()] += 1
        elif player.role == Roles.GOOD_PLAYER:
            self.actions_good[action.get_name()] += 1

    def get_truth(self):
        return self.num_truth / (8*5)


class BriscolaGame:
    """Provide game APIs for env to run BRISCOLA and get corresponding state
    information.
    """

    def __init__(
        self, allow_step_back=False, print_game=False, communication_say_truth=False
    ):
        self.allow_step_back = allow_step_back
        self.np_random = np.random.RandomState()
        self.num_players = 5
        self.print_game = print_game
        self.communication_say_truth = communication_say_truth

    def init_game(self):
        """Initialize players and state.

        Returns:
            dict: first state in one game
            int: current player's id
        """
        # initialize public variables
        self.winner_id = None
        self.history = []

        # initialize players
        self.players = [
            BriscolaPlayer(num, self.np_random) for num in range(self.num_players)
        ]

        # Distribute cards
        self.dealer = BriscolaDealer(self.np_random)
        self.dealer.shuffle()
        self.dealer.deal_cards(self.players)

        # Initial bet
        self.dummy_bet()
        self.callee_id = self.get_callee_id()
        self.players[self.caller_id].role = Roles.CALLER
        self.players[self.callee_id].role = Roles.CALLEE

        self.round = BriscolaRound(self.caller_id, self.briscola_suit)

        # initialize judger
        self.judger = BriscolaJudger(
            self.players, self.caller_id, self.callee_id, self.caller_points_bet
        )

        # Public state visible to everyone
        self.public = BriscolaPublicState(
            self.caller_id, self.caller_points_bet, self.called_card
        )
        self.public.round_order = self.round.player_order

        # get state of first player
        player_id = self.round.current_player
        self.state = self.get_state(player_id)

        if self.print_game:
            for i in range(5):
                print(self.get_state(i))
            print(
                f"Briscola: {self.briscola_suit} called_card: {self.called_card}")

        self.stats_comms = ComStats()
        return self.state, player_id

    def log(self, m):
        if self.print_game:
            print(m)

    def dummy_bet(self):
        self.caller_id = self.np_random.choice(5, 1)[0]
        self.caller_points_bet = None

        hand_strengthens = {"S": 0, "H": 0, "D": 0, "C": 0}
        for card in self.players[self.caller_id].current_hand:
            if card.rank == "A":
                hand_strengthens[card.suit] += 40
            elif card.rank == "3":
                hand_strengthens[card.suit] += 39
            elif card.rank == "K":
                hand_strengthens[card.suit] += 14
            elif card.rank == "Q":
                hand_strengthens[card.suit] += 13
            elif card.rank == "J":
                hand_strengthens[card.suit] += 12
            else:
                hand_strengthens[card.suit] += int(card.rank)

        self.briscola_suit = max(hand_strengthens, key=hand_strengthens.get)

        current_hand_briscola = list(
            filter(
                lambda c: c.suit == self.briscola_suit,
                self.players[self.caller_id].current_hand,
            )
        )
        rank_called_card = "A"
        rank_A_to_2 = CARD_RANK_WITHIN_SUIT.copy()
        rank_A_to_2.reverse()
        current_hand_briscola.sort(reverse=True)
        for i, card in enumerate(current_hand_briscola):
            if card.rank == rank_A_to_2[i]:
                rank_called_card = rank_A_to_2[i + 1]
            else:
                break

        self.called_card = Card(suit=self.briscola_suit, rank=rank_called_card)

    def get_callee_id(self):
        for player in self.players:
            current_hand = player.current_hand
            for c in current_hand:
                if self.called_card == c:
                    return player.player_id

        raise Exception("Called card not found!")

    def step(self, action):
        if self.round.communication_phase:
            player = self.players[self.round.current_player]

            self.stats_comms.update(player, action)
            self.log(
                f"Player {player.player_id} -> {action.message} saying {action.truth}")

            self.round.register_comm(player, action)
            self.round.update_current_player_comm()

            if not self.round.communication_phase:
                self.public.comms_round = self.round.comms
                self.public.trace_comms.append(self.round.comms)
                self.log("----- COMMS END ----")

            state = self.get_state(self.round.current_player)
            self.state = state
            self.log(state)

        else:
            player = self.players[self.round.current_player]
            player.play(action)
            self.log(f"Player {player.player_id} -> {action.card}")

            self.round.trace.append((player, action))
            self.round.update_current_player()

            self.public.trace_round = self.round.trace
            for player, card in self.round.trace:
                if card.card == self.called_card:
                    self.public.called_card_player = player.player_id

            # get next state
            if self.round.round_ended:
                winner, points = self.round.end_round()
                self.log(f"----- Round END ---- Won {winner} with {points}")
                self.round = BriscolaRound(winner, self.briscola_suit)
                self.judger.points[winner] += points
                # Update public stante on round end
                self.public.trace.append(self.public.trace_round)
                self.public.trace_round = []
                self.public.comms_round = []
                self.public.points = self.judger.points
                self.public.round_order = self.round.player_order

            state = self.get_state(self.round.current_player)
            self.state = state
            self.log(state)

        return state, self.round.current_player

    def get_state(self, player_id):
        player = self.players[player_id]
        other_hands = self._get_others_current_hand(player)

        if self.round.communication_phase:
            all_actions = [
                BriscolaCommsAction(truth=t, message=m)
                for m in Messages
                for t in [True, False]
            ]
            if self.communication_say_truth:
                available_actions = [
                    BriscolaCommsAction(truth=True, message=m) for m in Messages
                ]
            else:
                available_actions = all_actions
            state = player.get_state(
                self.public, other_hands, available_actions, all_actions
            )
        else:
            available_actions = player.available_actions_card()
            state = player.get_state(
                self.public, other_hands, available_actions)

        return state

    def is_over(self):
        """Judge whether a game is over

        Returns:
            Bool: True(over) / False(not over)
        """
        if len(self.public.trace) == 8:
            return True
        else:
            return False

    def _get_others_current_hand(self, player) -> List[Card]:
        other_cards = []
        for i in range(1, 5):
            p = self.players[(player.player_id + i) % len(self.players)]
            other_cards.extend(p.current_hand)

        other_cards.sort()

        return other_cards

    def seed(self, seed):
        self.np_random.seed(seed)
