from environment.briscola_immediate_reward.card import Card
from typing import List, Tuple
from environment.briscola_immediate_reward.round import BriscolaRound


class BriscolaPublicState:
    caller_id: int
    caller_points_bet: int
    called_card: Card

    """
    List of round traces, inner list has dimension 5
    """
    trace: List[List[Tuple[int, Card]]]
    trace_round: List[Tuple[int, Card]]
    points: List[int]

    called_card_player: int

    def __init__(self, caller_id, caller_points_bet, called_card) -> None:
        self.caller_points_bet = caller_points_bet
        self.caller_id = caller_id
        self.called_card = called_card
        self.trace = []
        self.trace_round = []
        self.points = [0, 0, 0, 0, 0]
        self.called_card_player = -1

    def update_state_on_round_end(self, points):
        self.trace.append(self.trace_round)
        self.trace_round = []
        self.points = points

    def update_state_on_round_step(self, round: BriscolaRound):
        self.trace_round = round.trace

        for (player, card) in round.trace:
            if (card.card == self.called_card):
                self.called_card_player = player.player_id
