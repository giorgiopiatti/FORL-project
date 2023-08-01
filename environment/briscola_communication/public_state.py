from environment.briscola_communication.actions import BriscolaCommsAction
from environment.briscola_communication.card import Card
from typing import List, Tuple
from environment.briscola_communication.round import BriscolaRound


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

    comms_round: List[Tuple[int, BriscolaCommsAction]]
    trace_comms: List[List[Tuple[int, BriscolaCommsAction]]]
    round_order : List[int]

    def __init__(self, caller_id, caller_points_bet, called_card) -> None:
        self.caller_points_bet = caller_points_bet
        self.caller_id = caller_id
        self.called_card = called_card
        self.trace = []
        self.trace_round = []
        self.points = [0, 0, 0, 0, 0]
        self.called_card_player = -1
        self.comms_round = []
        self.trace_comms = []