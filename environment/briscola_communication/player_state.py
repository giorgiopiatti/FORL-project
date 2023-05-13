from typing import List, Tuple
from environment.briscola_communication.card import Card
from environment.briscola_communication.actions import BriscolaAction, BriscolaCommsAction
from environment.briscola_communication.public_state import BriscolaPublicState


class BriscolaPlayerState:
    # Public Information
    caller_id: int
    caller_points_bet: int
    called_card: Card

    """
    List of round traces, inner list has dimension 5
    """
    trace: List[List[Tuple[int, Card]]]
    trace_round: List[Tuple[int, Card]]
    points: List[int]

    # Player private Information
    role: str
    player_id: int
    current_hand: List[str]
    other_hands: List[str]
    actions: List[BriscolaAction]
    actions_all_coms: List[BriscolaAction]

    comms_round:  List[Tuple[int, BriscolaCommsAction]]

    def __init__(self, public: BriscolaPublicState,
                 role: str, player_id: int,  current_hand: List[str],
                 other_hands: List[str], actions: List[str], actions_all_coms) -> None:
        self.caller_id = public.caller_id
        self.caller_points_bet = public.caller_points_bet
        self.called_card = public.called_card
        self.trace = public.trace
        self.trace_round = public.trace_round
        self.comms_round = public.comms_round
        self.points = public.points
        self.called_card_player = public.called_card_player

        # Private info
        self.role = role
        self.player_id = player_id
        self.current_hand = current_hand
        self.other_hands = other_hands
        self.actions = actions
        self.actions_all_coms = actions_all_coms

    def __str__(self) -> str:
        return f'PlayerID {self.player_id} Role {self.role} Points {self.points} CurrentHand {self.current_hand} Round {self.trace_round}'