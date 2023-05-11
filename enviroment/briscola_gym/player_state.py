from typing import List, Tuple
from enviroment.briscola_gym.card import Card
from enviroment.briscola_gym.actions import BriscolaAction
from enviroment.briscola_gym.public_state import BriscolaPublicState


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

    def __init__(self, public: BriscolaPublicState,
                 role: str, player_id: int,  current_hand: List[str],
                 other_hands: List[str], actions: List[str]) -> None:
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
        self.other_hands = other_hands
        self.actions = actions

    def __str__(self) -> str:
        return f'PlayerID {self.player_id} Role {self.role} Points {self.points} CurrentHand {self.current_hand} Round {self.trace_round}'

    def get_full_info(self):
        result = {
            'caller_id' : self.caller_id,
            'caller_points_bet' : self.caller_points_bet,
            'called_card' : self.called_card,
            'trace' : self.trace,
            'trace_round' : self.trace_round,
            'points' : self.points,

            'role' : self.role,
            'player_id' : self.player_id,
            'current_hand' : self.current_hand,
            'other_hands' : self.other_hands,
            'actions' : self.actions
        }
        return result