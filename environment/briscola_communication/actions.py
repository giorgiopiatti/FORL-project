from enum import Enum
from environment.briscola_communication.card import Card

from abc import (
    ABC,
    abstractmethod,
)

from environment.briscola_communication.utils import CARD_STR_TO_NUM, NUM_TO_CARD_STR


class BriscolaAction(ABC):
    action_type: str

    @abstractmethod
    def __str__(self) -> str:
        return super().__str__()

    @abstractmethod
    def to_action_id(self):
        pass


class PlayCardAction(BriscolaAction):
    def __init__(self, card: Card) -> None:
        self.action_type = "play_card"
        self.card = card

    def __str__(self) -> str:
        return "Play_" + str(self.card)

    @staticmethod
    def from_action_id(action_id):
        s = NUM_TO_CARD_STR[action_id]
        return PlayCardAction(card=Card.init_from_string(s))

    def to_action_id(self):
        return CARD_STR_TO_NUM[str(self.card)]

    def to_num(self):
        return CARD_STR_TO_NUM[str(self.card)]

    def __repr__(self) -> str:
        return f"{self.card}"


class Messages(Enum):
    CARICO_NOT_BRISCOLA = 0
    BRISCOLINO = 1
    BRISCOLA_FIGURA = 2
    BRISCOLA_CARICO = 3
    LISCIO = 4


class BriscolaCommsAction(BriscolaAction):
    NUM_MESSAGES = 5

    def __init__(self, truth, message: Messages) -> None:
        self.action_type = "comms"
        self.truth = truth
        self.message = message
        self.positive = None

    def set_meaning_from_player(self, player, briscola_suit):
        is_valid = False
        if self.message == Messages.CARICO_NOT_BRISCOLA:
            for c in player.current_hand:
                is_valid = is_valid or (
                    c.rank in ["A", "3"] and c.suit != briscola_suit
                )
        elif self.message == Messages.BRISCOLINO:
            for c in player.current_hand:
                is_valid = is_valid or (
                    c.rank in ["2", "4", "5", "6", "7"] and c.suit == briscola_suit
                )
        elif self.message == Messages.BRISCOLA_FIGURA:
            for c in player.current_hand:
                is_valid = is_valid or (
                    c.rank in ["K", "Q", "J"] and c.suit == briscola_suit
                )
        elif self.message == Messages.BRISCOLA_CARICO:
            for c in player.current_hand:
                is_valid = is_valid or (
                    c.rank in ["A", "3"] and c.suit == briscola_suit
                )
        elif self.message == Messages.LISCIO:
            for c in player.current_hand:
                is_valid = is_valid or (
                    c.rank in ["2", "4", "5", "6", "7"] and c.suit != briscola_suit
                )

        if (is_valid and self.truth) or (not is_valid and not self.truth):
            self.positive = True
        else:
            self.positive = False

    @staticmethod
    def from_action_id(action_id):
        action_id = action_id - len(CARD_STR_TO_NUM)
        truth = action_id < BriscolaCommsAction.NUM_MESSAGES
        message = Messages(action_id % BriscolaCommsAction.NUM_MESSAGES)
        return BriscolaCommsAction(truth, message)

    def to_action_id(self):
        offset = 0 if self.truth else BriscolaCommsAction.NUM_MESSAGES
        return len(CARD_STR_TO_NUM) + self.message.value + offset

    def __str__(self) -> str:
        return "Communicate_" + str(self.truth) + " " + str(self.message)

    def __repr__(self) -> str:
        return self.__str__()
