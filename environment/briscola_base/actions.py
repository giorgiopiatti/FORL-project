from environment.briscola_base.card import Card

from abc import (
    ABC,
    abstractmethod,
)


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
        self.action_type = 'play_card'
        self.card = card

    def __str__(self) -> str:
        return 'Play_' + str(self.card)

    @staticmethod
    def from_action_id(action_id):
        s = PLAY_ACTION_ID_TO_STR[action_id]
        return PlayCardAction(card=Card.init_from_string(s))

    def to_action_id(self):
        return PLAY_ACTION_STR_TO_ID[str(self.card)]

    def __repr__(self) -> str:
        return f'{self.card}'


PLAY_ACTION_ID_TO_STR = {0: 'AS', 1: '2S', 2: '3S', 3: '4S', 4: '5S', 5: '6S', 6: '7S', 7: 'JS', 8: 'QS', 9: 'KS', 10: 'AH', 11: '2H', 12: '3H', 13: '4H', 14: '5H', 15: '6H', 16: '7H', 17: 'JH', 18: 'QH',
                         19: 'KH', 20: 'AD', 21: '2D', 22: '3D', 23: '4D', 24: '5D', 25: '6D', 26: '7D', 27: 'JD', 28: 'QD', 29: 'KD', 30: 'AC', 31: '2C', 32: '3C', 33: '4C', 34: '5C', 35: '6C', 36: '7C', 37: 'JC', 38: 'QC', 39: 'KC'}
PLAY_ACTION_STR_TO_ID = {'AS': 0, '2S': 1, '3S': 2, '4S': 3, '5S': 4, '6S': 5, '7S': 6, 'JS': 7, 'QS': 8, 'KS': 9, 'AH': 10, '2H': 11, '3H': 12, '4H': 13, '5H': 14, '6H': 15, '7H': 16, 'JH': 17, 'QH': 18,
                         'KH': 19, 'AD': 20, '2D': 21, '3D': 22, '4D': 23, '5D': 24, '6D': 25, '7D': 26, 'JD': 27, 'QD': 28, 'KD': 29, 'AC': 30, '2C': 31, '3C': 32, '4C': 33, '5C': 34, '6C': 35, '7C': 36, 'JC': 37, 'QC': 38, 'KC': 39}
