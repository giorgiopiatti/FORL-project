
from enum import Enum
DEBUG_ENV = False

CARD_RANK_WITHIN_SUIT = [
    '2', '4', '5', '6', '7', 'J', 'Q', 'K', '3', 'A'
]

CARD_RANK_WITHIN_SUIT_INDEX = {
    '2': 0, '4': 1, '5': 2, '6': 3, '7': 4, 'J': 5, 'Q': 6, 'K': 7, '3': 8, 'A': 9
}

CARD_POINTS = {
    '2': 0, '4': 0, '5': 0, '6': 0, '7': 0, 'J': 2, 'Q': 3, 'K': 4, '3': 10, 'A': 11
}


def cards2str(cards):
    ''' Get the corresponding string representation of cards

    Args:
        cards (list): list of Card objects

    Returns:
        string: string representation of cards
    '''
    response = ''
    for card in cards:
        response += str(card)
    return response


# class syntax


class Roles(Enum):
    GOOD_PLAYER = 1
    CALLER = 2
    CALLEE = 3
