
CARD_RANK_WITHIN_SUIT = [
    '2', '4', '5', '6', '7', 'J', 'Q', 'K', '3', 'A'
]

CARD_RANK_WITHIN_SUIT_INXED = {
    '2': 0, '4': 1, '5': 2, '6': 3, '7': 4, 'J': 5, 'Q': 6, 'K': 7, '3': 8, 'A': 9
}

CARD_POINTS = {
    '2': 0, '4': 0, '5': 0, '6': 0, '7': 0, 'J': 2, 'Q': 3, 'K': 4, '3': 10, 'A': 11
}


def briscola_sort_card(card_1, card_2):
    ''' Compare the rank of two cards of Card object

    Args:
        card_1 (object): object of Card
        card_2 (object): object of card
    '''
    if card_1.rank == card_2.rank:
        return card_1.suit > card_2.suit
    else:
        return card_1.rank > card_2.rank


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
