# ''' Game-related base classes
# '''


# from enviroment.briscola.utils import CARD_RANK_WITHIN_SUIT_INDEX


# class Card:
#     '''
#     Card stores the suit and rank of a single card

#     Note:
#         The suit variable in a standard card game should be one of [S, H, D, C] meaning [Spades, Hearts, Diamonds, Clubs]
#         Similarly the rank variable should be one of [A, 2, 3, 4, 5, 6, 7, J, Q, K]
#     '''
#     suit = None
#     rank = None
#     valid_suit = ['S', 'H', 'D', 'C']
#     valid_rank = ['A', '2', '3', '4', '5',
#                   '6', '7', 'J', 'Q', 'K']

#     def __init__(self, suit, rank):
#         ''' Initialize the suit and rank of a card

#         Args:
#             suit: string, suit of the card, should be one of valid_suit
#             rank: string, rank of the card, should be one of valid_rank
#         '''
#         self.suit = suit
#         self.rank = rank

#     def __eq__(self, other):
#         if isinstance(other, Card):
#             return self.rank == other.rank and self.suit == other.suit
#         else:
#             # don't attempt to compare against unrelated types
#             return NotImplemented

#     def __hash__(self):
#         suit_index = Card.valid_suit.index(self.suit)
#         rank_index = Card.valid_rank.index(self.rank)
#         return rank_index + len(Card.valid_rank) * suit_index

#     def __str__(self):
#         ''' Get string representation of a card.

#         Returns:
#             string: the combination of rank and suit of a card. Eg: AS, 5H, JD, 3C, ...
#         '''
#         return self.rank + self.suit

#     def __repr__(self):
#         ''' Get string representation of a card.

#         Returns:
#             string: the combination of rank and suit of a card. Eg: AS, 5H, JD, 3C, ...
#         '''
#         return self.rank + self.suit

#     def get_index(self):
#         ''' Get index of a card.

#         Returns:
#             string: the combination of suit and rank of a card. Eg: 1S, 2H, AD, BJ, RJ...
#         '''
#         return self.suit+self.rank

#     def get_index_suit(self):
#         return Card.valid_suit.index(self.suit)

#     def get_index_rank(self):
#         return Card.valid_rank.index(self.rank)

#     @staticmethod
#     def init_from_string(card_string):
#         return Card(rank=card_string[0], suit=card_string[1])

#     def __lt__(self, other):
#         if self.suit == other.suit:
#             return CARD_RANK_WITHIN_SUIT_INDEX[self.rank] < CARD_RANK_WITHIN_SUIT_INDEX[other.rank]
#         else:
#             return self.suit < other.suit
