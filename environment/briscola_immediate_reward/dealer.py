import functools

from environment.briscola_immediate_reward.card import Card


def init_40_deck():
    '''
    Returns:
        (list): A list of Card object
    '''
    suit_list = ['S', 'H', 'D', 'C']
    rank_list = ['A', '2', '3', '4', '5', '6', '7', 'J', 'Q', 'K']
    res = [Card(suit, rank) for suit in suit_list for rank in rank_list]
    return res


class BriscolaDealer:
    ''' Dealer will shuffle, deal cards
    '''

    def __init__(self, np_random):
        '''Give dealer the deck

        '''
        self.np_random = np_random
        self.deck = init_40_deck()
        self.deck.sort()

    def shuffle(self):
        ''' Randomly shuffle the deck
        '''
        self.np_random.shuffle(self.deck)

    def deal_cards(self, players):
        ''' Deal cards to players

        Args:
            players (list): list of BriscolaPlayer objects
        '''
        hand_num = 8
        for index, player in enumerate(players):
            current_hand = self.deck[index*hand_num:(index+1)*hand_num]
            current_hand.sort()
            player.set_current_hand(current_hand)
            player.initial_hand = player.current_hand
