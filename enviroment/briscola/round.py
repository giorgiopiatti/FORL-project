from enviroment.briscola.utils import CARD_RANK_WITHIN_SUIT_INDEX
from enviroment.briscola.utils import CARD_POINTS


class BriscolaRound:

    def __init__(self, starting_player, briscola_suit):
        self.trace = []
        self.briscola_suit = briscola_suit
        self.current_player = starting_player
        self.round_ended = False

    def update_current_player(self):
        if len(self.trace) == 5:
            self.round_ended = True
        else:
            self.current_player = (self.current_player + 1) % 5

    def proceed_round(self, player, action):
        self.trace.append((player, action))

    def end_round(self):
        # compute winner
        winner_index = 0
        winner_card = self.trace[0][1].card
        for i in range(1, 5):
            current_card = self.trace[i][1].card
            if winner_card.suit == self.briscola_suit:
                if CARD_RANK_WITHIN_SUIT_INDEX[winner_card.rank] < CARD_RANK_WITHIN_SUIT_INDEX[current_card.rank] and current_card.suit == self.briscola_suit:
                    winner_card = current_card
                    winner_index = i
            else:
                if current_card.suit == self.briscola_suit or (CARD_RANK_WITHIN_SUIT_INDEX[current_card.rank] > CARD_RANK_WITHIN_SUIT_INDEX[winner_card.rank] and current_card.suit == winner_card.suit):
                    winner_card = current_card
                    winner_index = i

        # return winner & cardas
        points = 0
        for c in self.trace:
            points += CARD_POINTS[c[1].card.rank]

        return self.trace[winner_index][0].player_id, points
