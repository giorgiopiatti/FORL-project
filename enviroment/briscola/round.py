from briscola.utils import CARD_RANK_WITHIN_SUIT_INXED
from briscola.utils import CARD_POINTS


class BriscolaRound:

    def __init__(self, starting_player, briscola_suit):
        self.trace = []
        self.briscola_suit = briscola_suit
        self.current_player = starting_player

    def update_current_player(self):
        if len(self.trace) == 5:
            self.round_ended = True
        else:
            self.current_player = (self.current_player + 1) % 5

    def proceed_round(self, player, action):
        self.trace.append((player, action))

    def end_round(self):
        # compute winner
        winner = self.trace[0]
        winner_card = self.trace[1]
        for i in range(1, 5):
            if winner_card.suit == self.briscola_suit:
                if CARD_RANK_WITHIN_SUIT_INXED[winner_card.rank] < CARD_RANK_WITHIN_SUIT_INXED[self.trace[i].rank] and self.trace[i].suit == self.briscola_suit:
                    winner_card = self.trace[i]
                    winner = i
            else:
                if self.trace[i].suit == self.briscola_suit or (CARD_RANK_WITHIN_SUIT_INXED[self.trace[i].rank] > CARD_RANK_WITHIN_SUIT_INXED[winner_card.rank] and self.trace[i].suit == winner_card.suit):
                    winner_card = self.trace[i]
                    winner = i

        # return winner & cardas
        points = 0
        for c in self.trace:
            points += CARD_POINTS[c[1]]

        return winner, points
