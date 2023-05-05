from typing import Any, Dict, Optional, Union

import numpy as np

from tianshou.data import Batch
from tianshou.policy import BasePolicy
from enviroment.briscola_gym.actions import PLAY_ACTION_STR_TO_ID 
from enviroment.briscola_gym.utils import CARD_POINTS, CARD_RANK_WITHIN_SUIT_INDEX
from enviroment.briscola_gym.card import Card

def wins(card1, card2, briscola_suit):
        winner_card = card1
        prev_card = card2
        if winner_card.suit == briscola_suit:
            if CARD_RANK_WITHIN_SUIT_INDEX[winner_card.rank] < CARD_RANK_WITHIN_SUIT_INDEX[prev_card.rank] and prev_card.suit == briscola_suit:
                winner_card = prev_card
        else:
            if prev_card.suit == briscola_suit or (CARD_RANK_WITHIN_SUIT_INDEX[prev_card.rank] > CARD_RANK_WITHIN_SUIT_INDEX[winner_card.rank] and prev_card.suit == winner_card.suit):
                winner_card = prev_card
        return winner_card

class HeuristicAgent(BasePolicy):
    """A random agent used in multi-agent learning.

    It randomly chooses an action from the legal action.
    """

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        global_logits = np.zeros((len(batch), 40))

        raw_states = batch.info['raw_state']  # List of state pro env
        for j, raw_state in enumerate(raw_states):
            logits = np.zeros(40)
            briscola_suit = raw_state.called_card.suit
            mask = batch.obs.mask
            caller_id = raw_state.caller_id
            rank_list = ['A', '2', '3', '4', '5', '6', '7', 'J', 'Q', 'K']
            suit_list = ['S', 'C', 'D', 'H']

            points_in_round = 0
            caller_played = False
            is_last = len(raw_state.trace_round) == 4

            #winner_id, winner_card = HeuristicAgent.winner(raw_state.trace_round, briscola_suit)
            winner_index = 0
            if len(raw_state.trace_round) == 0:
                for suit in suit_list:
                    if suit != briscola_suit:
                        for rank in rank_list:
                            card = rank+suit
                            if CARD_POINTS[rank] < 10:
                                logits[PLAY_ACTION_STR_TO_ID[card]] += 5-CARD_POINTS[rank]
            else:
                winner_card = raw_state.trace_round[0][1].card
                for i, (prev_player, prev_card) in enumerate (raw_state.trace_round):
                    player_id = prev_player.player_id

                    if (i >= 1):
                        winner_card = wins(winner_card, prev_card.card, briscola_suit)
                    
                    points_in_round += CARD_POINTS[prev_card.card.rank]

                    if player_id== caller_id:
                        caller_played = True
                        if prev_card.card.suit == briscola_suit:
                            for rank in rank_list:
                                card = rank + briscola_suit 
                                logits[PLAY_ACTION_STR_TO_ID[card]] -= CARD_POINTS[rank]

                    
                if points_in_round >= 10 and not caller_played:
                    for rank in rank_list:
                        for suit in suit_list:
                            card = rank + suit
                            logits[PLAY_ACTION_STR_TO_ID[card]] += CARD_POINTS[rank]

                if points_in_round >= 10 and caller_played:
                    for rank in rank_list:
                        card = rank + briscola_suit
                        logits[PLAY_ACTION_STR_TO_ID[card]] += CARD_POINTS[rank]

                if points_in_round <= 6:
                    for rank in rank_list:
                        if CARD_POINTS[rank] == 0:
                            for suit in suit_list:
                                card = rank+suit
                                logits[PLAY_ACTION_STR_TO_ID[card]] += 5

                        if CARD_POINTS[rank] >= 2 and CARD_POINTS[rank] <= 4 :
                            for suit in suit_list:
                                card = rank+suit
                                logits[PLAY_ACTION_STR_TO_ID[card]] += 2

                if is_last and winner_card.suit != briscola_suit:
                    current_suit = raw_state.trace_round[0][1].card.suit
                    for rank in ['3', 'A']:
                        card = rank+current_suit
                        if (wins(winner_card, Card(current_suit, rank), briscola_suit)):
                            logits[PLAY_ACTION_STR_TO_ID[card]] += 50
                    
                if is_last and winner_index == caller_id:
                    for suit in suit_list:
                        if suit != briscola_suit:
                            for rank in rank_list:
                                card = rank+suit
                                logits[PLAY_ACTION_STR_TO_ID[card]] += 5*CARD_POINTS[rank]

            global_logits[j] = logits

        global_logits[~mask] = -np.inf
        return Batch(act=global_logits.argmax(axis=-1))

    def learn(self, batch: Batch, **kwargs: Any) -> Dict[str, float]:
        """Since a random agent learns nothing, it returns an empty dict."""
        return {}

    # @staticmethod
    # def winner(trace_round, briscola_suit): 
    #     winner_index = 0
    #     winner_card = trace_round.card
    #     for i, (prev_player, prev_card) in enumerate (trace_round):
    #         if i >= 1:
    #             if winner_card.suit == briscola_suit:
    #                 if CARD_RANK_WITHIN_SUIT_INDEX[winner_card.rank] < CARD_RANK_WITHIN_SUIT_INDEX[prev_card.rank] and prev_card.suit == briscola_suit:
    #                     winner_card = prev_card
    #                     winner_index = i
    #             else:
    #                 if prev_card.suit == briscola_suit or (CARD_RANK_WITHIN_SUIT_INDEX[prev_card.rank] > CARD_RANK_WITHIN_SUIT_INDEX[winner_card.rank] and prev_card.suit == winner_card.suit):
    #                     winner_card = prev_card
    #                     winner_index = i
        
    #     return trace_round[winner_index][0].player_id, winner_card


    
    