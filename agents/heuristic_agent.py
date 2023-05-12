from typing import Any, Dict, Optional, Union
import numpy as np
from environment.briscola_base.actions import PLAY_ACTION_STR_TO_ID, PlayCardAction
from environment.briscola_base.utils import CARD_POINTS, CARD_RANK_WITHIN_SUIT_INDEX
from environment.briscola_base.card import Card
from environment.briscola_base.utils import Roles

rank_list = ['A', '2', '3', '4', '5', '6', '7', 'J', 'Q', 'K']
suit_list = ['S', 'C', 'D', 'H']


def wins(card1, card2, index1, index2, briscola_suit):
    winner_card = card1
    winner_index = index1
    if winner_card.suit == briscola_suit:
        if CARD_RANK_WITHIN_SUIT_INDEX[winner_card.rank] < CARD_RANK_WITHIN_SUIT_INDEX[card2.rank] and card2.suit == briscola_suit:
            winner_card = card2
            winner_index = index2

    else:
        if card2.suit == briscola_suit or (CARD_RANK_WITHIN_SUIT_INDEX[card2.rank] > CARD_RANK_WITHIN_SUIT_INDEX[winner_card.rank] and card2.suit == winner_card.suit):
            winner_card = card2
            winner_index = index2
    return winner_card, winner_index


def next_pos(player_id):
    return (player_id+1) % 5


def prev_pos(player_id):
    return (player_id-1) % 5


class HeuristicAgent:
    def heuristic_caller(self, raw_state):
        logits = np.zeros(40)
        briscola_suit = raw_state.called_card.suit
        caller_id = raw_state.caller_id

        points_in_round = 0
        is_last = len(raw_state.trace_round) == 4

        #winner_id, winner_card = HeuristicAgent.winner(raw_state.trace_round, briscola_suit)
        callee_id = raw_state.called_card_player
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
                    winner_card, winner_index = wins(winner_card, prev_card.card, winner_index, player_id, briscola_suit)
                
                points_in_round += CARD_POINTS[prev_card.card.rank]

            
    
            if points_in_round >= 20 and winner_index != callee_id:
                for rank in rank_list:
                    card = rank + briscola_suit
                    logits[PLAY_ACTION_STR_TO_ID[card]] += 5*CARD_POINTS[rank]

            elif points_in_round >= 10:
                if callee_id == -1 or callee_id != winner_index:
                    if is_last:
                        for rank in rank_list:
                            card = rank + briscola_suit
                            w, _ = wins(winner_card, Card(briscola_suit, rank), winner_index, raw_state.player_id, briscola_suit)
                            if (w == Card(briscola_suit, rank)):
                                logits[PLAY_ACTION_STR_TO_ID[card]] += 15 - CARD_RANK_WITHIN_SUIT_INDEX[rank]
                            else:
                                logits[PLAY_ACTION_STR_TO_ID[card]] -= 15 
                
                    else:
                        for rank in ['J', 'Q', 'K']:
                            card = rank + briscola_suit
                            w, _ = wins(winner_card, Card(briscola_suit, rank), winner_index, raw_state.player_id, briscola_suit)
                            if (w == Card(briscola_suit, rank)):
                                logits[PLAY_ACTION_STR_TO_ID[card]] += 10
                            else:
                                logits[PLAY_ACTION_STR_TO_ID[card]] -= 20
                
                else:
                    if winner_card.rank in ['Q', 'K', '3', 'A'] or is_last:
                        for rank in rank_list:
                            for suit in suit_list:
                                if suit != briscola_suit:
                                    card = rank + suit
                                    logits[PLAY_ACTION_STR_TO_ID[card]] += 2*CARD_POINTS[rank]
                                    if is_last:
                                        logits[PLAY_ACTION_STR_TO_ID[card]] += 20
                    
                    else:
                        for rank in ['J', 'Q', 'K']:
                            for suit in suit_list:
                                if suit != briscola_suit:
                                    card = rank + suit
                                    logits[PLAY_ACTION_STR_TO_ID[card]] += 5
                    
            else:
                for suit in suit_list:
                    if suit != briscola_suit:
                        for rank in rank_list:
                            card = rank+suit
                            w, _ = wins(winner_card, Card(suit, rank), winner_index, raw_state.player_id, briscola_suit)
                            if (w == winner_card):
                                logits[PLAY_ACTION_STR_TO_ID[card]] += 5-CARD_POINTS[rank]
            
            if is_last and winner_card.suit != briscola_suit:
                current_suit = raw_state.trace_round[0][1].card.suit
                for rank in ['3', 'A']:
                    card = rank+current_suit
                    w, _ = wins(winner_card, Card(current_suit, rank), winner_index, raw_state.player_id, briscola_suit)
                    if (w == Card(current_suit, rank)):
                        logits[PLAY_ACTION_STR_TO_ID[card]] += 100
                    
        return logits

    def heuristic_callee(self, raw_state):
        logits = np.zeros(40)
        briscola_suit = raw_state.called_card.suit
        caller_id = raw_state.caller_id

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
                            logits[PLAY_ACTION_STR_TO_ID[card]] += 5 - \
                                CARD_POINTS[rank]
        else:
            winner_card = raw_state.trace_round[0][1].card
            for i, (prev_player, prev_card) in enumerate(raw_state.trace_round):
                prev_player_id = prev_player.player_id

                if (i >= 1):
                    winner_card, winner_index = wins(
                        winner_card, prev_card.card, winner_index, prev_player_id, briscola_suit)

                points_in_round += CARD_POINTS[prev_card.card.rank]

                if prev_player_id == caller_id:
                    caller_played = True
                    if prev_card.card.suit == briscola_suit:
                        for rank in rank_list:
                            card = rank + briscola_suit
                            logits[PLAY_ACTION_STR_TO_ID[card]
                                   ] -= CARD_POINTS[rank]

            if points_in_round >= 10 and not caller_played:
                for rank in rank_list:
                    for suit in suit_list:
                        card = rank + suit
                        logits[PLAY_ACTION_STR_TO_ID[card]
                               ] += CARD_POINTS[rank]

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

                    if CARD_POINTS[rank] >= 2 and CARD_POINTS[rank] <= 4:
                        for suit in suit_list:
                            card = rank+suit
                            logits[PLAY_ACTION_STR_TO_ID[card]] += 2

            if is_last and winner_card.suit != briscola_suit:
                current_suit = raw_state.trace_round[0][1].card.suit
                for rank in ['3', 'A']:
                    card = rank+current_suit
                    w, _ = wins(winner_card, Card(current_suit, rank),
                                winner_index, raw_state.player_id, briscola_suit)
                    if (w == Card(current_suit, rank)):
                        logits[PLAY_ACTION_STR_TO_ID[card]] += 50

            if is_last and winner_index == caller_id:
                for suit in suit_list:
                    if suit != briscola_suit:
                        for rank in rank_list:
                            card = rank+suit
                            logits[PLAY_ACTION_STR_TO_ID[card]] += 5 * \
                                CARD_POINTS[rank]

        return logits

    def heuristic_good(self, raw_state):
        logits = np.zeros(40)
        briscola_suit = raw_state.called_card.suit
        caller_id = raw_state.caller_id

        points_in_round = 0
        caller_played = False
        is_last = len(raw_state.trace_round) == 4
        before_caller = next_pos(raw_state.player_id) == raw_state.caller_id
        after_caller = prev_pos(raw_state.player_id) == raw_state.caller_id

        #winner_id, winner_card = HeuristicAgent.winner(raw_state.trace_round, briscola_suit)
        winner_index = 0
        if before_caller:
            for rank in rank_list:
                card = rank+briscola_suit
                if CARD_POINTS[rank] == 0:
                    logits[PLAY_ACTION_STR_TO_ID[card]] += 5

        if len(raw_state.trace_round) == 0:
            for suit in suit_list:
                if suit != briscola_suit:
                    for rank in rank_list:
                        card = rank+suit
                        if after_caller:  # means caller is last!
                            if CARD_POINTS[rank] < 10:
                                logits[PLAY_ACTION_STR_TO_ID[card]
                                       ] += 5-CARD_POINTS[rank]
                            else:
                                logits[PLAY_ACTION_STR_TO_ID[card]] -= 5

        else:
            winner_card = raw_state.trace_round[0][1].card
            for i, (prev_player, prev_card) in enumerate(raw_state.trace_round):
                player_id = prev_player.player_id

                if (i >= 1):
                    winner_card, winner_index = wins(
                        winner_card, prev_card.card, winner_index, player_id, briscola_suit)

                points_in_round += CARD_POINTS[prev_card.card.rank]

                if player_id == caller_id:
                    caller_played = True

            if points_in_round >= 10 and not caller_played:  # briscolino 20, briscolone 10, normale kariko -10
                for rank in rank_list:
                    for suit in suit_list:
                        card = rank + suit
                        logits[PLAY_ACTION_STR_TO_ID[card]
                               ] -= CARD_POINTS[rank]
                        if suit == briscola_suit:
                            logits[PLAY_ACTION_STR_TO_ID[card]] += 20

            if points_in_round >= 10 and winner_index == caller_id:
                for suit in suit_list:
                    for rank in rank_list:
                        card = rank + suit
                        if suit == briscola_suit:
                            w, _ = wins(winner_card, Card(suit, rank),
                                        winner_index, raw_state.player_id, suit)

                            if w == Card(suit, rank):
                                logits[PLAY_ACTION_STR_TO_ID[card]] += 30
                                if is_last:
                                    # take with smallest briscola if last
                                    logits[PLAY_ACTION_STR_TO_ID[card]
                                           ] += 11 - CARD_POINTS[rank]
                        else:
                            logits[PLAY_ACTION_STR_TO_ID[card]
                                   ] -= CARD_POINTS[rank]

            if points_in_round <= 6:
                for rank in rank_list:
                    for suit in suit_list:
                        card = rank + suit
                        if caller_played:
                            logits[PLAY_ACTION_STR_TO_ID[card]
                                   ] += CARD_POINTS[rank]
                        else:
                            logits[PLAY_ACTION_STR_TO_ID[card]
                                   ] -= CARD_POINTS[rank]

            if is_last and winner_card.suit != briscola_suit:
                current_suit = raw_state.trace_round[0][1].card.suit
                for rank in ['3', 'A']:
                    card = rank+current_suit
                    w, _ = wins(winner_card, Card(current_suit, rank),
                                winner_index, raw_state.player_id, briscola_suit)
                    if (w == Card(current_suit, rank)):
                        logits[PLAY_ACTION_STR_TO_ID[card]] += 100

        return logits

    def get_heuristic_action(self, raw_state, available_actions):
        if (raw_state.role == Roles.CALLEE):
            logits = self.heuristic_callee(raw_state)
        elif (raw_state.role == Roles.CALLER):
            logits = self.heuristic_caller(raw_state)
        else:
            logits = self.heuristic_good(raw_state)

        action_mask = np.zeros_like(logits, dtype=np.bool_)
        for i in available_actions:
            action_mask[i] = True

        logits[~action_mask] = -np.inf
        action = logits.argmax()
        return PlayCardAction.from_action_id(action)
