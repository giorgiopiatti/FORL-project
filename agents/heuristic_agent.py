from typing import Any, Dict, Optional, Union
import gymnasium as gym

import numpy as np

from tianshou.data import Batch
from tianshou.policy import BasePolicy
from tianshou.utils import MultipleLRSchedulers
from enviroment.briscola_gym.actions import PLAY_ACTION_STR_TO_ID 
from enviroment.briscola_gym.utils import CARD_POINTS, CARD_RANK_WITHIN_SUIT_INDEX
from enviroment.briscola_gym.card import Card
import torch 
from enviroment.briscola_gym.utils import Roles

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

class HeuristicAgent(BasePolicy):

    def __init__(self, device, observation_space = None, action_space = None, action_scaling: bool = False, action_bound_method: str = "", lr_scheduler = None) -> None:
        self.device = device
        super().__init__(observation_space, action_space, action_scaling, action_bound_method, lr_scheduler)
    """A random agent used in multi-agent learning.

    It randomly chooses an action from the legal action.
    """

    def heuristic_callee(self, batch, raw_state):
        logits = torch.zeros(40, device=self.device)
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
                            logits[PLAY_ACTION_STR_TO_ID[card]] += 5-CARD_POINTS[rank]
        else:
            winner_card = raw_state.trace_round[0][1].card
            for i, (prev_player, prev_card) in enumerate (raw_state.trace_round):
                prev_player_id = prev_player.player_id

                if (i >= 1):
                    winner_card, winner_index = wins(winner_card, prev_card.card, winner_index, prev_player_id, briscola_suit)
                
                points_in_round += CARD_POINTS[prev_card.card.rank]

                if prev_player_id== caller_id:
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
                    w, _ = wins(winner_card, Card(current_suit, rank), winner_index, raw_state.player_id, briscola_suit)
                    if (w == Card(current_suit, rank)):
                        logits[PLAY_ACTION_STR_TO_ID[card]] += 50
                
            if is_last and winner_index == caller_id:
                for suit in suit_list:
                    if suit != briscola_suit:
                        for rank in rank_list:
                            card = rank+suit
                            logits[PLAY_ACTION_STR_TO_ID[card]] += 5*CARD_POINTS[rank]
        
        return logits
    
    def heuristic_good(self, batch, raw_state):
        logits = torch.zeros(40, device=self.device)
        briscola_suit = raw_state.called_card.suit
        caller_id = raw_state.caller_id

        points_in_round = 0
        caller_played = False
        is_last = len(raw_state.trace_round) == 4
        before_caller = next_pos(raw_state.role) == raw_state.caller_id
        after_caller = prev_pos(raw_state.role) == raw_state.caller_id

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
                        if after_caller: #means caller is last!
                            if CARD_POINTS[rank] < 10:
                                logits[PLAY_ACTION_STR_TO_ID[card]] += 5-CARD_POINTS[rank]
                            else:
                                logits[PLAY_ACTION_STR_TO_ID[card]] -= 5

        else:
            winner_card = raw_state.trace_round[0][1].card
            for i, (prev_player, prev_card) in enumerate (raw_state.trace_round):
                player_id = prev_player.player_id

                if (i >= 1):
                    winner_card, winner_index = wins(winner_card, prev_card.card, winner_index, player_id, briscola_suit)
                
                points_in_round += CARD_POINTS[prev_card.card.rank]

                if player_id== caller_id:
                    caller_played = True

                
            if points_in_round >= 10 and not caller_played: #briscolino 20, briscolone 10, normale kariko -10
                for rank in rank_list:
                    for suit in suit_list:
                        card = rank + suit
                        logits[PLAY_ACTION_STR_TO_ID[card]] -= CARD_POINTS[rank]
                        if suit == briscola_suit:
                            logits[PLAY_ACTION_STR_TO_ID[card]] += 20

            if points_in_round >= 10 and winner_index == caller_id:
                for suit in suit_list:
                    for rank in rank_list:
                        card = rank + suit
                        if suit == briscola_suit:
                            w, _ = wins(winner_card, Card(current_suit, rank), winner_index, raw_state.player_id, suit)

                            if w == Card(current_suit, rank):
                                logits[PLAY_ACTION_STR_TO_ID[card]] += 30
                                if is_last:
                                    logits[PLAY_ACTION_STR_TO_ID[card]] += 11 - CARD_POINTS[rank] #take with smallest briscola if last
                        else:
                            logits[PLAY_ACTION_STR_TO_ID[card]] -= CARD_POINTS[rank]


            if points_in_round <= 6:
                for rank in rank_list:
                        for suit in suit_list:
                            if caller_played: logits[PLAY_ACTION_STR_TO_ID[card]] += CARD_POINTS[rank]
                            else: logits[PLAY_ACTION_STR_TO_ID[card]] -= CARD_POINTS[rank]

            if is_last and winner_card.suit != briscola_suit:
                current_suit = raw_state.trace_round[0][1].card.suit
                for rank in ['3', 'A']:
                    card = rank+current_suit
                    w, _ = wins(winner_card, Card(current_suit, rank), winner_index, raw_state.player_id, briscola_suit)
                    if (w == Card(current_suit, rank)):
                        logits[PLAY_ACTION_STR_TO_ID[card]] += 100
                
        return logits
    

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        global_logits = torch.zeros((len(batch), 40), device=self.device)
        mask = batch.obs.mask
        raw_states = batch.info['raw_state']  # List of state pro env

        for j, raw_state in enumerate(raw_states):
            if (raw_state.role == Roles.CALLEE):
                logits = self.heuristic_callee(batch, raw_state)
            elif (raw_state.role == Roles.CALLER):
                logits = self.heuristic_caller(batch, raw_state)
            else:
                logits = self.heuristic_good(batch, raw_state)
            global_logits[j] = logits

        mask = torch.tensor(mask, device=self.device)
        global_logits = global_logits.masked_fill(~mask, -torch.inf)
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


    
    