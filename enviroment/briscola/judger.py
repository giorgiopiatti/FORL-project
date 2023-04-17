# -*- coding: utf-8 -*-
''' Implement Doudizhu Judger class
'''
import numpy as np
import collections
from itertools import combinations
from bisect import bisect_left

from briscola.utils import CARD_RANK_STR, CARD_RANK_STR_INDEX
from briscola.utils import cards2str, contains_cards


class BriscolaJudger:
    ''' Determine what cards a player can play
    '''

    def __init__(self, players, called_points):
        self.players = players
        self.points = [0 for _ in len(self.players)]
        self.called_points = called_points

    @staticmethod
    def judge_game(players, player_id):
        ''' Judge whether the game is over

        Args:
            players (list): list of  BriscolaPlayer objects
            player_id (int): integer of player's id

        Returns:
            (bool): True if the game is over
        '''
        player = players[player_id]
        if not player.current_hand:
            return True
        return False

    def judge_payoffs(self, caller_id, callee_id):
        payoffs = np.array([-1, -1, -1, -1, -1])
        if self.points[caller_id] + self.points[callee_id] >= self.called_points:
            payoffs[caller_id] = 1
            payoffs[callee_id] = 1
        else:
            for i in range(5):
                if i != caller_id and i != callee_id:
                    payoffs[i] = 1
        return payoffs
