# -*- coding: utf-8 -*-
''' Implement Briscola Judger class
'''
import numpy as np
import collections
from itertools import combinations
from bisect import bisect_left


class BriscolaJudger:
    ''' Determine what cards a player can play
    '''

    def __init__(self,
                 players,
                 caller_id,
                 callee_id,
                 called_points):
        self.players = players
        self.points = [0 for _ in range(len(self.players))]
        self.called_points = called_points
        self.caller_id = caller_id
        self.callee_id = callee_id

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

    def _bet_payoffs(self, caller_id, callee_id):
        # NOTE Currently not used!
        payoffs = np.array([-1, -1, -1, -1, -1])
        if self.points[caller_id] + self.points[callee_id] >= self.called_points:
            payoffs[caller_id] = 1
            payoffs[callee_id] = 1
        else:
            for i in range(5):
                if i != caller_id and i != callee_id:
                    payoffs[i] = 1
        return payoffs

    def judge_payoffs(self, caller_id, callee_id):
        payoffs = np.zeros((5,))
        bad_team_points = 0
        good_team_points = 0

        for i in range(5):
            if i == caller_id:
                bad_team_points += self.points[i]
            elif i == callee_id:
                bad_team_points += self.points[i]
            else:
                good_team_points += self.points[i]

        for i in range(5):
            if i == caller_id:
                payoffs[i] = bad_team_points
            elif i == callee_id:
                payoffs[i] = bad_team_points
            else:
                payoffs[i] = good_team_points
        return payoffs
