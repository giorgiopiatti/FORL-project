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
        self.games_is_over = False
        self.callee_revealed = False
        self.round_winner = None

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
        if self.round_winner is None:
            return np.zeros((5,))

        # Game end logic
        if self.callee_revealed and not self.games_is_over:
            payoffs = np.zeros((5,))

            if self.round_winner == self.caller_id:
                payoffs[self.round_winner] = self.round_points
                payoffs[self.callee_id] = self.round_points
            elif self.round_winner == self.callee_id:
                payoffs[self.round_winner] = self.round_points
                payoffs[self.caller_id] = self.round_points
            else:
                for i in range(5):
                    if i != self.caller_id and i != self.callee_id:
                        payoffs[i] = self.round_points

        elif self.games_is_over:
            payoffs = np.zeros((5,))

            for i in range(5):
                if i == caller_id:
                    if self.round_winner == callee_id or self.round_winner == caller_id:
                        payoffs[i] = self.round_points
                    payoffs[i] += self.points_before_revelation[callee_id]
                elif i == callee_id:
                    if self.round_winner == callee_id or self.round_winner == caller_id:
                        payoffs[i] = self.round_points
                else:

                    for j in range(5):
                        if self.round_winner == j and j != callee_id and j != caller_id:
                            payoffs[i] = self.round_points

                    for j in range(5):
                        if j != callee_id and j != caller_id and i != j:
                            payoffs[i] += self.points_before_revelation[j]

        else:
            payoffs = np.zeros((5,))
            payoffs[self.round_winner] = self.round_points
            if self.round_winner == self.caller_id:
                payoffs[self.callee_id] = self.round_points

        self.round_winner = None
        return payoffs

    def callee_is_revealed(self):
        self.callee_revealed = True
        self.points_before_revelation = self.points.copy()

    def update_round(self, winner, points):
        self.round_points = points
        self.round_winner = winner

    def payoffs_end_game(self, caller_id, callee_id):
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
