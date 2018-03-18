from ..env_T_2 import make_game, T_lab_observation

from ..environment import BaseEnvironment

import numpy as np



class TLabyrinthEmulator(BaseEnvironment):
    def __init__(self, actor_id, args):
        self.game = make_game(True, None)
        self.legal_actions = set([0,1,2,3,4])
        self.noop = 'pass'
        self.id = actor_id
        
        self.game = make_game(True, None)
        obs_t, r_t, discount_t = self.game.its_showtime()
        self.observation_shape = T_lab_observation(obs_t).shape


    def get_initial_state(self):
        """Starts a new episode and returns its initial state"""
        matr_obs = []
        
        self.game = make_game(True, None)
        obs_t, r_t, discount_t = self.game.its_showtime()
        obs = T_lab_observation(obs_t)
        
        return obs

    def next(self, action):

        """
        Performs the given action.
        Returns the next state, reward, and terminal signal
        """
        
        if not self.game.game_over:
            observation, reward, discount = self.game.play(action)
        return observation, reward, discount


    def get_legal_actions(self):
        return self.legal_actions

    def get_noop(self):
        return self.noop

    def on_new_frame(self, frame):
        pass
