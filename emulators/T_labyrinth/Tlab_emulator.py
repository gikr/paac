from ..env_T_2 import make_game, T_lab_observation, T_lab_actions

from ..environment import BaseEnvironment

import numpy as np



class TLabyrinthEmulator(BaseEnvironment):
    def __init__(self, actor_id, args):
        self.game = make_game(True, None)
        self.legal_actions = T_lab_actions().shape
        #print(self.legal_actions)
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
        act = [i for i, x in enumerate(action) if x]
        
        if not self.game.game_over:
            obs, reward, discount = self.game.play(act[0])
        termination = 1-discount
        
        observation = T_lab_observation(obs)
        return observation, reward, termination


    def get_legal_actions(self):
        return self.legal_actions

    def get_noop(self):
        return self.noop

    def on_new_frame(self, frame):
        pass
