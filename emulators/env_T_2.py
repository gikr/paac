from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import curses
import sys
import numpy as np
import pandas as pd
import time

from pycolab import ascii_art
from pycolab import human_ui
from pycolab.prefab_parts import sprites as prefab_sprites
from pycolab.prefab_parts import drapes as prefab_drapes


GAME_ART = [
    ['#########',
     '#L     R#',
     '@#@# #@#@',
     '#@#@ @#@#',  # Testing environment v1.плюсики нужны для самого начала - от какой точки начинаем показывать
     '##@# #@##',
     '@### ###@',
     '@@## ##@@',
     '+#@@ @@##',
     '#@@# H@@#',
     '@##@A@##@',
     '#########'],

    ['#########',
     '#L     R#',
     '@#@# #@#@',
     '#@#@ @#@#',  # Testing environment v2
     '##@# #@##',
     '@### ###@',
     '@@## ##@@',
     '+#@@ @@##',
     '#@@H #@@#',
     '@##@A@##@',
     '#########']
]


AGENT_CHR = 'A'
GOAL_CHR1 = 'L'
GOAL_CHR2 = 'R'
HINT_CHR = 'H'

MOVEMENT_REWARD = -1 
GOAL_REWARD = 100
HINT_REWARD = 20



def make_game(is_testing, level_choice):
  global LEFT_REWARD 
  global RIGHT_REWARD
  if level_choice is None:
    if is_testing:
      # If the agent is in testing mode, randomly choose a Goal location.
      level_choice = np.random.choice([0, 1]) 
    else:
      level_choice = 0
  
  scrolly_info = prefab_drapes.Scrolly.PatternInfo(
      GAME_ART[level_choice], STAR_ART, board_northwest_corner_mark='+',
      what_lies_beneath=MAZES_WHAT_LIES_BENEATH[0],
       )
  if level_choice == 0:
     LEFT_REWARD = -100
     RIGHT_REWARD = 100
  else: 
     LEFT_REWARD = 100
     RIGHT_REWARD = -100
  game = GAME_ART[level_choice] 
  player_position = scrolly_info.virtual_position('A')
  left_goal_kwarg = scrolly_info.kwargs('L')
  right_goal_kwarg = scrolly_info.kwargs('R')
  hint_position = scrolly_info.kwargs('H')
  
 
  wall_1_kwargs = scrolly_info.kwargs('#')
  wall_2_kwargs = scrolly_info.kwargs('@')

  return ascii_art.ascii_art_to_game(
      STAR_ART, what_lies_beneath=' ',
      sprites={'A': ascii_art.Partial(AgentSprite, player_position)},
      drapes={'#': ascii_art.Partial(MazeDrape, **wall_1_kwargs),
              '@': ascii_art.Partial(MazeDrape, **wall_2_kwargs),
               'L': ascii_art.Partial(MazeDrape, **left_goal_kwarg),
               'R': ascii_art.Partial(MazeDrape, **right_goal_kwarg),
               'H': ascii_art.Partial(MazeDrape, **hint_position)},
      update_schedule=[['#', 'H','@'], ['A', 'L', 'R']]) #важно! для того, чтобы обозреваемая часть среды менялась. беэ этого иногда застревает
  
MAZES_WHAT_LIES_BENEATH = [  #что показывать вместо "+"
    '#'
]
  
STAR_ART = ['         ',
	    '    .    ',
	    '         ',
            '    .    ']   #схема того, какого размера часть хотим видеть




class AgentSprite(prefab_sprites.MazeWalker):
  
  def __init__(self, corner, position, character, virtual_position):
    """Inform superclass that we can't walk through walls."""
    super(AgentSprite, self).__init__(
        corner, position, character, egocentric_scroller=True, impassable={'#', 'H','@'})
    self._teleport(virtual_position)

  def update(self, actions, board, layers, backdrop, things, the_plot):
    del backdrop  # Unused.

 
    
    # Apply motion commands.
    if actions == 0:    # walk upward?
      self._north(board, the_plot)
      the_plot.add_reward(MOVEMENT_REWARD)
      
    elif actions == 1:  # walk downward?
      self._south(board, the_plot)
      the_plot.add_reward(MOVEMENT_REWARD)

    elif actions == 2:  # walk leftward?
      self._west(board, the_plot)
      the_plot.add_reward(MOVEMENT_REWARD)
     # if layers['H'][things['A'].position]  == True:   #вознаграждение за пребывание в подсказке
     #    the_plot.add_reward(HINT_REWARD)
     # else: the_plot.add_reward(MOVEMENT_REWARD)
      
    elif actions == 3:  # walk rightward?
      self._east(board, the_plot)
      the_plot.add_reward(MOVEMENT_REWARD)

    elif actions == 4:  # is the player doing nothing?
      self._stay(board, the_plot)
      the_plot.add_reward(MOVEMENT_REWARD)
    
    if layers['L'][things['A'].position] == True:
      the_plot.add_reward(LEFT_REWARD)
      the_plot.terminate_episode()

    if layers['R'][things['A'].position] == True:
      the_plot.add_reward(RIGHT_REWARD)
      the_plot.terminate_episode()
   
  #the_plot.terminate_episode()
   
class MazeDrape(prefab_drapes.Scrolly):
  def update(self, actions, board, layers, backdrop, things, the_plot):
    #del backdrop, things, layers  # Unused
																		#print(actions) - lovely Nones
    if actions == 0:    # is the player going upward?
      self._north(the_plot)
    elif actions == 1:  # is the player going downward?
      self._south(the_plot)
    elif actions == 2:  # is the player going leftward?
      self._west(the_plot)
      
    elif actions == 3:  # is the player going rightward?
      self._east(the_plot)
    elif actions == 4:  # is the player doing nothing?
      self._stay(the_plot)      
    
    
    

   

def main(argv=()):
  del argv  # Unused.
  
  # Build a game.
  
  game = make_game(True, None)
  

  # Make a CursesUi to play it with.
  ui = human_ui.CursesUi(
      keys_to_actions={curses.KEY_UP: 0, curses.KEY_DOWN:1,
                       curses.KEY_LEFT: 2, curses.KEY_RIGHT: 3,
                       -1: 4}, #curses.KEY_DOWN: not using
      delay=200)
  
  # Let the game begin!
  ui.play(game)
  
def print_obs(obs):
    matr = []
    obs, info = obs
    obs = obs.tolist()
    print("info", info)
    print("end info")
    print("obs", obs)
    print(type(info), type(obs))
    print("end obs")
    #a = np.array(pd.DataFrame.from_dict(info))
    #print(a)
    #matr.append(1*info['A']) #,1*info['H'], info['@'], info['#'], info['L'], info['R'] )
    #matr.append(1*info['H'])
    keys = list(info.keys())
    for i in range(len(keys)):
       key = keys[i]
       if all([key != '.', key != ' ']):
          print("I am here", key)
          matr.append(1*info[key])
    matr = np.array(matr)

       
    print(matr, type(matr))
    print(matr.shape)
    for i in range(len(obs)):
        obs[i] = ''.join([chr(ch) for ch in obs[i]])
        print(obs[i])


def dummy_episode():
    import numpy as np
    game = make_game(True, None)
    print(game.game_over )
    action_keys = ['up', 'down', 'left', 'right', 'noop']
    obs_t, r_t, discount_t = game.its_showtime()
    for t in range(1,101):
        a_t = None
        
        while a_t not in action_keys:
            a_t = input("Choose one of the following actions: {}:\n".format(action_keys))
        obs_t, r_t, discount_t = game.play(action_keys.index(a_t))
        print('r =', r_t, 'gamma = ', discount_t)
        #obs_t, r_t = game.play(action_keys.index(a_t))
        
        if discount_t == 0: break
        print('===========  Step #{}  ==========:'.format(t))
    
    print('Done!')
    
def T_lab_observation(obs_t):
	matr_obs = []
	obs,info = obs_t 
	keys = list(info.keys())
	for i in range(len(keys)):
		key = keys[i]
		if all([key != '.', key != ' ']):
			matr_obs.append(1*info[key])
	matr_obs = np.array(matr_obs)
	return matr_obs
	

def T_lab_actions():
	action_keys = [0, 1, 2, 3, 4]
	return(np.ndarray(action_keys))
	
	
if __name__ == '__main__':
  
  dummy_episode()
  #main(sys.argv)
 
  

	#game = make_game(True, None)
	#obs_t, r_t, discount_t = game.its_showtime()
	#obs,info = 0, 0
