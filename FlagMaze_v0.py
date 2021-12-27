import copy
import gym
import numpy as np
import random
from datetime import datetime
from gym import spaces

MAP = [[1, 1, 1, 1, 1, -1, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [1, 1, 1, 1, 1, 1, 2, -2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [1, 1, 1, 1, 9, 1, 2, 2, 2, 2, 2, 2, 2, 5, 5, -5, 5, 5],
       [6, 6, 6, 6, 9, 6, 7, 7, 7, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [6, 6, 6, 6, 0, 9, 9, 7, 9, 9, 2, 2, 2, 5, 5, 5, 5, 5],
       [6, 9, 6, 6, 6, 6, 7, 7, 7, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [4, 9, 4, 4, 4, 4, 7, 7, 7, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [4, 4, 4, -4, 4, 4, 7, 9, 7, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [4, 4, 4, 4, 4, 4, 3 , 9, 3 , 3 , 3 , 3 , 3 , 5, 5, 5, 5, 5],
       [4, 4, 4, 4, 4, 4, 3 , 3 , 3 , 3 , 3 , -3, 3 , 5, 5, 5, 5, 5],
       [4, 8, 4, 4, 4, 4, 3 , 3 , 3 , 3 , 3 , 3 , 9, 9, 5, 5, 5, 5],
       [4, 4, 4, 4, 4, 4, 3 , 3 , 3 , 3 , 3 , 3 , 3 , 5, 5, 5, 5, -5]]

DICT = {'-':'*', '0':'S', '1':'A', '2':'B', '3':'C', '4':'D', '5':'E', '6':'a', '7':'b', '8':'G', '9':'o', '10':'X'}

ACT_DICT = {'0':'Move up', '1':'Move down', '2':'Move left', '3':'Move right'}

class MazeEnv(gym.Env):

       """
       Map of the Maze as Stored
       -------------------------
       1 1 1 1 1 -1 2 2 2 2 2 2 2 5 5 5 5 5
       1 1 1 1 1 1 2 2 2 2 2 2 2 5 5 5 5 5
       1 1 1 1 1 1 2 -2 2 2 2 2 2 5 5 5 5 5
       1 1 1 1 9 1 2 2 2 2 2 2 2 5 5 -5 5 5
       6 6 6 6 9 6 7 7 7 2 2 2 2 5 5 5 5 5
       6 6 6 6 0 9 9 7 9 9 2 2 2 5 5 5 5 5
       6 9 6 6 6 6 7 7 7 2 2 2 2 5 5 5 5 5
       4 9 4 4 4 4 7 7 7 2 2 2 2 5 5 5 5 5
       4 4 4 -4 4 4 7 9 7 2 2 2 2 5 5 5 5 5
       4 4 4 4 4 4 3 9 3 3 3 3 3 5 5 5 5 5
       4 4 4 4 4 4 3 3 3 3 3 -3 3 5 5 5 5 5
       4 8 4 4 4 4 3 3 3 3 3 3 9 9 5 5 5 5
       4 4 4 4 4 4 3 3 3 3 3 3 3 5 5 5 5 -5

       1 - Room A, 6 - Hall A, 2 - Room B,
       7 - Hall B, 3 - Room C, 4 - Room D,
       5 - Room E, -x - Flag in room x, 0 - Start State,
       8 - Goal State, 9 - Door
       
       10 - Current position
       
       Map of the Maze as Displayed
       ----------------------------
       A A A A A * B B B B B B B E E E E E
       A A A A A A B B B B B B B E E E E E
       A A A A A A B * B B B B B E E E E E
       A A A A o A B B B B B B B E E * E E
       a a a a o a b b b B B B B E E E E E
       a a a a S o o b o o B B B E E E E E
       a o a a a a b b b B B B B E E E E E
       D o D D D D b b b B B B B E E E E E
       D D D * D D b o b B B B B E E E E E
       D D D D D D C o C C C C C E E E E E
       D D D D D D C C C C C * C E E E E E
       D G D D D D C C C C C C o o E E E E
       D D D D D D C C C C C C C E E E E *

       A - Room A, a - Hall A, B - Room B,
       b - Hall B, C - Room C, D - Room D,
       E - Room E, * - Flag, S - Start State,
       G - Goal State, o - Door
       
       X - Current position

       Can move up, down, left or right within a room.
       Can move to another room using doors.

       A reward of +1 for every flag collected.
       An episode ends when goal state is reached.
       
       Observation:
              Type: Box(3)
              Num Observation   Min  Max
              0   x-coordinate  0    17
              1   y-coordinate  0    12
              2   location      -7   9

       Actions:
              Type: Discrete(4)
              Num Action
              0   Move up
              1   Move down
              2   Move left
              3   Move right
       """

       metadata = {'render.modes':['human']}

       def __init__(self):
              super(MazeEnv, self).__init__()
              
              self.action_space = spaces.Discrete(4)
              low = np.array([0,0,-7],dtype=np.int_,)
              high = np.array([17,12,9],dtype=np.int_,)
              self.observation_space = spaces.Box(low, high, dtype=np.int)
              
              self.reset()
              
              print("Map:")              
              for i in self.maze:
                    string = ''
                    for j in i:
                          if str(j)[0] == '-':
                                j = '-'
                          string = string + DICT[str(j)] + ' '
                    print(string)

       def step(self, action):
              
              done = False
              reward = 0

              self.action = action
              new_x = self.x
              new_y = self.y
              
              if self.action == 0 and self.x != 0: # Move up
                     new_x -= 1
              if self.action == 1 and self.x != 17: # Move down
                     new_x += 1
              if self.action == 2 and self.y != 0: # Move left
                     new_y -= 1
              if self.action == 3 and self.y != 12: # Move right
                     new_y += 1
              
              cond1 = False
              cond2 = False
              
              if np.abs(self.maze[self.x][self.y]) == 9 or np.abs(self.maze[new_x][new_y]) == 9: # At the doorstep
                     cond1 = True
              if np.abs(self.maze[new_x][new_y]) == np.abs(self.loc): # In the same room
                     cond2 = True

              if cond1 or cond2:
                     self.maze[self.x][self.y] = self.loc
                     self.x = new_x
                     self.y = new_y
                     self.loc = self.maze[self.x][self.y]
                     self.maze[self.x][self.y] = 10
              
              self.nb_step += 1
              if self.nb_step == 1000:
                     done = True # Limit agent to 1000 steps
                     
              if self.loc == 8:
                     done = True # Return done if goal state is reached
              
              if self.loc < 0:
                     reward = 1 # Immediate reward of +1 on collecting the flag
                     self.loc = self.loc*(-1) # If flag present, then collect the flag and empty the cell

              observation = [self.x,self.y,self.loc]
              info = {}
              
              return observation, reward, done, info

       def reset(self):
              # MAP[5][4] = 10; DICT['10'] = 'S' -> Start State
              self.maze = copy.deepcopy(MAP)
              self.nb_step = 0
              self.x = 5
              self.y = 4
              self.loc = 6
              self.maze[self.x][self.y] = 10
              observation = [self.x,self.y,self.loc]
              return observation

       def render(self, mode='human', close=False):
              print(f"\nNext action:{ACT_DICT[str(self.action)]}")
              print("Map:")
              for i in self.maze:
                    string = ''
                    for j in i:
                          if str(j)[0] == '-':
                                j = '-'
                          string = string + DICT[str(j)] + ' '
                    print(string)
