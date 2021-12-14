import copy
import gym
import numpy as np
from gym import spaces

MAP = [[1, 1, 1, 1, 1, -1, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [1, 1, 1, 1, 1, 1, 2, -1, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [1, 1, 1, 1, 9, 1, 2, 2, 2, 2, 2, 2, 2, 5, 5, -1, 5, 5],
       [6, 6, 6, 6, 9, 6, 7, 7, 7, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [6, 6, 6, 6, 0, 9, 9, 7, 9, 9, 2, 2, 2, 5, 5, 5, 5, 5],
       [6, 9, 6, 6, 6, 6, 7, 7, 7, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [4, 9, 4, 4, 4, 4, 7, 7, 7, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [4, 4, 4, -1, 4, 4, 7, 9, 7, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [4, 4, 4, 4, 4, 4, 3 , 9, 3 , 3 , 3 , 3 , 3 , 5, 5, 5, 5, 5],
       [4, 4, 4, 4, 4, 4, 3 , 3 , 3 , 3 , 3 , -1, 3 , 5, 5, 5, 5, 5],
       [4, 8, 4, 4, 4, 4, 3 , 3 , 3 , 3 , 3 , 3 , 9, -1, 5, 5, 5, 5],
       [4, 4, 4, 4, 4, 4, 3 , 3 , 3 , 3 , 3 , 3 , 3 , 5, 5, 5, 5, -1]]

DICT = {'-1':'*', '0':'S', '1':'A', '2':'B', '3':'C', '4':'D', '5':'E', '6':'a', '7':'b', '8':'G', '9':'o'}

class MazeEnv(gym.Env):

       """
       Map of the Maze as Stored
       -------------------------
       1 1 1 1 1 -1 2 2 2 2 2 2 2 5 5 5 5 5
       1 1 1 1 1 1 2 2 2 2 2 2 2 5 5 5 5 5
       1 1 1 1 1 1 2 -1 2 2 2 2 2 5 5 5 5 5
       1 1 1 1 9 1 2 2 2 2 2 2 2 5 5 -1 5 5
       6 6 6 6 9 6 7 7 7 2 2 2 2 5 5 5 5 5
       6 6 6 6 0 9 9 7 9 9 2 2 2 5 5 5 5 5
       6 9 6 6 6 6 7 7 7 2 2 2 2 5 5 5 5 5
       4 9 4 4 4 4 7 7 7 2 2 2 2 5 5 5 5 5
       4 4 4 -1 4 4 7 9 7 2 2 2 2 5 5 5 5 5
       4 4 4 4 4 4 3 9 3 3 3 3 3 5 5 5 5 5
       4 4 4 4 4 4 3 3 3 3 3 -1 3 5 5 5 5 5
       4 8 4 4 4 4 3 3 3 3 3 3 9 9 5 5 5 5
       4 4 4 4 4 4 3 3 3 3 3 3 3 5 5 5 5 -1

       1 - Room A, 6 - Hall A, 2 - Room B,
       7 - Hall B, 3 - Room C, 4 - Room D,
       5 - Room E, -1 - Flag, 0 - Start State,
       8 - Goal State, 9 - Door
       
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

       Can move up, down, left or right within a room.
       Can move to another room using doors.

       A reward of +1 for every flag collected.
       An episode ends when goal state is reached.
       
       Observation:
              Type: Box(3)
              Num Observation   Min  Max
              0   x-coordinate  0    17
              1   y-coordinate  0    12
              2   location      -1   9

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

              # MAP[5][4] = 'S' -> Start State
              
              self.maze = copy.deepcopy(MAP)
              self.x = 5
              self.y = 4
              self.loc = maze[x][y]
              self.maze[x][y] = 'X'
              
              self.action_space = spaces.Discrete(4)
              self.observation_space = spaces.Box(-high, high, dtype=np.int_)

       def step(self, action):

       def reset(self):

       def render(self, mode='human', close=False):
