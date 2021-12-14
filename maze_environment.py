import copy
import gym
from gym import spaces

MAP = [['A', 'A', 'A', 'A', 'A', '*', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'E', 'E', 'E', 'E', 'E'],
['A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'E', 'E', 'E', 'E', 'E'],
['A', 'A', 'A', 'A', 'A', 'A', 'B', '*', 'B', 'B', 'B', 'B', 'B', 'E', 'E', 'E', 'E', 'E'],
['A', 'A', 'A', 'A', 'o', 'A', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'E', 'E', '*', 'E', 'E'],
['a', 'a', 'a', 'a', 'o', 'a', 'b', 'b', 'b', 'B', 'B', 'B', 'B', 'E', 'E', 'E', 'E', 'E'],
['a', 'a', 'a', 'a', 'S', 'o', 'o', 'b', 'o', 'o', 'B', 'B', 'B', 'E', 'E', 'E', 'E', 'E'],
['a', 'o', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'B', 'B', 'B', 'B', 'E', 'E', 'E', 'E', 'E'],
['D', 'o', 'D', 'D', 'D', 'D', 'b', 'b', 'b', 'B', 'B', 'B', 'B', 'E', 'E', 'E', 'E', 'E'],
['D', 'D', 'D', '*', 'D', 'D', 'b', 'o', 'b', 'B', 'B', 'B', 'B', 'E', 'E', 'E', 'E', 'E'],
['D', 'D', 'D', 'D', 'D', 'D', 'C', 'o', 'C', 'C', 'C', 'C', 'C', 'E', 'E', 'E', 'E', 'E'],
['D', 'D', 'D', 'D', 'D', 'D', 'C', 'C', 'C', 'C', 'C', '*', 'C', 'E', 'E', 'E', 'E', 'E'],
['D', 'G', 'D', 'D', 'D', 'D', 'C', 'C', 'C', 'C', 'C', 'C', 'o', 'o', 'E', 'E', 'E', 'E'],
['D', 'D', 'D', 'D', 'D', 'D', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'E', 'E', 'E', 'E', '*']]

class MazeEnv(gym.Env):

       """
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
              Type: Discrete(3)
              Num Observation
              0   x-coordinate
              1   y-coordinate
              2   location

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

       def step(self, action):

       def reset(self):

       def render(self, mode='human', close=False):
