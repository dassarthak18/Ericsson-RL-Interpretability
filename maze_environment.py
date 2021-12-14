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

# MAP[5][4] = 'S' -> Start State       

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

       Can move up, down, left or right.

       A reward of +1 for every flag collected.
       An episode ends when goal state is reached.
       """

       metadata = {'render.modes':['human']}
       maze = copy.deepcopy(MAP)
       x = 5
       y = 4

       def __init__(self, arg1, arg2):
              super(MazeEnv, self).__init__()
              self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

       def step(self, action):

       def reset(self):
              maze = copy.deepcopy(MAP)
              x = 5
              y = 4

       def render(self, mode='human', close=False):
