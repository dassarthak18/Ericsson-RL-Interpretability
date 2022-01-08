import copy
import gym
import numpy as np
import random
from gym import spaces

MAP = [[1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [1, 1, 1, 1, 9, 1, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [6, 6, 6, 6, 9, 6, 7, 7, 7, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [6, 6, 6, 6, 6, 9, 9, 7, 9, 9, 2, 2, 2, 5, 5, 5, 5, 5],
       [6, 9, 6, 6, 6, 6, 7, 7, 7, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [4, 9, 4, 4, 4, 4, 7, 7, 7, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [4, 4, 4, 4, 4, 4, 7, 9, 7, 2, 2, 2, 2, 5, 5, 5, 5, 5],
       [4, 4, 4, 4, 4, 4, 3, 9, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5],
       [4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5],
       [4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 9, 9, 5, 5, 5, 5],
       [4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]]

DICT = {'-':'*', '0':'S', '1':'A', '2':'B', '3':'C', '4':'D', '5':'E', '6':'a', '7':'b', '8':'G', '9':'o', '10':'X'}
ACT_DICT = {'0':'Move up', '1':'Move down', '2':'Move left', '3':'Move right'}

class ComplexMaze(gym.Env):

       """
       Map of the Maze as Stored
       -------------------------
       1 1 1 1 1 1 2 2 2 2 2 2 2 5 5 5 5 5
       1 1 1 1 1 1 2 2 2 2 2 2 2 5 5 5 5 5
       1 1 1 1 1 1 2 2 2 2 2 2 2 5 5 5 5 5
       1 1 1 1 9 1 2 2 2 2 2 2 2 5 5 5 5 5
       6 6 6 6 9 6 7 7 7 2 2 2 2 5 5 5 5 5
       6 6 6 6 6 9 9 7 9 9 2 2 2 5 5 5 5 5
       6 9 6 6 6 6 7 7 7 2 2 2 2 5 5 5 5 5
       4 9 4 4 4 4 7 7 7 2 2 2 2 5 5 5 5 5
       4 4 4 4 4 4 7 9 7 2 2 2 2 5 5 5 5 5
       4 4 4 4 4 4 3 9 3 3 3 3 3 5 5 5 5 5
       4 4 4 4 4 4 3 3 3 3 3 3 3 5 5 5 5 5
       4 4 4 4 4 4 3 3 3 3 3 3 9 9 5 5 5 5
       4 4 4 4 4 4 3 3 3 3 3 3 3 5 5 5 5 5

       1 - Room A, 6 - Hall A, 2 - Room B,
       7 - Hall B, 3 - Room C, 4 - Room D,
       5 - Room E, 9 - Door

       Alloted randomly:
       0 - Start State, 8 - Goal State, x - Flag in room x
       
       10 - Current position
       
       Map of the Maze as Displayed
       ----------------------------
       A A A A A A B B B B B B B E E E E E
       A A A A A A B B B B B B B E E E E E
       A A A A A A B B B B B B B E E E E E
       A A A A o A B B B B B B B E E E E E
       a a a a o a b b b B B B B E E E E E
       a a a a a o o b o o B B B E E E E E
       a o a a a a b b b B B B B E E E E E
       D o D D D D b b b B B B B E E E E E
       D D D D D D b o b B B B B E E E E E
       D D D D D D C o C C C C C E E E E E
       D D D D D D C C C C C E C E E E E E
       D D D D D D C C C C C C o o E E E E
       D D D D D D C C C C C C C E E E E E

       A - Room A, a - Hall A, B - Room B,
       b - Hall B, C - Room C, D - Room D,
       E - Room E, o - Door

       Alloted randomly:
       * - Flag, S - Start State, G - Goal State,
       
       X - Current position

       Can move up, down, left or right within a room.
       Can move to another room using doors.

       Deterministic or non-deterministic in nature.

       A penalty of -1 for every step taken, -5 if all the flags have been collected.
       A reward of +5 for every new room explored.
       A reward of +10 for every flag collected.
       A reward of +20 for reaching the goal state.

       An episode ends when either goal state or the maximum number of steps is reached, whichever earlier.
       
       Observation:
              Type: Box(3)
              Num Observation   Min  Max
              0   x-coordinate  0    12
              1   y-coordinate  0    17
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

       def __init__(self, deterministic, steps=-1): # Default -1; no maximum number of steps
              super(ComplexMaze, self).__init__()
              self.action_space = spaces.Discrete(4)
              self.observation_space = spaces.Box(np.array([0,0,-7],dtype=np.int_,), np.array([12,17,9],dtype=np.int_,), dtype=np.int)
              self.deterministic = deterministic
              self.max_step = steps
              self.reset()
              self.print()

       def reset(self):
              self.maze = copy.deepcopy(MAP)
              self.nb_step = 0
              self.nb_flags = 0
              self.unvisited = [1,2,3,4,5,7]
              cells = [(a,b) for a in range(13) for b in range(18)]
              # Choosing an S
              tup = random.choice(cells)
              cells.remove(tup)
              self.x = tup[0]
              self.y = tup[1]
              self.loc = self.maze[self.x][self.y]
              self.maze[self.x][self.y] = 10
              #Choosing a G
              tup = random.choice(cells)
              cells.remove(tup)
              self.maze[tup[0]][tup[1]] = 8
              # Choosing six flags
              for i in range(6):
                     tup = random.choice(cells)
                     cells.remove(tup)
                     self.maze[tup[0]][tup[1]] *= -1
              return [self.x,self.y,self.loc]

       def print(self):
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
              if self.nb_flags == 6: # Negative reward for every step taken
                     reward = -5
              else:
                     reward = -1
              if self.deterministic == True:
                     self.action = action
              else:
                     self.action = self.probability_matrix(action)

              new_x = self.x
              new_y = self.y
              
              if self.action == 0 and self.x != 0: # Move up
                     new_x -= 1
              if self.action == 1 and self.x != 12: # Move down
                     new_x += 1
              if self.action == 2 and self.y != 0: # Move left
                     new_y -= 1
              if self.action == 3 and self.y != 17: # Move right
                     new_y += 1

              # At the doorstep or movement within the room
              if np.abs(self.loc) == 9 or np.abs(self.maze[new_x][new_y]) == 9 or np.abs(self.maze[new_x][new_y]) == np.abs(self.loc):
                     self.maze[self.x][self.y] = self.loc
                     self.x = new_x
                     self.y = new_y
                     self.loc = self.maze[self.x][self.y]
                     self.maze[self.x][self.y] = 10

              if np.abs(self.loc) in self.unvisited: # Positive reward for exploring new room/hall
                     self.unvisited.remove(np.abs(self.loc))
                     reward += 5

              self.nb_step += 1
              if self.nb_step == self.max_step:
                     done = True # Limit agent to a fixed number of steps
              
              if self.loc < 0:
                     reward += 10 # Immediate reward of +10 on collecting the flag
                     self.loc = np.abs(self.loc) # If flag present, then collect the flag and empty the cell

              if self.loc == 8:
                     done = True # Return done = True if goal state is reached
                     reward += 20 # Immediate reward of +20 for reaching the goal state
                     self.nb_flags += 1
              
              return [self.x,self.y,self.loc], reward, done, {}
       
       def probability_matrix(self, action):
              """
              Introduces non-determinism - returns the correct action with 0.7 probability,
              and the other three actions with 0.1 probability each.
              """
              arr = [0,1,2,3]
              arr.remove(action)
              x = random.choice([1,2,3,4,5,6,7,8,9,10])
              if x < 8:
                     return action
              else:
                     return random.choice(arr)

       def render(self, mode='human', close=False):
              print(f"\nNext action:{ACT_DICT[str(self.action)]}")
              self.print()
