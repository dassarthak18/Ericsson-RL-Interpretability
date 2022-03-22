import gym
import numpy as np
import random
from gym import spaces

ACT_DICT = {'0':'Move up', '1':'Move down', '2':'Move left', '3':'Move right'}
DICT = {'0':'0', '1':'*', '2':'S', '3':'G', '4':'X'}

class SimpleMaze(gym.Env):

       """
       Map of the Maze as Stored
       -------------------------
       A m by n grid (mn > 1) with floor(sqrt(mn)) flags, a start state S and a goal state G.
       The flags, S and G are assigned at random.
       
       X - Current position

       Can move up, down, left or right.
       Deterministic or probabilistic in nature.

       For each step the following immediate rewards are offered:
              1. A penalty of -1 for every step taken.
              2. A penalty of -1 for visiting every already explored cell.
              3. A reward of +10 for every flag collected.

       An episode ends when either goal state or the maximum number of steps is reached, whichever earlier.
       
       Observation:
              Type: Box(3)
              Num Observation   Min  Max
              0   x-index       0    m-1
              1   y-index       0    n-1
              2   start         0     1
              3   goal          0     1
              4   flag          0     1

       Actions:
              Type: Discrete(4)
              Num Action
              0   Move up
              1   Move down
              2   Move left
              3   Move right
       """

       def __init__(self, m, n, deterministic, steps=-1): # Default -1; no maximum number of steps
              super(SimpleMaze, self).__init__()
              self.action_space = spaces.Discrete(4)
              low = np.array([0,0,0,0,0],dtype=np.int_,)
              high = np.array([m-1,n-1,1,1,1],dtype=np.int_,)
              self.observation_space = spaces.Box(low, high, dtype=np.int)
              self.m = m
              self.n = n
              self.deterministic = deterministic
              self.max_step = steps
              self.reset()
              print("Map:")      
              for i in self.maze:
                     string = ''
                     for j in i:
                            string = string + DICT[str(j)] + ' '
                     print(string)

       def reset(self):
              maze = []
              for i in range(self.m):
                     maze_row = []
                     for j in range(self.n):
                            maze_row.append(0)
                     maze.append(maze_row)
              self.maze = maze
              self.visited = []
              self.nb_step = 0
              arr = []
              for i in range(int(np.sqrt(self.m*self.n))):
                     arr.append(1) # 1 indicates flag
              arr.append(2) # 2 indicates S
              arr.append(3) # 3 indicates G
              while len(arr) > 0:
                     x = random.choice(arr)
                     i = random.choice(np.arange(self.m))
                     j = random.choice(np.arange(self.n))
                     if self.maze[i][j] == 0:
                            self.maze[i][j] = x
                            arr.remove(x)
                     if x == 2:
                            self.x = i
                            self.y = j
              self.maze[self.x][self.y] = 4 # 4 indicates X
              self.loc = 2
              self.visited.append((self.x,self.y))
              return [self.x,self.y,1,0,0]
       
       def render(self, mode='human', close=False):
              print(f"\nNext action:{ACT_DICT[str(self.action)]}")
              print("Map:")      
              for i in self.maze:
                     string = ''
                     for j in i:
                            string = string + DICT[str(j)] + ' '
                     print(string)
       
       def step(self, action):

              done = False
              reward = -1
              self.maze[self.x][self.y] = self.loc
              if self.deterministic == True:
                     self.action = action
              else:
                     self.action = self.probability_matrix(action)

              if self.action == 0 and self.x != 0: # Move up
                     self.x -= 1
              if self.action == 1 and self.x != self.m-1: # Move down
                     self.x += 1
              if self.action == 2 and self.y != 0: # Move left
                     self.y -= 1
              if self.action == 3 and self.y != self.n-1: # Move right
                     self.y += 1       

              self.loc = self.maze[self.x][self.y]
              self.maze[self.x][self.y] = 4

              if (self.x,self.y) in self.visited:
                     reward += -1 # Immediate reward of -1 on visiting explored cell
              else:       
                     self.visited.append((self.x,self.y))

              self.nb_step += 1

              # Set the start, goal and flag bits for the new location

              self.start = 0
              self.goal = 0
              self.flag = 0

              if self.loc == 1:
                     self.flag = 1
              if self.loc == 2:
                     self.start = 1
              if self.loc == 3:
                     self.goal = 1
              
              if self.flag:
                     self.loc = 0 # If flag present, then collect the flag and empty the cell
                     reward += 10 # Immediate reward of +10 on collecting a flag

              if self.goal:
                     done = True # Return done = True if goal state is reached
              
              if self.nb_step == self.max_step:
                     done = True # Limit agent to a fixed number of steps
              
              return [self.x,self.y,self.start,self.goal,self.flag], reward, done, {}

       def probability_matrix(self, action):
              """
              Introduces an element of probability - returns the intended action with
              0.7 probability, and the other three actions with 0.1 probability each.
              """
              arr = [0,1,2,3]
              arr.remove(action)
              x = random.choice([1,2,3,4,5,6,7,8,9,10])
              if x < 8:
                     return action
              else:
                     return random.choice(arr)
