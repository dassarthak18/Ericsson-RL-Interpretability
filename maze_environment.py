import gym
from gym import spaces

class MazeEnv(gym.Env):
  
  """
  A A A A A * B B B B B B B E E E E E
  A A A A A A B B B B B B B E E E E E
  A A A A A A B * B B B B B E E E E E
  A A A A o A B B B B B B B E E * E E
  a a a a o a b b b B B B B E E E E E
  a a a a S o o b b o o B B E E E E E
  a o a a a a b b b B B B B E E E E E
  D o D D D D b b b B B B B E E E E E
  D D D * D D b o b B B B B E E E E E
  D D D D D D C o C C C C C E E E E E
  D D D D D D C C C C C * C E E E E E
  D G D D D D C C C C C C o o E E E E
  D D D D D D C C C C C C C E E E E *

  A - Room A
  a - Hall A
  B - Room B
  b - Hall B
  C - Room C
  D - Room D
  E - Room E
  * - Flag
  S - Start State
  G - Goal State
  o - Door
  """
  
  metadata = {'render.modes':['human']}
  
  def __init__(self, arg1, arg2):
    super(MazeEnv, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)

  def step(self, action):
    # Execute one time step within the environment

  def reset(self):
    # Reset the state of the environment to an initial state

  def render(self, mode='human', close=False):
    # Render the environment to the screen
