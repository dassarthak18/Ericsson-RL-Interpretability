from copy import deepcopy
from src.data import output

ACT_DICT = {'0':u'\u2191', '1':u'\u2193', '2':u'\u2190', '3':u'\u2192'}

def print_maze(env,model):
  print("\nDirection with flags:\n")
  maze = deepcopy(env.maze)
  for i in range(len(maze)):
    for j in range(len(maze[0])):
      start = 0
      goal = 0
      flag = 0
      if maze[i][j] == 1:
        flag = 1
      if maze[i][j] == 4:
        start = 1
      if maze[i][j] == 3:
        goal = 1
      # Info of surroundings
      F_N = 0
      S_N = 0
      G_N = 0
      F_S = 0
      S_S = 0
      G_S = 0
      F_E = 0
      S_E = 0
      G_E = 0
      F_W = 0
      S_W = 0
      G_W = 0
      if i != 0 and env.maze[i-1][j] == 1:
             F_N = 1
      if i != 0 and env.maze[i-1][j] == 2:
             S_N = 1
      if i != 0 and env.maze[i-1][j] == 3:
             G_N = 1
      if i != len(maze)-1 and env.maze[i+1][j] == 1:
             F_S = 1
      if i != len(maze)-1 and env.maze[i+1][j] == 2:
             S_S = 1
      if i != len(maze)-1 and env.maze[i+1][j] == 3:
             G_S = 1
      if j != 0 and env.maze[i][j-1] == 1:
             F_E = 1
      if j != 0 and env.maze[i][j-1] == 2:
             S_E = 1
      if j != 0 and env.maze[i][j-1] == 3:
             G_E = 1
      if j != len(maze[0])-1 and env.maze[i][j+1] == 1:
             F_W = 1
      if j != len(maze[0])-1 and env.maze[i][j+1] == 2:
             S_W = 1
      if j != len(maze[0])-1 and env.maze[i][j+1] == 3:
             G_W = 1
      maze[i][j] = ACT_DICT[str(output(model,(S_N,G_N,F_N,S_S,G_S,F_S,S_E,G_E,F_E,S_W,G_W,F_W)))]

  for i in range(len(maze)):
    s = ''
    for j in range(len(maze[0])):
      s = s + str(maze[i][j]) + ' '
    print(s)

  print("\nDirection without flags:\n")
  maze = deepcopy(env.maze)
  for i in range(len(maze)):
    for j in range(len(maze[0])):
      start = 0
      goal = 0
      if maze[i][j] == 4:
        start = 1
      if maze[i][j] == 3:
        goal = 1
      # Info of surroundings
      F_N = 0
      S_N = 0
      G_N = 0
      F_S = 0
      S_S = 0
      G_S = 0
      F_E = 0
      S_E = 0
      G_E = 0
      F_W = 0
      S_W = 0
      G_W = 0
      if i != 0 and maze[i-1][j] == 1:
             F_N = 1
      if i != 0 and maze[i-1][j] == 2:
             S_N = 1
      if i != 0 and maze[i-1][j] == 3:
             G_N = 1
      if i != len(maze)-1 and env.maze[i+1][j] == 1:
             F_S = 1
      if i != len(maze)-1 and env.maze[i+1][j] == 2:
             S_S = 1
      if i != len(maze)-1 and env.maze[i+1][j] == 3:
             G_S = 1
      if j != 0 and env.maze[i][j-1] == 1:
             F_E = 1
      if j != 0 and env.maze[i][j-1] == 2:
             S_E = 1
      if j != 0 and env.maze[i][j-1] == 3:
             G_E = 1
      if j != len(maze[0])-1 and env.maze[i][j+1] == 1:
             F_W = 1
      if j != len(maze[0])-1 and env.maze[i][j+1] == 2:
             S_W = 1
      if j != len(maze[0])-1 and env.maze[i][j+1] == 3:
             G_W = 1
      maze[i][j] = ACT_DICT[str(output(model,(S_N,G_N,F_N,S_S,G_S,F_S,S_E,G_E,F_E,S_W,G_W,F_W)))]

  for i in range(len(maze)):
    s = ''
    for j in range(len(maze[0])):
      s = s + str(maze[i][j]) + ' '
    print(s)
