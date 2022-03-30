from src.data import output

ACT_DICT = {'0':u'\u2191', '1':u'\u2193', '2':u'\u2190', '3':u'\u2192'}

def print_maze(env,model):
  print("Direction with flags:\n")
  maze = env.maze
  for i in range(len(maze)):
    for j in range(len(maze[0])):
      start = 0
      goal = 0
      flag = 0
      if maze[i][j] == 1:
        flag = 1
      if maze[i][j] == 2:
        start = 1
      if maze[i][j] == 3:
        goal = 1
      maze[i][j] = ACT_DICT[str(output(model,(i,j,start,goal,flag)))]

  for i in range(len(maze)):
    s = ''
    for j in range(len(maze[0])):
      s = s + str(maze[i][j]) + ' '
    print(s)

  print("\nDirection without flags:\n")
  maze = env.maze
  for i in range(len(maze)):
    for j in range(len(maze[0])):
      start = 0
      goal = 0
      if maze[i][j] == 2:
        start = 1
      if maze[i][j] == 3:
        goal = 1
      maze[i][j] = ACT_DICT[str(output(model,(i,j,start,goal,0)))]

  for i in range(len(maze)):
    s = ''
    for j in range(len(maze[0])):
      s = s + str(maze[i][j]) + ' '
    print(s)
