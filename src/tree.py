import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from pandas import read_csv, qcut
from sklearn.tree import DecisionTreeClassifier, plot_tree
from src.data import output

def build_tree(env,filename):
  # Defining parameters
  low = env.observation_space.low
  high = env.observation_space.high
  n = env.observation_space.shape[0]

  # Extracting data from csv
  data = read_csv(filename)
  X = []
  for i in range(n):
    temp = data[f'Input {i}'].tolist()
    X.append(temp)
  Y = data['Output'].tolist()

  # Categorization of data
  X_encoded = []
  for i in range(n):
      l = int(low[i])
      if int(high[i]) == high[i]:
          h = high[i]
      else:
          h = high[i]+1
      temp = []
      for j in range(l,h):
          temp.append(f"{j}")
      X_encoded.append(qcut(X[i], len(temp), labels=temp))

  Y_encoded = np.array(Y)

  # Building the Decision Tree
  Tree = DecisionTreeClassifier()
  X1_encoded = []
  n = len(X_encoded[0])
  for i in range(n):
    arr = []
    for j in range(len(X_encoded)):
      arr.append(X_encoded[j][i])
    X1_encoded.append(arr)
  Tree.fit(X1_encoded, Y_encoded.reshape(-1,1).tolist())

  return Tree

# Visualizing the Decision Tree
def visualize_tree(env,Tree):
  n = env.action_space.n
  class_names = []
  for i in range(n):
    class_names.append(str(i))
  fig = plt.figure(figsize=(25,20))
  _ = plot_tree(Tree,class_names=class_names,filled=True)
  return plt

# Testing the Decision Tree
def test_tree(env,model,Tree,num=31830): # Test = floor(Train/pi), Default Train = 100000, Default Test = 31830
  low = env.observation_space.low
  high = env.observation_space.high
  n = env.observation_space.shape[0]

  count = 0

  for k in tqdm(range(num)):
    rand = []
    for i in range(n):
      rand.append(np.random.uniform(low[i],high[i]))
    true = output(model,tuple(rand))
    pred = Tree.predict([rand], check_input=True)[0]

    if true == pred:
      count += 1

  print(f"Instances checked: {num}\nPredictions matched: {count}\nAccuracy: {float(count*100/num)}%")