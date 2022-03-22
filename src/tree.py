import graphviz
import numpy as np
from tqdm import tqdm
from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
from src.data import output

def build_tree(env,filename,num=None):
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
  
  X_encoded = np.array(X)
  Y_encoded = np.array(Y)

  # Building the Decision Tree
  Tree = DecisionTreeClassifier(max_depth=num)
  X1_encoded = []
  n = len(X_encoded[0])
  for i in range(n):
    arr = []
    for j in range(len(X_encoded)):
      arr.append(X_encoded[j][i])
    X1_encoded.append(arr)
  Tree.fit(X1_encoded, Y_encoded.reshape(-1,1))

  return Tree

# Visualizing the Decision Tree
def visualize_tree(env,Tree):
  n = env.action_space.n
  class_names = []
  for i in range(n):
    class_names.append(str(i))
  data = export_graphviz(Tree,class_names=class_names,filled=True)
  graph = graphviz.Source(data, format="png")
  return graph

# Testing the Decision Tree
def test_tree(env,model,Tree):
  low = env.observation_space.low
  high = env.observation_space.high
  n = env.observation_space.shape[0]

  count = 0
  
  array = []
  tups = [()]
  for i in range(n):
    array.append(np.arange(low[i],high[i]+1).tolist())
  for i in range(n):
    tups = [tup + (a,) for tup in tups for a in array[i]]

  for k in tqdm(tups):
    true = output(model,tuple(k))
    pred = Tree.predict([k], check_input=True)[0]

    if true == pred:
      count += 1

  print(f"Instances checked: {len(tups)}\nPredictions matched: {count}\nAccuracy: {float(count*100/len(tups))}%")
