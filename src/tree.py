import numpy as np
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

def build_tree(filename,n):
  # Extracting data from csv
  data = read_csv("simple_maze_3_by_5_data.csv")
  X = []
  for i in range(n):
    temp = data[f'Input {i}'].tolist()
    X.append(temp)
  Y = data['Output'].tolist()

  # Label encoding for continuous data
  X_encoded = []
  for i in range(n):
    temp = LabelEncoder()
    X_encoded.append(temp.fit_transform(X[i]))

  temp = LabelEncoder()
  Y_encoded = temp.fit_transform(Y)

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