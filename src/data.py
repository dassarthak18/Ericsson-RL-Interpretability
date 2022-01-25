import numpy as np
from tqdm import tqdm

def output(model,tup):
       # Returning output from ANN model
       x = np.array([list(tup)]).reshape(1,1,len(tup))
       return np.argmax(model.predict(x))

def dataset(env,model,filename="agent_data.csv",num=100000):
       # Obtaining parameters for ANN to DT conversion
       low = env.observation_space.low.tolist()
       high = env.observation_space.high.tolist()
       n = env.observation_space.shape[0]

       # Building a comprehensive list of tuples
       array = []
       bnd_per_node = int(np.power(num, float(1/n))) # fix upper bound for len(tups)
       for i in range(n):
              array.append(np.arange(low[i],high[i],float((high[i] - low[i])/bnd_per_node)).tolist())
       tups = [()]
       for i in range(n):
              tups = [tup + (a,) for tup in tups for a in array[i]]

       # Saving the output in a csv file
       f = open(filename, 'w')
       string = ''
       for i in range(n):
           string = string + "Input " + str(i) + ','
       string = string + "Output\n"
       f.write(string)
       for tup in tqdm(tups):
           string = ''
           for i in range(n):
               string = string + str(tup[i]) + ','
           string = string + str(output(model,tup)) + "\n"
           f.write(string)
       f.close()