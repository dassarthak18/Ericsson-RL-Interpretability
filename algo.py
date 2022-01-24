import gym

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.agents.dqn import DQNAgent
from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import EpisodeParameterMemory, SequentialMemory

def cem(env):
       model = Sequential()
       model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(env.action_space.n))
       model.add(Activation('softmax'))
       cem = CEMAgent(model=model, nb_actions=env.action_space.n, memory=EpisodeParameterMemory(limit=50000, window_length=1), batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
       cem.compile()
       cem.fit(env, nb_steps=100000, visualize=False, verbose=1)
       return model, cem

def dqn(env):
       model = Sequential()
       model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(env.action_space.n))
       model.add(Activation('linear'))
       dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=SequentialMemory(limit=50000, window_length=1), nb_steps_warmup=100, target_model_update=1e-2, policy=BoltzmannQPolicy())
       dqn.compile(Adam(learning_rate=1e-3), metrics=['mse'])
       dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
       return model, dqn

def duel_dqn(env):
       model = Sequential()
       model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(env.action_space.n))
       model.add(Activation('linear'))
       dqn = DQNAgent(model=model, nb_actions=env.action_space.n, memory=SequentialMemory(limit=50000, window_length=1), nb_steps_warmup=100, enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=BoltzmannQPolicy())
       dqn.compile(Adam(learning_rate=1e-3), metrics=['mse'])
       dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
       return model, dqn

def sarsa(env):
       model = Sequential()
       model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(16))
       model.add(Activation('relu'))
       model.add(Dense(env.action_space.n))
       model.add(Activation('linear'))
       sarsa = SARSAAgent(model=model, nb_actions=env.action_space.n, nb_steps_warmup=100, policy=BoltzmannQPolicy())
       sarsa.compile(Adam(learning_rate=1e-3), metrics=['mse'])
       sarsa.fit(env, nb_steps=50000, visualize=False, verbose=1)
       return model, sarsa
