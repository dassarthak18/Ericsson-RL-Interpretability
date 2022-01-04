import gym

import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.agents.dqn import DQNAgent
from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import EpisodeParameterMemory, SequentialMemory

# Print the list of available OpenAI Gym environments
print("List of available OpenAI Gym environments.\n")
for i in gym.envs.registry.all():
	print(str(i)[8:-1])

# Get the environment name and number of episodes to run
string = input("\nEnter the name of the environment: ")
episodes = int(input("Enter the number of episodes to run: "))

# Create the environment and reset it to the initial state
env = gym.make(string)
np.random.seed(123)
env.seed(123)

# Train the RL agent
if type(env.action_space)==gym.spaces.box.Box:
	print("Cannot train RL agent for environments with Box() space.")
	exit(0)

nb_actions = env.action_space.n
choice = int(input("Choose a learning policy:\n1. Simple CEM\n2. Deep CEM\n3. Deep Q-Network\n4. Duel Deep Q-Network\n5. SARSA\n"))

# Simple Neural Network
S_model = Sequential()
S_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
S_model.add(Dense(nb_actions))
S_model.add(Activation('softmax'))

# Complex Neural Network
C_model = Sequential()
C_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
C_model.add(Dense(16))
C_model.add(Activation('relu'))
C_model.add(Dense(16))
C_model.add(Activation('relu'))
C_model.add(Dense(16))
C_model.add(Activation('relu'))
C_model.add(Dense(nb_actions))
C_model.add(Activation('softmax'))

# Complex Neural Network for DQN, SARSA
CD_model = Sequential()
CD_model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
CD_model.add(Dense(16))
CD_model.add(Activation('relu'))
CD_model.add(Dense(16))
CD_model.add(Activation('relu'))
CD_model.add(Dense(16))
CD_model.add(Activation('relu'))
CD_model.add(Dense(nb_actions))
CD_model.add(Activation('linear'))

# Boltzmann Q Policy
BQ_policy = BoltzmannQPolicy()

# Episode Parameter Memory
E_memory = EpisodeParameterMemory(limit=50000, window_length=1)

# Sequential Memory
S_memory = SequentialMemory(limit=100000, window_length=1)

if choice == 1:
	name = f'scem_{string}_params.h5f'
	cem = CEMAgent(model=S_model, nb_actions=nb_actions, memory=E_memory, batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
	cem.compile()
	cem.fit(env, nb_steps=100000, visualize=False, verbose=2)
	cem.save_weights(name, overwrite=True)
	cem.test(env, nb_episodes=episodes, visualize=True)

elif choice == 2:
	name = f'dcem_{string}_params.h5f'
	cem = CEMAgent(model=C_model, nb_actions=nb_actions, memory=E_memory, batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
	cem.compile()
	cem.fit(env, nb_steps=100000, visualize=False, verbose=2)
	cem.save_weights(name, overwrite=True)
	cem.test(env, nb_episodes=episodes, visualize=True)

elif choice == 3:
	name = f'dqn_{string}_params.h5f'
	dqn = DQNAgent(model=CD_model, nb_actions=nb_actions, memory=S_memory, nb_steps_warmup=100, target_model_update=1e-2, policy=BQ_policy)
	dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
	dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
	dqn.save_weights(name, overwrite=True)
	dqn.test(env, nb_episodes=episodes, visualize=True)

elif choice == 4:
	name = f'duel_dqn_{string}_weights.h5f'
	dqn = DQNAgent(model=CD_model, nb_actions=nb_actions, memory=S_memory, nb_steps_warmup=100, enable_dueling_network=True, dueling_type='avg', target_model_update=1e-2, policy=BQ_policy)
	dqn.compile(Adam(learning_rate=1e-3), metrics=['mae'])
	dqn.fit(env, nb_steps=50000, visualize=False, verbose=2)
	dqn.save_weights(name, overwrite=True)
	dqn.test(env, nb_episodes=episodes, visualize=True)

elif choice == 5:
	name = f'sarsa_{string}_weights.h5f'
	sarsa = SARSAAgent(model=CD_model, nb_actions=nb_actions, nb_steps_warmup=100, policy=BQ_policy)
	sarsa.compile(Adam(learning_rate=1e-3), metrics=['mae'])
	sarsa.fit(env, nb_steps=50000, visualize=False, verbose=2)
	sarsa.save_weights(name, overwrite=True)
	sarsa.test(env, nb_episodes=episodes, visualize=True)