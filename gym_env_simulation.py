'''
To install OpenAI Gym:

sudo apt-get install build-essential python3-dev swig python3-pygame git libosmesa6-dev libgl1-mesa-glx libglfw3
sudo pip3 install ale-py AutoROM.accept-rom-license box2d box2d-kengz gym lz4 opencv-python pyvirtualdisplay
sudo pip3 install pyglet importlib-resources Cython cffi glfw imageio lockfile pycparser pillow zipp

(There seems to be an error in installing the mujoco-py package, due to which several of the environments cannot be run.)

'''

# Step 1: Simulating AI Environment using OpenAI Gym

import gym
'''
# Print the list of available OpenAI Gym environments
print("List of available OpenAI Gym environments.\n")
for i in gym.envs.registry.all():
	print(str(i)[8:-1])
'''

# Get the environment name and timestep to simulate
string = input("\nEnter the name of the environment to simulate: ")
sim_steps = int(input("Enter the number of steps for simulation: "))

# Create the environment and reset it to the initial state
env = gym.make(string)
print(env.reset())

'''
# Simulate the environment
for _ in range(sim_steps):
	env.render()
	act = env.action_space.sample()
	print(act)
	arr = env.step(act)
	if arr[-2] == True:
		env.reset()
'''

episodes = 1

for i in range(episodes):
	observation = env.reset()
	done = False
	t = 0
	while not done:
		env.render()
		action = env.action_space.sample() # take a random action
		observation, reward, done, info = env.step(action)
		print(observation)
		t = t + 1
	print(f"Episode {i+1} finished after {t} timesteps.")