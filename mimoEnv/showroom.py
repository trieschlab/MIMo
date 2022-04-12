import gym
import time
import os
import numpy as np
import mimoEnv
from mimoEnv.envs.mimo_env import SCENE_DIRECTORY

showroom_xml = os.path.join(SCENE_DIRECTORY, "showroom.xml")

env = gym.make("MIMoDummy-v0", model_path=showroom_xml, show_sensors=True, print_space_sizes=True)

max_steps = 500

obs = env.reset()

start = time.time()
for step in range(max_steps):
    action = np.zeros(env.action_space.shape)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        env.reset()

print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps*env.dt)
env.close()
