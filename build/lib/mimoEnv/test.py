import sys,os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+"../")
print(os.path.dirname(os.path.realpath(__file__)) )
import gym
import numpy as np
import mimoEnv

env = gym.make("MIMo-v0")

max_steps = 125

obs = env.reset()

for step in range(max_steps):
    #action = env.action_space.sample()
    action = np.zeros(env.action_space.shape)
    obs, reward, done, info = env.step(action)
    env.render()
    #time.sleep(1)
    if done:
        env.reset()

env.close()
