import gym
import time
import numpy as np
import mimoEnv

env = gym.make("MIMoDemo-v0")

bodies = [
    {'name':'eyes', 'welds_idx':[11,12], 'actions_idx': [6,7,8,9]},
    {'name':'head', 'welds_idx':[0], 'actions_idx': [3,4,5]},
    {'name':'torso', 'welds_idx':[1,2,3,4,5,6], 'actions_idx': [0,1,2, 26,27,28, 33,34,35]},
    {'name':'arms', 'welds_idx':[3,4,5,6], 'actions_idx': np.arange(10,25)},
    {'name':'left_leg', 'welds_idx':[7,8], 'actions_idx': [33,34,35,36,37,38,39, 0,1,2]},
    {'name':'right_leg', 'welds_idx':[9,10], 'actions_idx': [26,27,28,29,30,31,32, 0,1,2]},
]

max_steps = 200 

obs = env.reset()
for body in bodies:
    name = body['name']
    welds_idx = body['welds_idx']
    actions_idx = body['actions_idx']
    env.sim.model.eq_active[welds_idx] = 0

    for step in range(max_steps):
        action_sample = env.action_space.sample()
        action = np.zeros(action_sample.shape)
        action[actions_idx] = action_sample[actions_idx]
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            env.reset()

    env.sim.model.eq_active[welds_idx] = 1
    env.reset()


env.close()