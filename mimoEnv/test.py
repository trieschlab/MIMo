import gym
import time
import numpy as np
import mimoEnv


def main():
    env = gym.make("MIMoSelfBody-v0")

    max_steps = 5000

    _ = env.reset()

    start = time.time()
    for step in range(max_steps):
        action = env.action_space.sample()
        #action = np.zeros(env.action_space.shape)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            env.reset()

    print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps*env.dt)
    env.close()


if __name__ == "__main__":
    main()
