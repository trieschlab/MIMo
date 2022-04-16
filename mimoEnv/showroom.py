import gym
import time
import numpy as np
import mimoEnv


def main():

    env = gym.make("MIMoShowroom-v0", show_sensors=False, print_space_sizes=True)

    max_steps = 2000

    _ = env.reset()

    start = time.time()
    for step in range(max_steps):
        action = np.zeros(env.action_space.shape)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            env.reset()

    print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps*env.dt)
    env.close()


if __name__ == "__main__":
    main()
