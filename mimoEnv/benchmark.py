import gym
import time
import copy
import cProfile
import mimoEnv
from mimoEnv.envs.mimo_env import DEFAULT_TOUCH_PARAMS


def run(env, max_steps):
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()


def benchmark():
    resolutions = [64, 128, 256, 512]
    scales = [0.25, 0.5, 1.0, 2.0]
    max_steps = 360000
    n_substeps = 2

# 1 hour simulation time, 1 minute episodes before reset. MIMO takes random actions.
    for resolution in resolutions:
        for scale in scales:
            VISION_PARAMS = {
                "eye_left": {"width": resolution, "height": resolution},
                "eye_right": {"width": resolution, "height": resolution},
            }
            TOUCH_PARAMS = copy.deepcopy(DEFAULT_TOUCH_PARAMS)
            for body in TOUCH_PARAMS["scales"]:
                TOUCH_PARAMS["scales"][body] = DEFAULT_TOUCH_PARAMS["scales"][body] / scale

            filename = "autobench_v{}_t{}.profile".format(resolution, scale)

            print("\n" + filename)
            pr = cProfile.Profile()
            pr.enable()
            init_start = time.time()
            env = gym.make("MIMo-v0", touch_params=TOUCH_PARAMS, vision_params=VISION_PARAMS, n_substeps=n_substeps)
            _ = env.reset()

            start = time.time()
            run(env, max_steps)
            env.close()
            pr.create_stats()
            pr.dump_stats(filename)

            print("Elapsed time: total", time.time() - init_start)
            print("Init time ", start - init_start)
            print("Non-init time", time.time() - start)
            print("Simulation time:", max_steps * env.dt, "\n")


if __name__ == "__main__":
    benchmark()
