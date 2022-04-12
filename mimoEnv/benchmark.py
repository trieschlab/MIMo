import gym
import time
import copy
import cProfile
import mimoEnv
from mimoEnv.envs.mimo_env import DEFAULT_TOUCH_PARAMS

resolutions = [512]
scales = [1.5, 1, 0.5, 0.25]
max_steps = 360000


def run(env, max_steps):
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()


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
        env = gym.make("MIMoDummy-v0", touch_params=TOUCH_PARAMS, vision_params=VISION_PARAMS)
        obs = env.reset()

        start = time.time()
        run(env, max_steps)
        env.close()
        pr.create_stats()
        pr.dump_stats(filename)

        print("Elapsed time: total", time.time() - init_start)
        print("Init time ", start - init_start)
        print("Non-init time", time.time() - start)
        print("Simulation time:", max_steps * env.dt, "\n")
