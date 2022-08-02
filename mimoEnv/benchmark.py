""" This module contains some functions to benchmark the performance of the simulation.
"""

import gym
import time
import cProfile
import mimoEnv
from mimoEnv.envs.mimo_env import DEFAULT_TOUCH_PARAMS, DEFAULT_TOUCH_PARAMS_V2
from mimoEnv.envs.dummy import BENCHMARK_XML_V2, BENCHMARK_XML


def run(env, max_steps):
    """ Runs an environment for a number of steps while taking random actions.

    Args:
        env (gym.Env): The environment.
        max_steps (int): The number of time steps that we run the environment for.
    """
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        if done:
            env.reset()


def benchmark():
    """ Benchmarks multiple configurations for the sensor modules.

    We use cProfile as the profiler. Multiple runs with different configurations are performed and their profiles saved
    to a directory. Simulation time for one configuration is 1 hour with 60 episodes of one minute each.

    Since vision and touch take up the majority of the runtime we focus on those. We test vision with resolutions of
    64, 128, 256, and 512. We test touch by altering the sensor resolution with scale factors of 0.25, 0.5, 1.0 and 2.0.
    Every combination of resolution and sensor scale is tested.

    """

    # 1 hour simulation time, 1 minute episodes before reset. MIMO takes random actions.
    environments = ["MIMoMuscle-v0"]#["MIMoBench-v0", "MIMoMuscle-v0"]
    xmls = {BENCHMARK_XML: DEFAULT_TOUCH_PARAMS,
            BENCHMARK_XML_V2: DEFAULT_TOUCH_PARAMS_V2}
    max_steps = 6000  # 100 steps per second -> 360000 steps for 1 hour of simulation time with a dt of .01

    for environment in environments:
        for xml in xmls:
            touch_params = xmls[xml]
            version = 1 if xml == BENCHMARK_XML else 2
            actuation = "torque" if environment == "MIMoBench-v0" else "muscle"
            filename = "autobench_{}_ver{}.profile".format(actuation, version)

            print("\n" + filename)
            pr = cProfile.Profile()
            pr.enable()
            init_start = time.time()
            env = gym.make(environment, model_path=xml, touch_params=touch_params)
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
