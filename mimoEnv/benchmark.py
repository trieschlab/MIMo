""" This module contains some functions to benchmark the performance of the simulation.

Classes and functions :cls:`FunctionProfile`, :cls:`StatsProfile` and :func:`get_stats_profile`  from
NAME HERE @ LINK
"""

import os
import math
import gym
import time
import copy
import cProfile
import mimoEnv
from mimoEnv.envs.mimo_env import DEFAULT_TOUCH_PARAMS

import pstats


def run(env, max_steps):
    """ Runs an environment for a number of steps while taking random actions.

    Args:
        env (gym.Env): The environment.
        max_steps (int): The number of time steps that we run the environment for.
    """
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, done, trunc, info = env.step(action)
        if done or trunc:
            env.reset()


def benchmark(configurations, output_file):
    """ Benchmarks multiple configurations for MIMo.

    We use cProfile as the profiler. Multiple runs with different configurations are performed and the runtime
    measurements saved to the specified output file. The profile for each run is also saved to a file named after the
    configuration.
    Configurations consist of an environment name and initialization parameters for that environment.
    Each configuration is run for the specified simulation time and the real time required for that is measured.
    MIMo takes random actions throughout.
    Measurements include the total runtime, the simulation time, the number of simulation steps, the time spent in
    environment initialization, the time spent on the physics simulation, and the time spent in each of the sensor
    modalities: touch, vision, proprioception and vestibular.

    Args:
        configurations: A list of tuples storing configurations to be benchmarked. Each tuple has four entries:
            An arbitrary name for the entry, the name of the gym environment that will be run, a dictionary with
            parameters for the environment, and finally the duration of the run in simulation seconds.
            Note that the parameter dictionary can be empty if you wish to use the default parameters for the
            environment.
        output_file: Runtime results are written to this file.

    """
    results_file = os.path.abspath(output_file)
    profile_dir = os.path.dirname(results_file)

    runtime_measurements = []
    runtime_measurements.append(["Config", "Runtime", "Simtime", "n_steps", "Init.", "Physics", "Touch",
                                 "Vision", "Proprioception", "Vestibular", "Other"])

    for configuration in configurations:
        # 1 hour simulation time
        config_name, env_name, config_dict, sim_time = configuration

        print("Running configuration:", config_name)

        profile_file_name = config_name + ".profile"
        profile_file = os.path.join(profile_dir, profile_file_name)

        pr = cProfile.Profile()
        pr.enable()
        init_start = time.time()
        env = gym.make(env_name, **config_dict)
        _ = env.reset()

        max_steps = math.floor(sim_time / env.dt)

        start = time.time()
        run(env, max_steps)
        env.close()

        end_time = time.time()

        pr.create_stats()
        pr.dump_stats(profile_file)

        total_time = end_time - init_start
        init_time = start - init_start
        runtime = end_time - start

        stats_object = pstats.Stats(pr).get_stats_profile()

        physics_time = stats_object.func_profiles["do_simulation"].cumtime
        touch_time = stats_object.func_profiles["get_touch_obs"].cumtime if "get_touch_obs" in stats_object.func_profiles else 0
        vision_time = stats_object.func_profiles["get_vision_obs"].cumtime if "get_vision_obs" in stats_object.func_profiles else 0
        proprio_time = stats_object.func_profiles["get_proprioception_obs"].cumtime if "get_proprioception_obs" in stats_object.func_profiles else 0
        vesti_time = stats_object.func_profiles["get_vestibular_obs"].cumtime if "get_vestibular_obs" in stats_object.func_profiles else 0
        other_time = runtime - physics_time - touch_time - vision_time - proprio_time - vesti_time

        runtime_measurements.append([config_name,
                                     "{:.2f}".format(total_time),
                                     "{:.2f}".format(max_steps * env.dt),
                                     "{}".format(max_steps),
                                     "{:.2f}".format(init_time),
                                     "{:.2f}".format(physics_time),
                                     "{:.2f}".format(touch_time),
                                     "{:.2f}".format(vision_time),
                                     "{:.2f}".format(proprio_time),
                                     "{:.2f}".format(vesti_time),
                                     "{:.2f}".format(other_time)])

        print("Elapsed time: total", total_time)
        print("Init time ", init_time)
        print("Non-init time", runtime)
        print("Simulation time:", max_steps * env.dt)
        print("Simulation steps:", max_steps, "\n")

    with open(results_file, "wt", encoding="utf8") as f:
        for measurement in runtime_measurements:
            f.write("\t".join(measurement) + "\n")


def run_paper_benchmarks():
    """ Performs the same benchmarks as used in the paper."""
    configurations = []
    resolutions = [64, 128, 256, 512]
    scales = [0.25, 0.5, 1.0, 2.0]
    for resolution in resolutions:
        for scale in scales:

            vision_params = {
                "eye_left": {"width": resolution, "height": resolution},
                "eye_right": {"width": resolution, "height": resolution},
            }

            touch_params = copy.deepcopy(DEFAULT_TOUCH_PARAMS)
            for body in touch_params["scales"]:
                touch_params["scales"][body] = DEFAULT_TOUCH_PARAMS["scales"][body] / scale

            config_name = "V: {}² T: {}".format(resolution, scale)
            configurations.append((config_name,
                                   "MIMoBench-v0",
                                   {"vision_params": vision_params, "touch_params": touch_params},
                                   3600))
    benchmark(configurations, "paper_results.txt")


if __name__ == "__main__":
    configurations = []
    configurations.append(("MIMoV1", "MIMoBench-v0", {}, 3600))
    configurations.append(("MIMoV2", "MIMoBenchV2-v0", {}, 3600))
    benchmark(configurations, "optimized_results")
