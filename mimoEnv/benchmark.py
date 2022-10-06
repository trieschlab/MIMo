""" This module contains some functions to benchmark the performance of the simulation.
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
from dataclasses import dataclass
from typing import Dict


@dataclass(unsafe_hash=True)
class FunctionProfile:
    ncalls: str
    tottime: float
    percall_tottime: float
    cumtime: float
    percall_cumtime: float
    file_name: str
    line_number: int


@dataclass(unsafe_hash=True)
class StatsProfile:
    """Class for keeping track of an item in inventory."""
    total_tt: float
    func_profiles: Dict[str, FunctionProfile]


def get_stats_profile(stats):
    """This method returns an instance of StatsProfile, which contains a mapping
    of function names to instances of FunctionProfile. Each FunctionProfile
    instance holds information related to the function's profile such as how
    long the function took to run, how many times it was called, etc...
    """
    func_list = stats.fcn_list[:] if stats.fcn_list else list(stats.stats.keys())
    if not func_list:
        return StatsProfile(0, {})

    total_tt = float(pstats.f8(stats.total_tt))
    func_profiles = {}
    stats_profile = StatsProfile(total_tt, func_profiles)

    for func in func_list:
        cc, nc, tt, ct, callers = stats.stats[func]
        file_name, line_number, func_name = func
        ncalls = str(nc) if nc == cc else (str(nc) + '/' + str(cc))
        tottime = float(pstats.f8(tt))
        percall_tottime = -1 if nc == 0 else float(pstats.f8(tt / nc))
        cumtime = float(pstats.f8(ct))
        percall_cumtime = -1 if cc == 0 else float(pstats.f8(ct / cc))
        func_profile = FunctionProfile(
            ncalls,
            tottime,  # time spent in this function alone
            percall_tottime,
            cumtime,  # time spent in the function plus all functions that this function called,
            percall_cumtime,
            file_name,
            line_number
        )
        func_profiles[func_name] = func_profile

    return stats_profile


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


def benchmark(configurations, output_file):
    """ Benchmarks multiple configurations for MIMo.

    We use cProfile as the profiler. Multiple runs with different configurations are performed and the runtime
    measurements saved to the specified output file. The profile for each run is also saved to a file named after the
    configuration.
    Configurations consist of an environment name and initialization parameters for that environment.
    Each configuration is run for the specified simulation time and the real time required for that is measured.
    MIMo takes random actions throughout.

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
    runtime_measurements.append(["Init.", "Physics", "Touch", "Vision", "Proprioception", "Vestibular", "Other"])

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

        stats_object = get_stats_profile(pstats.Stats(pr))

        physics_time = stats_object.func_profiles["do_simulation"].cumtime
        touch_time = stats_object.func_profiles["get_touch_obs"].cumtime
        vision_time = stats_object.func_profiles["get_vision_obs"].cumtime
        proprio_time = stats_object.func_profiles["get_proprioception_obs"].cumtime
        vesti_time = stats_object.func_profiles["get_vestibular_obs"].cumtime
        other_time = runtime - physics_time - touch_time - vision_time - proprio_time - vesti_time

        runtime_measurements.append(["{:.2f}".format(init_time),
                                     "{:.2f}".format(physics_time),
                                     "{:.2f}".format(touch_time),
                                     "{:.2f}".format(vision_time),
                                     "{:.2f}".format(proprio_time),
                                     "{:.2f}".format(vesti_time),
                                     "{:.2f}".format(other_time)])

        print("Elapsed time: total", total_time)
        print("Init time ", init_time)
        print("Non-init time", runtime)
        print("Simulation time:", max_steps * env.dt, "\n")

    with open(results_file, "wt", encoding="utf8") as f:
        for measurement in runtime_measurements:
            f.write("\t".join(measurement) + "\n")


def run_paper_benchmarks():
    configurations = []
    resolutions = [64, 128, 256, 512]
    scales = [0.25, 0.5, 1.0, 2.0]
    for resolution in resolutions:
        for scale in scales:

            VISION_PARAMS = {
                "eye_left": {"width": resolution, "height": resolution},
                "eye_right": {"width": resolution, "height": resolution},
            }

            TOUCH_PARAMS = copy.deepcopy(DEFAULT_TOUCH_PARAMS)
            for body in TOUCH_PARAMS["scales"]:
                TOUCH_PARAMS["scales"][body] = DEFAULT_TOUCH_PARAMS["scales"][body] / scale

            config_name = "V: {}² T: {}".format(resolution, scale)
            configurations.append((config_name,
                                   "MIMoBench-v0",
                                   {"vision_params": VISION_PARAMS, "touch_params": TOUCH_PARAMS}))
    benchmark(configurations, "paper_results.txt")


if __name__ == "__main__":
    configurations = []
    configurations.append(("MIMoV1", "MIMoBench-v0", {}, 3600))
    configurations.append(("MIMoV2", "MIMoBenchV2-v0", {}, 3600))
    benchmark(configurations, "optimized_results")
