""" This module contains some functions to benchmark the performance of the simulation.

Classes and functions :class:`FunctionProfile`, :class:`StatsProfile` and :func:`get_stats_profile`  from
NAME HERE @ LINK
"""

import os
import math
import gymnasium as gym
import re
import time
import copy
import gc
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib import patches

import cProfile
import pstats

import mimoEnv
from mimoEnv.envs.mimo_env import DEFAULT_TOUCH_PARAMS, DEFAULT_TOUCH_PARAMS_V2
from mimoEnv.envs.dummy import BENCHMARK_XML_V2, BENCHMARK_XML

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
        configurations (List[Tuple[str, str, Dict, int]]): A list of tuples storing configurations to be benchmarked.
            Each tuple has four entries:
            An arbitrary name for the entry, the name of the gym environment that will be run, a dictionary with
            parameters for the environment, and finally the duration of the run in simulation seconds.
            Note that the parameter dictionary can be empty if you wish to use the default parameters for the
            environment.
        output_file (str): Runtime results are written to this file.
    """
    results_file = os.path.abspath(output_file)
    profile_dir = os.path.dirname(results_file)

    runtime_measurements = [["Config", "Runtime", "Simtime", "n_steps", "Init.", "Physics", "Muscle", "Touch",
                            "Vision", "Proprioception", "Vestibular", "Other"]]

    for configuration in configurations:
        # 1 hour simulation time
        config_name, env_name, config_dict, sim_time = configuration

        print("Running configuration:", config_name)

        profile_file_name = re.sub(r"[\[\]/\\:\"?<>*|]", "", config_name)
        profile_file_name = re.sub(r"\s", "_", profile_file_name) + ".profile"
        profile_file = os.path.join(profile_dir, profile_file_name)

        pr = cProfile.Profile()
        pr.enable()
        init_start = time.time()
        env = gym.make(env_name, **config_dict)
        _ = env.reset()

        dt = env.dt

        max_steps = math.floor(sim_time / dt)

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
        if "_compute_muscle_action" in stats_object.func_profiles:
            step_time = stats_object.func_profiles["<method 'step' of 'mujoco_py.cymj.MjSim' objects>"].cumtime
            muscle_time = physics_time - step_time
            physics_time = step_time
        else:
            muscle_time = 0.0
        touch_time = stats_object.func_profiles["get_touch_obs"].cumtime \
            if "get_touch_obs" in stats_object.func_profiles else 0
        vision_time = stats_object.func_profiles["get_vision_obs"].cumtime \
            if "get_vision_obs" in stats_object.func_profiles else 0
        proprio_time = stats_object.func_profiles["get_proprioception_obs"].cumtime \
            if "get_proprioception_obs" in stats_object.func_profiles else 0
        vesti_time = stats_object.func_profiles["get_vestibular_obs"].cumtime \
            if "get_vestibular_obs" in stats_object.func_profiles else 0
        other_time = total_time - init_time - physics_time - muscle_time - touch_time - vision_time - proprio_time - vesti_time

        runtime_measurements.append([config_name,
                                     "{:.3e}".format(total_time),
                                     "{:.3e}".format(max_steps * dt),
                                     "{}".format(max_steps),
                                     "{:.3e}".format(init_time),
                                     "{:.3e}".format(physics_time),
                                     "{:.3e}".format(muscle_time),
                                     "{:.3e}".format(touch_time),
                                     "{:.3e}".format(vision_time),
                                     "{:.3e}".format(proprio_time),
                                     "{:.3e}".format(vesti_time),
                                     "{:.3e}".format(other_time)])

        print("Elapsed time: total", total_time)
        print("Init time ", init_time)
        print("Non-init time", runtime)
        print("Simulation time:", max_steps * dt)
        print("Simulation steps:", max_steps, "\n")

        # Clear everything to keep memory close
        del pr
        del env
        gc.collect()

    with open(results_file, "wt", encoding="utf8") as f:
        for measurement in runtime_measurements:
            f.write("\t".join(measurement) + "\n")


def load_benchmark_file(file_name) -> Dict:
    """ Loads a benchmark file in the format as produced by :func:`.benchmark` into a dictionary.

    Args:
        file_name (str): The input benchmark file.

    Returns:
        Dict[str, float]: The dictionary with loaded benchmark data.
    """
    # Assumes that the first column is a name/label for each run
    with open(file_name, "rt", encoding="utf8") as f:
        data = {}
        headers = f.readline().strip().split("\t")
        data_key = headers[1:]
        for line in f:
            line = line.strip().split("\t")
            run_name = line[0]
            run_data = {}
            data[run_name] = run_data
            for i in range(len(data_key)):
                run_data[data_key[i]] = float(line[i+1])
    return data


def make_stacked_bar_chart(data, labels: List[str], colors: Dict[str, str], ylabel, figsize=(6, 5), legend_loc="upper left"):
    """ Makes a stacked bar chart.

    Args:
        data: A dictionary of dictionaries with data for each stacked bar. High level dictionary stores the label for
            each bar as keys and the associated data dictionary as values. Low level dictionary contains the data
            for each stack with component labels as keys.
        labels: A list with the labels for each stack component. This also selects which components are plotted
                at all. The `colors` parameter must have an entry for every label.
        colors: A dictionary with colors for each stack component.
        ylabel: The y-axis label.
        figsize: A tuple with the figure size.
        legend_loc: Location of the legend.

    Returns:
        A tuple (fig, ax) with the plotted chart
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=200)

    bar_width = 0.15
    bar_sep = 0.015
    boundary = 0.005

    x = 0
    xl_ticks = []
    xl_labels = []
    for stacked_bar in data:
        x += bar_sep
        x += bar_width
        xl_ticks.append(x)
        xl_labels.append(stacked_bar)
        cur_bottom = 0
        for component in labels:
            ax.bar(x,
                   data[stacked_bar][component],
                   bar_width - boundary,
                   bottom=cur_bottom,
                   label=component,
                   color=colors[component])
            cur_bottom += data[stacked_bar][component]
    ax.set_xlim([xl_ticks[0]-bar_width/2-bar_sep, xl_ticks[-1] + bar_width/2 + bar_sep])
    ax.set_xticks(xl_ticks)
    ax.set_xticklabels(xl_labels, ha="center", rotation=-15)
    ax.set_ylabel(ylabel)

    legends = [patches.Rectangle((0, 0), 1.0, 1.0, color=colors[component], edgecolor=None) for component in labels]

    # Invert order so legend matches stack order.
    legends.reverse()
    labels.reverse()
    ax.legend(legends, labels, loc=legend_loc)
    return fig, ax


def plot_benchmarks(file_name, list_of_runs, output_file, label_list=None, color_dict=None, figsize=(6, 5)):
    """ Create benchmark plots.

    Loads data from a benchmark file and creates a stacked bar chart from the loaded data. Which runs are plotted can
    be selected with `list_of_runs`

    Args:
        file_name (str): The file containing the benchmark data.
        list_of_runs (List[str]): A list with the configuration names that will be plotted side by side.
        output_file (str): Output image file.
        label_list (List[str]): A list of the runtime components that will be plotted.
        color_dict (Dict[str, str]): A dictionary with the colors for each component listed above.
    """
    data = load_benchmark_file(file_name)

    # Colors and labels for each section
    if label_list is None:
        label_list = ["Init.", "Physics", "Muscle", "Touch", "Vision", "Proprioception", "Vestibular", "Other"]
    if color_dict is None:
        color_dict = {"Other": "limegreen",
                      "Vestibular": "cornflowerblue",
                      "Proprioception": "mediumblue",
                      "Vision": "purple",
                      "Touch": "red",
                      "Physics": "orange",
                      "Init.": "yellow",
                      "Muscle": "darkgoldenrod"}

    data_to_plot = {}
    for single_run in list_of_runs:
        data_to_plot[single_run] = dict([(key, value / 3600) for (key, value) in data[single_run].items()])

    fig, ax = make_stacked_bar_chart(data_to_plot, colors=color_dict, labels=label_list, figsize=figsize, ylabel="Realtime/Simtime")
    ax.axhline(1.0, color="k", linestyle="dashed", linewidth=1.0)
    fig.savefig(output_file, bbox_inches="tight")


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

            config_name = f"V: {resolution}² T: {scale}"
            configurations.append((config_name,
                                   "MIMoBench-v0",
                                   {"vision_params": vision_params, "touch_params": touch_params},
                                   3600))
    benchmark(configurations, "paper_results.txt")


def make_paper_plot(file_name, output_file):
    """ Creates the sensor benchmarking plot from the paper.

    Args:
        file_name (str): Input benchmark file.
        output_file (str): Output image file.
    """
    data = load_benchmark_file(file_name)

    label_list = ["Init.", "Physics", "Touch", "Vision", "Proprioception", "Vestibular", "Other"]
    color_dict = {"Other": "limegreen",
                  "Vestibular": "cornflowerblue",
                  "Proprioception": "mediumblue",
                  "Vision": "purple",
                  "Touch": "red",
                  "Physics": "orange",
                  "Init.": "yellow",
                  "Muscle": "darkgoldenrod"}
    runs_to_plot = ["V: 64² T: 0.25", "V: 64² T: 0.5", "V: 64² T: 1.0", "V: 64² T: 2.0", "V: 128² T: 1.0",
                    "V: 256² T: 1.0", "V: 512² T: 1.0"]
    data_to_plot = {}
    for run_to_plot in runs_to_plot:
        data_to_plot[run_to_plot] = dict([(key, value / 3600) for (key, value) in data[run_to_plot].items()])

    fig, ax = make_stacked_bar_chart(data_to_plot, colors=color_dict, labels=label_list, figsize=(9, 5), ylabel="Realtime/Simtime")
    ax.axhline(1.0, color="k", linestyle="dashed", linewidth=1.0)
    fig.savefig(output_file, bbox_inches="tight")


if __name__ == "__main__":
    run_paper_benchmarks()
    make_paper_plot("paper_results.txt", "paper_plot.pdf")
    configurations = []
    configurations.append(("MIMoV1",
                           "MIMoBench-v0",
                           {},
                           3600))
    configurations.append(("MIMoV1 - Muscle",
                           "MIMoMuscle-v0",
                           {"model_path": BENCHMARK_XML, "touch_params": DEFAULT_TOUCH_PARAMS},
                           3600))
    configurations.append(("MIMoV2",
                           "MIMoBenchV2-v0",
                           {},
                           3600))
    configurations.append(("MIMoV2 - Muscle",
                           "MIMoMuscle-v0",
                           {"model_path": BENCHMARK_XML_V2, "touch_params": DEFAULT_TOUCH_PARAMS_V2},
                           3600))
    # # configurations.append(("Reach", "MIMoReach-v0", {}, 3600))
    # # configurations.append(("Standup", "MIMoStandup-v0", {}, 3600))
    # # configurations.append(("SelfBody", "MIMoSelfBody-v0", {}, 3600))
    benchmark(configurations, "thesis_benchmarks.txt")
