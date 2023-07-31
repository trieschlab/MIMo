""" Functions in this model are designed to help transition actuation models for MIMo.

By default, MIMo uses direct torque motors for actuation with the maximum torques corresponding to the maximum voluntary
isometric torques. A second actuation model exists based on
`https://arxiv.org/abs/2207.03952 <https://arxiv.org/abs/2207.03952>`_. This second model more accurately
represents the position and velocity dependent force generating behaviour of real muscles. The second model requires
several adjustments to the actuation and joint parameters, which can be done using the functions in this module.
"""
import gymnasium as gym
import os
import matplotlib
from typing import Dict, List
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import cv2

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
from mimoEnv.utils import EPS

matplotlib.use("Agg")

LMAX = 1.6
LMIN = 0.5
FVMAX = 1.2
FPMAX = 1.3


def vectorized(fn):
    """ Simple vector wrapper for functions that clearly came from C.
    """
    def new_fn(vec):
        if hasattr(vec, "__iter__"):
            ret = []
            for x in vec:
                ret.append(fn(x))
            return np.array(ret, dtype=np.float32)
        else:
            return fn(vec)
    return new_fn


@vectorized
def fl(lce):
    """ Force length curve as implemented by MuJoCo.

    Args:
        lce (np.ndarray|float): Virtual muscle lengths.

    Returns:
        np.ndarray|float: The corresponding force-length multipliers.
    """
    return bump(lce, LMIN, 1, LMAX) + 0.15 * bump(lce, LMIN, 0.5 * (LMIN + 0.95), 0.95)


def bump(length, a, mid, b):
    """ Part of the force length relationship as implemented by MuJoCo.

    The parameters `a`, `mid` and `b` define the shape of the force-length curve. See
    `https://arxiv.org/abs/2207.03952 <https://arxiv.org/abs/2207.03952>`_ for more details.

    Args:
        length (np.ndarray): The current virtual muscle lengths.
        a (float): One of the parameters of the force-length equation.
        mid (float): One of the parameters of the force-length equation.
        b (float): One of the parameters of the force-length equation.

    Returns:
        np.ndarray: Resulting force-length multiplier.
    """
    left = 0.5 * (a + mid)
    right = 0.5 * (mid + b)

    if (length <= a) or (length >= b):
        return 0
    elif length < left:
        temp = (length - a) / (left - a)
        return 0.5 * temp * temp
    elif length < mid:
        temp = (mid - length) / (mid - left)
        return 1 - 0.5 * temp * temp
    elif length < right:
        temp = (length - mid) / (right - mid)
        return 1 - 0.5 * temp * temp
    else:
        temp = (b - length) / (b - right)
        return 0.5 * temp * temp


@vectorized
def fp(lce):
    """ Passive force component.

    Args:
        lce (np.ndarray|float): Virtual muscle lengths.

    Returns:
        np.ndarray|float: The corresponding passive force component.
    """
    b = 0.5 * (LMAX + 1)
    if lce <= 1:
        return 0
    elif lce <= b:
        temp = (lce - 1) / (b - 1)
        return 0.25 * FPMAX * temp * temp * temp
    else:
        temp = (lce - b) / (b - 1)
        return 0.25 * FPMAX * (1 + 3 * temp)


def fv_vec(lce_dot, vmax):
    """ Force-velocity curve.

    Args:
        lce_dot (np.ndarray): Array with virtual muscle velocities.
        vmax (np.ndarray|float): Array or float with the VMAX value.

    Returns:
        np.ndarray: The corresponding force-velocity multipliers.
    """
    c = FVMAX - 1
    return force_vel_v_vec(lce_dot, c, vmax, FVMAX)


def force_vel_v_vec(velocity, c, vmax, fvmax):
    """ Force velocity relationship as implemented by MuJoCo.

    Args:
        velocity (np.ndarray): Array with virtual muscle velocities.
        c (float): Virtual velocity at which the curve is 1. Determines the shape of the curve.
        vmax (np.ndarray|float): Scaling factor VMAX. Determines the shape of the curve.
        fvmax (float): Maximum multiplier due to velocity. Determines the shape of the curve.

    Returns:
        np.ndarray: The corresponding force-velocity multipliers.
    """
    eff_vel = velocity / vmax
    eff_vel_con1 = eff_vel[eff_vel <= c]
    eff_vel_con2 = eff_vel[eff_vel <= 0]

    output = np.full(velocity.shape, fvmax)
    output[eff_vel <= c] = fvmax - (c - eff_vel_con1) * (c - eff_vel_con1) / c
    output[eff_vel <= 0] = (eff_vel_con2 + 1) * (eff_vel_con2 + 1)
    output[eff_vel < -1] = 0

    return output


def vmax_calibration(env_name, n_episodes, save_dir, lr=0.1, lr_decay=0.8, decay_lr_every=100, make_plots=True):
    """ Iteratively calibrate VMAX parameters for the muscle model.

    We determine VMAX with an iterative procedure. Using an initial value we take random actions and measure the maximum
    achieved joint velocity.
    The initial VMAX values are then updated using learning rate `lr` and we continue with more random actions. The
    learning rate is updated every `decay_lr_every` episodes by factor `lr_decay`. The procedure continues for
    `n_episodes` episodes. Optionally VMAX can be plotted for every step by setting `make_plots` to ``True``.
    We use the environment as provided by `env_name`. For MIMo these are fixed environments in which MIMo is hovering
    in the air with gravity disabled entirely.
    Muscle actions do not use the full range of inputs, instead we randomly set maximum or minimum inputs with no in
    between.
    The final VMAX values are saved to a file "vmax.npy" in the plotting directory.

    Args:
        env_name (str): The name of the environment to be used for the calibration. Must use the muscle model.
        n_episodes (int): The total number of episodes.
        save_dir (str): The directory where the final VMAX and any plots will be saved.
        lr (float): The learning rate used to update VMAX every episode. Default 0.1.
        lr_decay (float): The learning rate is multiplied by this factor every `decay_lr_every` episodes. Default 0.8.
        decay_lr_every (int): How often the learning rate is updated. Default 100.
        make_plots (bool): If ``True`` we plot the change in VMAX over time and save as a file in the plotting
            directory. Default ``True``.

    Returns:
        np.ndarray: A numpy array with the final VMAX values.
    """
    os.makedirs(save_dir, exist_ok=True)
    vmax_scale_factor = 1
    env: MIMoEnv = gym.make(env_name)
    max_vel = env.actuation_model.vmax
    if not isinstance(max_vel, np.ndarray):
        max_vel = np.ones_like(env.actuation_model.lce_dot_1) * max_vel
    vmaxes = np.zeros((n_episodes + 1, env.actuation_model.lce_dot_1.shape[0]))
    vmaxes[0, :] = max_vel.copy()
    print("Calibrating VMAX for {} episodes using initial lr {} with decay {} every {} episodes.".format(
        n_episodes, lr, lr_decay, decay_lr_every))
    # Perform iteration
    for ep in range(1, n_episodes + 1):
        # Set initial values for this iteration
        max_vel_episode = np.zeros_like(env.actuation_model.lce_dot_1) + EPS
        env.actuation_model.set_vmax(max_vel)
        _ = env.reset()
        ep_steps = 0
        action = np.zeros(env.action_space.shape)
        # Runs an episode
        while True:
            if not ep_steps % 200:
                action[:] = np.random.randint(0, 2, size=action.shape)
            state, rew, done, trunc, info = env.step(action)
            max_vel_episode = vmax_scale_factor * np.maximum(max_vel_episode, env.actuation_model.lce_dot_1)
            max_vel_episode = vmax_scale_factor * np.maximum(max_vel_episode, env.actuation_model.lce_dot_2)
            ep_steps += 1

            if done or trunc:
                break
        # Calculate new VMAX
        delta = max_vel_episode - max_vel
        max_vel = max_vel + lr * delta
        vmaxes[ep, :] = max_vel
        # Update learning rate
        if ep % decay_lr_every == 0:
            lr = lr * lr_decay
            norm_of_delta_over_lr = np.linalg.norm(vmaxes[ep, :] - vmaxes[ep - decay_lr_every], ord=2)
            print("{} episodes elapsed, updating lr to {:.6g}, "
                  "Norm of difference for VMAX since last: {:.6g}".format(ep, lr, norm_of_delta_over_lr))

    # Average vmax since that does deviate between runs
    max_vel = average_left_right(env, max_vel)
    np.save(os.path.join(save_dir, "vmax.npy"), max_vel)
    # VMAX is in relative quanitity, compute corresponding maximum qvel values (note that actual qvel may exceed this)
    qvmax = np.stack([max_vel / env.actuation_model.moment_1, max_vel / env.actuation_model.moment_2], axis=-1)
    np.save(os.path.join(save_dir, "qvmax.npy"), qvmax)
    np.save("vmax.npy", max_vel)
    # Make plot of vmax over time
    if make_plots:
        actuator_ids = env.mimo_actuators
        actuator_names = [env.model.actuator(act_id).name for act_id in actuator_ids]
        print("Creating vmax plots")
        # VMAX plot
        fig, axs = plt.subplots(len(actuator_names), 1, figsize=(10, env.n_actuators))
        for i, actuator_name in enumerate(actuator_names):
            data = vmaxes[:, i]
            ax = axs[i]
            ax.plot(data)
            ax.set_xlim([0, data.shape[0]])
            ax.hlines(y=0.0, xmin=0, xmax=data.shape[0], colors="tab:grey")
            actuator_name = actuator_name.replace("act:", "")
            ax.set_title(actuator_name + "_vmax")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "vmax.png"))
        fig.clear()
        plt.close(fig)
        # Change in VMAX plot
        fig, axs = plt.subplots(len(actuator_names), 1, figsize=(10, env.n_actuators))
        for i, actuator_name in enumerate(actuator_names):
            data = np.abs(vmaxes[1:, i] - vmaxes[:-1, i])
            ax = axs[i]
            ax.plot(data)
            ax.set_xlim([0, data.shape[0]])
            ax.hlines(y=0.0, xmin=0, xmax=data.shape[0], colors="tab:grey")
            actuator_name = actuator_name.replace("act:", "")
            ax.set_title(actuator_name + "_vmax_delta")
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, "vmax_delta.png"))
        fig.clear()
        plt.close(fig)
        env.close()
    return max_vel


def fmax_calibration(env_name, save_dir, n_iterations=3, make_plots=True):
    """ Calibrate FMAX parameters for the muscle model.

    The calibration procedure is as follows:
    We take the desired maximum force values from the actuator definitions in the scene XML.
    We then apply maximum control input in one direction for 500 steps, back off for 500 steps, and then maximum input
    in the opposite direction for 500 steps.
    The maximum torque actually generated during this is recorded for each direction and compared against the desired
    values. The FMAX parameter is then adjusted such that the generated and desired torques match.
    This is performed iteratively as all MuJoCo constraints are soft and even locked joints will change position
    slightly based on applied torque.

    This method requires a specialised scene to measure maximum voluntary isometric muscle torque in which all joints
    are locked in the angle at which the torque is to be measured.
    Note also that if the initial FMAX (set in MIMoMuscleEnv) is too large the motors may overcome the joint locking
    entirely, leading to NaNs and associated errors. In this case adjust the initial FMAX downwards.

    Args:
        env_name (str): The name of the environment to be used for the calibration. Must use the muscle model.
        save_dir (str): The directory where the final VMAX and any plots will be saved.
        n_iterations (int): How many iterations of the calibration to perform. Default 3.
        make_plots (bool): If ``True`` we plot muscle parameters during the last iteration. Default ``True``.

    Returns:
        np.ndarray: A numpy array with the final FMAX values.
    """
    os.makedirs(save_dir, exist_ok=True)
    env: MIMoEnv = gym.make(env_name)
    target_torque = np.concatenate([env.actuation_model.maximum_isometric_forces[:, 0],
                                    env.actuation_model.maximum_isometric_forces[:, 1]])
    fmax = env.actuation_model.fmax
    n_actuators = env.n_actuators
    if not isinstance(fmax, np.ndarray):
        fmax = np.ones_like(target_torque) * fmax

    actuator_ids = env.mimo_actuators
    actuator_names = [env.model.actuator(act_id).name for act_id in actuator_ids]
    muscle_data = []

    # Perform iteration
    for ep in range(n_iterations):
        # Setup for this iteration
        print("FMAX iteration {} of {}".format(ep + 1, n_iterations))
        _ = env.reset()
        env.actuation_model.set_fmax(fmax.copy())
        fmax_old = fmax.copy()
        ep_steps = 0
        max_unscaled_torque = np.zeros(env.action_space.shape)

        # 500 steps with maximum tension of muscles acting in negative direction
        action = np.zeros(env.action_space.shape)
        action[:n_actuators] = 1.0
        for j in range(500):
            _ = env.step(action)
            ep_steps += 1
            if make_plots and ep == n_iterations - 1:
                muscle_data.append(env.actuation_model.collect_data_for_actuators())
            # Once we have stabilized, start collecting MVF
            if j > 250:
                torque_1 = env.actuation_model.moment_1 * env.actuation_model.force_muscles_1
                torque_2 = - env.actuation_model.moment_2 * env.actuation_model.force_muscles_2
                unscaled_torque = np.concatenate([torque_1, torque_2])
                max_unscaled_torque = np.maximum(max_unscaled_torque, unscaled_torque)

        # 500 steps with no tension at all
        action = np.zeros(env.action_space.shape)
        for _ in range(500):
            _ = env.step(action)
            ep_steps += 1
            if make_plots and ep == n_iterations - 1:
                muscle_data.append(env.actuation_model.collect_data_for_actuators())

        # 500 steps with maximum tension of muscles acting in positive direction
        action = np.zeros(env.action_space.shape)
        action[n_actuators:] = 1.0
        for j in range(500):
            _ = env.step(action)
            ep_steps += 1
            if make_plots and ep == n_iterations - 1:
                muscle_data.append(env.actuation_model.collect_data_for_actuators())
            if j > 250:
                torque_1 = env.actuation_model.moment_1 * env.actuation_model.force_muscles_1
                torque_2 = - env.actuation_model.moment_2 * env.actuation_model.force_muscles_2
                unscaled_torque = np.concatenate([torque_1, torque_2])
                max_unscaled_torque = np.maximum(max_unscaled_torque, unscaled_torque)

        fmax = target_torque / (max_unscaled_torque + EPS)
        print("Norm of difference for fmax: {:.6g}".format(np.linalg.norm(fmax - fmax_old, ord=2)))

    fmax[:n_actuators] = average_left_right(env, fmax[:n_actuators])
    fmax[n_actuators:] = average_left_right(env, fmax[n_actuators:])
    np.save(os.path.join(save_dir, "fmax.npy"), fmax)

    # Plot the data
    if make_plots:
        print("Creating plots")
        muscle_data_dict = {}
        muscle_data = np.asarray(muscle_data)
        for i, act_name in enumerate(actuator_names):
            muscle_data_dict[act_name] = np.asarray(muscle_data[:, :, i])
        create_joint_plots(save_dir, muscle_data_dict, env.dt)

    env.close()
    np.save("fmax.npy", fmax)
    return fmax


def create_joint_plots(plot_dir, data, dt=None):
    """ Creates a series of plots for muscles data.

    This function is designed to be used with
    :meth:`~mimoEnv.envs.mimo_muscle_env.MIMoMuscleEnv.collect_data_for_actuator`.
    The `data` argument should be a dictionary with the data for each actuator saved as an array with the actuator name
    as the dictionary key. The structure of the array should have steps or time as the first dimension and the different
    return values as the second.

    Args:
        plot_dir (str): The directory where the plots will be saved.
        data (Dict[str, np.ndarray]): A dictionary containing the actuator data.
        dt (float|None): The time between data points. If not ``None`` the x-axis will be time instead of number of data
            points. Default ``None``.
    """

    for actuator_name in data:
        act_data = data[actuator_name]
        n_steps = act_data.shape[0]
        x = np.arange(n_steps)
        xlimit = n_steps
        if dt is not None:
            x = x * dt
            xlimit = xlimit * dt
        fig, axs = plt.subplots(10, 2, figsize=(10, 18))
        for i, ax in enumerate(axs.flat):
            ax.plot(x, act_data[:, i])
            ax.set_xlim([0, xlimit])
            ax.hlines(y=0.0, xmin=0, xmax=xlimit, colors="tab:grey")
        actuator_name = actuator_name.replace("act:", "")
        axs[0, 0].set_title("qpos")
        axs[0, 1].set_title("qvel")
        axs[1, 0].set_title("qpos_muscle")
        axs[1, 1].set_title("gear")
        axs[2, 0].set_title("action")
        axs[3, 0].set_title("activity")
        axs[4, 0].set_title("lce")
        axs[5, 0].set_title("lce_dot")
        axs[6, 0].set_title("muscle_force")
        axs[7, 0].set_title("FL")
        axs[8, 0].set_title("FV")
        axs[9, 0].set_title("FP")

        # Connect axis for flexor/extensor plots
        for i in range(2, 10):
            axs[i, 0].get_shared_y_axes().join(axs[i, 0], axs[i, 1])
            axs[i, 0].autoscale()
            axs[i, 0].set_xlim([0, xlimit])

        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, actuator_name + ".png"))
        fig.clear()
        plt.close(fig)


def average_left_right(env, array):
    """ Averages an array with actuator values between left and right side actuators of MIMo.

    Actuators without symmetric versions are left as is.

    Args:
        env (mimoEnv.envs.MIMoEnv): A MIMo environment.
        array (np.ndarray): An array with actuator values. Note that the first dimension must have the same size as the
            number of MIMo actuators in the environment.

    Returns:
        np.ndarray: The averaged array.
    """
    averaged_array = array.copy()
    actuator_ids = env.mimo_actuators
    actuator_names = [env.sim.model.actuator_id2name(act_id) for act_id in actuator_ids]
    lefts = [i for i, name in enumerate(actuator_names) if "left" in name]
    rights = [i for i, name in enumerate(actuator_names) if "right" in name]
    averages = (array[lefts] + array[rights]) / 2
    averaged_array[lefts] = averages
    averaged_array[rights] = averages
    dif = np.amax(np.abs(averaged_array - array))
    print("Max Deviation to average", dif)
    return averaged_array


def plotting_episode(env_name, save_dir):
    """ Performs a single episode, saving and creating joint value plots.

    We randomize action inputs to either maximum or minimum values every 200 steps.

    Args:
        env_name (str): The name of the environment to use.
        save_dir (str): The directory where the data will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("Collecting data for plots")
    env: MIMoEnv = gym.make(env_name)
    max_vel = env.actuation_model.vmax
    np.save(os.path.join(save_dir, 'vmax.npy'), max_vel)
    actuator_ids = env.mimo_actuators
    actuator_names = [env.model.actuator(act_id).name for act_id in actuator_ids]

    muscle_props = []
    muscle_data = []

    _ = env.reset()
    ep_steps = 0
    action = np.zeros(env.action_space.shape)
    while True:
        if not ep_steps % 200:
            action[:] = np.random.randint(0, 2, size=action.shape)

        state, rew, done, info = env.step(action)
        muscle_data.append(env.actuation_model.collect_data_for_actuators())
        ep_steps += 1
        muscle_props.append([env.actuation_model.lce_1.copy(),
                             env.actuation_model.lce_2.copy(),
                             env.actuation_model.lce_dot_1.copy(),
                             env.actuation_model.lce_dot_2.copy(),
                             env.actuation_model.force_muscles_1.copy(),
                             env.actuation_model.force_muscles_2.copy()])

        if done:
            break

    # Plot the data
    print("Creating plots")
    muscle_data_dict = {}
    muscle_data = np.asarray(muscle_data)
    for i, act_name in enumerate(actuator_names):
        muscle_data_dict[act_name] = muscle_data[:, :, i]
    create_joint_plots(save_dir, muscle_data_dict, env.dt)

    lengths_1 = [x[0] for x in muscle_props]
    vels_1 = [x[2] for x in muscle_props]
    forces_1 = [x[4] for x in muscle_props]
    fig, axs = plt.subplots(env.n_actuators, 3, figsize=(10, env.n_actuators))
    for midx in range(env.n_actuators):
        f = fl
        lengthes = np.linspace(0.5, 1.2, 100)
        axs[midx, 0].plot(lengthes, f(lengthes), color='tab:blue')
        axs[midx, 0].plot([x[midx] for x in lengths_1], [f(x[midx]) for x in lengths_1], 'x', color='tab:red')
        axs[midx, 0].set_xlim([0.6, 1.2])
        f = fv_vec
        vels = np.linspace(-max_vel, max_vel, 100)
        axs[midx, 1].plot(vels[:, midx], f(vels[:, midx], max_vel[midx]), color='tab:blue')
        axs[midx, 1].plot([x[midx] for x in vels_1], [f(x[midx], max_vel[midx]) for x in vels_1], 'x', color='tab:red')
        axs[midx, 1].set_xlim([-max_vel[midx], max_vel[midx]])
        axs[midx, 2].plot([x[midx] for x in forces_1], 'x', color='tab:red')
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "mimoflfvplots.png"))
    fig.clear()
    plt.close(fig)
    env.close()


def recording_episode(env_name, video_dir, env_params, video_width=500, video_height=500,
                      camera_name=None, make_joint_plots=True, binary_actions=False, interactive=False):
    """ Perform a single episode, saving joint data and creating a video recording.

    We randomize action inputsevery 200 steps.

    Args:
        env_name (str): The environment to use.
        video_dir (str): The directory where the video and any plots will be saved.
        env_params (Dict): A dictionary with parameters to the environment. Keys are parameter names.
        video_width (int): The width of the rendered video. Default 500.
        video_height (int): The height of the rendered video. Default 500.
        camera_name (str|None): The name of the camera to use for the video. If ``None``, the MuJoCo freecam is used
            (camera ID -1). Default ``None``.
        make_joint_plots (bool): If ``True`` we also save plots of joint and muscle parameters over time. Default
            ``True``.
        binary_actions (bool): If ``True``, actions are randomized to be minimal or maximal. Default ``False``.
        interactive (bool): If ``True``, an interactive window is also rendered. Default ``False``.
    """
    os.makedirs(video_dir, exist_ok=True)
    env: MIMoEnv = gym.make(env_name, **env_params)
    _ = env.reset()
    ep_steps = 0
    action = np.zeros(env.action_space.shape)
    images = []
    muscle_data = []
    actuator_ids = env.mimo_actuators
    actuator_names = [env.model.actuator(act_id).name for act_id in actuator_ids]
    while True:
        if ep_steps % 200 == 0:
            if binary_actions:
                action[:] = np.random.randint(0, 2, size=action.shape)
            else:
                action[:] = env.action_space.sample()
        ep_steps += 1
        obs, _, done, _ = env.step(action)
        img = env.mujoco_renderer.render(render_mode="rgb_array",
                                                   width=video_width,
                                                   height=video_height,
                                                   camera_name=camera_name)
        images.append(img)
        if make_joint_plots:
            muscle_data.append(env.actuation_model.collect_data_for_actuators())
        if interactive:
            env.render()
        if done:
            break
    framerate = 1 / env.dt
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(os.path.join(video_dir, 'test_video.avi'),
                            fourcc,
                            framerate, (video_width, video_height))
    for img in images:
        video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()
    video.release()
    if make_joint_plots:
        muscle_data_dict = {}
        muscle_data = np.asarray(muscle_data)
        for i, act_name in enumerate(actuator_names):
            muscle_data_dict[act_name] = muscle_data[:, :, i]
        create_joint_plots(video_dir, muscle_data_dict, env.dt)
    env.close()


def compliance_test():
    """ Performs the compliance test from the paper.
    """

    # Plotting function
    def plot_qpos_torque(qpos, motor_torques, net_torques, imgs, image_times, file_name):
        """ Make the qpos/torque/img timeline plots.

        Args:
            qpos (List[np.ndarray]): A list with joint qpos values for each time step.
            motor_torques (List[np.ndarray]): A list with motor torques for each time step.
            net_torques (List[np.ndarray]): A list with net actuation torques for each time step.
            imgs (List[np.ndarray]): A list images.
            image_times (List[int]): On which step each of the images in `imgs` was taken.
            file_name (str): The file where the plot will be saved.
        """
        fig = plt.figure(figsize=(12, 6), layout="constrained")
        heights = [2, 1, 1]
        fig.tight_layout()
        gs = GridSpec(3, 4, figure=fig, height_ratios=heights)

        # Plot images
        for i, plot_img in enumerate(imgs):
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(plot_img)
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)

        # Plot torque and qpos
        x = (np.arange(n_steps) + 1) * env.dt
        torque_plot = fig.add_subplot(gs[2, :])
        torque_plot.plot(x, motor_torques, color="tab:cyan", label="Active torque")
        torque_plot.plot(x, net_torques, color="tab:green", label="Net torque")
        torque_plot.set_xlim([np.min(x), np.max(x)])
        torque_plot.set_xlabel("Time (s)")
        torque_plot.set_ylabel("Torque (Nm)")
        for image_time in image_times:
            torque_plot.axvline(image_time * env.dt, color="tab:red", alpha=0.5)
        torque_plot.legend()
        torque_plot.grid(axis="y")

        qpos_plot = fig.add_subplot(gs[1, :], sharex=torque_plot)
        qpos_plot.plot(x, qpos, color="tab:cyan")
        qpos_plot.set_xlim([np.min(x), np.max(x)])
        qpos_plot.tick_params(labelbottom=False)
        qpos_plot.set_ylabel("Joint Position (rad)")
        for image_time in image_times:
            qpos_plot.axvline(image_time * env.dt, color="tab:red", alpha=0.5)
        qpos_plot.grid(axis="y")

        fig.savefig(file_name)
        fig.clear()
        plt.close(fig)

    def make_video(images, file_name):
        """ Saves the images as a video.

        Args:
            images (List[np.ndarray]): A list of images.
            file_name (str): The output video file.
        """
        video_width = 500
        video_height = 500
        framerate = 1 / env.dt
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(file_name,
                                fourcc,
                                framerate, (video_width, video_height))
        for img in images:
            video.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        cv2.destroyAllWindows()
        video.release()

    def collect_data(env, action, img_times):
        """ Collect joint data and video images.

        Args:
            env (MIMoEnv): The environment to use.
            action (np.ndarray): The input action. A constant action is used throughout.
            img_times (List[int]): A list of time steps when images will be taken for plotting.
        """
        qpos = []
        motor_torques = []
        net_torques = []
        plot_imgs = []
        video_imgs = []
        for i in range(n_steps):
            obs, _, done, trunc, _ = env.step(action)
            # Collect information on right shoulder joint
            qpos.append(env.data.qpos[shoulder_joint_qpos])
            motor_torque = env.data.ctrl[shoulder_actuator_id] * env.model.actuator_gear[shoulder_actuator_id, 0]
            motor_torques.append(motor_torque)
            stiffness_torque = env.model.jnt_stiffness[shoulder_joint_id] * env.data.qpos[shoulder_joint_qpos]
            damping_torque = env.model.dof_damping[shoulder_joint_dof] * env.data.qvel[shoulder_joint_qvel]
            net_torque = motor_torque - damping_torque - stiffness_torque
            net_torques.append(net_torque)
            img = env.mujoco_renderer.render(render_mode="rgb_array")
            video_imgs.append(img)
            if i + 1 in img_times:
                plot_imgs.append(img)
        return qpos, motor_torques, net_torques, plot_imgs, video_imgs

    n_steps = 500

    # First do the version without muscles
    env: MIMoEnv = gym.make("MIMoComplianceTest-v0")
    _ = env.reset()
    # Collect all the indices and what have you for the right shoulder joint
    shoulder_joint_id = env.model.joint("robot:right_shoulder_ad_ab").id
    shoulder_joint_qpos = mimoEnv.utils.get_joint_qpos_addr(env.model, shoulder_joint_id)
    shoulder_joint_qvel = mimoEnv.utils.get_joint_qvel_addr(env.model, shoulder_joint_id)
    shoulder_joint_dof = env.model.jnt_dofadr[shoulder_joint_id]
    shoulder_actuator_name = "act:right_shoulder_abduction"
    shoulder_actuator_id = env.model.actuator(shoulder_actuator_name).id

    # Fixed control input
    control_input = 0.1775
    action = np.zeros(env.action_space.shape)
    action[shoulder_actuator_id] = control_input
    img_times_motor = [45, 52, 73, 450]
    # Data collection
    qpos_motor, motor_torques_motor, net_torques_motor, plot_imgs_motor, \
        video_imgs_motor = collect_data(env, action, img_times_motor)
    # Plot the data
    plot_qpos_torque(qpos_motor, motor_torques_motor, net_torques_motor,
                     plot_imgs_motor, img_times_motor, "compliance_motor.png")
    make_video(video_imgs_motor, "compliance_motor.avi")
    env.close()

    # Muscle environment - soft
    env = gym.make("MIMoComplianceMuscleTest-v0")
    _ = env.reset()
    control_input_agonist = 0.139
    control_input_antagonist = 0.0
    action = np.zeros(env.action_space.shape)
    action[shoulder_actuator_id] = control_input_antagonist
    action[shoulder_actuator_id + env.n_actuators] = control_input_agonist
    img_times_soft = [45, 52, 77, 450]
    # Data collection
    qpos_soft, motor_torques_soft, net_torques_soft, plot_imgs_soft, \
        video_imgs_soft = collect_data(env, action, img_times_soft)
    # Plot the data
    plot_qpos_torque(qpos_soft, motor_torques_soft, net_torques_soft,
                     plot_imgs_soft, img_times_soft, "compliance_muscle_soft.png")
    make_video(video_imgs_soft, "compliance_muscle_soft.avi")

    # Muscle environment - hard
    _ = env.reset()
    control_input_agonist = 0.754
    control_input_antagonist = 0.9
    action = np.zeros(env.action_space.shape)
    action[shoulder_actuator_id] = control_input_antagonist
    action[shoulder_actuator_id + env.n_actuators] = control_input_agonist
    img_times_stiff = [45, 52, 56, 450]
    # Data collection
    qpos_stiff, motor_torques_stiff, net_torques_stiff, plot_imgs_stiff, \
        video_imgs_stiff = collect_data(env, action, img_times_stiff)
    # Plot the data
    plot_qpos_torque(qpos_stiff, motor_torques_stiff, net_torques_stiff,
                     plot_imgs_stiff, img_times_stiff, "compliance_muscle_stiff.png")
    make_video(video_imgs_stiff, "compliance_muscle_stiff.avi")

    # Muscle environment - softish
    _ = env.reset()
    control_input_agonist = 0.1738
    control_input_antagonist = 0.05
    action = np.zeros(env.action_space.shape)
    action[shoulder_actuator_id] = control_input_antagonist
    action[shoulder_actuator_id + env.n_actuators] = control_input_agonist
    img_times_softish = [45, 52, 74, 450]
    # Data collection
    qpos_softish, motor_torques_softish, net_torques_softish, \
        plot_imgs_softish, video_imgs_softish = collect_data(env, action, img_times_softish)
    # Plot the data
    plot_qpos_torque(qpos_softish, motor_torques_softish, net_torques_softish,
                     plot_imgs_softish, img_times_softish, "compliance_muscle_softish.png")
    make_video(video_imgs_softish, "compliance_muscle_softish.avi")

    # Muscle environment - medium
    env = gym.make("MIMoComplianceMuscleTest-v0")
    _ = env.reset()
    control_input_agonist = 0.2763
    control_input_antagonist = 0.2
    action = np.zeros(env.action_space.shape)
    action[shoulder_actuator_id] = control_input_antagonist
    action[shoulder_actuator_id + env.n_actuators] = control_input_agonist
    img_times_medium = [45, 52, 66, 450]
    # Data collection
    qpos_medium, motor_torques_medium, net_torques_medium, \
        plot_imgs_medium, video_imgs_medium = collect_data(env, action, img_times_medium)
    # Plot the data
    plot_qpos_torque(qpos_medium, motor_torques_medium, net_torques_medium,
                     plot_imgs_medium, img_times_medium, "compliance_muscle_medium.png")
    make_video(video_imgs_medium, "compliance_muscle_medium.avi")

    # Make paper plots:
    # 4 Shots showing maximum deflection for Motor, muscle soft, muscle medium and muscle stiff
    # One position plot -> Motor, Muscle soft, Muscle medium, Muscle stiff
    # One torque plot -> Motor, Muscle soft, Muscle medium, Muscle stiff, net-torque only for muscle, both for motor
    # Write Strong explanation as to torque plots
    # Plot colors: Red motor, muscle shades of blue-green plots
    imgs = plot_imgs_medium
    img_times = img_times_medium
    img_labels = ["A", "B", "C", "D"]
    img_label_x_positions = [(img_time + 1) * env.dt for img_time in img_times]
    img_label_x_positions[0] -= 0.1  # Adjust for narrow spacing by putting first text on left
    fig = plt.figure(figsize=(8, 5), layout="constrained")
    fig.tight_layout()
    heights = [1.5, 1, 1]
    gs = GridSpec(3, 4, figure=fig, height_ratios=heights)

    # Plot images
    for i, plot_img in enumerate(imgs):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(plot_img[100:-100, 125:-75, :])  # Slicing does a crop
        ax.plot(0, 0, "-", color="tab:gray", label=img_labels[i])
        ax.legend(handlelength=0, loc="upper left", handletextpad=-0.2)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

    # Plot torque and qpos
    x = (np.arange(n_steps) + 1) * env.dt
    torque_plot = fig.add_subplot(gs[2, :])
    torque_plot.plot(x, net_torques_motor, color="red", label="Motor", alpha=0.8)
    torque_plot.plot(x, net_torques_soft, color="cyan", label="Muscle - soft", alpha=0.8)
    torque_plot.plot(x, net_torques_medium, color="darkturquoise", label="Muscle - medium", alpha=0.8)
    torque_plot.plot(x, net_torques_stiff, color="darkcyan", label="Muscle - stiff", alpha=0.8)
    torque_plot.set_xlim([np.min(x), np.max(x)])
    torque_plot.set_xlabel("Time (s)")
    torque_plot.set_ylabel("Torque (Nm)")
    torque_plot.legend()
    torque_plot.grid(axis="y")

    qpos_plot = fig.add_subplot(gs[1, :], sharex=torque_plot)
    qpos_plot.plot(x, qpos_motor, color="red", label="Motor", alpha=0.8)
    qpos_plot.plot(x, qpos_soft, color="cyan", label="Muscle - soft", alpha=0.8)
    qpos_plot.plot(x, qpos_medium, color="darkturquoise", label="Muscle - medium", alpha=0.8)
    qpos_plot.plot(x, qpos_stiff, color="darkcyan", label="Muscle - stiff", alpha=0.8)
    qpos_plot.set_xlim([np.min(x), np.max(x)])
    qpos_plot.tick_params(labelbottom=False)
    qpos_plot.set_ylabel("Joint Position (rad)")
    for i, image_time in enumerate(img_times):
        qpos_plot.axvline(image_time * env.dt, color="tab:gray", alpha=0.5)
        qpos_plot.text(img_label_x_positions[i], 0.1, img_labels[i], rotation=0, verticalalignment='center',
                       color="tab:gray", alpha=0.9)
    qpos_plot.grid(axis="y")

    fig.savefig("paperplot.png")
    fig.clear()
    plt.close(fig)

    env.close()

# ========================== Workflow functions ========================================
# ======================================================================================


def calibrate_full(save_dir,
                   n_fmax=3,
                   n_vmax=30,
                   n_episodes_per_it=50,
                   n_episodes_video=5,
                   lr_initial=0.1,
                   lr_decay=0.7,
                   fmax_scene="MIMoMuscleStaticTest-v0",
                   vmax_scene="MIMoVelocityMuscleTest-v0",
                   video_scene=None):
    """ Determine muscle parameters for a given model.

    Performs FMAX and VMAX calibrations. Afterward some scenes can be recorded to video with the new muscle parameters.
    Note that the parameter calibration requires specialised scenes, see the documentation for
    :func:`~.fmax_calibration` and :func:`~.vmax_calibration` for more information.

    Args:
        save_dir (str): The directory where output files and subdirectories will be created.
        n_fmax (int): The number of iterations for the FMAX calibration. Default 3.
        n_vmax (int): The number of iterations for the VMAX calibration. Default 20.
        n_episodes_per_it (int): The number of episodes for each VMAX iteration. Default 20.
        n_episodes_video (int): After calibration, this many episodes will be recorded to video using the new parameters.
        lr_initial (float): The initial learning rate for the VMAX iteration. Default 0.1.
        lr_decay (float): Decay factor after each VMAX iteration. Default 0.7.
        fmax_scene (str): The environment, by name, to use for the FMAX calibration.
        vmax_scene (str): The environment, by name, to use for the VMAX calibration.
        video_scene (str): The environment, by name, that will be used to record videos. If ``None``, no video is
            recorded. Default ``None``.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The new FMAX and VMAX parameters.
    """

    if os.path.exists("vmax.npy"):
        os.remove("vmax.npy")
    if os.path.exists("fmax.npy"):
        os.remove("fmax.npy")

    vmax_total_episodes = n_vmax * n_episodes_per_it

    # fmax adjusting
    fmax = fmax_calibration(fmax_scene,
                            os.path.join(save_dir, "fmax"),
                            n_iterations=n_fmax,
                            make_plots=True)
    # vmax
    vmax = vmax_calibration(vmax_scene,
                            vmax_total_episodes,
                            os.path.join(save_dir, "vmax"),
                            lr=lr_initial,
                            lr_decay=lr_decay,
                            decay_lr_every=n_episodes_per_it,
                            make_plots=True)
    plotting_episode(vmax_scene, os.path.join(save_dir, "vmax"))
    recording_env_params = {
        "touch_params": None,
        "vision_params": None,
    }
    if video_scene is not None:
        for i in range(n_episodes_video):
            if i < n_episodes_video / 2:
                binary_actions = False
            else:
                binary_actions = True
            recording_episode(video_scene,
                              os.path.join(save_dir, "video_{}".format(i)),
                              env_params=recording_env_params,
                              binary_actions=binary_actions)

    return fmax, vmax


def repeatability_test(save_dir,
                       n_fmax=3,
                       n_vmax=30,
                       n_episodes_per_it=50,
                       n_episodes_video=5,
                       lr_initial=0.1,
                       lr_decay=0.7,
                       fmax_scene="MIMoMuscleStaticTest-v0",
                       vmax_scene="MIMoVelocityMuscleTest-v0",
                       video_scene=None,
                       n_repeats=3
                       ):
    """ Performs multiple full calibrations and compares the results against one another for repeatability.

    Args:
        save_dir (str): The directory where output files and subdirectories will be created.
        n_fmax (int): The number of iterations for the FMAX calibration. Default 3.
        n_vmax (int): The number of iterations for the VMAX calibration. Default 30.
        n_episodes_per_it (int): The number of episodes for each VMAX iteration. Default 50.
        n_episodes_video (int): After calibration, this many episodes will be recorded to video using the new
            parameters.
        lr_initial (float): The initial learning rate for the VMAX iteration. Default 0.1.
        lr_decay (float): Decay factor after each VMAX iteration. Default 0.7.
        fmax_scene (str): The environment, by name, to use for the FMAX calibration.
        vmax_scene (str): The environment, by name, to use for the VMAX calibration.
        video_scene (str): The environment, by name, that will be used to record videos. If ``None``, no video is
            recorded. Default ``None``.
        n_repeats (int): The number of repetitions.
    """
    fmaxes = []
    vmaxes = []
    for i in range(n_repeats):
        fmax, vmax = calibrate_full(os.path.join(save_dir, "test{}".format(i)),
                                    n_fmax=n_fmax,
                                    n_vmax=n_vmax,
                                    n_episodes_per_it=n_episodes_per_it,
                                    n_episodes_video=n_episodes_video,
                                    lr_initial=lr_initial,
                                    lr_decay=lr_decay,
                                    fmax_scene=fmax_scene,
                                    vmax_scene=vmax_scene,
                                    video_scene=video_scene)
        fmaxes.append(fmax)
        vmaxes.append(vmax)

    if os.path.exists("vmax.npy"):
        os.remove("vmax.npy")
    if os.path.exists("fmax.npy"):
        os.remove("fmax.npy")

    max_dif = np.zeros_like(vmax)
    for i in range(n_repeats-1):
        for j in range(i+1, n_repeats):
            dif = np.abs(2 * (vmaxes[i] - vmaxes[j]) / (vmaxes[i] + vmaxes[j]))
            max_dif = np.maximum(max_dif, dif)

    fmax_max_dif = np.zeros_like(fmax)
    for i in range(n_repeats-1):
        for j in range(i+1, n_repeats):
            dif = np.abs(2 * (fmaxes[i] - fmaxes[j]) / (fmaxes[i] + fmaxes[j]))
            fmax_max_dif = np.maximum(fmax_max_dif, dif)

    print("\n=== FMAX ===")
    print("Maximum fmax difference", np.amax(fmax_max_dif))

    print("\n=== VMAX ===")
    print("Maximum deviation for each joint: ", max_dif)
    print("Maximum deviation total", np.amax(max_dif))
    print("Average deviation", np.sum(max_dif) / max_dif.shape[0])


def make_flfvfp_plots():
    """ Creates a set of plots to show the FL, FV and FP curves."""
    qvmin = 0.75
    qvmax = 1.05
    l_min = 0.5
    l_max = 1.6
    fl_y_limits = [0, 1.2]
    fv_y_limits = [0, 1.3]
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    l_range = np.linspace(l_min, l_max, 100)
    v_range = np.linspace(-1.2, 0.4, 100)
    fl_torque = fl(l_range)
    fp_torque = fp(l_range)
    fv_torque = fv_vec(v_range, 1.0)
    fl_plot = axs[0]
    fv_plot = axs[1]
    fl_plot.plot(l_range, fl_torque, color='tab:cyan', label="FL")
    fl_plot.plot(l_range, fp_torque, color='tab:olive', label="FP")
    fl_plot.plot(l_range, fl_torque+fp_torque, color='tab:green', label="FL+FP")
    fl_plot.fill_between(l_range, fl_y_limits[0], fl_y_limits[1], where=(l_range <= qvmax) & (l_range >= qvmin),
                         alpha=0.3, color="tab:red")
    fl_plot.set_xlim([l_min, l_max])
    fl_plot.set_ylim(fl_y_limits)
    fl_plot.set_xlabel("Virtual muscle length")
    fl_plot.set_title("Force-length curves")
    fl_plot.grid()
    fl_plot.legend()
    fv_plot.plot(v_range, fv_torque, color='tab:blue', label="FV")
    fv_plot.set_xlim([-1.2, 0.4])
    fv_plot.set_ylim(fv_y_limits)
    fv_plot.set_xlabel("Virtual muscle speed")
    fv_plot.set_title("Force-velocity curves")
    fv_plot.legend()
    fv_plot.grid()
    fig.tight_layout()
    fig.savefig("flfvplots.png")
    fig.clear()
    plt.close(fig)


if __name__ == "__main__":

    # Experiments to ensure repeatability:
    # Measure for difference between two runs: 2 * (run1 - run2) / (run1 + run2)
    # 8 n_episodes_per_it = 20,  lr = 0.1, lr_decay = 0.8,  n_iterations = 20, max error ~12%
    # 9 n_episodes_per_it = 100, lr = 0.1, lr_decay = 0.8,  n_iterations = 20, max error ~ 8.27%, average error 2.58%
    # 10 n_episodes_per_it = 20,  lr = 0.1, lr_decay = 0.7,  n_iterations = 20, max error ~18.25%, average error 4.82%
    # 11 n_episodes_per_it = 100, lr = 0.1, lr_decay = 0.7,  n_iterations = 20, max error ~ 7.30%, average error 1.97%
    # 12 n_episodes_per_it = 50,  lr = 0.1, lr_decay = 0.75, n_iterations = 30, max error ~ 8.23%, average error 2.52%
    # 13 n_episodes_per_it = 50,  lr = 0.1, lr_decay = 0.7,  n_iterations = 30, max error ~ 7.36%, average error 2.34%
    V1_static_scene = "MIMoMuscleStaticTest-v0"
    V1_velocity_scene = "MIMoVelocityMuscleTest-v0"
    V2_static_scene = "MIMoMuscleStaticTestV2-v0"
    V2_velocity_scene = "MIMoVelocityMuscleTestV2-v0"
    video_scene = "MIMoMuscle-v0"
    n_iterations_fmax = 3
    n_iterations_vmax = 30
    n_episodes_per_it = 50
    n_recording_episodes = 6
    lr = 0.1
    lr_decay = 0.70
    plotting_dir = "video_check"

    compliance_test()
