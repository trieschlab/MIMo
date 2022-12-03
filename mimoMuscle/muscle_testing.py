""" Functions in this model are designed to help transition actuation models for MIMo.

By default MIMo uses direct torque motors for actuation with the maximum torques corresponding to the maximum voluntary
isometric torques. A second actuation model exists based on ZITAT ZU PAPIER. This second model more accurately
represents the position and velocity dependent force generating behaviour of real muscles. The second model requires
several adjustments to the actuation and joint parameters, which can be done using the functions in this module.
"""
import gym
import os
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2

import mimoEnv
from mimoEnv.utils import EPS

matplotlib.use("Agg")

LMAX = 1.6
LMIN = 0.5
FVMAX = 1.2
FPMAX = 1.3


# TODO: Provide an option in MIMoDummy and to pass through camera parameters for viewer_setup
# TODO: Documentation for these functions and mimo_muscle_env, general cleanup


def vectorized(fn):
    """
    Simple vector wrapper for functions that clearly came from C
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
def FL(lce):
    """
    Force length
    """
    return bump(lce, LMIN, 1, LMAX) + 0.15 * bump(lce, LMIN, 0.5 * (LMIN + 0.95), 0.95)


@vectorized
def FP(lce):
    """
    Force passive
    """
    b = 0.5 * (LMAX + 1)
    return passive_force(lce, b)


def FV_vec(lce_dot, vmax):
    """
    Force velocity
    """
    c = FVMAX - 1
    return force_vel_v_vec(lce_dot, c, vmax, FVMAX)


def force_vel_v_vec(velocity, c, vmax, fvmax):
    """
    Force velocity relationship as implemented by MuJoCo.
    """
    eff_vel = velocity / vmax
    eff_vel_con1 = eff_vel[eff_vel <= c]
    eff_vel_con2 = eff_vel[eff_vel <= 0]

    output = np.full(velocity.shape, fvmax)
    output[eff_vel <= c] = fvmax - (c - eff_vel_con1) * (c - eff_vel_con1) / c
    output[eff_vel <= 0] = (eff_vel_con2 + 1) * (eff_vel_con2 + 1)
    output[eff_vel < -1] = 0

    return output


def bump(length, A, mid, B):
    """
    Force length relationship as implemented by MuJoCo.
    """
    left = 0.5 * (A + mid)
    right = 0.5 * (mid + B)
    temp = 0

    if (length <= A) or (length >= B):
        return 0
    elif length < left:
        temp = (length - A) / (left - A)
        return 0.5 * temp * temp
    elif length < mid:
        temp = (mid - length) / (mid - left)
        return 1 - 0.5 * temp * temp
    elif length < right:
        temp = (length - mid) / (right - mid)
        return 1 - 0.5 * temp * temp
    else:
        temp = (B - length) / (B - right)
        return 0.5 * temp * temp


def passive_force(length, b):
    """Parallel elasticity (passive muscle force) as implemented
    by MuJoCo.
    """
    temp = 0

    if length <= 1:
        return 0
    elif length <= b:
        temp = (length - 1) / (b - 1)
        return 0.25 * FPMAX * temp * temp * temp
    else:
        temp = (length - b) / (b - 1)
        return 0.25 * FPMAX * (1 + 3 * temp)


def force_vel(velocity, c, VMAX, FVMAX):
    """
    Force velocity relationship as implemented by MuJoCo.
    """
    eff_vel = velocity / VMAX
    if eff_vel < -1:
        return 0
    elif eff_vel <= 0:
        return (eff_vel + 1) * (eff_vel + 1)
    elif eff_vel <= c:
        return FVMAX - (c - eff_vel) * (c - eff_vel) / c
    else:
        return FVMAX


def vmax_calibration(env_name, n_episodes, save_dir, lr=0.1, lr_decay=0.8, decay_lr_every=100, make_plots=True):
    """ Iteratively calibrate VMAX parameters for the muscle model.

    We determine VMAX with an iterative procedure. From an initial value we take random actions and measure the maximum
    achieved joint velocity.
    The initial value is then updating using learning rate `lr` and we continue with more
    random actions. The learning rate is updated every `decay_lr_every` episodes by factor `lr_decay`. The
    procedure continues for `n_episodes` episodes. Optionally VMAX can be plotted for every step, using the
    parameter `plot_vmax`.
    We use the environment as provided by `env_name`. For MIMo these are fixed environments in
    which MIMo is hovering in the air with gravity disabled entirely.
    Muscle actions do not use the full range of inputs, instead we randomly set maximum or minimum inputs with no in
    between.
    The final VMAX values are saved to a file "vmax.npy" in the plotting directory.

    Args:
        env_name: The name of the environment to be used for the calibration.
        n_episodes: The total number of episodes.
        save_dir: The directory where the final VMAX and any plots will be saved.
        lr: The learning rate used to update VMAX every episode. Default 0.1.
        lr_decay: The learning rate is multiplied by this factor every `decay_lr_every` episodes. Default 0.8.
        decay_lr_every: How often the learning rate is updated. Default 100.
        make_plots: If `True` we plot the change in VMAX over time and save as a file in the plotting directory.
            Default `True`.

    Returns:
        A numpy array with the final VMAX values.
    """
    os.makedirs(save_dir, exist_ok=True)
    env = gym.make(env_name)
    max_vel = env.vmax
    if not isinstance(max_vel, np.ndarray):
        max_vel = np.ones_like(env.lce_dot_1) * max_vel
    vmaxes = np.zeros((n_episodes + 1, env.lce_dot_1.shape[0]))
    vmaxes[0, :] = max_vel.copy()
    print("Calibrating VMAX for {} episodes using initial lr {} with decay {} every {} episodes.".format(n_episodes, lr, lr_decay, decay_lr_every))
    # Perform iteration
    for ep in range(1, n_episodes + 1):
        # Set initial values for this iteration
        max_vel_episode = np.zeros_like(env.lce_dot_1) + EPS
        env.vmax = max_vel
        _ = env.reset()
        ep_steps = 0
        action = np.zeros(env.action_space.shape)
        # Runs an episode
        while True:
            if not ep_steps % 200:
                action[:] = np.random.randint(0, 2, size=action.shape)
            state, rew, done, info = env.step(action)
            max_vel_episode = np.maximum(max_vel_episode, env.lce_dot_1)
            max_vel_episode = np.maximum(max_vel_episode, env.lce_dot_2)
            ep_steps += 1

            if done:
                break
        # Calculate new VMAX
        delta = max_vel_episode - max_vel
        max_vel = max_vel + lr * delta
        vmaxes[ep, :] = max_vel
        # Update learning rate
        if ep % decay_lr_every == 0:
            lr = lr * lr_decay
            norm_of_delta_over_lr = np.linalg.norm(vmaxes[ep, :] - vmaxes[ep - decay_lr_every], ord=2)
            print("{} episodes elapsed, updating lr to {:.6g}".format(ep, lr))
            print("Norm of difference for VMAX since last: {:.6g}".format(norm_of_delta_over_lr))

    np.save(os.path.join(save_dir, "vmax.npy"), max_vel)
    np.save("vmax.npy", max_vel)
    # Make plot of vmax over time
    if make_plots:
        actuator_ids = env.mimo_actuators
        actuator_names = [env.sim.model.actuator_id2name(act_id) for act_id in actuator_ids]
        print("Creating vmax plots")
        # VMAX plot
        fig, axs = plt.subplots(len(actuator_names), 1, figsize=(10, 90))
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
        fig, axs = plt.subplots(len(actuator_names), 1, figsize=(10, 90))
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

    We take the desired maximum force values from the actuator definitions in the scene XML.
    We then apply maximum control input in one direction for 500 steps, back off for 500 steps, and then maximum input
    in the opposite direction for 500 steps.
    The maximum torque actually generated during this is recorded for each direction and compared against the desired
    values. The FMAX parameter is then adjusted such that the generated torque matches the desired.
    This is performed iteratively as all MuJoCo constraints are soft and even locked joints will change position
    slightly based on applied torque.

    This method requires a specialised scene to measure maximum voluntary isometric muscle torque in which all joints
    are locked in the angle at which the torque is to be measured.
    Note also that if the initial FMAX (set in MIMoMuscleEnv) is too large the motors may overcome the joint locking
    entirely, leading to NaNs and associated errors. In this case adjust the initial FMAX downwards.

    Args:
        env_name: The name of the environment to be used for the calibration.
        save_dir: The directory where the final VMAX and any plots will be saved.
        n_iterations: How many iterations of the calibration to perform. Default 5.
        make_plots: If `True` we plot muscle parameters during the last iteration. Default `True`.

    Returns:
        A numpy array with the final FMAX values.
    """
    os.makedirs(save_dir, exist_ok=True)
    env = gym.make(env_name)
    target_torque = np.concatenate([env.maximum_isometric_forces[:, 0], env.maximum_isometric_forces[:, 1]])
    fmax = env.fmax
    if not isinstance(fmax, np.ndarray):
        fmax = np.ones_like(target_torque) * fmax

    actuator_ids = env.mimo_actuators
    actuator_names = [env.sim.model.actuator_id2name(act_id) for act_id in actuator_ids]
    muscle_data = []

    # Perform iteration
    for ep in range(n_iterations):
        # Setup for this iteration
        print("FMAX iteration {} of {}".format(ep + 1, n_iterations))
        _ = env.reset()
        env.fmax = fmax.copy()
        fmax_old = fmax.copy()
        n_actuators = env.n_actuators
        ep_steps = 0
        max_unscaled_torque = np.zeros(env.action_space.shape)

        # 500 steps with maximum tension of muscles acting in negative direction
        action = np.zeros(env.action_space.shape)
        action[:n_actuators] = 1.0
        for j in range(500):
            _ = env.step(action)
            ep_steps += 1
            if make_plots and ep == n_iterations - 1:
                muscle_data.append(env.collect_data_for_actuators())
            # Once we have stabilized, start collecting MVF
            if j > 250:
                torque_1 = env.moment_1 * env.force_muscles_1
                torque_2 = - env.moment_2 * env.force_muscles_2
                unscaled_torque = np.concatenate([torque_1, torque_2])
                max_unscaled_torque = np.maximum(max_unscaled_torque, unscaled_torque)

        # 500 steps with no tension at all
        action = np.zeros(env.action_space.shape)
        for _ in range(500):
            _ = env.step(action)
            ep_steps += 1
            if make_plots and ep == n_iterations - 1:
                muscle_data.append(env.collect_data_for_actuators())

        # 500 steps with maximum tension of muscles acting in positive direction
        action = np.zeros(env.action_space.shape)
        action[n_actuators:] = 1.0
        for j in range(500):
            _ = env.step(action)
            ep_steps += 1
            if make_plots and ep == n_iterations - 1:
                muscle_data.append(env.collect_data_for_actuators())
            if j > 250:
                torque_1 = env.moment_1 * env.force_muscles_1
                torque_2 = - env.moment_2 * env.force_muscles_2
                unscaled_torque = np.concatenate([torque_1, torque_2])
                max_unscaled_torque = np.maximum(max_unscaled_torque, unscaled_torque)

        fmax = target_torque / (max_unscaled_torque + EPS)
        print("Norm of difference for fmax: {:.6g}".format(np.linalg.norm(fmax - fmax_old, ord=2)))

    np.save(os.path.join(save_dir, "fmax.npy"), fmax)

    # Plot the data
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

    This function is designed to be used with :meth:`~mimoEnv.envs.mimo_muscle_env.MIMoMuscleEnv.collect_data_for_actuator`.
    `data` should be a dictionary with the data for each actuator saved as an array with the actuator name as the
    dictionary key. The structure of the array should have steps or time as the first dimension and the different
    return values as the second.

    Args:
        plot_dir: The directory where the plots will be saved.
        data: A dictionary containing the actuator data.
        dt: Optional. The time between data points. If provided the x-axis will be time instead of number of data points.

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
        fig.tight_layout()
        fig.savefig(os.path.join(plot_dir, actuator_name + ".png"))
        fig.clear()
        plt.close(fig)


def plotting_episode(save_dir):
    os.makedirs(save_dir, exist_ok=True)

    print("Collecting data for plots")
    env = gym.make("MIMoVelocityMuscleTest-v0")
    max_vel = env.vmax
    np.save(os.path.join(save_dir, 'vmax.npy'), max_vel)
    actuator_ids = env.mimo_actuators
    actuator_names = [env.sim.model.actuator_id2name(act_id) for act_id in actuator_ids]

    muscle_props = []
    muscle_data = []

    _ = env.reset()
    ep_steps = 0
    action = np.zeros(env.action_space.shape)
    while True:
        if not ep_steps % 200:
            action[:] = np.random.randint(0, 2, size=action.shape)

        state, rew, done, info = env.step(action)
        muscle_data.append(env.collect_data_for_actuators())
        ep_steps += 1
        muscle_props.append([env.lce_1.copy(),
                             env.lce_2.copy(),
                             env.lce_dot_1.copy(),
                             env.lce_dot_2.copy(),
                             env.force_muscles_1.copy(),
                             env.force_muscles_2.copy()])

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
    N = 90
    fig, axs = plt.subplots(90, 3, figsize=(10, 90))
    for midx in range(N):
        f = FL
        lengthes = np.linspace(0.5, 1.2, 100)
        axs[midx, 0].plot(lengthes, f(lengthes), color='tab:blue')
        axs[midx, 0].plot([x[midx] for x in lengths_1], [f(x[midx]) for x in lengths_1], 'x', color='tab:red')
        axs[midx, 0].set_xlim([0.6, 1.2])
        f = FV_vec
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


def recording_episode(env_name: str, video_dir: str, env_params={}, video_width=500, video_height=500,
                      camera_name=None, make_joint_plots=True, binary_actions=False, interactive=False):
    os.makedirs(video_dir, exist_ok=True)
    env = gym.make(env_name, **env_params)
    _ = env.reset()
    ep_steps = 0
    action = np.zeros(env.action_space.shape)
    images = []
    muscle_data = []
    actuator_ids = env.mimo_actuators
    actuator_names = [env.sim.model.actuator_id2name(act_id) for act_id in actuator_ids]
    while True:
        if ep_steps % 200 == 0:
            if binary_actions:
                action[:] = np.random.randint(0, 2, size=action.shape)
            else:
                action[:] = env.action_space.sample()
        ep_steps += 1
        obs, _, done, _ = env.step(action)
        img = env.render(mode="rgb_array", width=video_width, height=video_height, camera_name=camera_name)
        images.append(img)
        if make_joint_plots:
            muscle_data.append(env.collect_data_for_actuators())
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


# We perform a static test, locking all joints to determine maximum voluntary muscle force. The FMAX
# parameter is then adjusted to bring the voluntary muscle force in line with literature from biology/medicine.
# This is done by applying maximum control input for one set of muscles until muscle activation states have converged
# and measuring the resuling output torque before moving to the next set of muscles

# Then we iteratively determine a stable vmax, by starting the environment with some fixed vmax, then taking random
# actions to measure new values. These are then fed back into the environment and more actions are taken.
# We have n_episodes_random episodes before feeding back vmax, and we feed back n_vmax_iterations times before moving
# to the static test


if __name__ == "__main__":
    n_iterations_fmax = 3
    n_iterations_vmax = 20
    n_episodes_lr = 20
    lr = 0.1
    lr_decay = 0.8
    vmax_total_episodes = n_iterations_vmax * n_episodes_lr
    plotting_dir = "temp_test"

    # fmax adjusting
    fmax_calibration("MIMoMuscleStaticTest-v0",
                     os.path.join(plotting_dir, "fmax"),
                     n_iterations=n_iterations_fmax,
                     make_plots=True)
    vmax_calibration("MIMoVelocityMuscleTest-v0",
                     vmax_total_episodes,
                     os.path.join(plotting_dir, "vmax"),
                     lr=lr,
                     lr_decay=lr_decay,
                     decay_lr_every=n_episodes_lr,
                     make_plots=True)
    # Scale vmax by 2 (manual tweak)
    #np.save('vmax.npy', vmax * 2)
    plotting_episode(os.path.join(plotting_dir, "vmax"))
    recording_env_params = {
        "touch_params": None,
        "vision_params": None,
        "print_space_sizes": None,
    }
    recording_episode("MIMoMuscle-v0", os.path.join(plotting_dir, "final"), env_params=recording_env_params)
