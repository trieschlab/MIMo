import gym
import os
import mimoEnv
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

matplotlib.use("Agg")

LMAX = 1.6
LMIN = 0.5
FVMAX = 1.2
# TODO atm VMAX was just measured from random movements, will be adapted for mimo
FPMAX = 1.3


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


def vmax_iteration(n_episodes):
    env = gym.make("MIMoVelocityMuscleTest-v0")
    max_vel = np.zeros_like(env.lce_dot_1)
    min_vel = np.ones_like(env.lce_dot_1) * 1000
    for ep in range(n_episodes):
        _ = env.reset()
        ep_steps = 0
        action = np.zeros(env.action_space.shape)
        while True:
            if not ep_steps % 200:
                action[:] = np.random.randint(0, 2, size=action.shape)
            state, rew, done, info = env.step(action)
            max_vel = np.maximum(max_vel, env.lce_dot_1)
            max_vel = np.maximum(max_vel, env.lce_dot_2)
            min_vel = np.minimum(min_vel, env.lce_dot_1)
            min_vel = np.minimum(min_vel, env.lce_dot_2)
            # env.render()
            ep_steps += 1

            if done or ep_steps >= max_steps:
                break

    np.save('vmax.npy', max_vel)
    env.close()
    return max_vel


def plotting_episode(plot_dir):
    os.makedirs(plot_dir, exist_ok=True)

    print("Collecting data for plots")
    env = gym.make("MIMoVelocityMuscleTest-v0")
    max_vel = env.vmax
    np.save(os.path.join(plot_dir, 'vmax.npy'), max_vel)
    actuator_ids = env.mimo_actuators
    actuator_names = [env.sim.model.actuator_id2name(act_id) for act_id in actuator_ids]

    muscle_props = []
    muscle_data = {}
    for actuator_name in actuator_names:
        muscle_data[actuator_name] = []

    _ = env.reset()
    ep_steps = 0
    action = np.zeros(env.action_space.shape)
    while True:
        if not ep_steps % 200:
            action[:] = np.random.randint(0, 2, size=action.shape)

        state, rew, done, info = env.step(action)
        for i, actuator_name in enumerate(actuator_names):
            muscle_data[actuator_name].append(env.collect_data_for_actuator(actuator_ids[i]))
        ep_steps += 1

        muscle_props.append([env.lce_1.copy(),
                             env.lce_2.copy(),
                             env.lce_dot_1.copy(),
                             env.lce_dot_2.copy(),
                             env.force_muscles_1.copy(),
                             env.force_muscles_2.copy()])

        if done or ep_steps >= max_steps:
            break

    x = np.arange(ep_steps) * env.dt
    # Plot the data
    print("Creating plots")
    for actuator_name in muscle_data:
        data = np.asarray(muscle_data[actuator_name], dtype=np.float32)
        fig, axs = plt.subplots(10, 2, figsize=(10, 18))
        for i, ax in enumerate(axs.flat):
            ax.plot(x, data[:, i])
            ax.set_xlim([0, ep_steps * env.dt])
            ax.hlines(y=0.0, xmin=0, xmax=ep_steps * env.dt, colors="tab:grey")
        actuator_name = actuator_name.replace("act:", "")
        axs[0, 0].set_title("qpos")
        axs[0, 1].set_title("qvel")
        axs[1, 0].set_title("gear")
        axs[1, 1].set_title("torque")
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

    lengths_1 = [x[0] for x in muscle_props]
    lengths_2 = [x[1] for x in muscle_props]
    vels_1 = [x[2] for x in muscle_props]
    vels_2 = [x[3] for x in muscle_props]
    forces_1 = [x[4] for x in muscle_props]
    forces_2 = [x[5] for x in muscle_props]
    N = 90
    fig, axs = plt.subplots(90, 3, figsize=(10, 90))
    for midx in range(N):
        f = FL
        lengthes = np.linspace(0.5, 1.2, 100)
        axs[midx, 0].plot(lengthes, f(lengthes), color='tab:blue')
        axs[midx, 0].plot([x[midx] for x in lengths_1], [f(x[midx]) for x in lengths_1], 'x', color='tab:red')
        # axs[midx, 0].plot([x[midx] for x in lengths_2], [f(x[midx]) for x in lengths_2])
        axs[midx, 0].set_xlim([0.6, 1.2])
        f = FV_vec
        vels = np.linspace(-max_vel, max_vel, 100)
        axs[midx, 1].plot(vels[:, midx], f(vels[:, midx], max_vel[midx]), color='tab:blue')
        axs[midx, 1].plot([x[midx] for x in vels_1], [f(x[midx], max_vel[midx]) for x in vels_1], 'x', color='tab:red')
        axs[midx, 1].set_xlim([-max_vel[midx], max_vel[midx]])
        axs[midx, 2].plot([x[midx] for x in forces_1], 'x', color='tab:red')
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, "mimoflfvplots.png"))
    fig.clear()
    plt.close(fig)
    env.close()


def fmax_adjust(plot_dir):
    os.makedirs(plot_dir, exist_ok=True)
    env = gym.make("MIMoMuscleStaticTest-v0")
    # Max action on the first few
    _ = env.reset()
    n_actuators = env.n_actuators
    ep_steps = 0
    target_torque = np.concatenate([env.maximum_isometric_forces[:, 0], env.maximum_isometric_forces[:, 1]])
    max_unscaled_torque = np.zeros(env.action_space.shape)

    actuator_ids = env.mimo_actuators
    actuator_names = [env.sim.model.actuator_id2name(act_id) for act_id in actuator_ids]
    muscle_data = {}
    for actuator_name in actuator_names:
        muscle_data[actuator_name] = []

    # 500 steps with maximum tension of muscles acting in negative direction
    action = np.zeros(env.action_space.shape)
    action[:n_actuators] = 1.0
    for j in range(500):
        _ = env.step(action)
        ep_steps += 1
        for i, actuator_name in enumerate(actuator_names):
            muscle_data[actuator_name].append(env.collect_data_for_actuator(actuator_ids[i]))
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
        for i, actuator_name in enumerate(actuator_names):
            muscle_data[actuator_name].append(env.collect_data_for_actuator(actuator_ids[i]))

    # 500 steps with maximum tension of muscles acting in positive direction
    action = np.zeros(env.action_space.shape)
    action[n_actuators:] = 1.0
    for j in range(500):
        _ = env.step(action)
        ep_steps += 1
        for i, actuator_name in enumerate(actuator_names):
            muscle_data[actuator_name].append(env.collect_data_for_actuator(actuator_ids[i]))
        if j > 250:
            torque_1 = env.moment_1 * env.force_muscles_1
            torque_2 = - env.moment_2 * env.force_muscles_2
            unscaled_torque = np.concatenate([torque_1, torque_2])
            max_unscaled_torque = np.maximum(max_unscaled_torque, unscaled_torque)

    fmax = target_torque / max_unscaled_torque
    np.save("fmax.npy", fmax)
    np.save(os.path.join(plot_dir, "fmax.npy"), fmax)

    x = np.arange(ep_steps) * env.dt
    # Plot the data
    print("Creating plots")
    for actuator_name in muscle_data:
        data = np.asarray(muscle_data[actuator_name], dtype=np.float32)
        fig, axs = plt.subplots(10, 2, figsize=(10, 18))
        for i, ax in enumerate(axs.flat):
            ax.plot(x, data[:, i])
            ax.set_xlim([0, ep_steps * env.dt])
            ax.hlines(y=0.0, xmin=0, xmax=ep_steps * env.dt, colors="tab:grey")
        actuator_name = actuator_name.replace("act:", "")
        axs[0, 0].set_title("qpos")
        axs[0, 1].set_title("qvel")
        axs[1, 0].set_title("gear")
        axs[1, 1].set_title("torque")
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
    env.close()
    return fmax


# We iteratively determine a stable vmax, by starting the environment with some fixed vmax, then taking random actions
# to measure new values. These are then fed back into the environment and more actions are taken.
# We have n_episodes_random episodes before feeding back vmax, and we feed back n_vmax_iterations times before moving
# to the static test

# Then we perform a static test, locking all joints to determine maximum voluntary muscle force. The FMAX
# parameter is then adjusted to bring the voluntary muscle force in line with literature from biology/medicine.
# This is done by applying maximum control input for one set of muscles until muscle activation states have converged
# and measuring the resuling output torque before moving to the next set of muscles

n_iterations = 1
n_vmax_iterations = 1
n_episodes_random = 1
max_steps = 5000

if __name__ == "__main__":
    vmax = np.ones((90,)) * 0.1
    fmax = np.ones((180,)) * 0.1
    for i in range(n_iterations):
        print("iteration", i+1)
        for _ in range(n_vmax_iterations):
            vmax_new = vmax_iteration(n_episodes_random)
            print("Norm of difference for vmax", np.linalg.norm(vmax - vmax_new, ord=2))
            vmax = vmax_new
        print("vmax post iteration", vmax)
        plotting_episode("autoimgs/iteration_{}/vmax".format(i))

        fmax_new = fmax_adjust("autoimgs/iteration_{}/fmax".format(i))
        print("Norm of difference for fmax", np.linalg.norm(fmax - fmax_new, ord=2))
        fmax = fmax_new
        print("fmax post iteration", fmax)
    plotting_episode("autoimgs/final")
