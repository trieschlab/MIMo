import gym
import mimoEnv
from matplotlib import pyplot as plt
import numpy as np

LMAX = 1.6
LMIN = 0.5
FVMAX = 1.2
# TODO atm VMAX was just measured from random movements, will be adapted for mimo
VMAX = 0.1
FPMAX = 1.3
FMAX = 50

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
def FV(lce_dot):
    """
    Force velocity
    """
    c = FVMAX - 1
    return force_vel(lce_dot, c, VMAX, FVMAX)


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

N_episodes = 10
env = gym.make("MIMoMuscle-v0")

muscle_props = []
for ep in range(N_episodes):
    max_vel = np.zeros_like(env.lce_dot_1)
    min_vel = np.ones_like(env.lce_dot_1) * 1000
    try:
        max_vel = np.load("vmax.npy")
    except:
        pass
    print(f'Episode {ep} of {N_episodes}')
    env.vmax = max_vel
    state = env.reset()
    ep_steps = 0
    action = env.action_space.sample()
    while True:
        if not ep_steps % 50:
            action[:] = np.random.randint(0, 2, size=action.shape)
            print(f'{ep_steps=}')
            #action[:] = 0.5
            #action[0] = 1
        state, rew, done, info = env.step(action)
        max_vel = np.maximum(max_vel, env.lce_dot_1)
        max_vel = np.maximum(max_vel, env.lce_dot_2)
        min_vel = np.minimum(min_vel, env.lce_dot_1)
        min_vel = np.minimum(min_vel, env.lce_dot_2)
        env.render()
        ep_steps += 1
        muscle_props.append([env.lce_1.copy(),
                             env.lce_2.copy(),
                             env.lce_dot_1.copy(),
                             env.lce_dot_2.copy(),
                             env.force_muscles_1.copy(),
                             env.force_muscles_2.copy()])

        if done or ep_steps >= 1000:
            break
    np.save('vmax.npy', max_vel)
    print(max_vel)
lengths_1 = [x[0] for x in muscle_props]
lengths_2 = [x[1] for x in muscle_props]
vels_1 = [x[2] for x in muscle_props]
vels_2 = [x[3] for x in muscle_props]
forces_1 = [x[4] for x in muscle_props]
forces_2 = [x[5] for x in muscle_props]
N = 90
fig, axs = plt.subplots(90, 3, figsize=(20, 80))
for midx in range(N):
    f = FL
    lengthes = np.linspace(0.5, 1.2, 100)
    axs[midx, 0 ].plot(lengthes, f(lengthes), color='tab:blue')
    axs[midx, 0].plot([x[midx] for x in lengths_1], [f(x[midx]) for x in lengths_1], 'x', color='tab:red')
    #axs[midx, 0].plot([x[midx] for x in lengths_2], [f(x[midx]) for x in lengths_2])
    axs[midx, 0].set_xlim([0.6, 1.2])
    f = FV_vec
    vels = np.linspace(-max_vel, max_vel, 100)
    axs[midx, 1].plot(vels[:, midx], f(vels[:, midx], max_vel[midx]), color='tab:blue')
    f = FV_vec
    axs[midx, 1].plot([x[midx] for x in vels_1], [f(x[midx], max_vel[midx]) for x in vels_1], 'x', color='tab:red')
    axs[midx, 1].set_xlim([-max_vel[midx], max_vel[midx]])
    #axs[midx, 1].plot([x[midx] for x in vels_2], [f(x[midx]) for x in vels_2])
    axs[midx, 2].plot([x[midx] for x in forces_1], 'x', color='tab:red')
plt.savefig('mimoflfvplots.png')



