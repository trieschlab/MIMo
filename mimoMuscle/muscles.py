import gym
import numpy as np
from matplotlib import pyplot as plt
import mimoEnv
import time


# TODO Overall: tune parameters (min-max angles and lengths, velocities, forces
# make sure actuated joints are all sequentially in qpos to call data.qpos[:model.nu] for MIMo
# Alternatively: Call a mujoco function that gives you actuated joints directly
# Transform observations such that muscle information gets added correctly to MIMo


# MuJoCo internal parameters that are used to compute muscle properties (FL, FV, FP, curves)
LMAX = 1.6
LMIN = 0.5
FVMAX = 1.2
# TODO atm VMAX was just measured from random movements, will be adapted for mimo
VMAX = 5.0
FPMAX = 1.3
FMAX = 50


def vectorized(fn):
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
    # return 1.0
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


class MuscleWrapper(gym.Wrapper):
    """
    Temporary wrapper class that shows how muscles can be activated in an environment.
    We assume that all actuated joints come first in data.qpos such that we can call
    data.qpos[:model.nu] to get them. There are more complex mujoco functions we can
    call if that doesn't work out.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._set_muscles()

    def step(self, action):
        action = self._compute_muscle_action(action)
        state, reward, done, info = self.env.step(action)
        return self._add_muscle_state(state), reward, done, info

    def reset(self, *args, **kwargs):
        state = super().reset(*args, **kwargs)
        return self._add_muscle_state(state)

    def _set_muscles(self):
        """
        Activate muscles for this environment. Attention, this reset the sim state.
        So far we assume that there is 1 action per moveable joint and that all of them
        will be replaced by muscles. This also multiplies the action space by 2,
        because we have 2 antagonistic muscles per joint.
        """
        self._compute_parametrization()
        self._set_max_forces()
        self.unwrapped.do_simulation = self.do_simulation
        self._set_action_space()
        self._set_initial_muscle_state()
        self._set_observation_space()

    def _set_max_forces(self):
        self.maximum_isometric_forces = self.sim.model.actuator_gear[:, 0].copy()
        self.sim.model.actuator_gear[:, 0] = 1.0

    def _set_initial_muscle_state(self):
        """
        Activity needs to be twice the number of actuated joints.
        Only works with action space shape if we have adjusted it beforehand.
        """
        self.activity = np.zeros(shape=self.action_space.shape)
        self.prev_action = np.zeros(self.action_space.shape)
        self._update_muscle_state()

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for i in range(n_frames):
            self._compute_muscle_action(update_action=False)
            # print(self.muscle_activations)
            self.sim.step()

    def _add_muscle_state(self, state):
        """
        Add muscle specific quantities to the state
        :param state: Original muscle-free state
        :return: Updated observation vector
        """
        return np.concatenate(
            [
                state,
                self.muscle_lengths,
                self.muscle_velocities,
                self.muscle_activations,
                self.muscle_forces,
            ],
            dtype=np.float32,
        ).copy()

    def _update_muscle_state(self):
        self._update_activity()
        self._update_virtual_lengths()
        self._update_virtual_velocities()
        self._update_torque()

    def _compute_muscle_action(self, action=None, update_action=True):
        """
        Take in the muscle action, compute all virtual quantities and return the correct torques.
        """
        assert not (update_action and action is None)
        if update_action:
            self.prev_action = np.clip(action, 0, 1).copy()
        self._update_muscle_state()
        self._apply_torque()
        return np.ones_like(self.joint_torque)

    def _apply_torque(self):
        """
        Slightly hacky force application:
        We adjust the gear (which is just a scalar force multiplier) to be equal to the torque we want to apply,
        then we output an action-vector that is 1 everywhere. With this, we don't have to change anything
        inside the environment.step(action) function.
        # TODO scale appropriately to equal max isometric forces, this FMAX value was taken randomly
        # TODO could basically just divide somewhere by moment again to get FMAX back as maximumisometricforce
        """
        self.sim.model.actuator_gear[:, 0] = self.joint_torque.copy() * FMAX

    def _update_torque(self):
        """
        Torque times maximum isometric force wouldnt normally result in a torque, but in the one-dimensional
        scalar case there is no difference. (I.e. they are commutative)
        The minus sign at the end is a MuJoCo convention. A positive force multiplied by a positive moment results then in a NEGATIVE torque,
        (as muscles pull, they dont push) and the joint velocity gives us the correct muscle fiber velocities.
        """
        self.force_muscles_1 = FL(self.lce_1) * FV(self.lce_dot_1) * self.activity[
            : self.sim.model.nu
        ] + FP(self.lce_1)
        self.force_muscles_2 = FL(self.lce_2) * FV(self.lce_dot_2) * self.activity[
            self.sim.model.nu :
        ] + FP(self.lce_2)
        torque = (
            self.moment_1 * self.force_muscles_1 + self.moment_2 * self.force_muscles_2
        )
        self.joint_torque = -torque * self.maximum_isometric_forces

    def _update_activity(self):
        """
        Very simple low-pass filter, even simpler than MuJoCo internal, update in the future.
        The time scale parameter is hard-coded so far to 100. which corresponds to tau=0.01.
        """
        self.activity += 1 * self.dt * (self.prev_action - self.activity)
        self.activity = np.clip(self.activity, 0, 1)

    def _compute_parametrization(self):
        """
        Compute parameters for muscles from angle and muscle fiber length ranges.
        """
        # user parameters
        # TODO this is tuned by hand for HalfCheetah, adapt for mimo
        self.phi_min = -1.5
        self.phi_max = 1.5
        self.lce_min = 0.75
        self.lce_max = 1.05
        # compute remaining parameters
        eps = 0.001
        self.moment_1 = (self.lce_max - self.lce_min + eps) / (
            self.phi_max - self.phi_min + eps
        )
        self.lce_1_ref = self.lce_min - self.moment_1 * self.phi_min
        self.moment_2 = (self.lce_max - self.lce_min + eps) / (
            self.phi_min - self.phi_max + eps
        )
        self.lce_2_ref = self.lce_min - self.moment_2 * self.phi_max

    def _update_virtual_lengths(self):
        """
        Update the muscle lengths from current joint angles.
        """
        self.lce_1 = (
            self.sim.data.qpos[: self.sim.model.nu] * self.moment_1 + self.lce_1_ref
        )
        self.lce_2 = (
            self.sim.data.qpos[: self.sim.model.nu] * self.moment_2 + self.lce_2_ref
        )

    def _update_virtual_velocities(self):
        """
        Update the muscle lengths from current joint angle velocities.
        """
        self.lce_dot_1 = self.moment_1 * self.sim.data.qvel[: self.sim.model.nu]
        self.lce_dot_2 = self.moment_2 * self.sim.data.qvel[: self.sim.model.nu]

    def _set_action_space(self):
        """
        Action space doubles in size because we have 2 muscles per actuated joint.
        """
        assert not type(self.action_space) == gym.spaces.Discrete
        previous_shape = self.action_space.shape[0]
        self.action_space = gym.spaces.Box(
            low=-0.0, high=1.0, shape=(previous_shape * 2,), dtype=np.float32
        )

    def _set_observation_space(self):
        obs = self.reset()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=obs.shape)

    @property
    def muscle_forces(self):
        return np.concatenate(
            [self.force_muscles_1, self.force_muscles_2], dtype=np.float32
        ).copy()

    @property
    def muscle_velocities(self):
        return np.concatenate([self.lce_dot_1, self.lce_dot_2], dtype=np.float32).copy()

    @property
    def muscle_activations(self):
        return self.activity.copy()

    @property
    def muscle_lengths(self):
        return np.concatenate([self.lce_1, self.lce_2], dtype=np.float32).copy()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip


def run_env(env, render=False):
    for _ in range(1):
        env.reset()
        ep_steps = 0
        while True:
            if not ep_steps % 1:
                action = env.action_space.sample()
            next_state, reward, done, info = env.step(action)
            ep_steps += 1
            if render:
                env.render()
            if done or ep_steps > 1000:
                break


if __name__== '__main__':
    ENV = "HalfCheetah-v3"
    # Not tuned well, goes crazy
    #ENV = "HumanoidStandup-v2"
    env = gym.make(ENV)
    print('Before wrapper')
    print('Action space:')
    print(env.action_space.shape)
    print('Observation space:')
    print(env.observation_space.shape)
    obs = env.reset()
    print(f'Real observation shape: {obs.shape=}')
    env = MuscleWrapper(env)
    print('After wrapper')
    print('Action space:')
    print(env.action_space.shape)
    print('Observation space:')
    print(env.observation_space.shape)
    obs = env.reset()
    print(f'Real observation shape: {obs.shape=}')
    run_env(env, render=False)

