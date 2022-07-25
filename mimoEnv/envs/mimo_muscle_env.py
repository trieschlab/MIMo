""" This module defines the base MIMo environment.

The abstract base class is :class:`~mimoEnv.envs.mimo_env.MIMoEnv`. Default parameters for all the sensory modalities
are provided as well.
"""
import numpy as np

from gym import spaces

from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as mimo_utils


# MuJoCo internal parameters that are used to compute muscle properties (FL, FV, FP, curves)
# These would be found in a GENERAL actuator
LMAX = 1.6
LMIN = 0.5
FVMAX = 1.2
# TODO atm VMAX was just measured from random movements, will be adapted for mimo
VMAX = 5.0
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


class MIMoMuscleEnv(MIMoEnv):
    """ This is the abstract base class for all muscle mimo.
    """

    def __init__(self,
                 model_path,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=None,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=None,
                 goals_in_observation=True,
                 done_active=False):

        # user parameters
        # TODO this is tuned by hand for HalfCheetah, adapt for mimo
        # TODO Implement asymmetric forces for single motor
        # TODO Ask about qpos and qvel in muscles.py: self.sim.data.qpos[: self.sim.model.nu] does not seem to make sense?
        self.phi_min = -1.5
        self.phi_max = 1.5
        self.lce_min = 0.75
        self.lce_max = 1.05

        self.lce_1_ref = None
        self.lce_2_ref = None

        self.moment_1 = None
        self.moment_2 = None
        self.lce_1 = None
        self.lce_2 = None
        self.lce_dot_1 = None
        self.lce_dot_2 = None
        self.activity = None
        self.prev_action = None
        self.force_muscles_1 = None
        self.force_muscles_2 = None

        self.maximum_isometric_forces = None

        self.mimo_actuated_joints = None
        self.mimo_actuated_qpos = None
        self.mimo_actuated_qvel = None

        super().__init__(model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)

    def _env_setup(self, initial_qpos):
        super()._env_setup(initial_qpos)
        # Also perform all the muscle setup
        self.maximum_isometric_forces = self._set_max_forces()
        self._get_actuated_joints()
        self._set_muscles()

    def _get_actuated_joints(self):
        self.mimo_actuated_joints = self.sim.model.actuator_trnid[self.mimo_actuators, 0]
        actuated_qpos_idx = [mimo_utils.get_joint_qpos_addr(self.sim.model, idx) for idx in self.mimo_actuated_joints]
        actuated_qvel_idx = [mimo_utils.get_joint_qvel_addr(self.sim.model, idx) for idx in self.mimo_actuated_joints]
        self.mimo_actuated_qpos = np.asarray(actuated_qpos_idx, dtype=np.int32)
        self.mimo_actuated_qvel = np.asarray(actuated_qvel_idx, dtype=np.int32)

    def _set_muscles(self):
        """
        Activate muscles for this environment. Attention, this reset the sim state.
        So far we assume that there is 1 action per moveable joint and that all of them
        will be replaced by muscles. This also multiplies the action space by 2,
        because we have 2 antagonistic muscles per joint.
        """
        self._compute_parametrization()
        #self.unwrapped.do_simulation = self.do_simulation
        self._set_action_space()
        self._set_initial_muscle_state()

    def _set_max_forces(self):
        """
        Collect maximum isometric forces from mujoco actuator gears.
        """
        force_ranges = self.sim.model.actuator_forcerange[self.mimo_actuators, :].copy()
        return self.sim.model.actuator_gear[self.mimo_actuators, 0].copy()

    def _compute_parametrization(self):
        """
        Compute parameters for muscles from angle and muscle fiber length ranges.
        """
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

    def _set_action_space(self):
        self.action_space = spaces.Box(
            low=-0.0, high=1.0, shape=(self.n_actuators * 2,), dtype=np.float32
        )

    def _set_initial_muscle_state(self):
        """
        Activity needs to be twice the number of actuated joints.
        Only works with action space shape if we have adjusted it beforehand.
        """
        self.activity = np.zeros(shape=self.action_space.shape)
        self.prev_action = np.zeros(self.action_space.shape)
        self._update_muscle_state()

    def _update_muscle_state(self):
        self._update_activity()
        self._update_virtual_lengths()
        self._update_virtual_velocities()
        self._update_torque()

    def _update_activity(self):
        """
        Very simple low-pass filter, even simpler than MuJoCo internal, update in the future.
        The time scale parameter is hard-coded so far to 100. which corresponds to tau=0.01.
        """
        self.activity += 1 * self.dt * (self.prev_action - self.activity)
        self.activity = np.clip(self.activity, 0, 1)

    def _update_virtual_lengths(self):
        """
        Update the muscle lengths from current joint angles.
        """
        self.lce_1 = self.sim.data.qpos[self.mimo_actuated_qpos].flatten() * self.moment_1 + self.lce_1_ref
        self.lce_2 = self.sim.data.qpos[self.mimo_actuated_qpos].flatten() * self.moment_2 + self.lce_2_ref

    def _update_virtual_velocities(self):
        """
        Update the muscle lengths from current joint angle velocities.
        """
        self.lce_dot_1 = self.moment_1 * self.sim.data.qvel[self.mimo_actuated_qvel].flatten()
        self.lce_dot_2 = self.moment_2 * self.sim.data.qvel[self.mimo_actuated_qvel].flatten()

    def _update_torque(self):
        """
        Torque times maximum isometric force wouldnt normally result in a torque, but in the one-dimensional
        scalar case there is no difference. (I.e. they are commutative)
        The minus sign at the end is a MuJoCo convention. A positive force multiplied by a positive moment results then in a NEGATIVE torque,
        (as muscles pull, they dont push) and the joint velocity gives us the correct muscle fiber velocities.
        """
        self.force_muscles_1 = FL(self.lce_1) * FV(self.lce_dot_1) * self.activity[:self.n_actuators] + FP(self.lce_1)
        self.force_muscles_2 = FL(self.lce_2) * FV(self.lce_dot_2) * self.activity[self.n_actuators:] + FP(self.lce_2)
        torque = self.moment_1 * self.force_muscles_1 + self.moment_2 * self.force_muscles_2
        self.joint_torque = -torque * self.maximum_isometric_forces

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

    @property
    def muscle_forces(self):
        return np.concatenate([self.force_muscles_1, self.force_muscles_2], dtype=np.float32).copy()

    @property
    def muscle_velocities(self):
        return np.concatenate([self.lce_dot_1, self.lce_dot_2], dtype=np.float32).copy()

    @property
    def muscle_activations(self):
        return self.activity.copy()

    @property
    def muscle_lengths(self):
        return np.concatenate([self.lce_1, self.lce_2], dtype=np.float32).copy()

    def do_simulation(self, action, n_frames):
        """
        Overwrite do_simulation because we need to be able to call the muscle model
        at each physical time step, not only when the policy is called.
        """
        self._set_action(action)
        for i in range(n_frames):
            self._compute_muscle_action(update_action=False)
            self.sim.step()

    def _set_action(self, action):
        """ Set the control inputs for the next step.

        Control values are clipped to the control range limits defined the MuJoCo xmls and normalized to be even in
        both directions, i.e. an input of 0 corresponds to the center of the control range, rather than the default or
        neutral control position. The control ranges for the MIMo xmls are set up to be symmetrical, such that an input
        of 0 corresponds to no motor torque.

        Args:
            action (numpy.ndarray): A numpy array with control values.
        """
        action = self._compute_muscle_action(action)
        self.sim.data.ctrl[self.mimo_actuators] = action

    def _get_obs(self):
        obs = super()._get_obs()
        # Add muscle state

        obs["observation"] = np.concatenate(
            [
                obs["observation"],
                self.muscle_lengths.flatten(),
                self.muscle_velocities.flatten(),
                self.muscle_activations,
                self.muscle_forces,
            ],
            dtype=np.float32,
        )

        return obs
