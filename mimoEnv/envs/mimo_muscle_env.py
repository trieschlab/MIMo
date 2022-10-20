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
#VMAX = 0.1
FPMAX = 1.3
#FMAX = 50
FMAX = 50



def FL(lce):
    """
    Force length
    """
    return bump_v(lce, LMIN, 1, LMAX) + 0.15 * bump_v(lce, LMIN, 0.5 * (LMIN + 0.95), 0.95)


def FV(lce_dot, vmax):
    """
    Force velocity
    """
    c = FVMAX - 1
    return force_vel_v(lce_dot, c, vmax, FVMAX)


def FP(lce):
    """
    Force passive
    """
    b = 0.5 * (LMAX + 1)
    return passive_force_v(lce, b)


def bump_v(length, A, mid, B):
    """
    Force length relationship as implemented by MuJoCo.
    """
    left = 0.5 * (A + mid)
    right = 0.5 * (mid + B)

    a_dif = np.square(length - A) * 0.5
    b_dif = np.square(length - B) * 0.5
    mid_dif = np.square(length - mid) * 0.5

    output = b_dif / ((B-right) * (B-right))

    output[length < right] = 1 - mid_dif[length < right] / ((right - mid) * (right - mid))
    output[length < mid] = 1 - mid_dif[length < mid] / ((mid - left) * (mid - left))
    output[length < left] = a_dif[length < left] / ((left - A) * (left - A))
    output[(length <= A) | (length >= B)] = 0

    return output


def passive_force_v(length, b):
    """Parallel elasticity (passive muscle force) as implemented
    by MuJoCo.
    """
    tmp = (length[length <= b] - 1) / (b - 1)

    output = 0.25 * FPMAX * (1 + 3 * (length - b) / (b - 1))
    output[length <= b] = 0.25 * FPMAX * tmp * tmp * tmp
    output[length <= 1] = 0

    return output


def force_vel_v(velocity, c, vmax, fvmax):
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
        # user parameters ------------------------------
        self.lce_min = 0.75
        self.lce_max = 1.05
        self.vmax = np.load('../mimoMuscle/vmax.npy')
        #print(self.vmax)

        # Placeholders that gets overwritten later
        self.phi_min = -1
        self.phi_max = 1

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
        self.joint_torque = None

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

        self.phi_min = self.sim.model.jnt_range[self.mimo_actuated_joints, 0]
        self.phi_max = self.sim.model.jnt_range[self.mimo_actuated_joints, 1]
        self._set_muscles()

    def _env_setup(self, initial_qpos):
        super()._env_setup(initial_qpos)
        # Also perform all the muscle setup
        self._set_max_forces()
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
        self._set_action_space()
        self._set_initial_muscle_state()

    def _set_max_forces(self):
        """
        Collect maximum isometric forces from mujoco actuator gears.
        """
        force_ranges = np.abs(self.sim.model.actuator_forcerange[self.mimo_actuators, :]).copy()
        # Have to disable force limits afterwards
        self.sim.model.actuator_forcelimited[self.mimo_actuators] = np.zeros_like(self.sim.model.actuator_forcelimited[self.mimo_actuators])
        gears = self.sim.model.actuator_gear[self.mimo_actuators, 0].copy()
        self.maximum_isometric_forces = (force_ranges.T * gears).T

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
        self.force_muscles_1 = (FL(self.lce_1) * FV(self.lce_dot_1, self.vmax) * self.activity[:self.n_actuators] + FP(self.lce_1)) \
                               * self.maximum_isometric_forces[:, 0]
        self.force_muscles_2 = FL(self.lce_2) * FV(self.lce_dot_2, self.vmax) * self.activity[self.n_actuators:] + FP(self.lce_2) \
                               * self.maximum_isometric_forces[:, 1]
        torque = self.moment_1 * self.force_muscles_1 + self.moment_2 * self.force_muscles_2
        self.joint_torque = -torque

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
        # maximum isometric force is not enough because we have to appropriately scale the moment arms.
        """
        self.sim.model.actuator_gear[self.mimo_actuators, 0] = self.joint_torque.copy() * FMAX

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
            self._substep_callback()

    def _set_action(self, action):
        """ Set the control inputs for the next step.

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

    def reset(self):
        obs = super().reset()
        self._set_initial_muscle_state()
        return obs

    def collect_data_for_joint(self, joint_name):
        joint_id = self.sim.model.joint_name2id(joint_name)
        actuator_index = np.nonzero(self.mimo_actuators == joint_id)

        actuator_qpos = self.mimo_actuated_qpos[actuator_index]
        actuator_qvel = self.mimo_actuated_qvel[actuator_index]

        activity_neg = self.activity[actuator_index]
        activity_pos = self.activity[self.n_actuators + actuator_index]

        lce_neg = self.lce_1[actuator_index]
        lce_pos = self.lce_2[actuator_index]

        lce_neg_dot = self.lce_dot_1[actuator_index]
        lce_pos_dot = self.lce_dot_2[actuator_index]

        force_neg = self.force_muscles_1[actuator_index]
        force_pos = self.force_muscles_2[actuator_index]

        fl_neg = FL(self.lce_1)[actuator_index]
        fv_neg = FV(self.lce_dot_1, self.vmax)[actuator_index]
        fp_neg = FP(self.lce_1)[actuator_index]
        fl_pos = FL(self.lce_2)[actuator_index]
        fv_pos = FV(self.lce_dot_2, self.vmax)[actuator_index]
        fp_pos = FP(self.lce_2)[actuator_index]

        torque = self.joint_torque[actuator_index]
        actuator_gear = self.sim.model.actuator_gear[actuator_index, 0]
