""" This module defines the base class for the muscle actuation model.

Authors: Pierre Schumacher, Dominik Mattern
"""
import warnings
import numpy as np
from gymnasium import spaces

from mimoActuation.actuation import ActuationModel
import mimoEnv.utils as mimo_utils


class MuscleModel(ActuationModel):
    """ Class for the muscle actuation model.

    Implementation of the muscle model as seen in
    `https://arxiv.org/abs/2207.03952 <https://arxiv.org/abs/2207.03952>`_. Each actuator is internally modeled as two
    opposing muscles. These follow the force-length and force-velocity curves as described in the paper.
    Torque is applied in the simulation by setting the gear ratio to the computed output torque and applying a dummy
    control signal of 1.
    There are many parameters in this model, two of which were tweaked for MIMo specifically. The function used for
    this is :func:`~mimoActuation.muscle_testing.calibrate_full`.

    This model loads and modifies data from the actuators and joints the MIMo XML, which are effectively part of the
    specifications for this model. Changing them before this model is initialized might have unintended consequences
    for the actuation!

    Attributes:
        lmax (float): Determines the shape of the force-length curve.
        lmin (float): Determines the shape of the force-length curve.
        fvmax (float): The highest multiplier due to the force-velocity curve.
        fpmax (float): Multiplier for the passive force component.
        lce_min (float): Minimum virtual muscle length.
        lce_max (float): Maximum virtual muscle length.
        tau (float): Time constant for the activity. A higher tau means muscle activity takes longer to build up to the
            control signal.
        fmax (float|np.ndarray): Force multiplier to translate the normalised force-length and force-velocity curves
            into appropriate ranges.
        vmax (float|np.ndarray): Reference velocity for the force-velocity curve. A higher vmax leads to increased force
            at high virtual muscle velocities.
        target_activity (np.ndarray): Current control input. :attr:`.activity` will approach this value over time.
        activity (np.ndarray): Muscle activity.
    """
    def __init__(self, env, actuators):
        super().__init__(env, actuators)
        self.control_input = None
        self.lmax = 1.6
        self.lmin = 0.5
        self.fvmax = 1.2
        self.fpmax = 1.3
        self.lce_min = 0.75
        self.lce_max = 1.05
        self.tau = 0.01

        # Placeholders that gets overwritten later
        self.time_step = None
        self.vmax = None
        self.fmax = None
        self.phi_min = None
        self.phi_max = None

        self.lce_1_ref = None
        self.lce_2_ref = None

        self.moment_1 = None
        self.moment_2 = None
        self.lce_1 = None
        self.lce_2 = None
        self.lce_dot_1 = None
        self.lce_dot_2 = None
        self.activity = None
        self.target_activity = None
        self.force_muscles_1 = None
        self.force_muscles_2 = None
        self.joint_torque = None

        self.maximum_isometric_forces = None

        self.mimo_actuated_joints = None
        self.mimo_actuated_qpos = None
        self.mimo_actuated_qvel = None

        self._set_max_forces()
        self._get_actuated_joints()
        self._set_muscles()

    def get_action_space(self):
        """ Determines the actuation space attribute for the gym environment.

        The actuation space consists of two opposing muscles for each motor in the simulation, each with range [0, 1].

        Returns:
            spaces.Space: A gym spaces object with the actuation space.
        """
        action_space = spaces.Box(low=-0.0, high=1.0, shape=(self.n_actuators * 2,), dtype=np.float32)
        self.control_input = np.zeros(action_space.shape)  # Set initial control input to avoid NoneType Errors
        return action_space

    def action(self, action):
        """ Set the control inputs for the next step.

        Input values are clipped to the action space.

        Args:
            action (numpy.ndarray): A numpy array with control values.
        """
        self.control_input = np.clip(action, self.action_space.low, self.action_space.high)
        self.env.data.ctrl[self.actuators] = self._compute_muscle_action(self.control_input)

    def substep_update(self):
        """ Update muscle activity and torque.

        As activity is time-dependent we update activity and the output torque every physics step. The desired
        activity level (input action) is not changed during this."""
        self._compute_muscle_action(update_action=False)

    def observations(self):
        """ Returns muscle activations and forces for every actuator.

        Returns:
            np.ndarray: A flat array with the quantities described above.
        """
        return np.concatenate([self.muscle_activations.flatten(), self.muscle_forces.flatten() * self.fmax])

    def cost(self):
        """ Approximates the metabolic cost of muscle activations.

        Currently, it is given by :math:`\\sum_{i=1}^n \\frac{m_{a_i}^2 * f_{max_i}}{n \\sum_{i=1}^n f_{max_i}}`, where
        :math:`m_{a_i}` and :math:`f_{max_i}` are the activation and the maximum isometric muscle force of muscle
        :math:`i`, respectively, and :math:`n` is the number of muscles in the model.

        Returns:
            float: The actuation cost.
        """
        per_muscle_cost = self.activity * self.activity * self.fmax
        return np.abs(per_muscle_cost).sum() / (2 * self.n_actuators * self.fmax.sum())

    def reset(self):
        """ Set activity to zero and recompute muscle quantities. """
        self._set_initial_muscle_state()

    @property
    def muscle_activations(self):
        """ Activity for every muscle.

        Returns:
            np.ndarray: An array with copies of the activity for every muscle.
        """
        return self.activity.copy()

    @property
    def muscle_lengths(self):
        """ Virtual muscle lengths for all muscles.

        Returns:
            np.ndarray: An array with copies of the virtual muscle lengths.
        """
        return np.concatenate([self.lce_1, self.lce_2], dtype=np.float32).copy()

    @property
    def muscle_velocities(self):
        """ Virtual muscle speeds for all muscles.

        Returns:
            np.ndarray: An array with copies of the virtual muscle velocities."""
        return np.concatenate([self.lce_dot_1, self.lce_dot_2], dtype=np.float32).copy()

    @property
    def muscle_forces(self):
        """ Muscle force vectors.

        Returns:
            np.ndarray: An array with copies of the forces applied by each muscle."""
        return np.concatenate([self.force_muscles_1, self.force_muscles_2], dtype=np.float32).copy()

    def _set_max_forces(self):
        """ Collect maximum isometric forces from MuJoCo actuator gears.
        """
        force_ranges = np.abs(self.env.model.actuator_forcerange[self.actuators, :]).copy()
        gears = self.env.model.actuator_gear[self.actuators, 0].copy()
        self.maximum_isometric_forces = (force_ranges.T * gears).T
        # Have to disable force limits afterwards
        self.env.model.actuator_forcelimited[self.actuators] = \
            np.zeros_like(self.env.model.actuator_forcelimited[self.actuators])

    def _get_actuated_joints(self):
        """ Populates references to MIMo's joints and their indices in the qpos and qvel arrays.
        """
        self.mimo_actuated_joints = self.env.model.actuator_trnid[self.actuators, 0]
        actuated_qpos_idx = [mimo_utils.get_joint_qpos_addr(self.env.model, idx) for idx in
                             self.mimo_actuated_joints]
        actuated_qvel_idx = [mimo_utils.get_joint_qvel_addr(self.env.model, idx) for idx in
                             self.mimo_actuated_joints]
        self.mimo_actuated_qpos = np.asarray(actuated_qpos_idx, dtype=np.int32)
        self.mimo_actuated_qvel = np.asarray(actuated_qvel_idx, dtype=np.int32)

    def _set_muscles(self):
        """ Activate muscles for this environment.

        We replace all of MIMo's actuators, marked by an "act:" prefix, with actuators according to our muscle model.
        This adjusts muliple parameters of the simulation, such as joint damping.
        This also multiplies the action space by 2, as we have 2 antagonistic muscles per joint.

        Also resets the actuation state.
        """
        self._compute_parametrization()
        self._set_initial_muscle_state()

    def _compute_parametrization(self):
        """ Compute parameters for muscles from angle and muscle fiber length ranges.

        The muscle model converts joint positions and velocities into virtual muscle lengths and speeds. The parameters
        for this conversion are computed from fixed variables set
        """
        # Collect joint range from model, using springref as "neutral" position
        self.phi_min = self.env.model.jnt_range[self.mimo_actuated_joints, 0] - self.env.model.qpos_spring[
            self.mimo_actuated_qpos].flatten()
        self.phi_max = self.env.model.jnt_range[self.mimo_actuated_joints, 1] - self.env.model.qpos_spring[
            self.mimo_actuated_qpos].flatten()
        # Calculate values for muscle model
        self.moment_1 = (self.lce_max - self.lce_min + mimo_utils.EPS) / (
                self.phi_max - self.phi_min + mimo_utils.EPS
        )
        self.lce_1_ref = self.lce_min - self.moment_1 * self.phi_min
        self.moment_2 = (self.lce_max - self.lce_min + mimo_utils.EPS) / (
                self.phi_min - self.phi_max - mimo_utils.EPS
        )
        self.lce_2_ref = self.lce_min - self.moment_2 * self.phi_max
        # Adjust joint parameters: stiffness and damping
        self.env.model.jnt_stiffness[self.env.mimo_joints] = np.zeros_like(
            self.env.model.jnt_stiffness[self.env.mimo_joints])
        mimo_dof_ids = self.env.model.jnt_dofadr[self.env.mimo_joints]
        self.env.model.dof_damping[mimo_dof_ids] = self.env.model.dof_damping[mimo_dof_ids] / 20
        self._collect_muscle_parameters()

    def _collect_muscle_parameters(self):
        """ Loads the muscle model parameters from the XML.

        We store the FMAX and VMAX parameters in the user fields for the actuators. The first value is VMAX, the second
        value FMAX for the muscle acting in the negative joint direction, the third value FMAX for the positive joint
        direction.
        If there are no use user arguments, we print a warning and proceed with default values.
        If the values in the user arguments are invalid we raise ValueErrors.
        """
        if self.env.model.nuser_actuator >= 3:
            print("Reading Muscle parameters from XML")
            vmax = self.env.model.actuator_user[self.actuators, 0]
            fmax_neg = self.env.model.actuator_user[self.actuators, 1]
            fmax_pos = self.env.model.actuator_user[self.actuators, 2]
            self.vmax = vmax
            self.fmax = np.concatenate([fmax_neg, fmax_pos])
            if np.any(self.vmax < mimo_utils.EPS):
                raise ValueError("Illegal VMAX, VMAX values must be greater than 0!")
            if not np.isfinite(self.vmax).all():
                raise ValueError("NaN or Inf in VMAX!")
            if not np.isfinite(self.fmax).all():
                raise ValueError("NaN or Inf in FMAX!")
        else:
            warnings.warn("Muscle parameters missing from MIMo actuators! Trying calibration files...")
            try:
                self.vmax = np.load('../mimoActuation/vmax.npy')
            except FileNotFoundError:
                warnings.warn("No vmax calibration file, using default value 1!", RuntimeWarning)
                self.vmax = 1
            try:
                self.fmax = np.load("../mimoActuation/fmax.npy")
            except FileNotFoundError:
                warnings.warn("No fmax calibration file, using default value of 5!", RuntimeWarning)
                self.fmax = 5

    def _set_initial_muscle_state(self):
        """ Sets activity to zero and recomputes all muscle quantities.
        """
        self.activity = np.zeros(shape=self.action_space.shape)
        self.target_activity = np.zeros(self.action_space.shape)
        self._update_muscle_state()

    def _update_muscle_state(self):
        """ Computes the new muscle quantities needed to determine actuator torque.
        """
        self._update_activity()
        self._update_virtual_lengths()
        self._update_virtual_velocities()
        self._update_torque()

    def _update_activity(self):
        """ Updates the muscle activities for the current time step.
        """
        # Need sim timestep here rather than dt since we calculate this every physics step, while dt is the
        # duration of an environment step.
        self.activity += self.env.model.opt.timestep * (self.target_activity - self.activity) / self.tau
        self.activity = np.clip(self.activity, 0, 1)

    def _update_virtual_lengths(self):
        """ Update the muscle lengths from current joint angles.
        """
        self.lce_1 = (self.env.data.qpos[self.mimo_actuated_qpos].flatten() - self.env.model.qpos_spring[self.mimo_actuated_qpos].flatten()) * self.moment_1 + self.lce_1_ref
        self.lce_2 = (self.env.data.qpos[self.mimo_actuated_qpos].flatten() - self.env.model.qpos_spring[self.mimo_actuated_qpos].flatten()) * self.moment_2 + self.lce_2_ref

    def _update_virtual_velocities(self):
        """ Update the muscle velocities from current joint angle velocities.
        """
        self.lce_dot_1 = self.moment_1 * self.env.data.qvel[self.mimo_actuated_qvel].flatten()
        self.lce_dot_2 = self.moment_2 * self.env.data.qvel[self.mimo_actuated_qvel].flatten()

    def _update_torque(self):
        """ Compute muscle torques.

        The minus sign at the end is a MuJoCo convention. A positive force multiplied by a positive moment results then
        in a NEGATIVE torque, (as muscles pull, they don't push) and the joint velocity gives us the correct muscle
        fiber velocities.
        """
        self.force_muscles_1 = self.fl(self.lce_1) * self.fv(self.lce_dot_1) * self.activity[:self.n_actuators] \
                               + self.fp(self.lce_1)
        self.force_muscles_2 = self.fl(self.lce_2) * self.fv(self.lce_dot_2) * self.activity[self.n_actuators:] \
                               + self.fp(self.lce_2)
        if isinstance(self.fmax, np.ndarray):
            torque = self.moment_1 * self.force_muscles_1 * self.fmax[:self.n_actuators] \
                     + self.moment_2 * self.force_muscles_2 * self.fmax[self.n_actuators:]
        else:
            torque = self.moment_1 * self.force_muscles_1 * self.fmax + self.moment_2 * self.force_muscles_2 * self.fmax
        self.joint_torque = -torque

    def _apply_torque(self):
        """ Adjust MuJoCo values to actually apply muscle torque in the simulation.

        We adjust the MuJoCo actuator gear (which is just a scalar force multiplier) to be equal to the torque we want
        to apply, then we output an action-vector that is 1 everywhere. Doing this allows us to keep the same XMLs and
        gym interfaces for different actuation models.
        """
        self.env.model.actuator_gear[self.actuators, 0] = self.joint_torque.copy()

    def _compute_muscle_action(self, action=None, update_action=True):
        """ Take in the muscle action, compute all virtual quantities and set the correct torques.

        All-in-one function that updates muscle activity and computes muscle torques given current simulation state.

        Args:
            action: Either an array with control inputs or ``None``. If ``None`` we reuse the previous control input.
                Default ``None``. This argument is ignored if "update_action" is ``False``.
            update_action: If ``True`` we set :attr:`~.target_activity` to the `action` argument. If ``False`` the
                `action` argument is ignored.

        Returns:
            np.ndarray: A dummy array consisting only of ones. We apply our muscle torque by reusing the torque motors
            in the simulation. We set their gear ratio to our desired muscle torque and then apply a control input of 1.
        """
        assert not (update_action and action is None)
        if update_action:
            self.target_activity = np.clip(action, 0, 1).copy()
        self._update_muscle_state()
        self._apply_torque()
        return np.ones_like(self.joint_torque)

    def fl(self, lce):
        """ Force length curve as implemented by MuJoCo.

        Args:
            lce (np.ndarray): Virtual muscle lengths for MIMo.

        Returns:
            np.ndarray: An array with the force-length multipliers.
        """
        return bump(lce, self.lmin, 1, self.lmax) + 0.15 * bump(lce, self.lmin, 0.5 * (self.lmin + 0.95), 0.95)

    def fv(self, lce_dot):
        """ Force length curve as implemented by MuJoCo.

        Args:
            lce_dot (np.ndarray): Virtual muscle velocities for MIMo.

        Returns:
            np.ndarray: An array with the force-velocity multipliers.
        """
        c = self.fvmax - 1
        eff_vel = lce_dot / self.vmax
        eff_vel_con1 = eff_vel[eff_vel <= c]
        eff_vel_con2 = eff_vel[eff_vel <= 0]

        output = np.full(lce_dot.shape, self.fvmax)
        output[eff_vel <= c] = self.fvmax - (c - eff_vel_con1) * (c - eff_vel_con1) / c
        output[eff_vel <= 0] = (eff_vel_con2 + 1) * (eff_vel_con2 + 1)
        output[eff_vel < -1] = 0
        return output

    def fp(self, lce):
        """ Parallel elasticity (passive muscle force) as implemented by MuJoCo.

        Args:
            lce (np.ndarray): Virtual muscle lengths for MIMo.

        Returns:
            np.ndarray: An array with the passive force components.
        """
        b = 0.5 * (self.lmax + 1)
        tmp = (lce[lce <= b] - 1) / (b - 1)

        output = 0.25 * self.fpmax * (1 + 3 * (lce - b) / (b - 1))
        output[lce <= b] = 0.25 * self.fpmax * tmp * tmp * tmp
        output[lce <= 1] = 0
        return output

    def set_fmax(self, fmax):
        """ Setter for :attr:`~.fmax`.

        Args:
            fmax (np.ndarray|float): The new fmax value(s).
        """
        self.fmax = fmax

    def set_vmax(self, vmax):
        """ Setter for :attr:`~.vmax`.

        Args:
            vmax (np.ndarray|float): The new vmax value(s).
        """
        self.vmax = vmax

    def simulation_torque(self):
        """ Computes the currently applied torque for each motor in the simulation.

        Returns:
            np.ndarray: A numpy array with applied torques for each motor.
        """
        actuator_gear = self.env.model.actuator_gear[self.actuators, 0]
        control_input = self.env.data.ctrl[self.actuators]
        return actuator_gear * control_input

    def collect_data_for_actuators(self):
        """ Collect all muscle related values at the current timestep for all of MIMo's actuators.

        Returns:
            List[np.ndarray]: A list containing the joint position and velocity, corrected position, output torque,
            desired target muscle activity, actual current muscle activity, virtual muscle length, virtual muscle
            velocity, muscle force, FL factor, FV factor and the FP component for all muscles.
        """
        actuator_qpos = self.env.data.qpos[self.mimo_actuated_qpos].flatten()
        actuator_qvel = self.env.data.qvel[self.mimo_actuated_qvel].flatten()

        # This is the basic qpos adjusted for the springref parameter. This shifts the "neutral" position of the joint.
        joint_qpos = actuator_qpos - self.env.model.qpos_spring[self.mimo_actuated_qpos].flatten()

        actuator_gear = self.env.model.actuator_gear[self.actuators, 0].flatten()

        action_neg = self.target_activity[:self.n_actuators]
        action_pos = self.target_activity[self.n_actuators:]

        activity_neg = self.activity[:self.n_actuators]
        activity_pos = self.activity[self.n_actuators:]

        lce_neg = self.lce_1
        lce_pos = self.lce_2

        lce_neg_dot = self.lce_dot_1
        lce_pos_dot = self.lce_dot_2

        force_neg = self.force_muscles_1
        force_pos = self.force_muscles_2

        fl1 = self.fl(self.lce_1)
        fl2 = self.fl(self.lce_2)
        fl_neg = fl1
        fl_pos = fl2

        fv1 = self.fv(self.lce_dot_1)
        fv2 = self.fv(self.lce_dot_2)
        fv_neg = fv1
        fv_pos = fv2

        fp1 = self.fp(self.lce_1)
        fp2 = self.fp(self.lce_2)
        fp_neg = fp1
        fp_pos = fp2

        output = [actuator_qpos, actuator_qvel,
                  joint_qpos, actuator_gear,
                  action_neg, action_pos,
                  activity_neg, activity_pos,
                  lce_neg, lce_pos,
                  lce_neg_dot, lce_pos_dot,
                  force_neg, force_pos,
                  fl_neg, fl_pos,
                  fv_neg, fv_pos,
                  fp_neg, fp_pos]
        return output


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

    a_dif = np.square(length - a) * 0.5
    b_dif = np.square(length - b) * 0.5
    mid_dif = np.square(length - mid) * 0.5

    output = b_dif / ((b - right) * (b - right))

    output[length < right] = 1 - mid_dif[length < right] / ((right - mid) * (right - mid))
    output[length < mid] = 1 - mid_dif[length < mid] / ((mid - left) * (mid - left))
    output[length < left] = a_dif[length < left] / ((left - a) * (left - a))
    output[(length <= a) | (length >= b)] = 0

    return output
