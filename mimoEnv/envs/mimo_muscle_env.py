""" This module defines the base class for the muscle actuation model.

The abstract base class is :class:`~mimoEnv.envs.mimo_muscle_env.MIMoMuscleEnv`. It is a direct drop in replacement for
:class:`~mimoEnv.envs.mimo_env.MIMoEnv`.
"""
import warnings
import numpy as np

from gym import spaces

from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as mimo_utils


def bump(length, A, mid, B):
    """
    Part of the force length relationship as implemented by MuJoCo.
    """
    left = 0.5 * (A + mid)
    right = 0.5 * (mid + B)

    a_dif = np.square(length - A) * 0.5
    b_dif = np.square(length - B) * 0.5
    mid_dif = np.square(length - mid) * 0.5

    output = b_dif / ((B - right) * (B - right))

    output[length < right] = 1 - mid_dif[length < right] / ((right - mid) * (right - mid))
    output[length < mid] = 1 - mid_dif[length < mid] / ((mid - left) * (mid - left))
    output[length < left] = a_dif[length < left] / ((left - A) * (left - A))
    output[(length <= A) | (length >= B)] = 0

    return output


class MIMoMuscleEnv(MIMoEnv):
    """ This is the base class for the muscle actuated version of MIMo.

    It is a direct drop-in replacement for :class:`~mimoEnv.envs.mimo_env.MIMoEnv`. and has the same attributes and
    functions, with some extra ones to handle muscle functionality.
    """
    def __init__(self,
                 model_path,
                 initial_qpos=None,
                 n_substeps=2,
                 proprio_params=None,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=None,
                 goals_in_observation=True,
                 done_active=False):
        # user parameters ------------------------------
        # Muscle parameters. These are the same as the MuJoCo internal parameters that are used to compute muscle
        # properties (FL, FV, FP, curves).
        # These would be found in a GENERAL actuator
        self.lmax = 1.6
        self.lmin = 0.5
        self.fvmax = 1.2
        self.fpmax = 1.3
        self.lce_min = 0.75
        self.lce_max = 1.05
        # TODO: LMAX, LMIN, etc also in here.
        self.vmax = None
        self.fmax = None

        self.time_step = None
        self.tau = 0.01

        # Placeholders that gets overwritten later
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
        """ This function initializes muscle model and the sensory components of the model.

        Calls the setup functions for the sensory components, sets the initial positions of the joints according
        to the qpos dictionary and configures the muscle parameters.

        Args:
            initial_qpos (dict): A dictionary with the intial joint position for each joint. Keys are the joint names.
        """
        super()._env_setup(initial_qpos)
        # Also perform all the muscle setup
        self._set_max_forces()
        self._get_actuated_joints()
        self._set_muscles()

    def _get_actuated_joints(self):
        """ Populates references to MIMo's joints and their indices in the qpos and qvel arrays.
        """
        self.mimo_actuated_joints = self.sim.model.actuator_trnid[self.mimo_actuators, 0]
        actuated_qpos_idx = [mimo_utils.get_joint_qpos_addr(self.sim.model, idx) for idx in self.mimo_actuated_joints]
        actuated_qvel_idx = [mimo_utils.get_joint_qvel_addr(self.sim.model, idx) for idx in self.mimo_actuated_joints]
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

    def _set_max_forces(self):
        """ Collect maximum isometric forces from MuJoCo actuator gears.
        """
        force_ranges = np.abs(self.sim.model.actuator_forcerange[self.mimo_actuators, :]).copy()
        gears = self.sim.model.actuator_gear[self.mimo_actuators, 0].copy()
        self.maximum_isometric_forces = (force_ranges.T * gears).T
        # Have to disable force limits afterwards
        self.sim.model.actuator_forcelimited[self.mimo_actuators] = np.zeros_like(self.sim.model.actuator_forcelimited[self.mimo_actuators])

    def _compute_parametrization(self):
        """ Compute parameters for muscles from angle and muscle fiber length ranges.

        The muscle model converts joint positions and velocities into virtual muscle lengths and speeds. The parameters
        for this conversion are computed from fixed variables set
        """
        # Collect joint range from model, using springref as "neutral" position
        self.phi_min = self.sim.model.jnt_range[self.mimo_actuated_joints, 0] - self.sim.model.qpos_spring[self.mimo_actuated_qpos].flatten()
        self.phi_max = self.sim.model.jnt_range[self.mimo_actuated_joints, 1] - self.sim.model.qpos_spring[self.mimo_actuated_qpos].flatten()
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
        self.sim.model.jnt_stiffness[self.mimo_joints] = np.zeros_like(self.sim.model.jnt_stiffness[self.mimo_joints])
        mimo_dof_ids = self.sim.model.jnt_dofadr[self.mimo_joints]
        self.sim.model.dof_damping[mimo_dof_ids] = self.sim.model.dof_damping[mimo_dof_ids] / 20
        self._collect_muscle_parameters()

    def _collect_muscle_parameters(self):
        """ Loads the muscle model parameters from the XML.

        We store the FMAX and VMAX parameters in the user fields for the actuators. The first value is VMAX, the second
        value FMAX for the muscle acting in the negative joint direction, the third value FMAX for the positive joint
        direction.
        If there are no use user arguments, we print a warning and proceed with default values.
        If the values in the user arguments are invalid we raise ValueErrors.
        """
        if self.sim.model.nuser_actuator >= 3:
            print("Reading Muscle parameters from XML")
            vmax = self.sim.model.actuator_user[self.mimo_actuators, 0]
            fmax_neg = self.sim.model.actuator_user[self.mimo_actuators, 1]
            fmax_pos = self.sim.model.actuator_user[self.mimo_actuators, 2]
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
                self.vmax = np.load('../mimoMuscle/vmax.npy')
            except FileNotFoundError:
                warnings.warn("No vmax calibration file, using default value 1!", RuntimeWarning)
                self.vmax = 1
            try:
                self.fmax = np.load("../mimoMuscle/fmax.npy")
            except FileNotFoundError:
                warnings.warn("No fmax calibration file, using default value of 5!", RuntimeWarning)
                self.fmax = 5

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
        self.target_activity = np.zeros(self.action_space.shape)
        self._update_muscle_state()

    def _update_muscle_state(self):
        self._update_activity()
        self._update_virtual_lengths()
        self._update_virtual_velocities()
        self._update_torque()

    def _update_activity(self):
        """
        Very simple low-pass filter, even simpler than MuJoCo internal, update in the future.
        """
        # Need sim timestep here rather than dt since we take calculate this every physics step, while dt is the
        # duration of an environment step.
        self.activity += self.sim.model.opt.timestep * (self.target_activity - self.activity) / self.tau
        self.activity = np.clip(self.activity, 0, 1)

    def _update_virtual_lengths(self):
        """
        Update the muscle lengths from current joint angles.
        """
        self.lce_1 = (self.sim.data.qpos[self.mimo_actuated_qpos].flatten() - self.sim.model.qpos_spring[self.mimo_actuated_qpos].flatten()) * self.moment_1 + self.lce_1_ref
        self.lce_2 = (self.sim.data.qpos[self.mimo_actuated_qpos].flatten() - self.sim.model.qpos_spring[self.mimo_actuated_qpos].flatten()) * self.moment_2 + self.lce_2_ref

    def _update_virtual_velocities(self):
        """
        Update the muscle lengths from current joint angle velocities.
        """
        self.lce_dot_1 = self.moment_1 * self.sim.data.qvel[self.mimo_actuated_qvel].flatten()
        self.lce_dot_2 = self.moment_2 * self.sim.data.qvel[self.mimo_actuated_qvel].flatten()

    def set_fmax(self, fmax):
        """ Setter for :attr:`~.fmax`.

        Args:
            fmax: The new fmax array. Either a float or an array with shape (2*n_actuators, ).
        """
        self.fmax = fmax

    def set_vmax(self, vmax):
        """ Setter for :attr:`~.vmax`.

        Args:
            vmax: The new vmax array. Either a float or an array with shape (n_actuators, ).
        """
        self.vmax = vmax

    def fl(self, lce):
        """ Force length curve as implemented by MuJoCo.

        Args:
            lce (np.ndarray): Virtual muscle lengths for MIMo.

        Returns:
            An array with the force-length multipliers.
        """
        return bump(lce, self.lmin, 1, self.lmax) + 0.15 * bump(lce, self.lmin, 0.5 * (self.lmin + 0.95), 0.95)

    def fv(self, lce_dot):
        """ Force length curve as implemented by MuJoCo.

        Args:
            lce_dot (np.ndarray): Virtual muscle velocities for MIMo.

        Returns:
            An array with the force-velocity multipliers.
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
            An array with the passive force components.
        """
        b = 0.5 * (self.lmax + 1)
        tmp = (lce[lce <= b] - 1) / (b - 1)

        output = 0.25 * self.fpmax * (1 + 3 * (lce - b) / (b - 1))
        output[lce <= b] = 0.25 * self.fpmax * tmp * tmp * tmp
        output[lce <= 1] = 0
        return output

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

    def _compute_muscle_action(self, action=None, update_action=True):
        """ Take in the muscle action, compute all virtual quantities and apply the correct torques.

        All-in-one function that updates muscle activity and computes muscle torques given current simulation state.

        Args:
            action: Either an array with control inputs or `None`. If `None` we reuse the previous control input.
                    Default `None`. This argument is ignored if "update_action" is `False`.
            update_action: If `True` we set :attr:`~.target_activity` to the "action" argument. If `False` the action
                    is ignored.

        Returns:
            A dummy array consisting only of ones. We apply our muscle torque by reusing the torque motors in the
            simulation. We set their gear ratio to our desired muscle torque and then apply a control input of 1.
        """
        assert not (update_action and action is None)
        if update_action:
            self.target_activity = np.clip(action, 0, 1).copy()
        self._update_muscle_state()
        self._apply_torque()
        return np.ones_like(self.joint_torque)

    def _apply_torque(self):
        """ Adjust MuJoCo values to actually apply muscle torque in the simulation.

        We adjust the MuJoCo actuator gear (which is just a scalar force multiplier) to be equal to the torque we want
        to apply, then we output an action-vector that is 1 everywhere. Doing this allows us to keep the same XMLs and
        gym interfaces for both versions of MIMo.
        """
        self.sim.model.actuator_gear[self.mimo_actuators, 0] = self.joint_torque.copy()

    @property
    def muscle_activations(self):
        """ Activity for every muscle. """
        return self.activity.copy()

    @property
    def muscle_lengths(self):
        """ Virtual muscle lengths for all muscles. """
        return np.concatenate([self.lce_1, self.lce_2], dtype=np.float32).copy()

    @property
    def muscle_velocities(self):
        """ Virtual muscle speeds for all muscles. """
        return np.concatenate([self.lce_dot_1, self.lce_dot_2], dtype=np.float32).copy()

    @property
    def muscle_forces(self):
        """ Muscle force vector. """
        return np.concatenate([self.force_muscles_1, self.force_muscles_2], dtype=np.float32).copy()

    def do_simulation(self, action, n_frames):
        """ Step simulation forward for n_frames number of steps.

        The control input is set as the desired muscle activation level and then "n_frames" number of physics steps
        are performed. Muscle activities and torques are updated on each physics step.

        Args:
            action (np.ndarray): The control input for the actuators.
            n_frames (int): The number of physics steps to perform.
        """
        self._set_action(action)
        for i in range(n_frames):
            self._compute_muscle_action(update_action=False)
            self.sim.step()
            self._substep_callback()

    def _set_action(self, action):
        """ Set the control inputs for the next step.

        The input action is used to compute muscle activities and then discarded. We adjust the MuJoCo actuator gear
        (which is just a scalar force multiplier) to be equal to the torque we want to apply, then we output an
        action-vector that is 1 everywhere. Doing this allows us to keep the same XMLs and gym interfaces for both
        versions of MIMo.

        Args:
            action (numpy.ndarray): A numpy array with control values.
        """
        action = self._compute_muscle_action(action)
        self.sim.data.ctrl[self.mimo_actuators] = action

    def _get_obs(self):
        """Returns the observation.

        Like :meth:`~mimoEnv.envs.mimo_env.MIMoEnv._get_obs`, but adds the current state of the muscles to the output.

        Returns:
            dict: A dictionary containing simulation outputs with separate entries for each sensor modality.
        """
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
        """ Attempt to reset the simulator and sample a new goal.

        Same as :meth:`~mimoEnv.envs.mimo_env.MIMoEnv.reset`, but additionally resets all muscle states.

        Returns:
            dict: The observations after reset.
        """
        obs = super().reset()
        self._set_initial_muscle_state()
        return obs

    def collect_data_for_actuator(self, actuator_index):
        """ Collect all muscle related values at the current timestep for a given actuator.

        Args:
            actuator_index: The index of the actuator. Note that this is the MuJoCo index and might change between
                            runs if more actuators are added.

        Returns:
            A list containing the joint position and velocity, corrected position, output torque, desired target muscle
            activity, actual current muscle activity, virtual muscle length, virtual muscle velocity, muscle force,
            FL factor, FV factor and the FP component.
        """
        actuator_qpos = self.sim.data.qpos[self.mimo_actuated_qpos[actuator_index]].item()
        actuator_qvel = self.sim.data.qvel[self.mimo_actuated_qvel[actuator_index]].item()

        # This is the basic qpos adjusted for the springref parameter. This shifts the "neutral" position of the joint.
        joint_qpos = actuator_qpos - self.sim.model.qpos_spring[self.mimo_actuated_qpos[actuator_index]].item()

        actuator_gear = self.sim.model.actuator_gear[actuator_index, 0]

        action_neg = self.target_activity[actuator_index]
        action_pos = self.target_activity[self.n_actuators + actuator_index]

        activity_neg = self.activity[actuator_index]
        activity_pos = self.activity[self.n_actuators + actuator_index]

        lce_neg = self.lce_1[actuator_index]
        lce_pos = self.lce_2[actuator_index]

        lce_neg_dot = self.lce_dot_1[actuator_index]
        lce_pos_dot = self.lce_dot_2[actuator_index]

        force_neg = self.force_muscles_1[actuator_index]
        force_pos = self.force_muscles_2[actuator_index]

        fl1 = self.fl(self.lce_1)
        fl2 = self.fl(self.lce_2)
        fl_neg = fl1[actuator_index]
        fl_pos = fl2[actuator_index]

        fv1 = self.fv(self.lce_dot_1)
        fv2 = self.fv(self.lce_dot_2)
        fv_neg = fv1[actuator_index]
        fv_pos = fv2[actuator_index]

        fp1 = self.fp(self.lce_1)
        fp2 = self.fp(self.lce_2)
        fp_neg = fp1[actuator_index]
        fp_pos = fp2[actuator_index]

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

    def collect_data_for_actuators(self):
        """ Collect all muscle related values at the current timestep for all of MIMo's actuators.

        Returns:
            A list containing the joint position and velocity, corrected position, output torque, desired target muscle
            activity, actual current muscle activity, virtual muscle length, virtual muscle velocity, muscle force,
            FL factor, FV factor and the FP component for all muscles.
        """
        actuator_qpos = self.sim.data.qpos[self.mimo_actuated_qpos].flatten()
        actuator_qvel = self.sim.data.qvel[self.mimo_actuated_qvel].flatten()

        # This is the basic qpos adjusted for the springref parameter. This shifts the "neutral" position of the joint.
        joint_qpos = actuator_qpos - self.sim.model.qpos_spring[self.mimo_actuated_qpos].flatten()

        actuator_gear = self.sim.model.actuator_gear[self.mimo_actuators, 0].flatten()

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
