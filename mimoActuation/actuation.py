"""This module defines the actuation model interface and provides two implementations.

The interface is defined as an abstract class in :class:`~mimoActuation.actuation.ActuationModel`.
The spring-damper model is defined in :class:`~mimoActuation.actuation.TorqueMotorModel`.
A second implementation using direct positional control is :class:`~mimoActuation.actuation.PositionalModel`.
"""
import numpy
import numpy as np
from gymnasium import spaces

from mimoEnv.utils import set_joint_locking_angle


class ActuationModel:
    """ Abstract base class for MIMo's actuation model.

    This class defines the functions that all implementing classes must provide.

    Control inputs have two conceptual levels: The desired control input (i.e. maximum output in one direction),
    and the actual control input to the simulation motors. In the simulation the motor response is linear and
    instantaneous, but this may not be desired. Actuation models can model time-dependent or non-linear torque
    generation by taking the desired control input and altering it before passing it to the simulation.
    Actuation models can define an arbitrary control method, but must compute control inputs for the actual simulation
    motors as defined in the XMLs.

    The key functions are:

    - :meth:`.get_action_space` determines the actuation space attribute for the gym environment. This should have the
      shape of the input to the abstract model motors.
    - :meth:`.action` computes the actual control inputs to the simulation motors from a control input to the abstract
      motors. :meth:`.substep_update` is called on every physics step and allows torques to be updated between
      environment steps.
    - :meth:`.observations` should return any actuation-related quantities that could reasonably be used as
      observations for the gym environment. Note that these will only actually be included if the proprioception
      module is appropriately configured.
    - :meth:`.cost` should return the cost of the current activations. This can represent the metabolic cost or an
      action penalty. This function is not used by default, but environments may use it as they wish, for example during
      reward calculation.
    - :meth:`.reset` should reset whatever internal quantities the model uses to the value at the start of the
      simulation.

    Args:
        env (MIMoEnv): The environment to which this model will be attached.
        actuators (np.ndarray): An array with the actuators, by ID, to include in this model.

    Attributes:
        env (gym.Env): The environment to which this module will be attached.
        actuators (np.ndarray): The simulation motors, by ID, to include in this model.
        n_actuators (int): The number of actuators controlled by this model.
        action_space (spaces.Box): The action space for this model. This is set by :meth:`~.get_action_space`
    """
    def __init__(self, env, actuators, *args):
        self.env = env
        self.actuators = actuators
        self.n_actuators = self.actuators.shape[0]
        self.action_space = self.get_action_space()

    def get_action_space(self):
        """ Determines the actuation space attribute for the gym environment.

        Note that his action space must be a Box!

        Returns:
            gym.spaces.Box: A gym spaces object with the actuation space.
        """
        raise NotImplementedError

    def action(self, action):
        """ Converts abstract control inputs into actual motor inputs.

        This function is called during every environment step and sets the actual motor inputs for each included actuator.

        Args:
            action (numpy.ndarray): A numpy array with control values.
        """
        raise NotImplementedError

    def substep_update(self):
        """ Like action, but called on every physics step instead of every environment step.

        This allows for torques to be updated every physics step.
        """
        pass

    def observations(self):
        """ Collect any quantities for the observations.

        Returns:
            np.ndarray: A flat numpy array with these quantities.
        """
        raise NotImplementedError

    def cost(self):
        """ Returns the "cost" of the current action.

        This function may be used as an action penalty.

        Returns:
            float: The cost of the action.
        """
        raise NotImplementedError

    def reset(self):
        """ Reset actuation model to the initial state.
        """
        raise NotImplementedError


class SpringDamperModel(ActuationModel):
    """ Class for the Spring-Damper actuation model.

    In this model, MIMo's muscles are represented by torque motors with linear and instantaneous control response, i.e.
    the abstract model directly matches the in-simulation definitions.
    The force-velocity and force-length relationships of real muscles is approximated using damping and spring
    components in the joint definitions of MIMo. The maximum torque of the motors is set to the maximum voluntary
    isometric torque along the corresponding axis, with a control input of 1 representing maximum torque.

    In addition to the attributes from the base actuation class, there are two extra attributes:

    Attributes:
        control_input (np.ndarray): Contains the current control input.
        max_torque (np.ndarray): The maximum motor torques.
    """
    def __init__(self, env, actuators):
        super().__init__(env, actuators)
        self.control_input = None
        self.max_torque = self.env.model.actuator_gear[self.actuators, 0]

    def get_action_space(self):
        """ Determines the actuation space attribute for the gym environment.

        The actuation space directly corresponds to the control range of the simulations motors. Unless modified, this
        will be [-1, 1] for all motors.

        Returns:
            gym.spaces.Space: The actuation space.
        """
        bounds = self.env.model.actuator_ctrlrange.copy().astype(np.float32)[self.actuators]
        low, high = bounds.T
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.control_input = np.zeros(action_space.shape)  # Set initial control input to avoid NoneType Errors
        return action_space

    def action(self, action):
        """ Set the control inputs for the next step.

        Control values are clipped to the control range limits defined the MuJoCo XMLs and normalized to be even in
        both directions, i.e. an input of 0 corresponds to the center of the control range, rather than the default or
        neutral control position. The control ranges for the MIMo XMLs are set up to be symmetrical, such that an input
        of 0 corresponds to no motor torque.

        Args:
            action (numpy.ndarray): A numpy array with control values.
        """
        self.control_input = np.clip(action, self.action_space.low, self.action_space.high)
        self.env.data.ctrl[self.actuators] = self.control_input

    def observations(self):
        """ Control input and output torque for each motor at this time step.

        Returns:
            np.ndarray: A flat array with the control inputs and output torques.
        """
        torque = self.simulation_torque().flatten()
        return np.concatenate([self.control_input.flatten(), torque])

    def cost(self):
        """ Provides a cost function for current motor usage.

        The cost is given by given by :math:`\\sum_{i=1}^n \\frac{u_i^2 * T_{max_i}}{n \\sum_{i=1}^n T_{max_i}}`, where
        :math:`u_i` and :math:`T_{max_i}` are the control signal and maximum motor torque of motor
        :math:`i`, respectively, and :math:`n` is the number of motors in the model.

        Returns:
            float: The cost as described above.
        """
        per_actuator_cost = self.control_input * self.control_input * self.max_torque
        return np.abs(per_actuator_cost).sum() / (self.n_actuators * self.max_torque.sum())

    def simulation_torque(self):
        """ Computes the currently applied torque for each motor in the simulation.

        Returns:
            np.ndarray: An array with applied torques for each motor.
        """
        actuator_gear = self.env.model.actuator_gear[self.actuators, 0]
        control_input = self.env.data.ctrl[self.actuators]
        return actuator_gear * control_input

    def reset(self):
        """ Reset actuation model to the initial state.
        """
        self.action(numpy.zeros_like(self.control_input))


class PositionalModel(ActuationModel):
    """ This model allows posing MIMo or moving his joints along pre-determined trajectories .

    The 'action' input represents desired joint positions. MIMo will be locked into these at each timestep. Unlike the
    other actuation models this doesn't use the MuJoCo actuators in the scene but instead adjusts the equality
    constraints used to lock each joint into position. To determine which joints should be included we use the joints
    associated with the actuators in the 'actuators' parameter. Note that this requires that there is an equality
    constraint in the XMLs for each actuated joint. This is true for MIMo by default.

    In addition to the attributes from the base actuation class, there is three extra attributes.

    Attributes:
        control_input (np.ndarray): Contains the current control input.
        actuated_joints (np.npdarray): Contains an array of joint IDs associated with the actuators.
        constraints (np.ndarray): Contains an array of constraint IDs belonging to the joints in 'actuated_joints'.
    """
    def __init__(self, env, actuators):
        super().__init__(env, actuators)
        self.control_input = None
        self.actuated_joints = self.env.model.actuator_trnid[actuators, 0]
        self.constraints = self.get_constraints()

    def get_constraints(self):
        """ Collects the constraints associated with the actuated joints in the scene.

        Returns:
            np.ndarray: An array with the constraint IDs.
        """
        constraints = []
        # Iterate over all constraints, check that they belong to an actuated joint and are type 'joint'
        for i in range(self.env.model.neq):
            if self.env.model.eq_type[i] == 2 and self.env.model.eq_obj1id[i] in self.actuated_joints:
                self.env.model.eq_active[i] = True
                constraints.append(i)
        return numpy.asarray(constraints)

    def get_action_space(self):
        """ Determines the actuation space attribute for the gym environment.

        The actuation space directly corresponds to the range of motion of the joints in radians.

        Returns:
            gym.space.Spaces: A gym spaces object with the actuation space.
        """
        bounds = self.env.model.jnt_range.copy().astype(np.float32)[self.actuated_joints]
        low, high = bounds.T
        action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.control_input = np.zeros(action_space.shape)  # Set initial control input to avoid NoneType Errors
        return action_space

    def action(self, action):
        """ Locks the joints into the positions provided by 'action'.

        Control values are clipped to the joint range of motion.

        Args:
            action (numpy.ndarray): A numpy array with desired joint positions.
        """
        self.control_input = np.clip(action, self.action_space.low, self.action_space.high)
        set_joint_locking_angle(self.env.model, "", angle=self.control_input, constraint_id=self.constraints)

    def observations(self):
        """ Returns the current control input, i.e. the locked positions.

        Returns:
            np.ndarray: A flat numpy array with the control input.
        """
        return self.control_input.flatten()

    def reset(self):
        """ Reset actuation model to the initial state.
        """
        self.action(numpy.zeros_like(self.control_input))

    def cost(self):
        """ Dummy function.

        Returns:
            float: Always returns 0.
        """
        return 0
