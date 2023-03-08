import numpy
import numpy as np
from gym import spaces


class ActuationModel:
    """ Abstract base class for MIMo's actuation model.

    This class defines the functions that all implementing classes must provide.

    Control inputs have two conceptual levels: The desired control input (i.e. maximum output in one direction),
    and the actual control input to the simulation motors. In the simulation the motor response is linear and
    instantaneous, but this may not be desired. Actuation models can model time-dependent or non-linear torque
    generation by taking the desired control input and altering it before passing it to the simulation.
    Actuation models can define an arbitrary control method, but must compute control inputs for the actual simulation
    motors as defined in the XMLs.
    - :meth:`.get_action_space` determines the actuation space attribute for the gym environment. This should have the
      shape of the input to the abstract model motors.
    - :meth:`.action` computes the actual control inputs to the simulation motors from a control input to the abstract
      motors. :meth:`.substep_update` is called on every physics step and allows torques to be updated between
      environment steps.
    - :meth:`.observations` should return any actuation-related quantities that could reasonably be used as
      observations for the gym environment. Note that these will only actually be included if the proprioception
      module is appropriately configured.
    - :meth:`.reset` should reset whatever internal quantities the model uses to the value at the start of the
      simulation.

    Attributes:
        env: The environment to which this module will be attached.
        actuators: The simulation motors to include in this model.

    """
    def __init__(self, env, actuators):
        self.env = env
        self.actuators = actuators

    def get_action_space(self):
        """ Determines the actuation space attribute for the gym environment.

        Returns:
            A gym spaces object with the actuation space.
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
            A flat numpy array with these quantities.
        """
        raise NotImplementedError

    def reset(self):
        """ Reset actuation model to the initial state.
        """
        raise NotImplementedError


class TorqueMotorModel(ActuationModel):
    """ Class for the torque actuation model.

    In this model, MIMo's muscles are represented by torque motors with linear and instantaneous control response, i.e.
    the abstract model directly matches the in-simulation definitions.
    The force-velocity and force-length relationships of real muscles is approximated using damping and spring
    components in the joint definitions of MIMo. The maximum torque of the motors is set to the maximum voluntary
    isometric torque along the corresponding axis, with a control input of 1 representing maximum torque.

    In addition to the attributes from the base actuation class, there is one extra attribute:
    Attributes:
        control_input: Contains the current control input.
    """
    def __init__(self, env, actuators):
        super().__init__(env, actuators)
        self.control_input = None

    def get_action_space(self):
        """ Determines the actuation space attribute for the gym environment.

        The actuation space directly corresponds the control range of the simulations motors. Unless modified, this
        will be [-1, 1] for all motors.

        Returns:
            A gym spaces object with the actuation space.
        """
        bounds = self.env.sim.model.actuator_ctrlrange.copy().astype(np.float32)[self.actuators]
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
        self.control_input = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        self.env.sim.data.ctrl[self.actuators] = self.control_input

    def observations(self):
        """ Control input and output torque for each motor for this time step.

        Returns:
            A flat numpy array with the control inputs and output torques.
        """
        torque = self.simulation_torque().flatten()
        return np.concatenate([self.control_input.flatten(), torque])

    def simulation_torque(self):
        """ Computes the currently applied torque for each motor in the simulation.

        Returns:
            A numpy array with applied torques for each motor.
        """
        actuator_gear = self.env.sim.model.actuator_gear[self.actuators, 0]
        control_input = self.env.sim.data.ctrl[self.actuators]
        return actuator_gear * control_input

    def reset(self):
        """ Reset actuation model to the initial state.
        """
        self.action(numpy.zeros_like(self.control_input))
