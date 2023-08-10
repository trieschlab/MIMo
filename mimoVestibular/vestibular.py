""" This module defines the interface for the vestibular module and a simple implementation.

The interface is defined as an abstract class in :class:`~mimoVestibular.vestibular.Vestibular`.
A simple implementation treating using one 3D gyro and one 3D accelerometer is in
:class:`~mimoVestibular.vestibular.SimpleVestibular`.

"""

import numpy as np
from typing import Dict, List

from mimoEnv.utils import get_sensor_addr
from gymnasium.envs.mujoco import MujocoEnv


class Vestibular:
    """ Abstract base class for the vestibular system.

    This class defines the functions that all implementing classes must provide.
    The constructor takes two arguments: `env`, which is the environment we are working with, and
    `vestibular_parameters`, which can be used to supply implementation specific parameters.

    There is only one function that implementations must provide:
    :meth:`.get_vestibular_obs` should produce the sensor outputs that will be returned to the environment. These
    outputs should also be stored in :attr:`.sensor_outputs`.

    Attributes:
        env (MujocoEnv): The environment to which this module will be attached.
        vestibular_parameters (Dict): A dictionary containing the configuration. The exact from will depend on the
            specific implementation.
        sensor_outputs (np.ndarray): A list of outputs corresponding to the configuration dictionary. This should be
            populated by :meth:`.get_vestibular_obs`.
    """
    def __init__(self, env, vestibular_parameters):
        self.env = env
        self.vestibular_parameters = vestibular_parameters
        self.sensor_outputs = []

    def get_vestibular_obs(self):
        """ Produce the vestibular sensor outputs.

        This function should perform the whole sensory pipeline and return the vestibular output as defined in
        :attr:`.vestibular_parameters`. Exact return value and functionality will depend on the implementation, but
        should always be a flat numpy array.

        Returns:
            np.ndarray: An array containing the vestibular sensations.
        """
        raise NotImplementedError


class SimpleVestibular(Vestibular):
    """
    A simple implementation directly reading MuJoCo sensors.

    This class reads all sensors provided in the configuration and stores their outputs in :attr:`.sensor_outputs`. The
    constructor takes two arguments: `env`, the environment we are working with, and `vestibular_parameters`, which is
    a dictionary containing the configuration. The dictionary structure should be a single entry 'sensors' containing a
    list with the sensor names to be used for the output::

        {
            "sensors": ["vestibular_acc", "vestibular_gyro"],
        }

    The default model has two sensors, 'vestibular_acc' and 'vestibular_gyro', that can be used for the accelerometer
    and the gyro, both located in the head.

    Attributes:
        env (MujocoEnv): The environment to which this module should be attached.
        vestibular_parameters (Dict[str, List[str]]): A dictionary containing the configuration.
        sensor_outputs (np.ndarray): A list of outputs corresponding to the configuration dictionary. This is populated
            by :meth:`.get_vestibular_obs`.
    """
    def __init__(self, env, vestibular_parameters):
        super().__init__(env, vestibular_parameters)
        self.sensors = self.vestibular_parameters["sensors"]
        self.sensor_ids = [self.env.model.sensor(name).id for name in self.sensors]
        self.sensor_addrs = np.asarray([get_sensor_addr(self.env.model, idx) for idx in self.sensor_ids])

    def get_vestibular_obs(self):
        """ Produce the vestibular sensor outputs.

        Directly reads the sensor values from the MuJoCo sensors provided in the configuration.

        Returns:
            np.ndarray: An array containing the concatenated sensor values.
        """
        self.sensor_outputs = self.env.data.sensordata[self.sensor_addrs].flatten()
        return np.asarray(self.sensor_outputs)
