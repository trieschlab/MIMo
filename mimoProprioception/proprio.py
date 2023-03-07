""" This module provides the interface and a simple implementation for proprioception.

The interface is defined as an abstract class in :class:`~mimoProprioception.proprio.Proprioception`.
A simple implementation directly reading values is in :class:`~mimoProprioception.proprio.SimpleProprioception`.

"""

import numpy as np
from mimoEnv.utils import get_joint_qpos_addr, get_joint_qvel_addr, get_sensor_addr


class Proprioception:
    """ Abstract base class for proprioception

    This class defines the functions that implementing classes must provide.
    :meth:`.get_proprioception_obs` should return the complete sensor outputs and additionally store them in
    :attr:`.sensor_outputs`.

    Attributes:
        env: The environment to which this module will be attached.
        proprio_parameters: A dictionary containing the configuration. The exact from will depend on the specific
            implementation.
        output_components: A list containing all the proprioceptive components that should be put in the output. This
            attribute is populated by `proprio_parameters`. These components must be in :attr:`VALID_COMPONENTS`.
        sensor_outputs: A dictionary containing the outputs produced by the sensors. Shape will depend on the specific
            implementation. This should be populated by :meth:`.get_proprioception_obs`.
    """

    #: Valid entries for the output components
    VALID_COMPONENTS = []

    def __init__(self, env, proprio_parameters):
        self.env = env
        self.proprio_parameters = proprio_parameters
        if proprio_parameters is None or "components" not in proprio_parameters:
            self.output_components = []
        else:
            self.output_components = proprio_parameters["components"]
        assert all([component in self.VALID_COMPONENTS for component in self.output_components])
        self.sensor_outputs = {}

    def get_proprioception_obs(self):
        """ Produce the proprioceptive sensor outputs.

        This function should perform the whole sensory pipeline and return the output as defined in
        :attr:`.proprio_parameters`. Exact return value and functionality will depend on the implementation, but
        should always be a flat numpy array.

        """
        raise NotImplementedError


class SimpleProprioception(Proprioception):
    """ Simple implementation reading all values directly from the physics simulation.

    This class can provide relative joint positions, joint velocities, joint torques and limit sensors on the joint
    range of motion. Torques are in newton-meters, all other values in radians. Joint positions are always part of the
    output, while the others can be added optionally through the configuration dictionary. Valid components are
    'velocity', 'torque', 'limits' and 'actuation'.
    The limit sensing increases linearly from 0 through 1 and beyond as the joint position moves within the threshold
    distance to the limit and then exceeds the limit. The threshold is part of the configuration.
    The 'actuation' component returns quantities from the actuation model. What these are depends on the specific
    implementation.
    The configuration dictionary should have the form::

        {
            'components': [list, of, components],
            'threshold': threshold_value,
        }

    Joint positions, velocities and limits are read from the simulation state directly, joint torques uses torque
    sensors placed between bodies in the scene. By default, MIMo has one sensor for each joint. Any torque sensor with
    the 'proprio' prefix is used for the output.

    Attributes:
        env: The environment to which this module will be attached.
        proprio_parameters: A dictionary containing the configuration.
        output_components: A list containing all the proprioceptive components that should be put in the output. This
            attribute is populated by `proprio_parameters`. These components must be in :attr:`VALID_COMPONENTS`.
        sensor_outputs: A dictionary containing the outputs produced by the sensors. Contains one entry for each output
            component. Populated by :meth:`.get_proprioception_obs`.
        sensors: A list containing all the torque sensors.
        sensor_names: A dictionary of lists that can be used to find the joint/sensor of the associated entry in the
            output. The ith value in the joint position output belongs to joint sensor_names['qpos'][i].
        limit_thresh: Threshold distance to joint limit. If the joint is more than this distance away from the limit,
            the output will be 0. Default value is .035
    """

    #: Valid entries for the output components
    VALID_COMPONENTS = ["velocity", "torque", "limits", "actuation"]

    def __init__(self, env, proprio_parameters):
        super().__init__(env, proprio_parameters)

        self.sensors = []
        self.sensor_names = {}

        for sensor_name in self.env.sim.model.sensor_names:
            if sensor_name.startswith("proprio:"):
                self.sensors.append(sensor_name)

        self.joint_names = [name for name in self.env.sim.model.joint_names if name.startswith("robot:")]
        self.joint_ids = [self.env.sim.model.joint_name2id(name) for name in self.joint_names]
        self.joint_qpos = np.asarray([get_joint_qpos_addr(self.env.sim.model, idx) for idx in self.joint_ids])
        self.joint_qvel = np.asarray([get_joint_qvel_addr(self.env.sim.model, idx) for idx in self.joint_ids])
        self.joint_limits = self.env.sim.model.jnt_range[self.joint_ids]

        self.sensor_ids = [self.env.sim.model.sensor_name2id(name) for name in self.sensors]
        self.sensor_addrs = np.asarray([get_sensor_addr(self.env.sim.model, idx) for idx in self.sensor_ids])

        self.sensor_names["qpos"] = self.joint_names
        if "velocity" in self.output_components:
            self.sensor_names["qvel"] = self.joint_names
        if "torque" in self.output_components:
            self.sensor_names["torque"] = self.sensors
        if "limits" in self.output_components:
            self.sensor_names["limit"] = self.joint_names

        if proprio_parameters is not None and "threshold" in proprio_parameters:
            self.limit_thresh = proprio_parameters["threshold"]
        else:
            self.limit_thresh = .035  # ~2 degrees in radians

    def get_proprioception_obs(self):
        """ Produce the proprioceptive sensor outputs.

        Collects the proprioceptive output according to the sensor components. :attr:`~.sensor_outputs` is populated by
        this function. The dictionary will always contain an entry 'qpos' with the joint positions and optionally
        includes 'torques', 'limits', and 'qvel' for the joint velocities. The return value is a concatenation of these
        entries.

        Returns:
            A numpy array containing the concatenation of all enabled outputs.

        """
        self.sensor_outputs = {}
        robot_qpos = self.env.sim.data.qpos[self.joint_qpos].flatten()
        self.sensor_outputs["qpos"] = robot_qpos
        if "velocity" in self.output_components:
            robot_qvel = self.env.sim.data.qvel[self.joint_qvel].flatten()
            self.sensor_outputs["qvel"] = robot_qvel
        if "torque" in self.output_components:
            torques = self.env.sim.data.sensordata[self.sensor_addrs].flatten()
            self.sensor_outputs["torques"] = torques

        # Limit sensor outputs 0 while the joint position is more than _limit_thresh away from its limits, then scales
        # from 0 to 1 at the limit and then beyond 1 beyond the limit
        if "limits" in self.output_components:
            l_dif = robot_qpos - (self.joint_limits[:, 0] + self.limit_thresh)
            u_dif = (self.joint_limits[:, 1] - self.limit_thresh) - robot_qpos
            response = np.minimum(l_dif, u_dif) / self.limit_thresh
            response = - np.minimum(response, 0)
            self.sensor_outputs["limits"] = response

        if "actuation" in self.output_components:
            self.sensor_outputs["actuation"] = self.env.actuation_model.observations().flatten()

        return np.concatenate([self.sensor_outputs[key] for key in sorted(self.sensor_outputs.keys())])
