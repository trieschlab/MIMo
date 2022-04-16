import numpy as np
from mimoEnv.utils import get_data_for_sensor


class Proprioception:

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
        raise NotImplementedError


class SimpleProprioception(Proprioception):

    VALID_COMPONENTS = ["velocity", "torque", "limits"]

    def __init__(self, env, proprio_parameters):
        super().__init__(env, proprio_parameters)

        self.sensors = []
        self.sensor_names = {}

        for sensor_name in self.env.sim.model.sensor_names:
            if sensor_name.startswith("proprio:"):
                self.sensors.append(sensor_name)

        self.sensor_names = {}
        self.joint_names = [name for name in self.env.sim.model.joint_names if name.startswith("robot")]

        self.sensor_names["qpos"] = self.joint_names
        if "velocity" in self.output_components:
            self.sensor_names["qvel"] = self.joint_names
        if "torque" in self.output_components:
            self.sensor_names["torque"] = self.sensors
        if "limits" in self.output_components:
            self.sensor_names["limit"] = self.joint_names

        self._limit_thresh = .035  # ~2 degrees in radians

    def get_proprioception_obs(self):
        self.sensor_outputs = {}
        robot_qpos = np.array([self.env.sim.data.get_joint_qpos(name) for name in self.joint_names])
        self.sensor_outputs["qpos"] = robot_qpos
        if "velocity" in self.output_components:
            robot_qvel = np.array([self.env.sim.data.get_joint_qvel(name) for name in self.joint_names])
            self.sensor_outputs["qvel"] = robot_qvel
        if "torques" in self.output_components:
            torques = []
            for sensor in self.sensors:
                sensor_output = get_data_for_sensor(self.env.sim, sensor)
                # Convert from child to parent frame? Report torque in terms of the axis of the relevant joints?
                torques.append(sensor_output)
            torques = np.concatenate(torques)
            self.sensor_outputs["torques"] = torques

        # Limit sensor outputs 0 while the joint position is more than _limit_thresh away from its limits, then scales
        # from 0 to 1 at the limit and then beyond 1 beyond the limit
        if "limits" in self.output_components:
            limits = []
            for i, joint_name in enumerate(self.joint_names):
                joint_id = self.env.sim.model.joint_name2id(joint_name)
                joint_limits = self.env.sim.model.jnt_range[joint_id]
                joint_position = robot_qpos[i]
                l_dif = joint_position - (joint_limits[0] + self._limit_thresh)
                u_dif = (joint_limits[1]-self._limit_thresh) - joint_position
                response = min(l_dif, u_dif) / self._limit_thresh  # Outputs negative of difference, scaled by thresh
                response = - min(response, 0)  # Clamp all positive values (have not reached the threshold) and invert
                limits.append(response)
            limit_response = np.asarray(limits)  # np.asarrays
            self.sensor_outputs["limits"] = limit_response

        return np.concatenate([self.sensor_outputs[key] for key in sorted(self.sensor_outputs.keys())])
