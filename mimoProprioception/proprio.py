import numpy as np
from mimoEnv.utils import get_data_for_sensor

class Proprioception:
    def __init__(self, env, proprio_parameters):
        self.env = env
        self.parameters = proprio_parameters
        self.sensors = []

    def get_proprioception_obs(self):
        raise NotImplementedError


class SimpleProprioception(Proprioception):
    def __init__(self, env, proprio_parameters):
        super().__init__(env, proprio_parameters)

        for sensor_name in self.env.sim.model.sensor_names:
            if sensor_name.startswith("proprio:"):
                self.sensors.append(sensor_name)

        self.obs = {}

        self.sensor_names = {}
        self.joint_names = [name for name in self.env.sim.model.joint_names if name.startswith("robot")]
        self.sensor_names["qpos"] = self.joint_names
        self.sensor_names["qvel"] = self.joint_names
        self.sensor_names["torque"] = self.sensors

    def get_proprioception_obs(self):
        robot_qpos = np.array([self.env.sim.data.get_joint_qpos(name) for name in self.joint_names])
        robot_qvel = np.array([self.env.sim.data.get_joint_qvel(name) for name in self.joint_names])
        torques = []
        for sensor in self.sensors:
            sensor_output = get_data_for_sensor(self.env.sim, sensor)
            # Convert from child to parent frame? Report torque in terms of the axis of the relevant joints?
            torques.append(sensor_output)
        torques = np.concatenate(torques)
        self.obs = {
            "qpos": robot_qpos,
            "qvel": robot_qvel,
            "torques": torques
        }
        return np.concatenate([self.obs[key] for key in self.obs])
