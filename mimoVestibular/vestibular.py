import numpy as np
from mimoEnv.utils import get_data_for_sensor


class Vestibular:
    def __init__(self, env, vestibular_parameters):
        self.env = env
        self.sensors = vestibular_parameters["sensors"]
        self.sensor_outputs = {}

    def get_vestibular_obs(self):
        raise NotImplementedError


class SimpleVestibular(Vestibular):
    def __init__(self, env, vestibular_parameters):
        super().__init__(env, vestibular_parameters)


    def get_vestibular_obs(self):
        data = []
        for sensor in self.sensors:
            sensor_output = get_data_for_sensor(self.env.sim, sensor)
            data.append(sensor_output)
        self.sensor_outputs = data
        return np.concatenate(data)
