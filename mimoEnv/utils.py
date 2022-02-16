
def get_data_for_sensor(sim, sensor_name):
    """ Get sensor data for sensor sensor_name"""
    sensor_id = sim.model.sensor_name2id(sensor_name)
    start = sim.model.sensor_adr[sensor_id]
    end = start + sim.model.sensor_dim[sensor_id]
    return sim.data.sensordata[start:end]
