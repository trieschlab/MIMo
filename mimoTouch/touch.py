import math
import numpy as np
import mujoco_py

import mimoEnv.utils as env_utils
from mimoEnv.utils import mulRotT, mulRot, EPS
from mimoTouch.sensorpoints import spread_points_box, spread_points_sphere, spread_points_cylinder, \
                                   spread_points_capsule

# Class that handles all of this
#   Initialized as part of gym env, links to gym env
#   Function to add a sensing body/geom,
#   Adding a body that was already added overwrites existing sensor points
#   Figures out where to put sensor points
#   Stores sensor locations for each geom
#   Function to get normal vectors at sensor points
#   Function to read out contacts
#   TODO: Convenience functions for cleanup of automatic sensor placement: Remove sensors located within another geom
#   TODO: Function for "slippage" at sensor point: Direction + magnitude
#   Find nearest sensors:
#       TODO: Should consider surface of mesh, opposite side of thing object should not be considered
#   Function to adjust force based on distance between contact and sensor point
#   TODO: Rework (with trimesh?). Sensor points should have well defined frames with normal vector
#   TODO: Biologically accurate outputs/delays(?)

GEOM_TYPES = {"PLANE": 0, "HFIELD": 1, "SPHERE": 2, "CAPSULE": 3, "ELLIPSOID": 4, "CYLINDER": 5, "BOX": 6, "MESH": 7}


class Touch:

    VALID_TOUCH_TYPES = {}
    VALID_ADJUSTMENTS = []

    def __init__(self, env, touch_params):
        """
        env should be an openAI gym environment using mujoco. Critically env should have an attribute sim which is a
        mujoco-py sim object

        Sensor positions is a dictionary where each key is the index of a sensing object and the corresponding value
        is a numpy array storing the sensor positions on that object: {object_id: ndarray((n_sensors, 3))}
        Sensor positions should be in relative coordinates for the object.

        The sensor position dictionary should be populated when an object is added.

        The sensor scale dictionary stores the "scale" parameter from the input. This is used to determine the distance
        between sensor points and to to scale the force based on the distance between a contact and the sensor.
        touch_type, touch_function and touch_size all relate to the touch output of the sensors. touch_type determines
        the function used to compute the output, touch_size should match the size of that output for a single sensor.
        VALID_TOUCH_TYPES should thus contain function members of this class that can be used to compute sensor outputs,
        with the size of those outputs as values.
        adjustment_type and adjustment_function relate to the post processing that is done on the raw mujoco force.
        They operate like touch_type and touch_function, but for the post processing.
        Usually this would be used to spread the mujoco point contacts out over a larger area, to support "area"
        sensing.
        """
        self.env = env

        print(touch_params)
        self.sensor_scales = {}
        for body_name in touch_params["scales"]:
            body_id = env.sim.model.body_name2id(body_name)
            self.sensor_scales[body_id] = touch_params["scales"][body_name]

        # Get all information for the touch force function: Function name, reference to function, output size
        self.touch_type = touch_params["touch_function"]
        assert self.touch_type in self.VALID_TOUCH_TYPES
        assert hasattr(self, self.touch_type)
        self.touch_function = getattr(self, self.touch_type)
        self.touch_size = self.VALID_TOUCH_TYPES[self.touch_type]

        # Get all information for the surface adjustment function: function name, reference to function
        self.adjustment_type = touch_params["adjustment_function"]
        assert self.adjustment_type in self.VALID_ADJUSTMENTS
        self.adjustment_function = getattr(self, self.adjustment_type)

        self.sensor_outputs = {}

    def get_touch_obs(self):
        """

        :return: Touch obsevations
        """
        raise NotImplementedError


class DiscreteTouch(Touch):

    VALID_TOUCH_TYPES = {
        "normal": 1,
        "force_vector": 3,
        "force_vector_global": 3,
    }

    VALID_ADJUSTMENTS = ["nearest", "spread_linear"]

    def __init__(self, env, touch_params):
        """
        A specific implementation of the Touch class, that uses mujoco geoms as the basic sensor object. Sensor points
        are simply spread evenly over individual geoms, with no care taken for intersections. Nearest sensors are
        determined by direct euclidean distance.
        """
        super().__init__(env, touch_params)
        self.m_data = env.sim.data
        self.m_model = env.sim.model

        self.sensor_positions = {}
        self.plotting_limits = {}
        self.plots = {}

        # Add sensors to bodies
        for body_id in self.sensor_scales:
            self.add_body(body_id, scale=self.sensor_scales[body_id])

        # Get touch obs once to ensure all output arrays are initialized
        self.get_touch_obs()
        
    def add_body(self, body_id: int = None, body_name: str = None, scale: float = math.inf):
        """Adds sensors to all geoms belonging to the given body. Returns the number of sensor points added. Scale is
        the approximate distance between sensor points"""
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)

        n_sensors = 0

        for geom_id in env_utils.get_geoms_for_body(self.m_model, body_id):
            g_body_id = self.m_model.geom_bodyid[geom_id]
            contype = self.m_model.geom_contype[geom_id]
            # Add a geom if it belongs to body and has collisions enabled (at least potentially)
            if g_body_id == body_id and contype > 0:
                n_sensors += self.add_geom(geom_id=geom_id, scale=scale)
        return n_sensors

    def add_geom(self, geom_id: int = None, geom_name: str = None, scale: float = math.inf):
        geom_id = env_utils.get_geom_id(self.m_model, geom_id=geom_id, geom_name=geom_name)

        if self.m_model.geom_contype[geom_id] == 0:
            raise RuntimeWarning("Added sensors to geom with collisions disabled!")
        return self._add_sensorpoints(geom_id, scale)

    @property
    def sensing_geoms(self):
        """ Returns the ids of all geoms with sensors """
        return list(self.sensor_positions.keys())

    def has_sensors(self, geom_id):
        """ Returns true if the geom has sensors """
        return geom_id in self.sensor_positions

    def get_sensor_count(self, geom_id):
        """ Returns the number of sensors for the geom """
        return self.sensor_positions[geom_id].shape[0]

    def get_total_sensor_count(self):
        """ Returns the total number of haptic sensors in the model """
        n_sensors = 0
        for geom_id in self.sensing_geoms:
            n_sensors += self.get_sensor_count(geom_id)
        return n_sensors

    def _add_sensorpoints(self, geom_id: int, scale: float):
        # Add sensor points for the given geom using given resolution
        # Returns the number of sensor points added
        # Also set the maximum size of the geom, for plotting purposes
        geom_type = self.m_model.geom_type[geom_id]
        size = self.m_model.geom_size[geom_id]
        limit = 1
        if geom_type == GEOM_TYPES["BOX"]:
            limit = np.max(size)
            points = spread_points_box(scale, size)
        elif geom_type == GEOM_TYPES["SPHERE"]:
            limit = size[0]
            points = spread_points_sphere(scale, size[0])
        elif geom_type == GEOM_TYPES["CAPSULE"]:
            limit = size[1] + size[0]
            points = spread_points_capsule(scale, 2*size[1], size[0])
        elif geom_type == GEOM_TYPES["CYLINDER"]:
            # Cylinder size 0 is radius, size 1 is half length
            limit = np.max(size)
            points = spread_points_cylinder(scale, 2*size[1], size[0])
        elif geom_type == GEOM_TYPES["PLANE"]:
            RuntimeWarning("Cannot add sensors to plane geoms!")
            return None
        elif geom_type == GEOM_TYPES["ELLIPSOID"]:
            raise NotImplementedError("Ellipsoids currently not implemented")
        elif geom_type == GEOM_TYPES["MESH"]:
            size = self.m_model.geom_rbound[geom_id]
            limit = size
            points = spread_points_sphere(scale, size)
        else:
            return None

        self.plotting_limits[geom_id] = limit
        self.sensor_positions[geom_id] = points

        return points.shape[0]  # Return the number of points we added

    def get_nearest_sensor(self, contact_id, geom_id):
        """ Given a contact and a geom, return the sensor on the geom closest to the contact.
        Returns the sensor index and the distance between contact and sensor"""
        relative_position = self.get_contact_position_relative(contact_id, geom_id)
        sensor_points = self.sensor_positions[geom_id]
        distances = np.linalg.norm(sensor_points - relative_position, axis=1)
        idx = np.argmin(distances)
        return idx, distances[idx]

    def get_k_nearest_sensors(self, contact_id, geom_id, k):
        relative_position = self.get_contact_position_relative(contact_id, geom_id)
        sensor_points = self.sensor_positions[geom_id]
        distances = np.linalg.norm(sensor_points - relative_position, axis=1)
        sorted_idxs = np.argpartition(distances, k)
        return sorted_idxs[:k], distances[sorted_idxs[:k]]

    def get_sensors_within_distance(self, contact_id, geom_id, distance):
        relative_position = self.get_contact_position_relative(contact_id, geom_id)
        sensor_points = self.sensor_positions[geom_id]
        distances = np.linalg.norm(sensor_points - relative_position, axis=1)
        within_distance = distances < distance
        idx_within_distance = within_distance.nonzero()[0]
        return idx_within_distance, distances[within_distance]

    # ======================== Positions and rotations ================================
    # =================================================================================

    def get_contact_position_world(self, contact_id):
        """ Get the position of a contact in world frame """
        return self.m_data.contact[contact_id].pos

    def get_contact_position_relative(self, contact_id, geom_id: int):
        """ Get the position of a contact in the geom frame """
        return env_utils.world_pos_to_geom(self.m_data, self.get_contact_position_world(contact_id), geom_id)

    # =============== Visualizations ==================================================
    # =================================================================================

    # Plot sensor points for single geom
    def plot_sensors_geom(self, geom_id: int = None, geom_name: str = None):
        geom_id = env_utils.get_geom_id(self.m_model, geom_id=geom_id, geom_name=geom_name)

        points = self.sensor_positions[geom_id]
        limit = self.plotting_limits[geom_id]
        title = self.m_model.geom_id2name(geom_id)
        env_utils.plot_points(points, limit=limit, title=title)

    # Plot forces for single geom
    def plot_force_geom(self, geom_id: int = None, geom_name: str = None):
        geom_id = env_utils.get_geom_id(self.m_model, geom_id=geom_id, geom_name=geom_name)

        sensor_points = self.sensor_positions[geom_id]
        force_vectors = self.sensor_outputs[geom_id]
        if force_vectors.shape[1] == 1:
            # TODO: Need proper sensor normals, can't do this until trimesh rework
            raise RuntimeWarning("Plotting of scalar forces not implemented!")
        else:
            env_utils.plot_forces(sensor_points, force_vectors, limit=np.max(sensor_points) + 0.5)

    def _get_plot_info_body(self, body_id):
        points = []
        forces = []
        for geom_id in env_utils.get_geoms_for_body(self.m_model, body_id):
            points_in_body = env_utils.geom_pos_to_body(self.m_data, self.sensor_positions[geom_id], geom_id, body_id)
            points.append(points_in_body)
            force_vectors = self.sensor_outputs[geom_id] / 100
            if force_vectors.shape[1] == 1:
                # TODO: Need proper sensor normals, can't do this until trimesh rework
                raise RuntimeWarning("Plotting of scalar forces not implemented!")
            world_forces = mulRot(np.transpose(force_vectors), env_utils.get_geom_rotation(self.m_data, geom_id))
            body_forces = np.transpose(mulRotT(world_forces, env_utils.get_body_rotation(self.m_data, body_id)))
            forces.append(body_forces)
        points_t = np.concatenate(points)
        forces_t = np.concatenate(forces)
        return points_t, forces_t

    # Plot forces for single body
    def plot_force_body(self, body_id: int = None, body_name: str = None):
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)

        points, forces = self._get_plot_info_body(body_id)
        # For every geom: Convert sensor points and forces to body frame ->

        env_utils.plot_forces(points, forces, limit=np.max(points) + 0.5)

    # TODO: Plot forces for body subtree

    # =============== Raw force and contact normal ====================================
    # =================================================================================

    def get_raw_force(self, contact_id, geom_id):
        """ Returns the full contact force in mujocos own contact frame. Output is a 3-d vector"""
        forces = np.zeros(6, dtype=np.float64)
        mujoco_py.functions.mj_contactForce(self.m_model, self.m_data, contact_id, forces)
        contact = self.m_data.contact[contact_id]
        if geom_id == contact.geom1:
            forces *= -1  # Convention is that normal points away from geom1
        elif geom_id == contact.geom2:
            pass
        else:
            RuntimeError("Mismatch between contact and geom")
        return forces[:3]

    def get_contact_normal(self, contact_id, geom_id):
        """ Returns the normal vector (unit vector in direction of normal) in geom frame"""
        contact = self.m_data.contact[contact_id]
        normal_vector = contact.frame[:3]
        if geom_id == contact.geom1:  # Mujoco vectors point away from geom1 by convention
            normal_vector *= -1
        elif geom_id == contact.geom2:
            pass
        else:
            RuntimeError("Mismatch between contact and geom")
        # contact frame is in global coordinate frame, rotate to geom frame
        normal_vector = mulRot(normal_vector, env_utils.get_geom_rotation(self.m_data, geom_id))
        return normal_vector

    # =============== Valid touch force functions =====================================
    # =================================================================================

    def normal(self, contact_id, geom_id) -> float:
        """ Returns the normal force as a scalar"""
        return self.get_raw_force(contact_id, geom_id)[0]

    def force_vector_global(self, contact_id, geom_id):
        """ Returns full contact force in world frame. Output is a 3-d vector"""
        contact = self.m_data.contact[contact_id]
        forces = self.get_raw_force(contact_id, geom_id)
        force_rot = np.reshape(contact.frame, (3, 3))
        global_forces = mulRotT(forces, force_rot)
        return global_forces

    def force_vector(self, contact_id, geom_id):
        """ Returns full contact force in the frame of the geom. Output is a 3-d vector"""
        global_forces = self.force_vector_global(contact_id, geom_id)
        relative_forces = mulRotT(global_forces, env_utils.get_geom_rotation(self.m_data, geom_id))
        return relative_forces

    # =============== Output functions ================================================
    # =================================================================================

    def get_contacts(self):
        """ Returns a tuple containing (contact_id, geom_id, forces) for each active contact with a sensing geom,
        where contact_id is the index of the contact in the mujoco arrays, geom_id is the index of the geom and forces
        is a numpy array of the raw output force, determined by self.touch_type"""
        contact_tuples = []
        for i in range(self.m_data.ncon):
            contact = self.m_data.contact[i]
            # Do we sense this contact at all
            if self.has_sensors(contact.geom1) or self.has_sensors(contact.geom2):
                rel_geoms = []
                if self.has_sensors(contact.geom1):
                    rel_geoms.append(contact.geom1)
                if self.has_sensors(contact.geom2):
                    rel_geoms.append(contact.geom2)

                raw_forces = self.get_raw_force(i, contact.geom1)
                if abs(raw_forces[0]) < 1e-9:  # Contact probably inactive
                    continue

                for rel_geom in rel_geoms:
                    forces = self.touch_function(i, rel_geom)
                    contact_tuples.append((i, rel_geom, forces))

        return contact_tuples

    def get_empty_sensor_dict(self, size):
        """ Returns a dictionary with empty sensor outputs. Keys are geom ids, corresponding values are the output
        arrays. For every geom with sensors, returns an empty numpy array of shape (n_sensors, size)"""
        sensor_outputs = {}
        for geom_id in self.sensor_positions:
            sensor_outputs[geom_id] = np.zeros((self.get_sensor_count(geom_id), size), dtype=np.float32)
        return sensor_outputs

    def flatten_sensor_dict(self, sensor_dict):
        """ Flattens a sensor dict, such as from get_empty_sensor_dict into a single large array in a deterministic
        fashion. Geoms with lower id come earlier, sensor outputs correspond to sensor_positions"""
        sensor_arrays = []
        for geom_id in sorted(self.sensor_positions):
            sensor_arrays.append(sensor_dict[geom_id])
        return np.concatenate(sensor_arrays)

    def get_touch_obs(self) -> np.ndarray:
        """ Does the full contact getting-processing process, such that we get the forces, as determined by the touch
        type and the adjustments, for each sensor.
        """
        contact_tuples = self.get_contacts()
        self.sensor_outputs = self.get_empty_sensor_dict(self.touch_size)  # Initialize output dictionary

        for contact_id, geom_id, forces in contact_tuples:
            # At this point we already have the forces for each contact, now we must attach/spread them to sensor
            # points, based on the adjustment function
            self.adjustment_function(contact_id, geom_id, forces)

        sensor_obs = self.flatten_sensor_dict(self.sensor_outputs)
        return sensor_obs

    # =============== Force adjustment functions ======================================
    # =================================================================================

    def spread_linear(self, contact_id, geom_id, force):
        # Get all sensors within distance (distance here is just double the sensor scale)
        body_id = self.m_model.geom_bodyid[geom_id]
        scale = self.sensor_scales[body_id]
        nearest_sensors, sensor_distances = self.get_sensors_within_distance(contact_id, geom_id, 2*scale)

        adjusted_forces = {}
        force_total = np.zeros(force.shape)
        for sensor_id, distance in zip(nearest_sensors, sensor_distances):
            sensor_adjusted_force = scale_linear(force, distance, scale=2*scale)
            force_total += sensor_adjusted_force
            adjusted_forces[sensor_id] = sensor_adjusted_force

        factors = force / (force_total + EPS)  # Add very small value to avoid divide by zero errors
        for sensor_id in adjusted_forces:
            rescaled_sensor_adjusted_force = adjusted_forces[sensor_id] * factors
            self.sensor_outputs[geom_id][sensor_id] += rescaled_sensor_adjusted_force

    def nearest(self, contact_id, geom_id, force):
        # Get the nearest sensor to this contact, add the force to it
        nearest_sensor, distance = self.get_nearest_sensor(contact_id, geom_id)
        self.sensor_outputs[geom_id][nearest_sensor] += force


# =============== Scaling functions ===============================================
# =================================================================================

def scale_linear(force, distance, scale, **kwargs):
    """ Adjusts the force by a simple factor, such that force falls linearly from full at distance = 0
    to 0 at distance >= scale"""
    factor = (scale-distance) / scale
    if factor < 0:
        factor = 0
    out_force = force * factor
    return out_force
