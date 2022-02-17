import math
import numpy as np
import mujoco_py
from matplotlib import pyplot as plt

from mimoEnv.utils import mulRotT, mulRot, plot_forces, get_geoms_for_body, get_geom_rotation, get_body_rotation, \
                           world_pos_to_geom, geom_pos_to_body, EPS
from mimoTouch.sensorpoints import spread_points_box, spread_points_sphere, spread_points_cylinder, spread_points_capsule

# Class that handles all of this
#   Initialized as part of gym env, links to gym env
#   Function to add a sensing body/geom,
#   Adding a body that was already added overwrites existing sensor points
#   Figures out where to put sensor points
#   Stores sensor locations for each geom
#   Function to get normal vectors at sensor points
#   TODO: Handles observation space automatically (???)
#   Function to read out contacts
#   TODO: Convenience functions for cleanup of automatic sensor placement: Remove sensors located within another geom
#   TODO: Function for "slippage" at sensor point: Direction + magnitude
#   Find nearest sensors:
#       TODO: Should consider surface of mesh, opposite side of thing object should not be considered
#   Function to adjust force based on distance between contact and sensor point
#   TODO: Rework (with trimesh?). Sensor points should have well defined frames with normal vector
#   TODO: Biologically accurate outputs/delays(?)
#   TODO: Properly encapsulate the sensor plotting


GEOM_TYPES = {"PLANE": 0, "HFIELD": 1, "SPHERE": 2, "CAPSULE": 3, "ELLIPSOID": 4, "CYLINDER": 5, "BOX": 6, "MESH": 7}
CONTACT_NEAREST_K = 3


class DiscreteTouch:

    def __init__(self, env, verbose=False):
        """
        env should be an openAI gym environment using mujoco. Critically env should have an attribute sim which is a
        mujoco-py sim object

        Sensor positions is a dictionary where each key is the index of a geom with sensors and the corresponding value
        is a numpy array storing the sensor positions on that geom: {geom_id: ndarray((n_sensors, 3))}
        Sensor positions are in relative coordinates for the geom.

        The sensor position dictionary is populated when a body/geom is added and can be overwritten afterwards.

        sensor_params stores geom-specific sensor parameters, such as the scale. This is used to scale the force
        based on the distance between a contact and the sensor
        """
        self.env = env
        self.m_data = env.sim.data
        self.m_model = env.sim.model

        self.sensor_positions = {}
        self.sensor_params = {}
        self.sensor_outputs = {}

        self.plotting_limits = {}
        self.plots = {}
        self.verbose = verbose
        
    def add_body(self, body_id: int = None, body_name: str = None, scale: float = math.inf):
        """Adds sensors to all geoms belonging to the given body. Returns the number of sensor points added. Scale is
        the approximate distance between sensor points"""
        if body_id is None and body_name is None:
            raise RuntimeError("DiscreteTouch.add_body called without name or id")
            
        if body_id is None:
            body_id = self.m_model.body_name2id(body_name)

        n_sensors = 0

        for geom_id in get_geoms_for_body(self.m_model, body_id):
            g_body_id = self.m_model.geom_bodyid[geom_id]
            contype = self.m_model.geom_contype[geom_id]
            # Add a geom if it belongs to body and has collisions enabled (at least potentially)
            if g_body_id == body_id and contype > 0:
                n_sensors += self.add_geom(geom_id=geom_id, scale=scale)
        return n_sensors

    def add_geom(self, geom_id: int = None, geom_name: str = None, scale: float = math.inf):
        if geom_id is None and geom_name is None:
            raise RuntimeError("DiscreteTouch.add_geom called without name or id")
            
        if geom_id is None:
            geom_id = self.m_model.geom_name2id(geom_name)

        if self.m_model.geom_contype[geom_id] == 0:
            raise RuntimeWarning("Added sensors to geom with collisions disabled!")

        if self.verbose:
            print("Geom {} name {} type {} ".format(
                  geom_id,
                  self.m_model.geom_id2name(geom_id),
                  self.m_model.geom_type[geom_id]))

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
        if self.verbose:
            print("Added {} points to geom".format(points.shape[0]))

        self.plotting_limits[geom_id] = limit
        self.sensor_positions[geom_id] = points
        if geom_id in self.sensor_params:
            self.sensor_params[geom_id]["scale"] = 2*scale
        else:
            self.sensor_params[geom_id] = {"scale": 2*scale}

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

    # ======================== Positions and rotations ================================
    # =================================================================================

    def get_contact_position_world(self, contact_id):
        """ Get the position of a contact in world frame """
        return self.m_data.contact[contact_id].pos

    def get_contact_position_relative(self, contact_id, geom_id: int):
        """ Get the position of a contact in the geom frame """
        return world_pos_to_geom(self.m_data, self.get_contact_position_world(contact_id), geom_id)

    # =============== Visualizations ==================================================
    # =================================================================================

    # Plot forces for single geom
    def plot_force_geom(self, geom_id: int = None, geom_name: str = None):
        if geom_id is None and geom_name is None:
            raise RuntimeError("DiscreteTouch.add_geom called without name or id")

        if geom_id is None:
            geom_id = self.m_model.geom_name2id(geom_name)

        sensor_points = self.sensor_positions[geom_id]
        force_vectors = self.sensor_outputs[geom_id]
        if force_vectors.shape[1] == 1:
            # TODO: Need proper sensor normals, can't do this until trimesh rework
            raise RuntimeWarning("Plotting of scalar forces not implemented!")
        else:
            plot_forces(sensor_points, force_vectors, limit=np.max(sensor_points) + 0.5)

    def _get_plot_info_body(self, body_id):
        points = []
        forces = []
        for geom_id in get_geoms_for_body(self.m_model, body_id):
            points_in_body = geom_pos_to_body(self.m_data, self.sensor_positions[geom_id], geom_id, body_id)
            points.append(points_in_body)
            force_vectors = self.sensor_outputs[geom_id] / 10000
            if force_vectors.shape[1] == 1:
                # TODO: Need proper sensor normals, can't do this until trimesh rework
                raise RuntimeWarning("Plotting of scalar forces not implemented!")
            world_forces = mulRot(np.transpose(force_vectors), get_geom_rotation(self.m_data, geom_id))
            body_forces = np.transpose(mulRotT(world_forces, get_body_rotation(self.m_data, body_id)))
            forces.append(body_forces)
        points_t = np.concatenate(points)
        forces_t = np.concatenate(forces)
        return points_t, forces_t

    # TODO: Plot forces for single body
    def plot_force_body(self, body_id: int = None, body_name: str = None):

        if body_id is None and body_name is None:
            raise RuntimeError("DiscreteTouch.add_body called without name or id")

        if body_id is None:
            body_id = self.m_model.body_name2id(body_name)

        points, forces = self._get_plot_info_body(body_id)
        # For every geom: Convert sensor points and forces to body frame ->

        plot_forces(points, forces, limit=np.max(points) + 0.5)

    # TODO: Plot forces for body subtree

    # =============== Various types of forces in various frames =======================
    # =================================================================================

    def get_force(self, contact_id, geom_id):
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

    def get_force_normal(self, contact_id, geom_id) -> float:
        """ Returns the normal force as a scalar"""
        return self.get_force(contact_id, geom_id)[0]

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
        normal_vector = mulRot(normal_vector, get_geom_rotation(self.m_data, geom_id))
        return normal_vector

    def get_force_world(self, contact_id, geom_id):
        """ Returns full contact force in world frame. Output is a 3-d vector"""
        contact = self.m_data.contact[contact_id]
        forces = self.get_force(contact_id, geom_id)
        force_rot = np.reshape(contact.frame, (3, 3))
        global_forces = mulRotT(forces, force_rot)
        return global_forces

    def get_force_relative(self, contact_id, geom_id):
        """ Returns full contact force in geom frame. Output is a 3-d vector"""
        global_forces = self.get_force_world(contact_id, geom_id)
        relative_forces = mulRotT(global_forces, get_geom_rotation(self.m_data, geom_id))
        return relative_forces

    def _adjust_force_for_sensor(self, force, contact_id, geom_id, sensor_id, adjustment_function, adjustment_params):
        """
        Adjusts contact forces based on the distance between the contact and the sensor. Additional parameters for the
        adjustment function can be passed using adjustment_params.
        i.e. output force = adjustment_function(force, distance, **adjustment_params)
        """
        contact_position = self.get_contact_position_relative(contact_id, geom_id)
        sensor_position = self.sensor_positions[geom_id][sensor_id]
        distance = np.linalg.norm(contact_position - sensor_position, ord=2)
        return adjustment_function(force, distance, **adjustment_params)

    def get_contacts(self, verbose: bool = None):
        """ Returns a tuple containing (contact_id, geom_id, sensor_ids) for each active contact with a sensing geom,
        where contact_id is the index of the contact in the mujoco arrays, geom_id is the index of the geom in the
        arrays and sensor_ids are the indices of the k sensors on the geom nearest to the contact"""
        if verbose is None:
            verbose = self.verbose
        contact_geom_tuples = []
        for i in range(self.m_data.ncon):
            contact = self.m_data.contact[i]
            # Do we sense this contact at all
            if self.has_sensors(contact.geom1) or self.has_sensors(contact.geom2):
                rel_geoms = []
                if self.has_sensors(contact.geom1):
                    rel_geoms.append(contact.geom1)
                if self.has_sensors(contact.geom2):
                    rel_geoms.append(contact.geom2)

                forces = self.get_force(i, contact.geom1)
                if abs(forces[0]) < 1e-8:  # Contact probably inactive
                    continue

                for rel_geom in rel_geoms:
                    rel_pos = self.get_contact_position_relative(i, rel_geom)
                    nearest_sensor_ids, distances = self.get_k_nearest_sensors(i, rel_geom, CONTACT_NEAREST_K)
                    contact_geom_tuples.append((i, rel_geom, nearest_sensor_ids))

                    if verbose:
                        print("Sensing with geom: ", rel_geom, self.m_model.geom_id2name(rel_geom))
                        print("Relative position: ", rel_pos)
                        print("Nearest sensors {} at position {} with distances {}".format(
                              nearest_sensor_ids,
                              self.sensor_positions[rel_geom][nearest_sensor_ids],
                              distances))
        return contact_geom_tuples

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

    def get_touch_obs(self, force_function, sensor_shape, adjustment_function=None) -> np.ndarray:
        """ Does the full contact getting-processing process, such that we get the force as a full vector on each sensor
        point for every sensor in the model.

        Forces are adjusted to nearby sensors using adjustment function. See _adjust_force_for_sensor for more.
        If adjustment function is None we default to linear scaling.
        sensor_shape must match the size of a single sensor output for the given force function,
        i.e. if the return is the normal force as a scalar sensor_shape should be 1, if it is a 6-d vector sensor_shape
        should be 6

        Returns a numpy array of shape (n_sensors_total, 3)"""
        contact_tuples = self.get_contacts(verbose=False)
        sensor_outputs = self.get_empty_sensor_dict(sensor_shape)

        if adjustment_function is None:
            adjustment_function = scale_linear

        for contact_id, geom_id, nearest_sensor_ids in contact_tuples:
            rel_forces = force_function(self, contact_id, geom_id)
            adjusted_forces = {}
            force_total = np.zeros(rel_forces.shape)
            for sensor_id in nearest_sensor_ids:
                sensor_adjusted_force = self._adjust_force_for_sensor(rel_forces, contact_id, geom_id, sensor_id,
                                                                      adjustment_function, self.sensor_params[geom_id])
                force_total += sensor_adjusted_force
                adjusted_forces[sensor_id] = sensor_adjusted_force

            factors = rel_forces / (force_total + EPS)   # Add very small value to avoid divide by zero errors
            for sensor_id in adjusted_forces:
                rescaled_sensor_adjusted_force = adjusted_forces[sensor_id] * factors
                sensor_outputs[geom_id][sensor_id] += rescaled_sensor_adjusted_force

        self.sensor_outputs = sensor_outputs
        sensor_obs = self.flatten_sensor_dict(sensor_outputs)
        return sensor_obs


# =============== Force adjustment functions ======================================
# =================================================================================

def scale_linear(force, distance, scale, **kwargs):
    """ Adjusts the force by a simple factor, such that force falls linearly from full at distance = 0
    to 0 at distance >= scale"""
    factor = (scale-distance) / scale
    if factor < 0:
        factor = 0
    out_force = force * factor
    return out_force
