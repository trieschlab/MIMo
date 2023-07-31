""" This module defines the touch system interface and provides two implementations.

The interface is defined as an abstract class in :class:`~mimoTouch.touch.Touch`.
A simple implementation with a cloud of sensor points is in :class:`~mimoTouch.touch.DiscreteTouch`.
A second implementation using trimesh objects is in :class:`~mimoTouch.touch.TrimeshTouch`.
This second implementation allows for consideration of sensor normals as well as surface distance, avoiding the issue
of a contact penetrating through to sensors on the opposite side of the sensing body.

Both of the implementations also have functions for visualizing the touch sensations.
"""

import math
import operator
from collections import deque
from typing import Dict, List, Tuple, Callable

import numpy as np
import mujoco
import trimesh
from trimesh import PointCloud
from cachetools import cachedmethod, LRUCache
import CacheToolsUtils
from matplotlib import pyplot as plt

import mimoEnv.utils as env_utils
from mimoEnv.utils import rotate_vector_transpose, rotate_vector, EPS
from mimoTouch.sensorpoints import spread_points_box, spread_points_sphere, spread_points_cylinder, \
                                   spread_points_capsule

#: A key to identify the geom type ids used by MuJoCo.
from mimoTouch.sensormeshes import mesh_box, mesh_sphere, mesh_capsule, mesh_cylinder, mesh_ellipsoid

from gymnasium.envs.mujoco import MujocoEnv


class Touch:
    """ Abstract base class for the touch system.

    This class defines the functions that all implementing classes must provide. :meth:`.get_touch_obs` should perform
    the whole sensory pipeline as defined in the configuration and return the output as a single array.
    Additionally, the output for each body part should be stored in :attr:`.sensor_outputs`. The exact definition of
    'body part' is left to the implementing class.

    The constructor takes two arguments, `env` and `touch_params`:
    `env` should be an openAI gym environment using MuJoCo, while `touch_params` is a configuration dictionary. The
    exact form will depend on the specific implementation, but it must contain these three entries:

    - 'scales', which lists the distance between sensor points for each body part.
    - 'touch_function', which defines the output type and must be in :attr:`.VALID_TOUCH_TYPES`.
    - 'response_function', which defines how the contact forces are distributed to the sensors. Must be one of
      :attr:`.VALID_RESPONSE_FUNCTIONS`.

    The sensor scales determines the density of the sensor points, while 'touch_function' and 'response_function'
    determine the type of contact output and how it is distributed. 'touch_function' and 'response_function' refer to
    implementation methods by name and must be listed in :attr:`.VALID_TOUCH_TYPES`. and
    :attr:`.VALID_RESPONSE_FUNCTIONS` respectively. Different touch functions should be used to support different types
    of output, such as normal force, frictional forces or contact slip. The purpose of the response function is to
    loosely simulate surface behaviour. How exactly these functions work and interact is left to the implementing class.
    Note that the bodies listed in 'scales' must actually exist in the scene to avoid errors!

    Attributes:
        env (MujocoEnv): The environment to which this module will be attached.
        sensor_scales (Dict[int, float]): A dictionary listing the sensor distances for each body part. Populated from
            `touch_params`.
        touch_type (str): The name of the member method that determines output type. Populated from `touch_params`.
        touch_function (Callable): A reference to the actual member method determined by `touch_type`.
        touch_size (int): The size of the output of a single sensor for the given touch type.
        response_type (str): The name of the member method that determines how the output is distributed over the
            sensors. Populated from `touch_params`.
        response_function (Callable): A reference to the actual member method determined by `response_type`.
        sensor_positions (Dict[int, np.ndarray]): A dictionary containing the positions of the sensor points for each
            body part. The coordinates should be in the frame of the associated body part.
        sensor_outputs (Dict[int, np.ndarray]): A dictionary containing the outputs produced by the sensors for each
            body part. Shape will depend on the specific implementation. This should be populated by
            :meth:`.get_touch_obs`. Note that this will differ from the touch output to the environment, which is
            flattened.
    """

    #: A dictionary listing valid touch output types and their sizes.
    VALID_TOUCH_TYPES = {}
    #: A list of valid surface response functions.
    VALID_RESPONSE_FUNCTIONS = []

    def __init__(self, env, touch_params):
        self.env = env

        self.sensor_scales = {}
        for body_name in touch_params["scales"]:
            body_id = env.model.body(body_name).id
            self.sensor_scales[body_id] = touch_params["scales"][body_name]

        # Get all information for the touch force function: Function name, reference to function, output size
        self.touch_type = touch_params["touch_function"]
        assert self.touch_type in self.VALID_TOUCH_TYPES
        assert hasattr(self, self.touch_type)
        self.touch_function: Callable = getattr(self, self.touch_type)
        self.touch_size = self.VALID_TOUCH_TYPES[self.touch_type]

        # Get all information for the surface adjustment function: function name, reference to function
        self.response_type = touch_params["response_function"]
        assert self.response_type in self.VALID_RESPONSE_FUNCTIONS
        self.response_function: Callable = getattr(self, self.response_type)

        self.sensor_outputs = {}
        self.sensor_positions = {}

    def get_touch_obs(self):
        """ Produces the current touch output.

        This function should perform the whole sensory pipeline as defined in `touch_params` and return the output as a
        single array. The per-body output should additionally be stored in :attr:`.sensor_outputs`.

        Returns:
            np.ndarray: A numpy array of shape (n_sensor_points, touch_size)
        """
        raise NotImplementedError


class DiscreteTouch(Touch):
    """ A simple touch class using MuJoCo geoms as the basic sensing component.

    Sensor points are simply spread evenly over individual geoms, with no care taken for cases where geoms or bodies
    intersect. Nearest sensors are determined by direct euclidean distance. The sensor positions in
    :attr:`.sensor_positions` are directly used for the output, so altering them will also alter the output. This can
    be used to post-process the positions from the basic uniform distribution. Supported output types are

    - 'normal': The normal force as a scalar.
    - 'force_vector': The contact force vector (normal and frictional forces) reported in the coordinate
      frame of the sensing geom.
    - 'force_vector_global': Like 'force_vector', but reported in the world coordinate frame instead.

    The units are newton by default, but MuJoCo does not explicitly define units. Treating the unit of distance as
    meters and the unit of mass as kg as we do in our models will lead to newtons as the unit of force.
    The output can be spread to nearby sensors in two different ways:

    - 'nearest': Directly add the output to the nearest sensor.
    - 'spread_linear': Spreads the force to nearby sensor points, such that it decreases linearly with distance to the
      contact point. The force drops to 0 at twice the sensor scale. The force per sensor is normalised such that the
      total force is conserved.

    Touch functions return their output, while response functions do not return anything and instead write their
    adjusted forces directly into the output dictionary.

    The following attributes are provided in addition to those of :class:`~mimoTouch.touch.Touch`.

    Attributes:
        m_data (mujoco.MjData): A direct reference to the MuJoCo simulation data object.
        m_model (mujoco.MjModel): A direct reference to the MuJoCo simulation model object.
        plotting_limits (Dict[int, float]): A convenience dictionary listing axis limits for plotting forces or sensor
            points for geoms.
    """

    VALID_TOUCH_TYPES = {
        "normal": 1,
        "force_vector": 3,
        "force_vector_global": 3,
    }

    VALID_RESPONSE_FUNCTIONS = ["nearest", "spread_linear"]

    def __init__(self, env, touch_params):
        super().__init__(env, touch_params)
        self.m_data = env.data
        self.m_model = env.model

        self.plotting_limits = {}

        # Add sensors to bodies
        for body_id in self.sensor_scales:
            self.add_body(body_id, scale=self.sensor_scales[body_id])

        # Get touch obs once to ensure all output arrays are initialized
        self.get_touch_obs()
        
    def add_body(self, body_id=None, body_name=None, scale=math.inf):
        """ Adds sensors to all geoms belonging to the given body.

        Given a body, either by ID or name, spread sensor points over all geoms and add them to the output. If both ID
        and name are provided the name is ignored. The distance between sensor points is determined by `scale`.
        The names are determined by the scene XMLs while the IDs are assigned during compilation.

        Args:
            body_id (int|None): ID of the body.
            body_name (str|None): Name of the body. If `body_id` is provided this parameter is ignored!
            scale (float): The distance between sensor points.

        Returns:
            int: The number of sensors added to this body.
        """
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)

        n_sensors = 0

        for geom_id in env_utils.get_geoms_for_body(self.m_model, body_id):
            g_body_id = self.m_model.geom(geom_id).bodyid.item()
            contype = self.m_model.geom(geom_id).contype.item()
            conaffinity = self.m_model.geom(geom_id).conaffinity.item()
            # Add a geom if it belongs to body and has collisions enabled (at least potentially)
            if g_body_id == body_id and (contype > 0 or conaffinity > 0):
                n_sensors += self.add_geom(geom_id=geom_id, scale=scale)
        return n_sensors

    def add_geom(self, geom_id=None, geom_name=None, scale=math.inf):
        """ Adds sensors to the given geom.

        Spreads sensor points over the geom identified either by ID or name and add them to the output. If both ID and
        name are provided the name is ignored. The distance between sensor points is determined by `scale`. The names
        are determined by the scene XMLs while the IDs are assigned during compilation.

        Args:
            geom_id (int|None): ID of the geom.
            geom_name (str|None): Name of the geom.  If `geom_id` is provided this parameter is ignored!
            scale (float): The distance between sensor points.

        Returns:
            int: The number of sensors added to this geom.
        """
        geom_id = env_utils.get_geom_id(self.m_model, geom_id=geom_id, geom_name=geom_name)

        if self.m_model.geom(geom_id).contype.item() == 0 and self.m_model.geom(geom_id).conaffinity.item() == 0:
            raise RuntimeWarning("Added sensors to geom with collisions disabled!")
        return self._add_sensorpoints(geom_id, scale)

    @property
    def sensing_geoms(self):
        """ Returns the IDs of all geoms with sensors.

        Returns:
            List[int]: The IDs for all geoms that have sensors.
        """
        return list(self.sensor_positions.keys())

    def has_sensors(self, geom_id):
        """ Returns True if the geom has sensors.

        Args:
            geom_id (int): The ID of the geom.

        Returns:
            bool: ``True`` if the geom has sensors, ``False`` otherwise.
        """
        return geom_id in self.sensor_positions

    def get_sensor_count(self, geom_id):
        """ Returns the number of sensors for the geom.

        Args:
            geom_id (int): The ID of the geom.

        Returns:
            int: The number of sensor points for this geom.
        """
        return self.sensor_positions[geom_id].shape[0]

    def get_total_sensor_count(self):
        """ Returns the total number of touch sensors in the model.

        Returns:
            int: The total number of touch sensors in the model.
        """
        n_sensors = 0
        for geom_id in self.sensing_geoms:
            n_sensors += self.get_sensor_count(geom_id)
        return n_sensors

    def _add_sensorpoints(self, geom_id, scale):
        """ Adds sensors to the given geom.

        Spreads sensor points over the geom identified either by ID. The distance between sensor points is determined
        by `scale`. We identify the type of geom using the MuJoCo API. This function populates
        both :attr:`.sensor_positions` and :attr:`.plotting_limits`.

        Args:
            geom_id (int): ID of the geom.
            scale (float): The distance between sensor points.

        Returns:
            int|None: The number of sensors added to this geom or ``None`` if no sensors could be added.
        """
        # Add sensor points for the given geom using given resolution
        # Returns the number of sensor points added
        # Also set the maximum size of the geom, for plotting purposes
        geom_type = self.m_model.geom(geom_id).type.item()
        size = self.m_model.geom(geom_id).size
        limit = 1
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            limit = np.max(size)
            points = spread_points_box(scale, size)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            limit = size[0]
            points = spread_points_sphere(scale, size[0])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            limit = size[1] + size[0]
            points = spread_points_capsule(scale, 2*size[1], size[0])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            # Cylinder size 0 is radius, size 1 is half of the length
            limit = np.max(size)
            points = spread_points_cylinder(scale, 2*size[1], size[0])
        elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            RuntimeWarning("Cannot add sensors to plane geoms!")
            return None
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            raise NotImplementedError("Ellipsoids currently not implemented")
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            size = self.m_model.geom(geom_id).rbound
            limit = size
            points = spread_points_sphere(scale, size)
        else:
            return None

        self.plotting_limits[geom_id] = limit
        self.sensor_positions[geom_id] = points

        return points.shape[0]  # Return the number of points we added

    def get_nearest_sensor(self, contact_id, geom_id):
        """ Given a contact and a geom, return the sensor on the geom closest to the contact.

        Contact IDs are a MuJoCo attribute, see their documentation for more detail on contacts.

        Args:
            contact_id (int): The ID of the contact.
            geom_id (int): The ID of the geom. The geom must have sensors!

        Returns:
            Tuple[int, float]: The index of the closest sensor and the distance between the contact and sensor.
        """
        relative_position = self.get_contact_position_relative(contact_id, geom_id)
        sensor_points = self.sensor_positions[geom_id]
        distances = np.linalg.norm(sensor_points - relative_position, axis=1)
        idx = np.argmin(distances)
        return idx, distances[idx]

    def get_k_nearest_sensors(self, contact_id, geom_id, k):
        """ Given a contact and a geom, find the k sensors on the geom closest to the contact.

        Contact IDs are a MuJoCo attribute, see their documentation for more detail on contacts.

        Args:
            contact_id (int): The ID of the contact.
            geom_id (int): The ID of the geom. The geom must have sensors!
            k (int): The number of sensors to return.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The indices of the `k` nearest sensors, as well as the distances between
            their positions and the contact.
        """
        relative_position = self.get_contact_position_relative(contact_id, geom_id)
        sensor_points = self.sensor_positions[geom_id]
        distances = np.linalg.norm(sensor_points - relative_position, axis=1)
        if distances.shape[0] <= k:
            return np.arange(distances.shape[0]), distances
        else:
            sorted_idxs = np.argpartition(distances, k)
            return sorted_idxs[:k], distances[sorted_idxs[:k]]

    def get_sensors_within_distance(self, contact_id, geom_id, distance):
        """ Finds all sensors on a geom that are within a given distance to a contact.

        The distance used is the direct euclidean distance. Contact IDs are a MuJoCo attribute, see their documentation
        for more detail on contacts.

        Args:
            contact_id (int): The ID of the contact.
            geom_id (int): The ID of the geom. The geom must have sensors!
            distance (np.ndarray): Sensors must be within this distance to the contact position to be included in the
                output.

        Returns:
            Tuple[np.ndarray, np.ndarray]: The indices of all sensors on the geom that are within `distance` to the
            contact, as well as the distances between their positions and the contact.
        """
        relative_position = self.get_contact_position_relative(contact_id, geom_id)
        sensor_points = self.sensor_positions[geom_id]
        distances = np.linalg.norm(sensor_points - relative_position, axis=1)
        within_distance = distances < distance
        idx_within_distance = within_distance.nonzero()[0]
        return idx_within_distance, distances[within_distance]

    # ======================== Positions and rotations ================================
    # =================================================================================

    def get_contact_position_world(self, contact_id):
        """ Get the position of a contact in the world frame.

        Note that this is halfway between the touching geoms. Since geoms can intersect this point will likely be
        located inside both.

        Args:
            contact_id (int): The ID of the contact.

        Returns:
            np.ndarray: An array with the position of the contact.
        """
        return self.m_data.contact[contact_id].pos

    def get_contact_position_relative(self, contact_id, geom_id):
        """ Get the position of a contact in the coordinate frame of a geom.

        This position is corrected for the intersection between geoms, such that the contact lies on the surface of the
        sensing geom.

        Args:
            contact_id (int): The ID of the contact.
            geom_id (int): The ID of the geom.

        Returns:
            np.ndarray: An array with the position of the contact.
        """
        body_pos = env_utils.world_pos_to_geom(self.m_data, self.get_contact_position_world(contact_id), geom_id)
        contact = self.m_data.contact[contact_id]
        if contact.dist < 0:
            # Have to correct contact position towards surface of our body.
            # Note that distance is negative for intersecting geoms and the normal vector points into the sensing geom.
            normal = self.get_contact_normal(contact_id, geom_id)
            body_pos = body_pos + normal * contact.dist / 2
        return body_pos

    # =============== Visualizations ==================================================
    # =================================================================================

    def plot_sensors_geom(self, geom_id=None, geom_name=None):
        """ Plots the sensor positions for a geom.

        Given either an ID or the name of a geom, plot the positions of the sensors on that geom.

        Args:
            geom_id (int|None): The ID of the geom.
            geom_name (str|None): The name of the geom. This is ignored if the ID is provided!

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple (fig, ax) with the pyplot figure and axis objects.
        """
        geom_id = env_utils.get_geom_id(self.m_model, geom_id=geom_id, geom_name=geom_name)

        points = self.sensor_positions[geom_id]
        limit = self.plotting_limits[geom_id]
        title = self.m_model.geom(geom_id).name
        fig, ax = env_utils.plot_points(points, limit=limit, title=title, show=False)
        return fig, ax

    def plot_force_geom(self, geom_id=None, geom_name=None):
        """ Plot the sensor output for a geom.

        Given either an ID or the name of a geom, plots the positions and outputs of the sensors on that geom.

        Args:
            geom_id (int|None): The ID of the geom.
            geom_name (str|None): The name of the geom. This is ignored if the ID is provided!

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple (fig, ax) with the pyplot figure and axis objects.
        """
        geom_id = env_utils.get_geom_id(self.m_model, geom_id=geom_id, geom_name=geom_name)

        sensor_points = self.sensor_positions[geom_id]
        force_vectors = self.sensor_outputs[geom_id]
        if force_vectors.shape[1] == 1:
            # TODO: Need proper sensor normals, can't do this until trimesh rework
            raise RuntimeWarning("Plotting of scalar forces not implemented!")
        else:
            fig, ax = env_utils.plot_forces(sensor_points, force_vectors, limit=np.max(sensor_points) + 0.5, show=False)
        return fig, ax

    def _get_plot_info_body(self, body_id):
        """ Collects sensor points and forces for a single body.

        Args:
            body_id (int): The ID of the body.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Two numpy arrays containing the sensor positions and their outputs.
        """
        points = []
        forces = []
        for geom_id in env_utils.get_geoms_for_body(self.m_model, body_id):
            points_in_body = env_utils.geom_pos_to_body(self.m_data, self.sensor_positions[geom_id], geom_id, body_id)
            points.append(points_in_body)
            force_vectors = self.sensor_outputs[geom_id] / 100
            if force_vectors.shape[1] == 1:
                # TODO: Need proper sensor normals, can't do this until trimesh rework
                raise RuntimeWarning("Plotting of scalar forces not implemented!")
            body_forces = env_utils.geom_rot_to_body(self.m_data, force_vectors, geom_id=geom_id, body_id=body_id)
            forces.append(body_forces)
        points_t = np.concatenate(points)
        forces_t = np.concatenate(forces)
        return points_t, forces_t

    def plot_force_body(self, body_id=None, body_name=None):
        """ Plots sensor points and output forces for all geoms in a body.

        Given either an ID or the name of a body, plots the positions and outputs of the sensors for all geoms
        associated with that body.

        Args:
            body_id (int|None): The ID of the body.
            body_name (str|None): The name of the body. This argument is ignored if the ID is provided.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple (fig, ax) with the pyplot figure and axis objects.
        """
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)

        points, forces = self._get_plot_info_body(body_id)

        fig, ax = env_utils.plot_forces(points, forces, limit=np.max(points) + 0.5, show=False)
        return fig, ax

    # =============== Raw force and contact normal ====================================
    # =================================================================================

    def get_raw_force(self, contact_id, geom_id):
        """ Collect the full contact force in MuJoCos own contact frame.

        By convention the normal force points away from the first geom listed, so the forces are inverted if the first
        geom is the sensing geom.

        Args:
            contact_id (int): The ID of the contact.
            geom_id (int): The relevant geom in the contact. Must be one of the geoms involved in the contact!

        Returns:
            np.ndarray: An array with shape (3,) containing the normal force and tangential frictional forces.
        """
        forces = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self.m_model, self.m_data, contact_id, forces)
        contact = self.m_data.contact[contact_id]
        if geom_id == contact.geom1:
            forces *= -1  # Convention is that normal points away from geom1
        elif geom_id == contact.geom2:
            pass
        else:
            RuntimeError("Mismatch between contact and geom")
        return forces[:3]

    def get_contact_normal(self, contact_id, geom_id):
        """ Returns the normal vector of contact (unit vector in direction of normal) in geom coordinate frame.

        Args:
            contact_id (int): The ID of the contact.
            geom_id (int): The ID of the geom.

        Returns:
            np.ndarray: An array with shape (3,) containing the normal vector.
        """
        contact = self.m_data.contact[contact_id]
        normal_vector = contact.frame[:3]
        if geom_id == contact.geom1:  # Mujoco vectors point away from geom1 by convention
            normal_vector *= -1
        elif geom_id == contact.geom2:
            pass
        else:
            RuntimeError("Mismatch between contact and geom")
        # contact frame is in global coordinate frame, rotate to geom frame
        normal_vector = rotate_vector(normal_vector, env_utils.get_geom_rotation(self.m_data, geom_id))
        return normal_vector

    # =============== Valid touch force functions =====================================
    # =================================================================================

    def normal(self, contact_id, geom_id):
        """ Touch function. Returns the normal force as a scalar.

        Given a contact and a geom, returns the normal force of the contact. The geom is required to account for the
        MuJoCo contact conventions.

        Args:
            contact_id (int): The ID of the contact.
            geom_id (int): The ID of the geom.

        Returns:
            float: The normal force as a float.
        """
        return self.get_raw_force(contact_id, geom_id)[0]

    def force_vector_global(self, contact_id, geom_id):
        """ Touch function. Returns the full contact force in world frame.

        Given a contact returns the full contact force, i.e. the vector sum of the normal force and the two tangential
        friction forces, in the world coordinate frame. The geom is required to account for MuJoCo conventions and
        convert coordinate frames.

        Args:
            contact_id (int): The ID of the contact.
            geom_id (int): The ID of the geom.

        Returns:
            np.ndarray: An array with shape (3,) containing the contact force as a vector.
        """
        contact = self.m_data.contact[contact_id]
        forces = self.get_raw_force(contact_id, geom_id)
        force_rot = np.reshape(contact.frame, (3, 3))
        global_forces = rotate_vector_transpose(forces, force_rot)
        return global_forces

    def force_vector(self, contact_id, geom_id):
        """ Touch function. Returns full contact force in the frame of the geom.

        Same as :meth:`.force_vector_global`, but the force is returned in the coordinate frame of the geom.

        Args:
            contact_id (int): The ID of the contact.
            geom_id (int): The ID of the geom.

        Returns:
            np.ndarray: An array with shape (3,) containing the contact force as a vector.
        """
        global_forces = self.force_vector_global(contact_id, geom_id)
        relative_forces = rotate_vector_transpose(global_forces, env_utils.get_geom_rotation(self.m_data, geom_id))
        return relative_forces

    # =============== Output functions ================================================
    # =================================================================================

    def get_contacts(self):
        """ Collects all active contacts involving geoms with touch sensors.

        For each active contact with a sensing geom we build a tuple ``(contact_id, geom_id, forces)``, where
        `contact_id` is the ID of the contact in the MuJoCo arrays, `geom_id` is the ID of the sensing geom and
        `forces` is a numpy array of the raw output force, as determined by :attr:`.touch_type`.

        Returns:
            List[Tuple[int, int, np.ndarray]]: A list of tuples with contact information.
        """
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
        """ Returns a dictionary with empty sensor outputs.

        Creates a dictionary with an array of zeros for each geom with sensors. A geom with 'n' sensors has an empty
        output array of shape (n, size). The output of this function is equivalent to the touch sensor output if
        there are no contacts.

        Args:
            size (int): The size of a single sensor output.

        Returns:
            Dict[int, np.ndarray]: The dictionary of empty sensor outputs.
        """
        sensor_outputs = {}
        for geom_id in self.sensor_positions:
            sensor_outputs[geom_id] = np.zeros((self.get_sensor_count(geom_id), size), dtype=np.float32)
        return sensor_outputs

    def flatten_sensor_dict(self, sensor_dict):
        """ Concatenates a touch output dictionary into a single large array in a deterministic fashion.

        Output dictionaries list the arrays of sensor outputs for each geom. This function concatenates these arrays
        together in a reproducible fashion to avoid key order anomalies. Geoms are sorted by their ID.

        Args:
            sensor_dict (Dict[int, np.ndarray]): The output dictionary to be flattened.

        Returns:
            np.ndarray: The concatenated numpy array.
        """
        sensor_arrays = []
        for geom_id in sorted(self.sensor_positions):
            sensor_arrays.append(sensor_dict[geom_id])
        return np.concatenate(sensor_arrays)

    def get_touch_obs(self):
        """ Produces the current touch sensor outputs.

        Does the full contact getting-processing process, such that we get the forces, as determined by
        :attr:`.touch_type` and :attr:`.response_type`, for each sensor. :attr:`.touch_function` is called to compute
        the raw output force, which is then distributed over the sensors using :attr:`.response_function`.

        The indices of the output dictionary :attr:`~mimoTouch.touch.DiscreteTouch.sensor_outputs` and the sensor
        dictionary :attr:`.sensor_positions` are aligned, such that the `i`th sensor on geom `j` has position
        ``.sensor_positions[j][i]`` and output in ``.sensor_outputs[j][i]``.

        Returns:
            np.ndarray: An array containing all the touch sensations.
        """
        contact_tuples = self.get_contacts()
        self.sensor_outputs = self.get_empty_sensor_dict(self.touch_size)  # Initialize output dictionary

        for contact_id, geom_id, forces in contact_tuples:
            # At this point we already have the forces for each contact, now we must attach/spread them to sensor
            # points, based on the adjustment function
            self.response_function(contact_id, geom_id, forces)

        sensor_obs = self.flatten_sensor_dict(self.sensor_outputs)
        return sensor_obs

    # =============== Force adjustment functions ======================================
    # =================================================================================

    def spread_linear(self, contact_id, geom_id, force):
        """ Response function. Distributes the output force linearly based on distance.

        For a contact and a raw force we get all sensors within a given distance to the contact point and then
        distribute the force such that the force reported at a sensor decreases linearly with distance between the
        sensor and the contact point. Finally, the total force is normalized such that the total force over all sensors
        for this contact is identical to the raw force. The scaling distance is given by double the distance between
        sensor points.

        Args:
            contact_id (int): The ID of the contact.
            geom_id (int): The ID of the sensing geom.
            force (np.ndarray): The raw force.
        """
        # Get all sensors within distance (distance here is just double the sensor scale)
        body_id = self.m_model.geom(geom_id).bodyid.item()
        scale = self.sensor_scales[body_id]
        nearest_sensors, sensor_distances = self.get_sensors_within_distance(contact_id, geom_id, 2*scale)

        sensor_adjusted_forces = scale_linear(force, sensor_distances, scale=2 * scale)
        force_total = abs(np.sum(sensor_adjusted_forces[:, 0]))

        factor = abs(force[0] / force_total) if force_total > EPS else 0
        self.sensor_outputs[body_id][nearest_sensors] += sensor_adjusted_forces * factor

    def nearest(self, contact_id, geom_id, force):
        """ Response function. Adds the output force directly to the nearest sensor.

        Args:
            contact_id (int): The ID of the contact.
            geom_id (int): The ID of the geom.
            force (np.ndarray): The raw output force.
        """
        # Get the nearest sensor to this contact, add the force to it
        nearest_sensor, distance = self.get_nearest_sensor(contact_id, geom_id)
        self.sensor_outputs[geom_id][nearest_sensor] += force


# =============== Scaling functions ===============================================
# =================================================================================

def scale_linear(force, distances, scale):
    """ Used to scale forces linearly based on distance.

    Adjusts the force by a simple factor, such that force falls linearly from full at `distance = 0`
    to 0 at `distance >= scale`.

    Args:
        force (np.ndarray): The unadjusted force.
        distances (np.ndarray): The adjusted force reduces linearly with increasing distance.
        scale (float): The scaling limit. If ``distance >= scale`` the return value is reduced to 0.

    Returns:
        np.ndarray: The scaled force.
    """
    factor = (scale-distances) / scale
    factor[factor < 0] = 0
    out_force = (force[:, np.newaxis] * factor).T
    return out_force


# =============== Trimesh based alternative implementation ========================
# =================================================================================


class TrimeshTouch(Touch):
    """ A touch class with sensor meshes using MuJoCo bodies as the basic sensing component.

    Sensor points are simply spread evenly over individual geoms. Geoms belonging to the same body are then merged,
    removing all intersecting sensors. Nearest sensors are determined through adjacency to the closest vertex, but
    distances are still euclidean distance instead of geodesic. For runtime reasons multiple datastructures are
    cached, so the sensor positions in :attr:`.sensor_positions` should not be altered as they are tied to the
    underlying sensor mesh. Trimesh is used for the mesh operations. Supported output types are

        - 'force_vector': The contact force vector (normal and frictional forces) reported in the coordinate frame of
          the sensing body.
        - 'force_vector_global': Like 'force_vector', but reported in the world coordinate frame instead.
        - 'normal_force': Returns the normal force only, as a vector in the frame of the sensing body.

    The output can be spread to nearby sensors in two different ways:

        - 'nearest': Directly add the output to the nearest sensor.
        - 'spread_linear': Spreads the force to nearby sensor points, such that it decreases linearly with distance to
          the contact point. The force drops to 0 at twice the sensor scale. The force per sensor is normalised such
          that the total force is conserved.

    Touch functions return their output, while response functions do not return anything and instead write their
    adjusted forces directly into the output dictionary.

    An LRU cache is used to speed up performance of the nearest sensor point searches. This cache persists through
    calls to :meth:`~mimoEnv.envs.mimo_env.MIMoEnv.reset`.

    The following attributes are provided in addition to those of :class:`~mimoTouch.touch.Touch`.

    Attributes:
        m_data (mujoco.MjData): A direct reference to the MuJoCo simulation data object.
        m_model (mujoco.MjModel): A direct reference to the MuJoCo simulation model object.
        meshes (Dict[int, trimesh.Trimesh]): A dictionary containing the sensor mesh objects for each body.
        active_vertices (Dict[int, np.ndarray]: A dictionary of masks. Not every sensor point will be active as they
            may intersect another geom on the same body. Only active vertices contribute to the output, but inactive
            ones are still required for mesh operations. If a sensor is active the associated entry in this dictionary
            will be ``True``, otherwise ``False``.
        plotting_limits (Dict[int, float]: A convenience dictionary listing axis limits for plotting forces or sensor
            points for geoms.
        contact_tuples (List[Tuple[int, int, np.ndarray]]): A list of tuples listing the contact index, the relevant
            sensing body and the raw contact forces for that contact. Note that a contact may appear twice if both
            involved bodies have sensors.
        _submeshes (Dict[int, List[trimesh.Trimesh]]): A dictionary like :attr:`.meshes`, but storing a list of the
            individual geom meshes instead.
        _active_subvertices (Dict[int, List[np.ndarray]): A dictionary like :attr:`.active_vertices`, but storing a
            list of masks for each geom mesh instead.
        _vertex_to_sensor_idx (Dict[int, List[np.ndarray]]): A dictionary that maps the indices for each active vertex.
            Calculations happen on submeshes, so the indices have to mapped onto the output array. This dictionary
            stores that mapping.
        _neighbour_cache (LRUCache): An LRU cache storing the results for the nearest neighbour searches. Hit rate and
            current size can be determined with ``._neighbour_cache.hits()`` and ``._neighbour_cache._cache.currsize``
            respectively.
        """

    VALID_TOUCH_TYPES = {
        "force_vector": 3,
        "force_vector_global": 3,
        "normal_force": 3,
    }

    VALID_RESPONSE_FUNCTIONS = ["nearest", "spread_linear"]

    def __init__(self, env, touch_params):
        super().__init__(env, touch_params=touch_params)
        self.m_model = self.env.model
        self.m_data = self.env.data

        # for each body, _submeshes stores the individual watertight meshes as a list and '_active_subvertices' stores
        # a boolean area of whether the vertices are active sensors. Inactive sensors do not output data and are not
        # included in the output arrays.
        # _vertex_to_sensor_idx stores the active sensor index for the associated vertex. This value will be nonsensical
        # for inactive vertices!
        self._submeshes: Dict[int, List[trimesh.Trimesh]] = {}
        self._active_subvertices = {}
        self._sensor_counts = {}
        self._sensor_counts_submesh = {}
        self._vertex_to_sensor_idx = {}

        self.meshes: Dict[int, trimesh.Trimesh] = {}
        self.active_vertices = {}
        self.contact_tuples = []

        self.plotting_limits = {}

        n_sensors = 0
        # Add sensors to bodies
        for body_id in self.sensor_scales:
            self.add_body(body_id=body_id, scale=self.sensor_scales[body_id])
            n_sensors += self.get_sensor_count(body_id)

        cache_size = n_sensors / 20 + len(self.sensor_scales)
        self._neighbour_cache = CacheToolsUtils.StatsCache(LRUCache(maxsize=cache_size))

        # Get touch obs once to ensure all output arrays are initialized
        self.get_touch_obs()

    # ======================== Sensor related functions ===============================
    # =================================================================================

    def add_body(self, body_id=None, body_name=None, scale=math.inf):
        """ Adds sensors to the given body.

        Given a body, either by ID or name, spread sensor meshes over all geoms and adds them to the output. If both ID
        and name are provided the name is ignored. The distance between sensor points is determined by `scale`.
        This function has to handle all the arrays required for quick access after initialization, so it populates the
        submesh, mesh, mask and index mapping dictionaries.
        The names of bodies are determined by the scene XMLs while the IDs are assigned during compilation.

        Args:
            body_id (int|None): ID of the body.
            body_name (str|None): Name of the body. If `body_id` is provided this parameter is ignored!
            scale (float): The distance between sensor points.
        """
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)
        meshes = []
        for geom_id in env_utils.get_geoms_for_body(self.m_model, body_id):
            contype = self.m_model.geom(geom_id).contype.item()
            conaffinity = self.m_model.geom(geom_id).conaffinity.item()
            if contype == 0 and conaffinity == 0:
                continue
            mesh = self._get_mesh(geom_id, scale)
            # Move meshes from geom into body frame
            mesh.vertices = env_utils.geom_pos_to_body(self.m_data, mesh.vertices.copy(), geom_id, body_id)
            meshes.append(mesh)
        self._submeshes[body_id] = meshes
        if len(meshes) > 1:
            # Can't use trimesh util concatenate since that fails with point clouds (such as single point bodies)
            # self.meshes[body_id] = trimesh.util.concatenate(meshes)
            vertex_offset = 0
            vertices = []
            faces = []
            for mesh in meshes:
                vertices.append(mesh.vertices)
                if hasattr(mesh, "faces"):
                    faces.append(mesh.faces + vertex_offset)
                vertex_offset += mesh.vertices.shape[0]
            vertices = np.concatenate(vertices)
            if faces:
                faces = np.concatenate(faces)
            else:
                faces = np.zeros((0, 3))
            self.meshes[body_id] = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        else:
            self.meshes[body_id] = meshes[0]

        active_vertices = []
        vertex_to_sensor_idxs = []
        submesh_offset = 0
        sensor_counts = []
        # Cleanup, removing all sensor points inside other sensors
        for mesh in meshes:
            mask = np.ones((mesh.vertices.shape[0],), dtype=bool)
            for other_mesh in meshes:
                if other_mesh == mesh or isinstance(other_mesh, PointCloud):
                    continue
                # If our vertices are contained in the other mesh, make them inactive
                contained = _contains(mesh.vertices, other_mesh)
                mask = np.logical_and(mask, np.invert(contained))
            active_vertices.append(mask)
            sensor_counts.append(np.count_nonzero(mask))
            vertex_offsets = np.cumsum(mask) - 1
            vertex_to_sensor_idxs.append(vertex_offsets + submesh_offset)
            submesh_offset += np.count_nonzero(mask)

        self._active_subvertices[body_id] = active_vertices
        self.active_vertices[body_id] = np.concatenate(active_vertices)
        self._sensor_counts[body_id] = np.count_nonzero(self.active_vertices[body_id])
        self._sensor_counts_submesh[body_id] = sensor_counts
        self._vertex_to_sensor_idx[body_id] = vertex_to_sensor_idxs

        self.sensor_positions[body_id] = self.meshes[body_id].vertices[self.active_vertices[body_id], :]

    def sensing_bodies(self):
        """ Returns the IDs of all bodies with sensors.

        Returns:
            List[int]: A list with the IDs for all bodies that have sensors.
        """
        return list(self.meshes.keys())

    def has_sensors(self, body_id):
        """ Returns True if the body has sensors.

        Args:
            body_id (int): The ID of the body.

        Returns:
            bool: ``True`` if the body has sensors, ``False`` otherwise.
        """
        return body_id in self._submeshes

    def get_sensor_count(self, body_id):
        """ Returns the number of sensors for the body.

        Args:
            body_id (int): The ID of the body.

        Returns:
            int: The number of sensor points for this body.
        """
        return self._sensor_counts[body_id]

    def _get_sensor_count_submesh(self, body_id, submesh_idx):
        """ Returns the number of sensors on this submesh.

        Each body can contain multiple submeshes, so submeshes are uniquely identified by the ID of their body and
        their index in the list of submeshes for that body.

        Args:
            body_id (int): The ID of the body.
            submesh_idx (int): The index of the submesh.

        Returns:
            int: The number of sensor points on this submesh.
        """
        return self._sensor_counts_submesh[body_id][submesh_idx]

    def _get_mesh(self, geom_id, scale):
        """ Creates a sensor mesh for the geom.

        Given a geom this creates a raw sensor mesh. These are not used directly, instead :meth:`.add_body` collects
        the meshes for all geoms in a body and then merges them, marking any sensor points that are located inside
        another geom as inactive.

        Args:
            geom_id (int): The ID of the geom.
            scale (float): The distance between sensor points in the mesh.

        Returns:
            trimesh.Trimesh|None: The new sensor mesh or ``None`` if no mesh could be created.
        """
        # Do sensorpoints for a single geom
        # TODO: Use our own normals instead of default from trimesh face estimation
        geom_type = self.m_model.geom(geom_id).type.item()
        size = self.m_model.geom(geom_id).size

        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = mesh_box(scale, size)
        elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            mesh = mesh_sphere(scale, size[0])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            mesh = mesh_capsule(scale, 2 * size[1], size[0])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            # Cylinder size 0 is radius, size 1 is half the length
            mesh = mesh_cylinder(scale, 2 * size[1], size[0])
        elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
            RuntimeWarning("Cannot add sensors to plane geoms!")
            return None
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            mesh = mesh_ellipsoid(scale, size)
        elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            # TODO: Use convex hull of mesh as sensor mesh? Would not have remotely consistent spacing
            size = self.m_model.geom(geom_id).rbound
            mesh = mesh_sphere(scale, size)
        else:
            return None
        return mesh

    def _convert_active_sensor_idx(self, body_id, submesh_idx, vertex_idx):
        """ Converts index types for active sensors.

        Active sensors need to keep track of two indices: Their index in the submesh and their index in the output
        arrays. This function allows one to convert the submesh index to the output index, by reading from the map
        created when the associated body was added to the model. In this documentation indices will be output indices
        unless explicitly stated otherwise.

        Args:
            body_id (int): The ID of the body.
            submesh_idx (int): The index of the submesh.
            vertex_idx (int): The index of the vertex to be converted.

        Returns:
            int: The index in the output arrays for the input vertex.
        """
        return self._vertex_to_sensor_idx[body_id][submesh_idx][vertex_idx]

    def _get_nearest_vertex(self, contact_pos, mesh):
        """ Get the vertex in the mesh closest to the position.

        Note that this vertex might not be an active sensor!

        Args:
            contact_pos (np.ndarray): The position. Should be a numpy array with shape (3,).
            mesh (trimesh.Trimesh): The mesh.

        Returns:
            Tuple[int, float]: The submesh index of the vertex closest to the position and the distance between the
                vertex and the position.
        """
        if mesh.vertices.shape[0] == 1:
            distance = np.linalg.norm(contact_pos - mesh.vertices[0])
            sub_idx = 0
        else:
            proximity_query = trimesh.proximity.ProximityQuery(mesh)
            distance, sub_idx = proximity_query.vertex(contact_pos)
        return sub_idx, distance

    def get_nearest_sensor(self, contact_pos, body_id):
        """ Given a position in space and a body, return the sensor on the body closest to the position.

        Args:
            contact_pos (np.ndarray): The position. Should be a numpy array of shape (3,).
            body_id (int): The ID of the body. The body must have sensors!

        Returns:
            Tuple[int, float]: The index of the closest sensor and the distance between contact and sensor.
        """
        # Get the closest active vertex on the whole body mesh. Does this by getting clostest active subvertex on each
        # submesh and then returning the closest of those.
        active_sensors = {}
        for i, mesh in enumerate(self._submeshes[body_id]):
            # Get the closest vertex on mesh
            sub_idx, distance = self._get_nearest_vertex(contact_pos, mesh)
            active_vertices_on_submesh = self._active_subvertices[body_id][i]
            # If closest vertex is active: It is the candidate for this submesh
            if active_vertices_on_submesh[sub_idx]:
                active_sensors[(i, sub_idx)] = distance
                continue
            # If the closest vertex is not active: iteratively check neighbors for active candidates
            else:
                graph = mesh.vertex_adjacency_graph
                candidates = deque()
                candidates.extend(graph[sub_idx])
                distances = self._sensor_distances(contact_pos, mesh)
                checked = {sub_idx}
                while len(candidates) > 0:
                    candidate = candidates.pop()
                    if active_vertices_on_submesh[candidate]:
                        # Still assume convex meshes, so distance must >= for further vertices
                        distance = distances[candidate]
                        active_sensors[(i, candidate)] = distance
                        checked.add(candidate)
                    else:
                        candidates.extend(set(graph[candidate]) - checked)
                        checked.update(graph[candidate])
                pass
        closest = -1
        closest_distance = math.inf
        for key in active_sensors:
            distance = active_sensors[key]
            if distance < closest_distance:
                closest = key
                closest_distance = distance
        return self._convert_active_sensor_idx(body_id, closest[0], closest[1]), closest_distance

    @cachedmethod(operator.attrgetter("_neighbour_cache"),
                  key=lambda inst, distances, body_id, submesh_id, vertex_id, k:
                  hash(("nearest_k", body_id, submesh_id, vertex_id, k)))
    def _nearest_k_search(self, distances, body_id: int, submesh_id: int, vertex_id: int, k: int):
        """ Find the `k` nearest sensor points using BFS.

        The results from this function are cached in :attr:`._neighbour_cache`.

        Args:
            distances (List[np.ndarray]): A list of arrays, storing the distances between the sensor points and a given
                contact point. This list is used to terminate search early if we have `k` candidates already and a new
                vertex is further away than the furthest so far.
            body_id (int): The id of the body on which we search for sensor points.
            submesh_id (int): The submesh of the vertex with the shortest distance. Only used for the cache.
            vertex_id (int): The id of the vertex with the shortest distance. Only used for the cache.
            k (int): How many closest neighbours we return.

        Returns:
            List[List[int]]: A list containing the candidate vertices for each submesh.
        """
        # Use trimesh meshes to get the nearest vertex on all submeshes, then get up to k extra candidates for each
        # submesh, then get the k closest from the candidates of all submeshes
        candidate_sensor_idxs = []
        for i, mesh in enumerate(self._submeshes[body_id]):
            mesh_distances = distances[i]
            sub_idx = np.argmin(mesh_distances)
            active_vertices_on_submesh = self._active_subvertices[body_id][i]

            # If the mesh only has a single vertex, and it is an active vertex, take it and go to next mesh
            if mesh.vertices.shape[0] == 1 and active_vertices_on_submesh[0]:
                candidate_sensor_idxs.append((i, sub_idx))
                continue

            # Perform nearest k search, caching results.
            graph = mesh.vertex_adjacency_graph
            nodes_to_check = deque()
            nodes_to_check.append(sub_idx)
            nodes_to_check.extend(graph[sub_idx])
            largest_distance_so_far = 0
            checked = set()
            while len(nodes_to_check) > 0:
                candidate = nodes_to_check.pop()
                if candidate in checked:
                    continue
                checked.add(candidate)
                distance = mesh_distances[candidate]
                # If we have enough candidates and the current node is further away than the furthest, skip this
                # node. Otherwise, put neighbours into queue to check. If node is also active, add it to candidates.
                if len(candidate_sensor_idxs) >= k and distance > largest_distance_so_far:
                    continue
                else:
                    nodes_to_check.extend(set(graph[candidate]) - checked)
                    if active_vertices_on_submesh[candidate]:
                        candidate_sensor_idxs.append((i, candidate))
                        if len(candidate_sensor_idxs) < k and distance > largest_distance_so_far:
                            largest_distance_so_far = distance
        return candidate_sensor_idxs

    def get_k_nearest_sensors(self, contact_pos, body_id, k, k_margin=1.4):
        """ Given a position and a body, find the `k` sensors on the body closest to the position.

        Uses a cache to speed up the simulation. For a given contact we determine the closest sensor vertex on a
        given body. If this vertex is located in the cache, the nearest neighbour search is skipped and instead
        pulled from the cache. To ensure that the cache is accurate even as the contact point moves around a vertex,
        we store slightly more than `k` candidate neighbours in the cache. How many more is determined by `k_margin`.

        Args:
            contact_pos (np.ndarray): The position. Should be a numpy array of shape (3,).
            body_id (int): The ID of the body. The body must have sensors!
            k (int): The number of sensors to return.
            k_margin (float): The factor by which k is increased for the cache.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple with two arrays, the first containing the indices of the `k` closest
            sensors and the second containing the distances between the sensors and the position.
        """
        # Cache operates on closest vertex overall (even inactive), so have to find that first to determine if cache
        # applies
        distances = []
        closest_id = (-1, -1)
        closest_distance = math.inf
        for i, mesh in enumerate(self._submeshes[body_id]):
            mesh_distances = self._sensor_distances(contact_pos, mesh)
            sub_idx = np.argmin(mesh_distances)
            distance = mesh_distances[sub_idx]
            if distance < closest_distance:
                closest_id = (i, sub_idx)
                closest_distance = distance
            distances.append(mesh_distances)

        submesh_idx, vertex_idx = closest_id

        k_search = int(math.floor(k * k_margin))

        candidate_sensors_idx = \
            self._nearest_k_search(distances, body_id, submesh_idx, vertex_idx, k_search)

        sensors_idx = np.asarray([self._convert_active_sensor_idx(body_id, i, vert_idx)
                                  for i, vert_idx in candidate_sensors_idx])
        candidate_sensor_distances = np.asarray([distances[i][vert_idx] for i, vert_idx in candidate_sensors_idx])

        # Get k closest from all of these candidates
        if sensors_idx.shape[0] <= k:
            return sensors_idx, candidate_sensor_distances
        else:
            sorted_idxs = np.argpartition(candidate_sensor_distances, k)
            return sensors_idx[sorted_idxs[:k]], candidate_sensor_distances[sorted_idxs[:k]]

    def _get_mesh_adjacency_graph(self, mesh):
        """ Grab the adjacency graph for the mesh.

        Currently just wraps trimeshes vertex adjacency function, since they already handle caching.

        Args:
            mesh (trimesh.Trimesh): The mesh.

        Returns:
            networkx.Graph: A networkx graph of the mesh.
        """
        return mesh.vertex_adjacency_graph

    @cachedmethod(operator.attrgetter("_neighbour_cache"),
                  key=lambda inst, distances, body_id, submesh_id, vertex_id, distance_limit:
                  hash(("within_distance", body_id, submesh_id, vertex_id, distance_limit)))
    def _nearest_within_distance_search(self, distances, body_id, submesh_id,
                                        vertex_id, distance_limit):
        """ Finds all sensor points within a distance limit using BFS on the sensor mesh.

        The results from this function are cached in :attr:`._neighbour_cache`.

        Args:
            distances (List[np.ndarray]): A list of arrays, storing the distances between the sensor points and a given
                contact point.
            body_id (int): The id of the body on which we search for sensor points.
            submesh_id (int): The submesh of the vertex with the shortest distance. Only used for the cache.
            vertex_id (int): The id of the vertex with the shortest distance. Only used for the cache.
            distance_limit (float): Sensor vertices further away than this limit are excluded from the output.

        Returns:
            List[List[int]]: A list containing the candidate vertices for each submesh.
        """
        # Use trimesh to get the nearest vertex on each submesh and then inspect neighbours from there. Have to check
        # submeshes since we don't have edges between submeshes
        candidate_sensors_idxs = []
        for i, mesh in enumerate(self._submeshes[body_id]):
            mesh_candidate_idxs = []
            candidate_sensors_idxs.append(mesh_candidate_idxs)
            mesh_distances = distances[i]
            sub_idx = np.argmin(mesh_distances)
            distance = mesh_distances[sub_idx]
            # If even the closest point is too far away we can skip submesh
            if distance > distance_limit:
                continue
            active_vertices_on_submesh = self._active_subvertices[body_id][i]
            # If we only have a single sensor point, and it is active we add it to the candidates and go to next
            # submesh
            if mesh.vertices.shape[0] == 1 and active_vertices_on_submesh[0]:
                mesh_candidate_idxs.append(sub_idx)
                continue

            # The actual search happens here now simply using BFS
            graph = self._get_mesh_adjacency_graph(mesh)
            nodes_to_check = deque()
            nodes_to_check.append(sub_idx)
            nodes_to_check.extend(graph[sub_idx])
            within_distance = mesh_distances < distance_limit
            checked = np.invert(within_distance)

            while len(nodes_to_check) > 0:
                candidate = nodes_to_check.pop()
                # If the point is beyond the distance or we checked it already we skip it, otherwise we add the
                # neighbours into the queue to check. If the point is also an active sensor point we add it to the
                # candidates.
                if checked[candidate]:
                    continue
                checked[candidate] = 1
                nodes_to_check.extend(graph[candidate])
                if active_vertices_on_submesh[candidate]:
                    mesh_candidate_idxs.append(candidate)

        return candidate_sensors_idxs

    def get_sensors_within_distance(self, contact_pos, body_id,
                                    distance_limit, distance_margin=1.5):
        """ Finds all sensors on a body that are within a given distance to a given contact.

        The distance used is the direct euclidean distance. A sensor is included in the output if and only if:

        - It is within the distance limit to the position.
        - There is a path from the sensor to the vertex closest to the position such that all vertices on that path are
          also within the distance limit.

        Uses a cache to speed up the simulation. For a given contact we determine the closest sensor point on a
        given body. If this point is located in the cache, the nearest neighbour search is skipped and instead
        pulled from the cache. If the point is not located in the cache we store it there. Currently, there is no
        pruning or limiting of the size of the cache.
        To facilitate accurate searches even as the contact point moves about, we search a slightly larger area on
        the first occurrence, which is then pruned on subsequent occurrences using the distance limit. The increase for
        the first search is determined by the factor `distance_margin`.

        Args:
            contact_pos (np.ndarray): The position. Should be a numpy array of shape (3,).
            body_id (int): The ID of the body. The body must have sensors!
            distance_limit (float): Sensors must be within this distance to the position to be included in the output.
            distance_margin (float): How much the search limit is increased on the first search. Default 1.5.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple with two arrays, the first containing indices of all sensors on the
            body that are within `distance` to the contact and the second containing the distances between the sensors
            and the position.
        """
        # Cache operates on closest vertex overall (even inactive), so have to find that first to determine if cache
        # applies
        distances = []
        closest_id = (-1, -1)
        closest_distance = math.inf
        for i, mesh in enumerate(self._submeshes[body_id]):
            mesh_distances = self._sensor_distances(contact_pos, mesh)
            sub_idx = np.argmin(mesh_distances)
            distance = mesh_distances[sub_idx]
            if distance < closest_distance:
                closest_id = (i, sub_idx)
                closest_distance = distance
            distances.append(mesh_distances)
        submesh_idx, vertex_idx = closest_id

        search_distance = distance_limit*distance_margin

        # Perform the search or collect from the cache.
        candidate_sensor_idxs = self._nearest_within_distance_search(distances, body_id, submesh_idx, vertex_idx,
                                                                     search_distance)

        # Get distances for the candidate sensors and filter out those further away than the limit
        sensor_idxs = []
        sensor_distances = []
        for i in range(len(self._submeshes[body_id])):
            index_map = self._vertex_to_sensor_idx[body_id][i]  # Get the mapping between index types
            sensor_idxs_for_submesh = index_map[candidate_sensor_idxs[i]]  # Get the candidate vertices as active sensor index.
            vertex_distances = distances[i][candidate_sensor_idxs[i]]  # Get the distances of all candidate vertices
            distance_mask = vertex_distances < distance_limit   # Filter out vertices
            sensor_idxs.append(sensor_idxs_for_submesh[distance_mask])
            sensor_distances.append(vertex_distances[distance_mask])
        sensor_idxs = np.concatenate(sensor_idxs)
        sensor_distances = np.concatenate(sensor_distances)

        return sensor_idxs, sensor_distances

    def _sensor_distances(self, point, mesh):
        """ Returns the distances between a point and all sensor points on a mesh.

        This is the function that is used to determine if a vertex is within distance of a contact position.
        Optimally this would be the exact geodesic distance, but currently this is direct euclidean distance.

        Args:
            point (np.ndarray): The position.
            mesh (trimesh.Trimesh): The mesh.

        Returns:
            np.ndarray: The distances between all vertices in the mesh and the point.
        """
        return np.linalg.norm(mesh.vertices - point, axis=-1, ord=2)

    # ======================== Positions and rotations ================================
    # =================================================================================

    def get_contact_position_world(self, contact_id):
        """ Get the position of a contact in the world frame.

        Note that this is halfway between the touching geoms. Since geoms can intersect this point will likely be
        located inside both.

        Args:
            contact_id (int): The ID of the contact.

        Returns:
            np.ndarray: An array with the contact position.
        """
        return self.m_data.contact[contact_id].pos

    def get_contact_position_relative(self, contact_id, body_id):
        """ Get the position of a contact in the coordinate frame of a body.

        This position is corrected for the intersection of the bodies, i.e. it will be located at the surface of the
        sensing body.

        Args:
            contact_id (int): The ID of the contact.
            body_id (int): The ID of the body.

        Returns:
            np.ndarray: An array with the contact position.
        """
        contact_pos = env_utils.world_pos_to_body(self.m_data, self.get_contact_position_world(contact_id), body_id)
        contact = self.m_data.contact[contact_id]
        if contact.dist < 0:
            # Have to correct contact position towards surface of our body.
            # Note that distance is negative for intersecting geoms and the normal vector points into the sensing geom.
            normal = self.get_contact_normal(contact_id, body_id)
            contact_pos = contact_pos + normal * contact.dist / 2
        return contact_pos

    # =============== Raw force and contact normal ====================================
    # =================================================================================

    def get_raw_force(self, contact_id, body_id):
        """ Collect the full contact force in MuJoCos own contact frame.

        By convention the normal force points away from the first geom listed, so the forces are inverted if the first
        geom is the sensing geom.

        Args:
            contact_id (int): The ID of the contact.
            body_id (int): The relevant body in the contact. One of the geoms belonging to this body must be involved
                in the contact!

        Returns:
            np.ndarray: An array with shape (3,) with the normal force and the two tangential friction forces.
        """
        forces = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self.m_model, self.m_data, contact_id, forces)
        contact = self.m_data.contact[contact_id]
        if contact.geom1 in env_utils.get_geoms_for_body(self.m_model, body_id=body_id):
            forces *= -1  # Convention is that normal points away from geom1
        elif contact.geom2 in env_utils.get_geoms_for_body(self.m_model, body_id=body_id):
            pass
        else:
            RuntimeError("Mismatch between contact and body")
        return forces[:3]

    def get_contact_normal(self, contact_id, body_id):
        """ Returns the normal vector of contact (unit vector in direction of normal) in body coordinate frame.

        Args:
            contact_id (int): The ID of the contact.
            body_id (int): The ID of the body.

        Returns:
            np.ndarray: An array of shape (3,) containing the normal vector.
        """
        contact = self.m_data.contact[contact_id]
        normal_vector = contact.frame[:3]
        # Mujoco vectors point away from geom1 by convention
        if contact.geom1 in env_utils.get_geoms_for_body(self.m_model, body_id=body_id):
            normal_vector *= -1
        elif contact.geom2 in env_utils.get_geoms_for_body(self.m_model, body_id=body_id):
            pass
        else:
            RuntimeError("Mismatch between contact and body")
        # contact frame is in global coordinate frame, rotate to body frame
        normal_vector = env_utils.rotate_vector_transpose(normal_vector, env_utils.get_body_rotation(self.m_data, body_id))
        return normal_vector

    # =============== Valid touch force functions =====================================
    # =================================================================================

    def force_vector_global(self, contact_id, body_id):
        """ Touch function. Returns the full contact force in world frame.

        Given a contact returns the full contact force, i.e. the vector sum of the normal force and the two tangential
        friction forces, in the world coordinate frame. The body is required to account for MuJoCo conventions and
        convert coordinate frames.

        Args:
            contact_id (int): The ID of the contact.
            body_id (int): The ID of the body.

        Returns:
            np.ndarray: An array of shape (3,) with the forces.
        """
        contact = self.m_data.contact[contact_id]
        forces = self.get_raw_force(contact_id, body_id)
        force_rot = np.reshape(contact.frame, (3, 3))
        global_forces = env_utils.rotate_vector_transpose(forces, force_rot)
        return global_forces

    def force_vector(self, contact_id, body_id):
        """ Touch function. Returns full contact force in the frame of the body.

        Same as :meth:`.force_vector_global`, but the force is returned in the coordinate frame of the body.

        Args:
            contact_id (int): The ID of the contact.
            body_id (int): The ID of the body.

        Returns:
            np.ndarray: An array of shape (3,) with the forces.
        """
        global_forces = self.force_vector_global(contact_id, body_id)
        relative_forces = env_utils.rotate_vector_transpose(global_forces, env_utils.get_body_rotation(self.m_data,
                                                                                                       body_id))
        return relative_forces

    def normal_force(self, contact_id, body_id):
        """ Touch function. Returns normal force in the frame of the body.

        Args:
            contact_id (int): The ID of the contact.
            body_id (int): The ID of the body.

        Returns:
            np.ndarray: An array of shape (3,) with the normal force.
        """
        contact = self.m_data.contact[contact_id]
        forces = self.get_raw_force(contact_id, body_id)
        force_rot = np.reshape(contact.frame, (3, 3))
        # Forces[0] is the magnitude of the normal force, while the first column of the contact frame is the normal
        # vector.
        normal_force = forces[0] * force_rot[:, 0]
        normal_force = env_utils.world_rot_to_body(self.m_data, normal_force, body_id)
        return normal_force

    # =============== Output related functions ========================================
    # =================================================================================

    def get_contacts(self):
        """ Collects all active contacts involving bodies with touch sensors.

        For each active contact with a sensing geom we build a tuple ``(contact_id, body_id, forces)``, where
        `contact_id` is the ID of the contact in the MuJoCo arrays, `body_id` is the ID of the sensing body and
        `forces` is a numpy array of the raw output force, as determined by :attr:`.touch_type`.

        Returns:
            List[Tuple[int, int, np.ndarray]]: A list of tuples with contact information.
        """
        contact_tuples = []
        for i in range(self.m_data.ncon):
            contact = self.m_data.contact[i]
            body1 = self.m_model.geom(contact.geom1).bodyid.item()
            body2 = self.m_model.geom(contact.geom2).bodyid.item()
            # Do we sense this contact at all
            if self.has_sensors(body1) or self.has_sensors(body2):
                rel_bodies = []
                if self.has_sensors(body1):
                    rel_bodies.append(body1)
                if self.has_sensors(body2):
                    rel_bodies.append(body2)

                raw_forces = self.get_raw_force(i, body1)
                if abs(raw_forces[0]) < 1e-9:  # Contact probably inactive
                    continue

                for rel_body in rel_bodies:
                    forces = self.touch_function(i, rel_body)
                    contact_tuples.append((i, rel_body, forces))

        self.contact_tuples = contact_tuples
        return contact_tuples

    def get_empty_sensor_dict(self, size):
        """ Returns a dictionary with empty sensor outputs.

        Creates a dictionary with an array of zeros for each body with sensors. A body with `n` sensors has an empty
        output array of shape (n, size). The output of this function is equivalent to the touch sensor output if
        there are no contacts.

        Args:
            size (int): The size of a single sensor output.

        Returns:
            Dict[int, np.ndarray]: The dictionary of empty sensor outputs.
        """
        sensor_outputs = {}
        for body_id in self.meshes:
            sensor_outputs[body_id] = np.zeros((self.get_sensor_count(body_id), size), dtype=np.float32)
        return sensor_outputs

    def flatten_sensor_dict(self, sensor_dict):
        """ Concatenates a touch output dictionary into a single large array in a deterministic fashion.

        Output dictionaries list the arrays of sensor outputs for each body. This function concatenates these arrays
        together in a reproducible fashion to avoid key order anomalies. Bodies are sorted by their ID.

        Args:
            sensor_dict (Dict[int, np.ndarray]): The output dictionary to be concatenated.

        Returns:
            np.ndarray: The concatenated array.
        """
        sensor_arrays = []
        for body_id in sorted(self.meshes):
            sensor_arrays.append(sensor_dict[body_id])
        return np.concatenate(sensor_arrays)

    def get_touch_obs(self):
        """ Produces the current touch sensor outputs.

        Does the full contact getting-processing process, such that we get the forces, as determined by
        :attr:`.touch_type` and :attr:`.response_type`, for each sensor. :attr:`.touch_function` is called to compute
        the raw output force, which is then distributed over the sensors using :attr:`.response_function`.

        The indices of the output dictionary :attr:`~mimoTouch.touch.TrimeshTouch.sensor_outputs` and the sensor
        dictionary :attr:`.sensor_positions` are aligned, such that the ith sensor on `body` has position
        ``.sensor_positions[body][i]`` and output in ``.sensor_outputs[body][i]``.

        Returns:
            np.ndarray: An array containing all the touch sensations.
        """
        contact_tuples = self.get_contacts()
        self.sensor_outputs = self.get_empty_sensor_dict(self.touch_size)  # Initialize output dictionary

        for contact_id, body_id, forces in contact_tuples:
            # At this point we already have the forces for each contact, now we must attach/spread them to sensor
            # points, based on the adjustment function
            self.response_function(contact_id, body_id, forces)

        sensor_obs = self.flatten_sensor_dict(self.sensor_outputs)
        return sensor_obs

    # =============== Force adjustment functions ======================================
    # =================================================================================

    def spread_linear(self, contact_id, body_id, force):
        """ Response function. Distributes the output force linearly based on distance.

        For a contact and a raw force we get all sensors within a given distance to the contact point and then
        distribute the force such that the force reported at a sensor decreases linearly with distance between the
        sensor and the contact point. Finally, the total force is normalized such that the total force over all sensors
        for this contact is identical to the raw force. The scaling distance is given by double the distance between
        sensor points.

        Args:
            contact_id (int): The ID of the contact.
            body_id (int): The ID of the sensing body.
            force (np.ndarray): The raw force.
        """
        # Get all sensors within distance (distance here is just double the sensor scale)
        scale = self.sensor_scales[body_id]
        search_scale = 3*scale
        adjustment_scale = 2*scale
        contact_pos = self.get_contact_position_relative(contact_id=contact_id, body_id=body_id)
        nearest_sensors, sensor_distances = self.get_sensors_within_distance(contact_pos, body_id, search_scale)

        sensor_adjusted_forces = scale_linear(force, sensor_distances, scale=adjustment_scale)
        force_total = abs(np.sum(sensor_adjusted_forces[:, 0]))

        factor = abs(force[0] / force_total) if force_total > EPS else 0
        self.sensor_outputs[body_id][nearest_sensors] += sensor_adjusted_forces * factor

    def nearest(self, contact_id, body_id, force):
        """ Response function. Adds the output force directly to the nearest sensor.

        Args:
            contact_id (int): The ID of the contact.
            body_id (int): The ID of the body.
            force (np.ndarray): The raw output force.
        """
        # Get the nearest sensor to this contact, add the force to it
        contact_pos = self.get_contact_position_relative(contact_id=contact_id, body_id=body_id)
        nearest_sensor, distance = self.get_nearest_sensor(contact_pos, body_id)
        self.sensor_outputs[body_id][nearest_sensor] += force

    # =============== Visualizations ==================================================
    # =================================================================================

    # Plot sensor points for single geom
    def plot_sensors_body(self, body_id=None, body_name=None, title=""):
        """ Plots the sensor positions for a body.

        Given either an ID or the name of a body, plot the positions of the sensors on that body.

        Args:
            body_id (int|None): The ID of the body.
            body_name (str|None): The name of the body. This is ignored if the ID is provided!
            title (str): The title of the plot. Empty by default.
        """
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)

        points = self.meshes[body_id].vertices
        limit = self.plotting_limits[body_id]
        title = self.m_model.body(body_id).name
        env_utils.plot_points(points, limit=limit, title=title)

    # Plot forces for single body
    def plot_force_body(self, body_id=None, body_name=None, title=""):
        """ Plot the sensor output for a body.

        Given either an ID or the name of a body, plots the positions and outputs of the sensors on that body.

        Args:
            body_id (int|None): The ID of the body.
            body_name (str|None): The name of the body. This is ignored if the ID is provided!
            title (str): The title of the plot. Empty by default.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple (fig, ax) with the pyplot figure and axis objects.
        """
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)

        sensor_points = self.sensor_positions[body_id]
        force_vectors = self.sensor_outputs[body_id] / 20
        title = self.m_model.body(body_id).name + " forces"
        if force_vectors.shape[1] == 1:
            normals = self.meshes[body_id].vertex_normals[self.active_vertices[body_id], :]
            force_vectors = force_vectors * normals
        fig, ax = env_utils.plot_forces(sensor_points, force_vectors, limit=np.amax(sensor_points)*1.2, title=title,
                                        show=False)
        return fig, ax

    # Plot forces for list of bodies.
    def plot_force_bodies(self, body_ids=[], body_names=[],
                          title="", focus="world", show_contact_points=True):
        """ Plot the sensor output for a list of bodies.

        Given a list of bodies, either by ID or by name, plot the positions and outputs of all sensors on the bodies.
        The current relative positions and orientations of the bodies in the simulation are respected.
        The parameter `focus` determines how the coordinates are centered. Two options exist:
        - 'world':   In this setting all the coordinates are translated into global coordinates
        - 'first':   In this setting all the coordinates are translated into the frame of the first body in the list.

        Args:
            body_ids (List[int]): A list of IDs of the bodies that should be plotted.
            body_names (List[str]): A list of the names of the bodies that should be plotted. This is ignored if `body_ids` is
                provided!
            title (str): The title of the plot. Empty by default.
            focus (str): Coordinates are moved into a consistent reference frame. This parameter determines that
                reference frame. Must be one of ``["world", "first"]``. Default "world".
            show_contact_points (bool): If ``True`` the actual contact points are also plotted. Note that these are
                corrected for intersecting bodies. Default ``True``.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple (fig, ax) with the pyplot figure and axis objects.
        """
        assert len(body_ids) > 0 or len(body_names) > 0
        assert focus in ["world", "first"]
        if len(body_ids) == 0:
            body_ids = [env_utils.get_body_id(self.m_model, body_name=body_name) for body_name in body_names]

        points = []
        forces = []
        for body_id in body_ids:
            sensor_points = self.sensor_positions[body_id]
            force_vectors = self.sensor_outputs[body_id] / 20
            if force_vectors.shape[1] == 1:
                normals = self.meshes[body_id].vertex_normals[self.active_vertices[body_id], :]
                force_vectors = force_vectors * normals
            # Convert sensor locations and force vectors for this body into appropriate common coordinates
            if focus == "world":
                sensor_points_common = env_utils.body_pos_to_world(self.m_data, position=sensor_points, body_id=body_id)
                forces_common = env_utils.body_rot_to_world(self.m_data, vector=force_vectors, body_id=body_id)
            else:       # Focus == first
                sensor_points_common = env_utils.body_pos_to_body(self.m_data, position=sensor_points,
                                                                  body_id_source=body_id, body_id_target=body_ids[0])
                forces_common = env_utils.body_rot_to_body(self.m_data, vector=force_vectors,
                                                           body_id_source=body_id, body_id_target=body_ids[0])
            points.append(sensor_points_common)
            forces.append(forces_common)
        points = np.concatenate(points)
        forces = np.concatenate(forces)
        limit = np.amax(np.abs(points))*1.2
        fig, ax = env_utils.plot_forces(points=points, vectors=forces, limit=limit, title=title, show=False)

        if show_contact_points:
            for contact_id, contact_body_id, forces in self.contact_tuples:
                if contact_body_id in body_ids:
                    contact_position = self.get_contact_position_relative(contact_id, contact_body_id)
                    if focus == "world":
                        contact_position = env_utils.body_pos_to_world(self.m_data,
                                                                       position=contact_position,
                                                                       body_id=contact_body_id)
                    else:
                        contact_position = env_utils.body_pos_to_body(self.m_data,
                                                                      position=contact_position,
                                                                      body_id_source=contact_body_id,
                                                                      body_id_target=body_ids[0])
                    ax.scatter(contact_position[0], contact_position[1], contact_position[2],
                               color="y", s=15, depthshade=True, alpha=0.8)
        return fig, ax

    def plot_force_body_subtree(self, body_id=None, body_name=None, title="", show_contact_points=False):
        """ Plot the sensor output for the kinematic subtree with the given body at its root.

        Given a body, collects all descendant bodies in the kinematic tree and  plot the positions and outputs of their
        sensors. The current relative positions and orientations of the bodies in the simulation are respected and all
        coordinates are moved into the coordinate frame of the root body.

        Args:
            body_id (int|None): The ID of the root body for the subtree. Either this or `body_name` must be supplied.
            body_name (str|None): The name of the root body. Either this or `body_id` must be supplied. If both are
                provided, `body_name` is ignored.
            title (str): The title of the plot. Empty by default.
            show_contact_points (bool): If ``True``, the actual rigid body contact points are also plotted.
                Default ``False``.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple (fig, ax) with the pyplot figure and axis objects.
        """
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)
        # Go through all bodies and note their child bodies
        subtree = env_utils.get_child_bodies(self.m_model, body_id)
        fig, ax = self.plot_force_bodies(body_ids=subtree, title=title, focus="world",
                                         show_contact_points=show_contact_points)
        return fig, ax

    def visualize_contacts_subtree(self, root_id=None, root_name=None, show_contact_points=False, focus_body=None,
                                   camera_offset=None):
        """ Generates a neat visualization of parts of MIMo and their contact forces.

        Starting from a root body, renders the child parts of MIMo and the contact sensors on them. Contact forces are
        shown by coloring the sensor points and increasing their size corresponding to the contact force. Unlike the
        other force plotting functions this one only visualized the magnitude of the contact force, not the direction.

        For convenience the render can be centered on a body, with an additional offset for the camera. This allows
        controlling the camera placement.

        Args:
            root_id (int|None): The id of the root body. Either this or `root_name` must be supplied.
            root_name (str|None): The name of the root body. Either this or `root_id` must be supplied. If both are
                provided, `root_name` is ignored.
            show_contact_points (bool): Whether to render the MuJoCo point contacts. Default ``False``.
            focus_body (str|None): The body, by name, on which the render will be centered. If ``None``, the bodies
                will be at their coordinates as in the scene.
            camera_offset (np.ndarray|None): An array with a camera position offset. This allows moving the
                camera relative to the bodies. Must have shape (3,). If ``None``, there is no offset. Default ``None``.

        Returns:
            Tuple[plt.Figure, plt.Axes]: A tuple (fig, ax) with the pyplot figure and axis objects.
        """

        def get_bodypart_vis(body_id, offsets=0):
            meshes = []
            for geom_id in env_utils.get_geoms_for_body(self.m_model, body_id):
                mesh = self._get_mesh(geom_id, scale=.001)
                mesh.vertices = env_utils.geom_pos_to_body(self.m_data, mesh.vertices.copy(), geom_id, body_id)
                meshes.append(mesh)
            body_mesh = trimesh.util.concatenate(meshes)
            vertex_position = env_utils.body_pos_to_world(self.m_data, position=body_mesh.vertices,
                                                          body_id=body_id) + offsets
            triangles = body_mesh.faces
            return vertex_position, triangles

        root_id = env_utils.get_body_id(self.m_model, body_id=root_id, body_name=root_name)
        # Go through all bodies and note their child bodies
        subtree = env_utils.get_child_bodies(self.m_model, root_id)
        points_no_contact = []
        points_contact = []
        contact_magnitudes = []
        for body_id in subtree:
            sensor_points = self.sensor_positions[body_id]
            force_vectors = self.sensor_outputs[body_id]

            force_magnitude = np.linalg.norm(force_vectors, axis=-1, ord=2)
            no_touch_points = sensor_points[force_magnitude <= 1e-7]
            touch_points = sensor_points[force_magnitude > 1e-7]
            no_touch_points = env_utils.body_pos_to_world(self.m_data, position=no_touch_points, body_id=body_id)
            touch_points = env_utils.body_pos_to_world(self.m_data, position=touch_points, body_id=body_id)

            points_no_contact.append(no_touch_points)
            points_contact.append(touch_points)
            contact_magnitudes.append(force_magnitude[force_magnitude > 1e-7])

        points_gray = np.concatenate(points_no_contact)
        points_red = np.concatenate(points_contact)
        forces = np.concatenate(contact_magnitudes)
        size_min = 10
        size_max = 80
        sizes = forces / np.amax(forces) * (size_max - size_min) + size_min
        opacity_min = 0.4
        opacity_max = 0.8
        opacities = forces / np.amax(forces) * (opacity_max - opacity_min) + opacity_min
        # Opacities can't be set as an array, so must be set using color array
        red_colors = np.tile(np.array([1.0, 0, 0, 0]), (points_red.shape[0], 1))
        red_colors[:, 3] = opacities

        if focus_body:
            target_pos = self.m_data.get_body_xpos(focus_body)
        else:
            target_pos = np.zeros((3,))

        # Subtract all by ball position to center on ball
        xs_gray = points_gray[:, 0] - target_pos[0] + camera_offset[0]
        ys_gray = points_gray[:, 1] - target_pos[1] + camera_offset[1]
        zs_gray = points_gray[:, 2] - target_pos[2] + camera_offset[2]

        xs_red = points_red[:, 0] - target_pos[0] + camera_offset[0]
        ys_red = points_red[:, 1] - target_pos[1] + camera_offset[1]
        zs_red = points_red[:, 2] - target_pos[2] + camera_offset[2]

        fig = plt.figure(figsize=(5, 5), dpi=200)
        ax = fig.add_subplot(111, projection='3d')
        # Draw sensor points
        ax.scatter(xs_gray, ys_gray, zs_gray, color="k", s=15, depthshade=False, alpha=.15)
        ax.scatter(xs_red, ys_red, zs_red, color=red_colors, s=sizes, depthshade=False)

        # Draw contact points
        if show_contact_points:
            for body_id in subtree:
                for contact_id, contact_body_id, forces in self.contact_tuples:
                    if body_id == contact_body_id:
                        contact_position = self.get_contact_position_relative(contact_id, body_id)
                        contact_position = env_utils.body_pos_to_world(self.m_data,
                                                                       position=contact_position,
                                                                       body_id=body_id)
                        ax.scatter(contact_position[0], contact_position[1], contact_position[2],
                                   color="y", s=15, depthshade=True, alpha=0.8)

        # Draw body parts
        for body_id in subtree:
            vertex_pos, tris = get_bodypart_vis(body_id, offsets=camera_offset - target_pos)
            ax.plot_trisurf(vertex_pos[:, 0], vertex_pos[:, 1], triangles=tris, Z=vertex_pos[:, 2], color="tab:orange",
                            alpha=0.05, shade=True)

        dists = np.linalg.norm(np.stack([xs_gray, ys_gray, zs_gray], axis=-1), axis=-1, ord=2)
        limit = np.amax(np.abs(dists)) * 4
        ax.set_xlim([-limit, limit])
        ax.set_ylim([-limit, limit])
        ax.set_zlim([-limit, limit])
        ax.set_box_aspect((1, 1, 1))
        ax.set_axis_off()

        return fig, ax


_rng = np.random.default_rng()
_vectors = env_utils.normalize_vectors(_rng.normal(size=(10, 3)))


def _contains(points, mesh, directions=_vectors, tol=1e-10):
    """ Check whether points are inside a mesh.

    Points are checked elementwise. The check is performed by performing the ray intersection count using multiple rays.

    Args:
        points (np.ndarray): An array of points. Should have shape (n, 3).
        mesh (trimesh.Trimesh): The mesh.
        directions (np.ndarray): A set of directions for the check rays. Randomized by default.
        tol (float): Rays are sent from the points, offset by this tolerance. This prevents points on a surface from
             intersecting with that surface.

    Returns:
        np.ndarray: A boolean array array of shape (n,). Entry `i` is ``True`` if sensor point `i` is located inside
        the mesh.
    """
    contains = np.zeros(points.shape[0], dtype=bool)
    for i in range(directions.shape[0]):
        direction = np.tile(directions[i], (points.shape[0], 1))
        origins = points - direction * tol
        intersections, ray_indices, _ = mesh.ray.intersects_location(
            ray_origins=origins,
            ray_directions=direction,
        )
        ray_hits = np.bincount(ray_indices, minlength=points.shape[0])

        contained = np.mod(ray_hits, 2) == 1
        contains = np.logical_or(contains, contained)

    return contains
