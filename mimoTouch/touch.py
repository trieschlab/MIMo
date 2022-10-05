""" This module defines the touch system interface and provides two implementations.

The interface is defined as an abstract class in :class:`~mimoTouch.touch.Touch`.
A simple implementation with a cloud of sensor points is in :class:`~mimoTouch.touch.DiscreteTouch`.
A second implementation using trimesh objects is in :class:`~mimoTouch.touch.TrimeshTouch`.
This second implementation allows for consideration of sensor normals as well as surface distance, avoiding the issue
of a contact penetrating through to sensors on the opposite side of the sensing body.

Both of the implementations also have functions for visualizing the touch sensations.
"""

import math
from collections import deque
from typing import Dict, List

import numpy as np
import mujoco_py
from mujoco_py.generated import const
import trimesh
from trimesh import PointCloud

import mimoEnv.utils as env_utils
from mimoEnv.utils import rotate_vector_transpose, rotate_vector, EPS
from mimoTouch.sensorpoints import spread_points_box, spread_points_sphere, spread_points_cylinder, \
                                   spread_points_capsule

#: A key to identify the geom type ids used by MuJoCo.
from mimoTouch.sensormeshes import mesh_box, mesh_sphere, mesh_capsule, mesh_cylinder, mesh_ellipsoid


class Touch:
    """ Abstract base class for the touch system.

    This class defines the functions that all implementing classes must provide. :meth:`.get_touch_obs` should perform
    the whole sensory pipeline as defined in the configuration and return the output as a single array.
    Additionally the output for each body part should be stored in :attr:`.sensor_outputs`. The exact definition of
    'body part' is left to the implementing class.

    The constructor takes two arguments, `env` and `touch_params`:
    `env` should be an openAI gym environment using MuJoCo, while `touch_params` is a configuration dictionary. The
    exact form will depend on the specific implementation, but it must contain these three entries:

    - "scales", which lists the distance between sensor points for each body part.
    - "touch_function", which defines the output type and must be in :attr:`.VALID_TOUCH_TYPES`.
    - "response_function", which defines how the contact forces are distributed to the sensors. Must be one of
      :attr:`.VALID_RESPONSE_FUNCTIONS`.

    The sensor scales determines the density of the sensor points, while `touch_function` and `response_function`
    determine the type of contact output and how it is distributed. `touch_function` and `response_function` refer to
    class methods by name and must be listed in :attr:`.VALID_TOUCH_TYPES`. and :attr:`.VALID_RESPONSE_FUNCTIONS`
    respectively. Different touch functions should be used to support different types of output, such as normal force,
    frictional forces or contact slip. The purpose of the response function is to loosely simulate surface behaviour.
    How exactly these functions work and interact is left to the implementing class.
    Note that the bodies listed in "scales" must actually exist in the scene to avoid errors!

    Attributes:
        env: The environment to which this module will be attached.
        sensor_scales: A dictionary listing the sensor distances for each body part. Populated from `touch_params`.
        touch_type: The name of the member method that determines output type. Populated from `touch_params`.
        touch_function: A reference to the actual member method determined by `touch_type`.
        touch_size: The size of the output of a single sensor for the given touch type.
        response_type: The name of the member method that determines how the output is distributed over the sensors.
            Populated from `touch_params`.
        response_function: A reference to the actual member method determined by `response_type`.
        sensor_positions: A dictionary containing the positions of the sensor points for each body part. The
            coordinates should be in the frame of the associated body part.
        sensor_outputs: A dictionary containing the outputs produced by the sensors. Shape will depend on the specific
            implementation. This should be populated by :meth:`.get_touch_obs`

    """

    #: A dictionary listing valid touch output types and their sizes.
    VALID_TOUCH_TYPES = {}
    #: A list of valid surface response functions.
    VALID_RESPONSE_FUNCTIONS = []

    def __init__(self, env, touch_params):
        self.env = env

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
        self.response_type = touch_params["response_function"]
        assert self.response_type in self.VALID_RESPONSE_FUNCTIONS
        self.response_function = getattr(self, self.response_type)

        self.sensor_outputs = {}
        self.sensor_positions = {}

    def get_touch_obs(self):
        """ Produces the current touch output.

        This function should perform the whole sensory pipeline as defined in `touch_params` and return the output as a
        single array. The per-body output should additionally be stored in :attr:`.sensor_outputs`.

        Returns:
            A numpy array of shape (n_sensor_points, touch_size)
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
        m_data: A direct reference to the MuJoCo simulation data object.
        m_model: A direct reference to the MuJoCo simulation model object.
        plotting_limits: A convenience dictionary listing axis limits for plotting forces or sensor points for geoms.

    """

    VALID_TOUCH_TYPES = {
        "normal": 1,
        "force_vector": 3,
        "force_vector_global": 3,
    }

    VALID_RESPONSE_FUNCTIONS = ["nearest", "spread_linear"]

    def __init__(self, env, touch_params):

        super().__init__(env, touch_params)
        self.m_data = env.sim.data
        self.m_model = env.sim.model

        self.plotting_limits = {}

        # Add sensors to bodies
        for body_id in self.sensor_scales:
            self.add_body(body_id, scale=self.sensor_scales[body_id])

        # Get touch obs once to ensure all output arrays are initialized
        self.get_touch_obs()
        
    def add_body(self, body_id: int = None, body_name: str = None, scale: float = math.inf):
        """ Adds sensors to all geoms belonging to the given body.

        Given a body, either by ID or name, spread sensor points over all geoms and add them to the output. If both ID
        and name are provided the name is ignored. The distance between sensor points is determined by `scale`.
        The names are determined by the scene XMLs while the IDs are assigned during compilation.

        Args:
            body_id: ID of the body.
            body_name: Name of the body. If `body_id` is provided this parameter is ignored!
            scale: The distance between sensor points.

        Returns:
            The number of sensors added to this body.

        """
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)

        n_sensors = 0

        for geom_id in env_utils.get_geoms_for_body(self.m_model, body_id):
            g_body_id = self.m_model.geom_bodyid[geom_id]
            contype = self.m_model.geom_contype[geom_id]
            conaffinity = self.m_model.geom_conaffinity[geom_id]
            # Add a geom if it belongs to body and has collisions enabled (at least potentially)
            if g_body_id == body_id and (contype > 0 or conaffinity > 0):
                n_sensors += self.add_geom(geom_id=geom_id, scale=scale)
        return n_sensors

    def add_geom(self, geom_id: int = None, geom_name: str = None, scale: float = math.inf):
        """ Adds sensors to the given geom.

        Spreads sensor points over the geom identified either by ID or name and add them to the output. If both ID and
        name are provided the name is ignored. The distance between sensor points is determined by `scale`. The names
        are determined by the scene XMLs while the IDs are assigned during compilation.

        Args:
            geom_id: ID of the geom.
            geom_name: Name of the geom.  If `geom_id` is provided this parameter is ignored!
            scale: The distance between sensor points.

        Returns:
            The number of sensors added to this geom.

        """
        geom_id = env_utils.get_geom_id(self.m_model, geom_id=geom_id, geom_name=geom_name)

        if self.m_model.geom_contype[geom_id] == 0 and self.m_model.geom_conaffinity[geom_id] == 0:
            raise RuntimeWarning("Added sensors to geom with collisions disabled!")
        return self._add_sensorpoints(geom_id, scale)

    @property
    def sensing_geoms(self):
        """ Returns the IDs of all geoms with sensors.

        Returns:
            A list with the IDs for all geoms that have sensors.

        """
        return list(self.sensor_positions.keys())

    def has_sensors(self, geom_id):
        """ Returns True if the geom has sensors.

        Args:
            geom_id: The ID of the geom.

        Returns:
            True if the geom has sensors, False otherwise.

        """
        return geom_id in self.sensor_positions

    def get_sensor_count(self, geom_id):
        """ Returns the number of sensors for the geom.

        Args:
            geom_id: The ID of the geom.

        Returns:
            The number of sensor points for this geom.

        """
        return self.sensor_positions[geom_id].shape[0]

    def get_total_sensor_count(self):
        """ Returns the total number of touch sensors in the model.

        Returns:
            The total number of touch sensors in the model.

        """
        n_sensors = 0
        for geom_id in self.sensing_geoms:
            n_sensors += self.get_sensor_count(geom_id)
        return n_sensors

    def _add_sensorpoints(self, geom_id: int, scale: float):
        """ Adds sensors to the given geom.

        Spreads sensor points over the geom identified either by ID. The distance between sensor points is determined
        by `scale`. We identify the type of geom using the MuJoCo API. This function populates
        both :attr:`.sensor_positions` and :attr:`.plotting_limits`.

        Args:
            geom_id: ID of the geom.
            scale: The distance between sensor points.

        Returns:
            The number of sensors added to this geom.

        """
        # Add sensor points for the given geom using given resolution
        # Returns the number of sensor points added
        # Also set the maximum size of the geom, for plotting purposes
        geom_type = self.m_model.geom_type[geom_id]
        size = self.m_model.geom_size[geom_id]
        limit = 1
        if geom_type == const.GEOM_BOX:
            limit = np.max(size)
            points = spread_points_box(scale, size)
        elif geom_type == const.GEOM_SPHERE:
            limit = size[0]
            points = spread_points_sphere(scale, size[0])
        elif geom_type == const.GEOM_CAPSULE:
            limit = size[1] + size[0]
            points = spread_points_capsule(scale, 2*size[1], size[0])
        elif geom_type == const.GEOM_CYLINDER:
            # Cylinder size 0 is radius, size 1 is half length
            limit = np.max(size)
            points = spread_points_cylinder(scale, 2*size[1], size[0])
        elif geom_type == const.GEOM_PLANE:
            RuntimeWarning("Cannot add sensors to plane geoms!")
            return None
        elif geom_type == const.GEOM_ELLIPSOID:
            raise NotImplementedError("Ellipsoids currently not implemented")
        elif geom_type == const.GEOM_MESH:
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

        Contact IDs are a MuJoCo attribute, see their documentation for more detail on contacts.

        Args:
            contact_id: The ID of the contact.
            geom_id: The ID of the geom. The geom must have sensors!

        Returns:
            The index of the closest sensor and the distance between contact and sensor.

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
            contact_id: The ID of the contact.
            geom_id: The ID of the geom. The geom must have sensors!
            k: The number of sensors to return.

        Returns:
            The indices of the k-nearest sensors to the contact.

        """
        relative_position = self.get_contact_position_relative(contact_id, geom_id)
        sensor_points = self.sensor_positions[geom_id]
        distances = np.linalg.norm(sensor_points - relative_position, axis=1)
        sorted_idxs = np.argpartition(distances, k)
        return sorted_idxs[:k], distances[sorted_idxs[:k]]

    def get_sensors_within_distance(self, contact_id, geom_id, distance):
        """ Finds all sensors on a geom that are within a given distance to a contact.

        The distance used is the direct euclidean distance. Contact IDs are a MuJoCo attribute, see their documentation
        for more detail on contacts.

        Args:
            contact_id: The ID of the contact.
            geom_id: The ID of the geom. The geom must have sensors!
            distance: Sensors must be within this distance to the contact position to be included in the output.

        Returns:
            The indices of all sensors on the geom that are within `distance` to the contact.

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
            contact_id: The ID of the contact.

        Returns:
            A numpy array with the position of the contact.

        """
        return self.m_data.contact[contact_id].pos

    def get_contact_position_relative(self, contact_id, geom_id: int):
        """ Get the position of a contact in the coordinate frame of a geom.

        Args:
            contact_id: The ID of the contact.
            geom_id: The ID of the geom.

        Returns:
            A numpy array with the position of the contact.

        """
        body_pos = env_utils.world_pos_to_geom(self.m_data, self.get_contact_position_world(contact_id), geom_id)
        contact = self.m_data.contact[contact_id]
        if contact.dist < 0:
            # Have to correct contact position towards surface of our body.
            # Note that distance is negative for intersecting geoms and the normal vector points into the sensing geom.
            normal = self.get_contact_normal(contact_id, geom_id)
            body_pos = body_pos + normal * contact.dis / 2
        return body_pos

    # =============== Visualizations ==================================================
    # =================================================================================

    def plot_sensors_geom(self, geom_id: int = None, geom_name: str = None):
        """ Plots the sensor positions for a geom.

        Given either an ID or the name of a geom, plot the positions of the sensors on that geom.

        Args:
            geom_id: The ID of the geom.
            geom_name: The name of the geom. This is ignored if the ID is provided!

        """
        geom_id = env_utils.get_geom_id(self.m_model, geom_id=geom_id, geom_name=geom_name)

        points = self.sensor_positions[geom_id]
        limit = self.plotting_limits[geom_id]
        title = self.m_model.geom_id2name(geom_id)
        env_utils.plot_points(points, limit=limit, title=title)

    def plot_force_geom(self, geom_id: int = None, geom_name: str = None):
        """ Plot the sensor output for a geom.

        Given either an ID or the name of a geom, plots the positions and outputs of the sensors on that geom.

        Args:
            geom_id: The ID of the geom.
            geom_name: The name of the geom. This is ignored if the ID is provided!

        """
        geom_id = env_utils.get_geom_id(self.m_model, geom_id=geom_id, geom_name=geom_name)

        sensor_points = self.sensor_positions[geom_id]
        force_vectors = self.sensor_outputs[geom_id]
        if force_vectors.shape[1] == 1:
            # TODO: Need proper sensor normals, can't do this until trimesh rework
            raise RuntimeWarning("Plotting of scalar forces not implemented!")
        else:
            env_utils.plot_forces(sensor_points, force_vectors, limit=np.max(sensor_points) + 0.5)

    def _get_plot_info_body(self, body_id):
        """ Collects sensor points and forces for a single body.

        Args:
            body_id: The ID of the body.

        Returns:
            (points, forces) Two numpy arrays containing the sensor positions and their outputs.
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

    def plot_force_body(self, body_id: int = None, body_name: str = None):
        """ Plots sensor points and output forces for all geoms in a body.

        Given either an ID or the name of a body, plots the positions and outputs of the sensors for all geoms
        associated with that body.

        Args:
            body_id: The ID of the body.
            body_name: The name of the body. This argument is ignored if the ID is provided.

        """
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)

        points, forces = self._get_plot_info_body(body_id)

        env_utils.plot_forces(points, forces, limit=np.max(points) + 0.5)

    # TODO: Plot forces for body subtree

    # =============== Raw force and contact normal ====================================
    # =================================================================================

    def get_raw_force(self, contact_id, geom_id):
        """ Collect the full contact force in MuJoCos own contact frame.

        By convention the normal force points away from the first geom listed, so the forces are inverted if the first
        geom is the sensing geom.

        Args:
            contact_id: The ID of the contact.
            geom_id: The relevant geom in the contact. Must be one of the geoms involved in the contact!

        Returns:
            A 3d vector containing the normal force and the two tangential friction forces.

        """
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
        """ Returns the normal vector of contact (unit vector in direction of normal) in geom coordinate frame.

        Args:
            contact_id: The ID of the contact.
            geom_id: The ID of the geom.

        Returns:
            A 3d vector containing the normal vector.

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

    def normal(self, contact_id, geom_id) -> float:
        """ Touch function. Returns the normal force as a scalar.

        Given a contact and a geom, returns the normal force of the contact. The geom is required to account for the
        MuJoCo contact conventions.

        Args:
            contact_id: The ID of the contact.
            geom_id: The ID of the geom.

        Returns:
            The normal force as a float.
        """
        return self.get_raw_force(contact_id, geom_id)[0]

    def force_vector_global(self, contact_id, geom_id):
        """ Touch function. Returns the full contact force in world frame.

        Given a contact returns the full contact force, i.e. the vector sum of the normal force and the two tangential
        friction forces, in the world coordinate frame. The geom is required to account for MuJoCo conventions and
        convert coordinate frames.

        Args:
            contact_id: The ID of the contact.
            geom_id: The ID of the geom.

        Returns:
            A 3d vector of the forces.

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
            contact_id: The ID of the contact.
            geom_id: The ID of the geom.

        Returns:
            A 3d vector of the forces.

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
            A list of tuples with contact information.

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
        output array of shape `(n, size)`. The output of this function is equivalent to the touch sensor output if
        there are no contacts.

        Args:
            size: The size of a single sensor output.

        Returns:
            The dictionary of empty sensor outputs.

        """
        sensor_outputs = {}
        for geom_id in self.sensor_positions:
            sensor_outputs[geom_id] = np.zeros((self.get_sensor_count(geom_id), size), dtype=np.float32)
        return sensor_outputs

    def flatten_sensor_dict(self, sensor_dict):
        """ Flattens a touch output dictionary into a single large array in a deterministic fashion.

        Output dictionaries list the arrays of sensor outputs for each geom. This function concatenates these arrays
        together in a reproducible fashion to avoid key order anomalies. Geoms are sorted by their ID.

        Args:
            sensor_dict: The output dictionary to be flattened.

        Returns:
            A single concatenated numpy array.

        """
        sensor_arrays = []
        for geom_id in sorted(self.sensor_positions):
            sensor_arrays.append(sensor_dict[geom_id])
        return np.concatenate(sensor_arrays)

    def get_touch_obs(self) -> np.ndarray:
        """ Produces the current touch sensor outputs.

        Does the full contact getting-processing process, such that we get the forces, as determined by
        :attr:`.touch_type` and :attr:`.response_type`, for each sensor. :attr:`.touch_function` is called to compute
        the raw output force, which is then distributed over the sensors using :attr:`.response_function`.

        The indices of the output dictionary :attr:`~mimoTouch.touch.DiscreteTouch.sensor_outputs` and the sensor
        dictionary :attr:`.sensor_positions` are aligned, such that the ith sensor on `geom` has position
        ``.sensor_positions[geom][i]`` and output in ``.sensor_outputs[geom][i]``.

        Returns:
            A numpy array containing all the touch sensations.

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
        sensor and the contact point. Finally the total force is normalized such that the total force over all sensors
        for this contact is identical to the raw force. The scaling distance is given by double the distance between
        sensor points.

        Args:
            contact_id: The ID of the contact.
            geom_id: The ID of the sensing geom.
            force: The raw force.

        """
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
        """ Response function. Adds the output force directly to the nearest sensor.

        Args:
            contact_id: The ID of the contact.
            geom_id: The ID of the geom.
            force: The raw output force.

        """
        # Get the nearest sensor to this contact, add the force to it
        nearest_sensor, distance = self.get_nearest_sensor(contact_id, geom_id)
        self.sensor_outputs[geom_id][nearest_sensor] += force


# =============== Scaling functions ===============================================
# =================================================================================

def scale_linear(force, distance, scale):
    """ Used to scale forces linearly based on distance.

    Adjusts the force by a simple factor, such that force falls linearly from full at `distance = 0`
    to 0 at `distance >= scale`.

    Args:
        force: The unadjusted force.
        distance: The adjusted force reduces linearly with increasing distance.
        scale: The scaling limit. If 'distance >= scale' the return value is reduced to 0.

    Returns:
        The scaled force.

    """
    factor = (scale-distance) / scale
    if factor < 0:
        factor = 0
    out_force = force * factor
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
          the sensing geom.
        - 'force_vector_global': Like 'force_vector', but reported in the world coordinate frame instead.

        The output can be spread to nearby sensors in two different ways:

        - 'nearest': Directly add the output to the nearest sensor.
        - 'spread_linear': Spreads the force to nearby sensor points, such that it decreases linearly with distance to
          the contact point. The force drops to 0 at twice the sensor scale. The force per sensor is normalised such
          that the total force is conserved.

        Touch functions return their output, while response functions do not return anything and instead write their
        adjusted forces directly into the output dictionary.

        The following attributes are provided in addition to those of :class:`~mimoTouch.touch.Touch`.

        Attributes:
            m_data: A direct reference to the MuJoCo simulation data object.
            m_model: A direct reference to the MuJoCo simulation model object.
            meshes: A dictionary containing the sensor mesh objects for each body.
            active_vertices: A dictionary of masks. Not every sensor point will be active as they may intersect another
                geom on the same body. Only active vertices contribute to the output, but inactive ones are still
                required for mesh operations. If a sensor is active the associated entry in this dictionary will be
                `True`, otherwise `False`.
            plotting_limits: A convenience dictionary listing axis limits for plotting forces or sensor points for
                geoms.
            _submeshes: A dictionary like :attr:`.meshes`, but storing a list of the individual geom meshes instead.
            _active_subvertices: A dictionary like :attr:`.active_vertices`, but storing a list of masks for each geom
                mesh instead.
            _vertex_to_sensor_idx: A dictionary that maps the indices for each active vertex. Calculations happen on
                submeshes, so the indices have to mapped onto the output array. This dictionary stores that mapping.
            _neighbour_cache: A dictionary with (body_id, sensor_id) tuples as key, storing the nearest neighbours for
                the given sensor as a list.

        """

    VALID_TOUCH_TYPES = {
        "force_vector": 3,
        "force_vector_global": 3,
    }

    VALID_RESPONSE_FUNCTIONS = ["nearest", "spread_linear"]

    def __init__(self, env, touch_params):
        super().__init__(env, touch_params=touch_params)
        self.m_model = self.env.sim.model
        self.m_data = self.env.sim.data

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
        self.sensor_positions = {}

        self._neighbour_cache = {}

        self.plotting_limits = {}

        # Add sensors to bodies
        for body_id in self.sensor_scales:
            self.add_body(body_id=body_id, scale=self.sensor_scales[body_id])

        # Get touch obs once to ensure all output arrays are initialized
        self.get_touch_obs()

    # ======================== Sensor related functions ===============================
    # =================================================================================

    def add_body(self, body_id: int = None, body_name: str = None, scale: float = math.inf):
        """ Adds sensors to the given body.

        Given a body, either by ID or name, spread sensor meshes over all geoms and adds them to the output. If both ID
        and name are provided the name is ignored. The distance between sensor points is determined by `scale`.
        This function has to handle all the arrays required for quick access after initialization, so it populates the
        submesh, mesh, mask and index mapping dictionaries.
        The names are determined by the scene XMLs while the IDs are assigned during compilation.

        Args:
            body_id: ID of the body.
            body_name: Name of the body. If `body_id` is provided this parameter is ignored!
            scale: The distance between sensor points.

        """
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)
        meshes = []
        for geom_id in env_utils.get_geoms_for_body(self.m_model, body_id):
            contype = self.m_model.geom_contype[geom_id]
            conaffinity = self.m_model.geom_conaffinity[geom_id]
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
        # TODO: More filtering to remove vertices too close to one another near geom intersections
        for mesh in meshes:
            mask = np.ones((mesh.vertices.shape[0],), dtype=bool)
            for other_mesh in meshes:
                if other_mesh == mesh or isinstance(other_mesh, PointCloud):
                    continue
                # If our vertices are contained in the other mesh, make them inactive
                # TODO: This has weird inconcistencies, check those
                contained = other_mesh.contains(mesh.vertices)
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
            A list with the IDs for all bodies that have sensors.

        """
        return list(self.meshes.keys())

    def has_sensors(self, body_id):
        """ Returns True if the body has sensors.

        Args:
            body_id: The ID of the body.

        Returns:
            True if the body has sensors, False otherwise.

        """
        return body_id in self._submeshes

    def get_sensor_count(self, body_id):
        """ Returns the number of sensors for the body.

        Args:
            body_id: The ID of the body.

        Returns:
            The number of sensor points for this body.

        """
        return self._sensor_counts[body_id]

    def _get_sensor_count_submesh(self, body_id, submesh_idx):
        """ Returns the number of sensors on this submesh.

        Each body can contain multiple submeshes, so submeshes are uniquely identified by the ID of their body and
        their index in the list of submeshes for that body.

        Args:
            body_id: The ID of the body.
            submesh_idx: The index of the submesh.

        Returns:
            The number of sensor points on this submesh.

        """
        return self._sensor_counts_submesh[body_id][submesh_idx]

    def _get_mesh(self, geom_id: int, scale: float):
        """ Creates a sensor mesh for the geom.

        Given a geom this creates a raw sensor mesh. These are not used directly, instead :meth:`.add_body` collects
        the meshes for all geoms in a body and then merges them, marking any sensor points that are located inside
        another geom as inactive. Meshes are `trimesh` objects.

        Args:
            geom_id: The ID of the geom.
            scale: The distance between sensor points in the mesh.

        Returns:
            The new sensor mesh.

        """
        # Do sensorpoints for a single geom
        # TODO: Use our own normals instead of default from trimesh face estimation
        geom_type = self.m_model.geom_type[geom_id]
        size = self.m_model.geom_size[geom_id]

        if geom_type == const.GEOM_BOX:
            mesh = mesh_box(scale, size)
        elif geom_type == const.GEOM_SPHERE:
            mesh = mesh_sphere(scale, size[0])
        elif geom_type == const.GEOM_CAPSULE:
            mesh = mesh_capsule(scale, 2 * size[1], size[0])
        elif geom_type == const.GEOM_CYLINDER:
            # Cylinder size 0 is radius, size 1 is half length
            mesh = mesh_cylinder(scale, 2 * size[1], size[0])
        elif geom_type == const.GEOM_PLANE:
            RuntimeWarning("Cannot add sensors to plane geoms!")
            return None
        elif geom_type == const.GEOM_ELLIPSOID:
            mesh = mesh_ellipsoid(scale, size)
        elif geom_type == const.GEOM_MESH:
            # TODO: Use convex hull of mesh as sensor mesh? Would not have remotely consistent spacing
            size = self.m_model.geom_rbound[geom_id]
            mesh = mesh_sphere(scale, size)
        else:
            return None
        return mesh

    def _convert_active_sensor_idx(self, body_id, submesh_idx, vertex_idx):
        """ Converts index types for active sensors.

        Active sensors need to keep track of two indices: Their index in the submesh and their index in the output
        arrays. This function allows one to convert the submesh index to the output index, by reading from the map
        created when the associated body was added to the model. In this documentation indices will be output indices
        unless explicitely said otherwise.

        Args:
            body_id: The ID of the body.
            submesh_idx: The index of the submesh.
            vertex_idx: The index of the vertex to be converted.

        Returns:
            The index in the output arrays for the input vertex.
        """
        return self._vertex_to_sensor_idx[body_id][submesh_idx][vertex_idx]

    def _get_nearest_vertex(self, contact_pos, mesh):
        """ Get the vertex in the mesh closest to the position.

        Note that this vertex might not be an active sensor!

        Args:
            contact_pos: The position. Should be a shape `(3,)` numpy array.
            mesh: The mesh. Should be a trimesh object.

        Returns:
            `(distance, index)`, where `distance` is the distance between the position and closest vertex and `index`
            is the submesh index of the closest vertex.

        """
        if mesh.vertices.shape[0] == 1:
            distance = np.linalg.norm(contact_pos - mesh.vertices[0])
            sub_idx = 0
        else:
            proximity_query = trimesh.proximity.ProximityQuery(mesh)
            distance, sub_idx = proximity_query.vertex(contact_pos)
        return distance, sub_idx

    def get_nearest_sensor(self, contact_pos, body_id):
        """ Given a position in space and a body, return the sensor on the body closest to the position.

        Args:
            contact_pos: The position. Should be a numpy array of shape `(3,)`
            body_id: The ID of the body. The body must have sensors!

        Returns:
            The index of the closest sensor and the distance between contact and sensor.

        """
        # Get closest active vertex on the whole body mesh. Does this by getting clostest active subvertex on each
        # submesh and then returning the closest of those.
        active_sensors = {}
        for i, mesh in enumerate(self._submeshes[body_id]):
            # Get closest vertex on mesh
            distance, sub_idx = self._get_nearest_vertex(contact_pos, mesh)
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

    def get_k_nearest_sensors(self, contact_pos, body_id, k):
        """ Given a position and a body, find the k sensors on the body closest to the position.

        TODO: Uses a cache to speed up the simulation. For a given contact we determine the closest sensor point on a
            given body. If this point is located in the cache, the nearest neighbour search is skipped and instead
            pulled from the cache. If the point is not located in the cache we store it there. Currently there is no
            pruning or limiting of the cache.
            The cache also stores the "k" factor used during the previous search, so changing k mid simulation will
            reduce performance but not affect accuracy.

        Args:
            contact_pos: The position. Should be a numpy array of shape `(3,)`
            body_id: The ID of the body. The body must have sensors!
            k: The number of sensors to return.

        Returns:
            The indices of the k-nearest sensors to the contact and the distances between them and the position.

        """
        # Use trimesh meshes to get nearest vertex on all submeshes, then get up to k extra candidates for each submesh,
        # then get the k closest from the candidates of all submeshes
        candidate_sensors_idx = []
        candidate_sensor_distances = []
        for i, mesh in enumerate(self._submeshes[body_id]):
            mesh_distances = self._sensor_distances(contact_pos, mesh)
            sub_idx = np.argmin(mesh_distances)
            distance = mesh_distances[sub_idx]
            active_vertices_on_submesh = self._active_subvertices[body_id][i]

            # If the mesh only has a single vertex and it is an active vertex, take it and go to next mesh
            if mesh.vertices.shape[0] == 1 and active_vertices_on_submesh[0]:
                candidate_sensors_idx.append(self._convert_active_sensor_idx(body_id, i, sub_idx))
                candidate_sensor_distances.append(distance)
                continue

            # If the search for this sensor was already performed with the same k: grab from cache and go to next mesh
            if (body_id, i, sub_idx) in self._neighbour_cache:
                cached_candidate_sensor_idxs, cached_candidate_sensor_distances, cached_k = self._neighbour_cache[(body_id, i, sub_idx)]
                if cached_k == k:
                    candidate_sensors_idx.extend(cached_candidate_sensor_idxs)
                    candidate_sensor_distances.extend(cached_candidate_sensor_distances)
                    continue

            # If neither of the two above apply, perform nearest k search, caching results.
            graph = mesh.vertex_adjacency_graph
            nodes_to_check = deque()
            nodes_to_check.append(sub_idx)
            nodes_to_check.extend(graph[sub_idx])
            candidate_sensor_idxs_submesh = []
            candidate_sensor_distances_submesh = []
            largest_distance_so_far = 0
            checked = set()
            while len(nodes_to_check) > 0:
                candidate = nodes_to_check.pop()
                if candidate in checked:
                    continue
                checked.add(candidate)
                distance = mesh_distances[candidate]
                # If we have enough candidates and the current node is further away than the furthest, skip this node
                # Otherwise add it to found candidates and put neighbours into queue to check
                if len(candidate_sensor_idxs_submesh) >= k and distance > largest_distance_so_far:
                    continue
                else:
                    if active_vertices_on_submesh[candidate]:
                        candidate_sensor_idxs_submesh.append(self._convert_active_sensor_idx(body_id, i, candidate))
                        candidate_sensor_distances_submesh.append(distance)
                        if len(candidate_sensors_idx) < k:
                            largest_distance_so_far = distance
                    nodes_to_check.extend(set(graph[candidate]) - checked)
            # Cache results
            self._neighbour_cache[(body_id, i, sub_idx)] = \
                (candidate_sensor_idxs_submesh, candidate_sensor_distances_submesh, k)

            # Add results to current search
            candidate_sensor_distances.extend(candidate_sensor_distances_submesh)
            candidate_sensors_idx.extend(candidate_sensor_idxs_submesh)

        sensor_idx = np.asarray(candidate_sensors_idx)
        distances = np.asanyarray(candidate_sensor_distances)
        # Get k closest from all of these candidates
        sorted_idxs = np.argpartition(distances, k)
        return sensor_idx[sorted_idxs[:k]], distances[sorted_idxs[:k]]

    def _get_mesh_adjacency_graph(self, mesh):
        """ Grab the adjacency graph for the mesh.

        Currently just wraps trimeshes vertex adjacency function, since they already handle caching smartly.

        Args:
            mesh: The mesh.

        Returns:
            A networkx graph of the mesh.
        """
        return mesh.vertex_adjacency_graph

    def get_sensors_within_distance(self, contact_pos, body_id, distance_limit):
        """ Finds all sensors on a body that are within a given distance to a position.

        The distance used is the direct euclidean distance. A sensor is included in the output if and only if:

        - It is within the distance limit to the position.
        - There is a path from the sensor to the vertex closest to the position such that all vertices on that path are
          also within the distance limit.

        TODO: Uses a cache to speed up the simulation. For a given contact we determine the closest sensor point on a
            given body. If this point is located in the cache, the nearest neighbour search is skipped and instead
            pulled from the cache. If the point is not located in the cache we store it there. Currently there is no
            pruning or limiting of the cache.
            To facilitate accurate searches even as the contact point moves about, we search a slightly larger area on
            the first occurrence, which is then pruned on subsequent occurrences using the distance measure.

        Args:
            contact_pos: The position. Should be a numpy array of shape `(3,)`
            body_id: The ID of the body. The body must have sensors!
            distance_limit: Sensors must be within this distance to the position to be included in the output.

        Returns:
            The indices of all sensors on the body that are within `distance` to the contact as well as the distances
            between the sensors and the position.

        """
        # Use trimesh to get nearest vertex on each submesh and then inspect neighbours from there. Have to check
        # submeshes since we dont have edges between submeshes
        candidate_sensors_idx = []
        candidate_sensor_distances = []
        for i, mesh in enumerate(self._submeshes[body_id]):
            mesh_distances = self._sensor_distances(contact_pos, mesh)
            sub_idx = np.argmin(mesh_distances)
            distance = mesh_distances[sub_idx]
            if distance > distance_limit:
                continue
            active_vertices_on_submesh = self._active_subvertices[body_id][i]
            index_map = self._vertex_to_sensor_idx[body_id][i]
            if mesh.vertices.shape[0] == 1 and active_vertices_on_submesh[0]:
                candidate_sensors_idx.append(index_map[sub_idx])
                candidate_sensor_distances.append(distance)
                continue
            graph = self._get_mesh_adjacency_graph(mesh)
            candidates = deque()
            candidates.append(sub_idx)
            candidates.extend(graph[sub_idx])

            within_distance = mesh_distances < distance_limit
            checked = np.invert(within_distance)

            while len(candidates) > 0:
                candidate = candidates.pop()
                if checked[candidate]:
                    continue
                checked[candidate] = 1
                # If the sensor is an output sensor, we still need more candidates, or it is closer than another:
                #   Grab this sensor as a candidate
                if not within_distance[candidate]:
                    continue
                else:
                    if active_vertices_on_submesh[candidate]:
                        candidate_sensors_idx.append(index_map[candidate])
                        candidate_sensor_distances.append(mesh_distances[candidate])
                    candidates.extend([node for node in graph[candidate] if not checked[candidate]])

        sensor_idx = np.asarray(candidate_sensors_idx)
        distances = np.asanyarray(candidate_sensor_distances)
        return sensor_idx, distances

    def _sensor_distances(self, point, mesh):
        """ Returns the distances between a point and all sensor points on a mesh.

        This is the function that is used to determine if a vertex is within distance of a contact position.
        Optimally this would be the exact geodesic distance, but currently this is direct euclidean distance.

        Args:
            point: The position.
            mesh: The mesh.

        Returns:
            The distances between all vertices in the mesh and the point.

        """
        return np.linalg.norm(mesh.vertices - point, axis=-1, ord=2)

    # ======================== Positions and rotations ================================
    # =================================================================================

    def get_contact_position_world(self, contact_id):
        """ Get the position of a contact in the world frame.

        Note that this is halfway between the touching geoms. Since geoms can intersect this point will likely be
        located inside both.

        Args:
            contact_id: The ID of the contact.

        Returns:
            A numpy array with the position of the contact.

        """
        return self.m_data.contact[contact_id].pos

    def get_contact_position_relative(self, contact_id, body_id: int):
        """ Get the position of a contact in the coordinate frame of a body.

        Args:
            contact_id: The ID of the contact.
            body_id: The ID of the body.

        Returns:
            A numpy array with the position of the contact.

        """
        body_pos = env_utils.world_pos_to_body(self.m_data, self.get_contact_position_world(contact_id), body_id)
        contact = self.m_data.contact[contact_id]
        if contact.dist < 0:
            # Have to correct contact position towards surface of our body.
            # Note that distance is negative for intersecting geoms and the normal vector points into the sensing geom.
            normal = self.get_contact_normal(contact_id, body_id)
            body_pos = body_pos + normal * contact.dis / 2
        return body_pos

    # =============== Raw force and contact normal ====================================
    # =================================================================================

    def get_raw_force(self, contact_id, body_id):
        """ Collect the full contact force in MuJoCos own contact frame.

        By convention the normal force points away from the first geom listed, so the forces are inverted if the first
        geom is the sensing geom.

        Args:
            contact_id: The ID of the contact.
            body_id: The relevant body in the contact. One of the geoms belonging to this body must be involved in the
                contact!

        Returns:
            A 3d vector containing the normal force and the two tangential friction forces.

        """
        forces = np.zeros(6, dtype=np.float64)
        mujoco_py.functions.mj_contactForce(self.m_model, self.m_data, contact_id, forces)
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
            contact_id: The ID of the contact.
            body_id: The ID of the body.

        Returns:
            A 3d vector containing the normal vector.

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
        normal_vector = env_utils.rotate_vector(normal_vector, env_utils.get_body_rotation(self.m_data, body_id))
        return normal_vector

    # =============== Valid touch force functions =====================================
    # =================================================================================

    def force_vector_global(self, contact_id, body_id):
        """ Touch function. Returns the full contact force in world frame.

        Given a contact returns the full contact force, i.e. the vector sum of the normal force and the two tangential
        friction forces, in the world coordinate frame. The body is required to account for MuJoCo conventions and
        convert coordinate frames.

        Args:
            contact_id: The ID of the contact.
            body_id: The ID of the body.

        Returns:
            A 3d vector of the forces.

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
            contact_id: The ID of the contact.
            body_id: The ID of the body.

        Returns:
            A 3d vector of the forces.

        """
        global_forces = self.force_vector_global(contact_id, body_id)
        relative_forces = env_utils.rotate_vector_transpose(global_forces, env_utils.get_body_rotation(self.m_data,
                                                                                                       body_id))
        return relative_forces

    # =============== Output related functions ========================================
    # =================================================================================

    def get_contacts(self):
        """ Collects all active contacts involving bodies with touch sensors.

        For each active contact with a sensing geom we build a tuple ``(contact_id, body_id, forces)``, where
        `contact_id` is the ID of the contact in the MuJoCo arrays, `body_id` is the ID of the sensing body and
        `forces` is a numpy array of the raw output force, as determined by :attr:`.touch_type`.

        Returns:
            A list of tuples with contact information.

        """
        contact_tuples = []
        for i in range(self.m_data.ncon):
            contact = self.m_data.contact[i]
            body1 = self.m_model.geom_bodyid[contact.geom1]
            body2 = self.m_model.geom_bodyid[contact.geom2]
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

        return contact_tuples

    def get_empty_sensor_dict(self, size):
        """ Returns a dictionary with empty sensor outputs.

        Creates a dictionary with an array of zeros for each body with sensors. A body with 'n' sensors has an empty
        output array of shape `(n, size)`. The output of this function is equivalent to the touch sensor output if
        there are no contacts.

        Args:
            size: The size of a single sensor output.

        Returns:
            The dictionary of empty sensor outputs.

        """
        sensor_outputs = {}
        for body_id in self.meshes:
            sensor_outputs[body_id] = np.zeros((self.get_sensor_count(body_id), size), dtype=np.float32)
        return sensor_outputs

    def flatten_sensor_dict(self, sensor_dict):
        """ Flattens a touch output dictionary into a single large array in a deterministic fashion.

        Output dictionaries list the arrays of sensor outputs for each body. This function concatenates these arrays
        together in a reproducible fashion to avoid key order anomalies. Bodies are sorted by their ID.

        Args:
            sensor_dict: The output dictionary to be flattened.

        Returns:
            A single concatenated numpy array.
        """
        sensor_arrays = []
        for body_id in sorted(self.meshes):
            sensor_arrays.append(sensor_dict[body_id])
        return np.concatenate(sensor_arrays)

    def get_touch_obs(self) -> np.ndarray:
        """ Produces the current touch sensor outputs.

        Does the full contact getting-processing process, such that we get the forces, as determined by
        :attr:`.touch_type` and :attr:`.response_type`, for each sensor. :attr:`.touch_function` is called to compute
        the raw output force, which is then distributed over the sensors using :attr:`.response_function`.

        The indices of the output dictionary :attr:`~mimoTouch.touch.TrimeshTouch.sensor_outputs` and the sensor
        dictionary :attr:`.sensor_positions` are aligned, such that the ith sensor on `body` has position
        ``.sensor_positions[body][i]`` and output in ``.sensor_outputs[body][i]``.

        Returns:
            A numpy array containing all the touch sensations.
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
        sensor and the contact point. Finally the total force is normalized such that the total force over all sensors
        for this contact is identical to the raw force. The scaling distance is given by double the distance between
        sensor points.

        Args:
            contact_id: The ID of the contact.
            body_id: The ID of the sensing body.
            force: The raw force.
        """
        # Get all sensors within distance (distance here is just double the sensor scale)
        scale = self.sensor_scales[body_id]
        contact_pos = self.get_contact_position_relative(contact_id=contact_id, body_id=body_id)
        nearest_sensors, sensor_distances = self.get_sensors_within_distance(contact_pos, body_id, 2*scale)

        adjusted_forces = {}
        force_total = 0  # This variable c
        for sensor_id, distance in zip(nearest_sensors, sensor_distances):
            sensor_adjusted_force = scale_linear(force, distance, scale=2*scale)
            force_total += sensor_adjusted_force[0]
            adjusted_forces[sensor_id] = sensor_adjusted_force

        factor = force[0] / force_total if abs(force_total) > EPS else 0
        for sensor_id in adjusted_forces:
            self.sensor_outputs[body_id][sensor_id] += adjusted_forces[sensor_id] * factor

    def nearest(self, contact_id, body_id, force):
        """ Response function. Adds the output force directly to the nearest sensor.

        Args:
            contact_id: The ID of the contact.
            body_id: The ID of the body.
            force: The raw output force.
        """
        # Get the nearest sensor to this contact, add the force to it
        contact_pos = self.get_contact_position_relative(contact_id=contact_id, body_id=body_id)
        nearest_sensor, distance = self.get_nearest_sensor(contact_pos, body_id)
        self.sensor_outputs[body_id][nearest_sensor] += force

    # =============== Visualizations ==================================================
    # =================================================================================

    # Plot sensor points for single geom
    def plot_sensors_body(self, body_id: int = None, body_name: str = None):
        """ Plots the sensor positions for a body.

        Given either an ID or the name of a body, plot the positions of the sensors on that body.

        Args:
            body_id: The ID of the body.
            body_name: The name of the body. This is ignored if the ID is provided!
        """
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)

        points = self.meshes[body_id].vertices
        limit = self.plotting_limits[body_id]
        title = self.m_model.body_id2name(body_id)
        env_utils.plot_points(points, limit=limit, title=title)

    # Plot forces for single body
    def plot_force_body(self, body_id: int = None, body_name: str = None):
        """ Plot the sensor output for a body.

        Given either an ID or the name of a body, plots the positions and outputs of the sensors on that body.

        Args:
            body_id: The ID of the body.
            body_name: The name of the body. This is ignored if the ID is provided!
        """
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)

        sensor_points = self.sensor_positions[body_id]
        force_vectors = self.sensor_outputs[body_id] / 20
        title = self.m_model.body_id2name(body_id) + " forces"
        if force_vectors.shape[1] == 1:
            normals = self.meshes[body_id].vertex_normals[self.active_vertices[body_id], :]
            force_vectors = force_vectors * normals
        env_utils.plot_forces(sensor_points, force_vectors, limit=np.amax(sensor_points)*1.2, title=title)

    # Plot forces for list of bodies.
    def plot_force_bodies(self, body_ids: List[int] = [], body_names: List[str] = [],
                          title: str = "", focus: str = "world"):
        """ Plot the sensor output for a list of bodies.

        Given a list of bodies, either by ID or by name, plot the positions and outputs of all sensors on the bodies.
        The current relative positions and orientations of the bodies in the simulation are respected.
        The parameter `focus` determines how the coordinates are centered. Two options exist:
        - 'world':   In this setting all the coordinates are translated into global coordinates
        - 'first':   In this setting all the coordinates are translated into the frame of the first body in the list.

        Args:
            body_ids: A list of IDs of the bodies that should be plotted.
            body_names: A list of the names of the bodies that should be plotted. This is ignored if `body_ids` is
                provided!
            title: The title of the plot.
            focus: Coordinates are moved into a consistent reference frame. This parameter determines that reference
                frame. Must be one of ``["world", "first"]``.
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
        env_utils.plot_forces(points=points, vectors=forces, limit=limit, title=title)

    def plot_force_body_subtree(self, body_id: int = None, body_name: str = None, title=""):
        """ Plot the sensor output for the kinematic subtree with the given body at its root.

        Given a body, collects all descendent bodies in the kinematic tree and  plot the positions and outputs of their
        sensors. The current relative positions and orientations of the bodies in the simulation are respected and all
        coordinates are moved into the coordinate frame of the root body.

        Args:
            body_id: The ID of the root body for the subtree.
            body_name: The names of the root bodies. This is ignored if an ID is provided!
            title: The title of the plot.
        """
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)
        # Go through all bodies and note their child bodies
        subtree = env_utils.get_child_bodies(self.m_model, body_id)
        self.plot_force_bodies(body_ids=subtree, title=title, focus="first")
