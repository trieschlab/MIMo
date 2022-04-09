""" This module defines the touch system interface and provides a simple implementation.

The interface is defined as an abstract class in :class:`~mimoTouch.touch.Touch`.
A simple implementation with a cloud of sensor points is in :class:`~mimoTouch.touch.DiscreteTouch`.

"""

import math
import numpy as np
import mujoco_py

import mimoEnv.utils as env_utils
from mimoEnv.utils import rotate_vector_transpose, rotate_vector, EPS
from mimoTouch.sensorpoints import spread_points_box, spread_points_sphere, spread_points_cylinder, \
                                   spread_points_capsule

#: A key to identify the geom type ids used by MuJoCo.
GEOM_TYPES = {"PLANE": 0, "HFIELD": 1, "SPHERE": 2, "CAPSULE": 3, "ELLIPSOID": 4, "CYLINDER": 5, "BOX": 6, "MESH": 7}


class Touch:
    """ Abstract base class for the touch system.

    This class defines the functions that all implementing classes must provide. :meth:`.get_touch_obs` should perform
    the whole sensory pipeline as defined in the configuration and return the output as a single array.
    Additionally the output for each body part should be stored in :attr:`.sensor_outputs`. The exact definition of
    'body part' is left to the implementing class.

    The constructor takes two arguments, `env` and `touch_params`:
    `env` should be an openAI gym environment using MuJoCo, while `touch_params` is a configuration dictionary. The
    exact form will depend on the specific implementation, but it must contain these three entries:

    - 'scales', which lists the distance between sensor points for each body part.
    - 'touch_function', which defines the output type and must be in :attr:`.VALID_TOUCH_TYPES`.
    - 'response_function', which defines how the contact forces are distributed to the sensors. Must be one of
      :attr:`.VALID_RESPONSE_FUNCTIONS`.

    The sensor scales determines the density of the sensor points, while `touch_function` and `response_function`
    determine the type of contact output and how it is distributed. `touch_function` and `response_function` refer to
    class methods by name and must be listed in :attr:`.VALID_TOUCH_TYPES`. and :attr:`.VALID_RESPONSE_FUNCTIONS`
    respectively. Different touch functions should be used to support different types of output, such as normal force,
    frictional forces or contact slip. The purpose of the response function is to loosely simulate surface behaviour.
    How exactly these functions work and interact is left to the implementing class.

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
    - 'force_vector': The contact force vector (normal and frictional forces) reported in the coordinate frame of the
      sensing geom.
    - 'force_vector_global': Like 'force_vector', but reported in the world coordinate frame instead.

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
            # Add a geom if it belongs to body and has collisions enabled (at least potentially)
            if g_body_id == body_id and contype > 0:
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

        if self.m_model.geom_contype[geom_id] == 0:
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
        by `scale`. We identify the type of geom using the MuJoCo API and :data:`GEOM_TYPES`. This function populates
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
        return env_utils.world_pos_to_geom(self.m_data, self.get_contact_position_world(contact_id), geom_id)

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

        The indices of the output dictionary :attr:`.sensor_outputs` and the sensor dictionary :attr:`.sensor_positions`
        are aligned, such that the ith sensor on `geom` has position ``.sensor_positions[geom][i]`` and output in
        ``.sensor_outputs[geom][i]``.

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
