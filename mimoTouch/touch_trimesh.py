import math
import numpy as np
import mujoco_py
import trimesh
import trimesh.util
import trimesh.proximity
import trimesh.graph
from collections import deque

from typing import List, Dict

from mimoTouch.touch import Touch, scale_linear, GEOM_TYPES
from mimoTouch.sensormeshes import mesh_sphere, mesh_ellipsoid, mesh_cylinder, mesh_box, mesh_capsule
import mimoEnv.utils as env_utils
from mimoEnv.utils import EPS


class TrimeshTouch(Touch):

    VALID_TOUCH_TYPES = {
        "force_vector": 3,
        "force_vector_global": 3,
    }

    VALID_ADJUSTMENTS = ["nearest", "spread_linear"]

    """ This class uses bodies as the base touch object. Sensors are part of a mesh (using trimesh).

    This class use mujoco bodies as the basis for the sensor points and output dictionaries, unlike DiscreteTouch which
    uses geoms."""

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
        self._vertex_to_sensor_idx = {}

        self.meshes: Dict[int, trimesh.Trimesh] = {}
        self.active_vertices = {}
        self.sensor_positions = {}

        self.plotting_limits = {}

        # Add sensors to bodies
        for body_id in self.sensor_scales:
            self.add_body(body_id=body_id, scale=self.sensor_scales[body_id])

        # Get touch obs once to ensure all output arrays are initialized
        self.get_touch_obs()

    # ======================== Sensor related functions ===============================
    # =================================================================================

    def add_body(self, body_id: int = None, body_name: str = None, scale: float = math.inf):
        """ Add sensors to this body with a distance between sensors of around scale """
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)
        meshes = []
        for geom_id in env_utils.get_geoms_for_body(self.m_model, body_id):
            mesh = self._get_mesh(geom_id, scale)

            # Move meshes from geom into body frame
            mesh.vertices = env_utils.geom_pos_to_body(self.m_data, mesh.vertices.copy(), geom_id, body_id)
            meshes.append(mesh)
        self._submeshes[body_id] = meshes
        if len(meshes) > 1:
            self.meshes[body_id] = trimesh.util.concatenate(meshes)
        else:
            self.meshes[body_id] = meshes[0]

        active_vertices = []
        vertex_to_sensor_idxs = []
        submesh_offset = 0
        # Cleanup, removing all sensor points inside other sensors
        # TODO: More filtering to remove vertices too close to one another near geom intersections
        for mesh in meshes:
            mask = np.ones((mesh.vertices.shape[0],), dtype=bool)
            for other_mesh in meshes:
                if other_mesh == mesh:
                    continue
                # If our vertices are contained in the other mesh, make them inactive
                # TODO: This has weird inconcistencies, check those
                contained = other_mesh.contains(mesh.vertices)
                mask = np.logical_and(mask, np.invert(contained))
            active_vertices.append(mask)
            vertex_offsets = np.cumsum(mask) - 1
            vertex_to_sensor_idxs.append(vertex_offsets + submesh_offset)
            submesh_offset += np.count_nonzero(mask)

        self._active_subvertices[body_id] = active_vertices
        self.active_vertices[body_id] = np.concatenate(active_vertices)
        self._vertex_to_sensor_idx[body_id] = vertex_to_sensor_idxs

        self.sensor_positions[body_id] = self.meshes[body_id].vertices[self.active_vertices[body_id], :]

        #print("{} total vertices, {} active sensors on body {}".format(self.meshes[body_id].vertices.shape[0],
        #                                                               np.count_nonzero(self.active_vertices[body_id]),
        #                                                               self.m_model.body_id2name(body_id)))

    def has_sensors(self, body_id):
        """ Returns true if the geom has sensors """
        return body_id in self._submeshes

    def get_sensor_count(self, body_id):
        """ Returns the number of sensors for the geom """
        return np.count_nonzero(self.active_vertices[body_id])

    def _get_sensor_count_submesh(self, body_id, submesh_idx):
        return np.count_nonzero(self._active_subvertices[body_id][submesh_idx])

    def _get_mesh(self, geom_id: int, scale: float):
        # Do sensorpoints for a single geom
        # TODO: Use our own normals instead of default from trimesh face estimation
        geom_type = self.m_model.geom_type[geom_id]
        size = self.m_model.geom_size[geom_id]

        if geom_type == GEOM_TYPES["BOX"]:
            mesh = mesh_box(scale, size)
        elif geom_type == GEOM_TYPES["SPHERE"]:
            mesh = mesh_sphere(scale, size[0])
        elif geom_type == GEOM_TYPES["CAPSULE"]:
            mesh = mesh_capsule(scale, 2 * size[1], size[0])
        elif geom_type == GEOM_TYPES["CYLINDER"]:
            # Cylinder size 0 is radius, size 1 is half length
            mesh = mesh_cylinder(scale, 2 * size[1], size[0])
        elif geom_type == GEOM_TYPES["PLANE"]:
            RuntimeWarning("Cannot add sensors to plane geoms!")
            return None
        elif geom_type == GEOM_TYPES["ELLIPSOID"]:
            mesh = mesh_ellipsoid(scale, size)
        elif geom_type == GEOM_TYPES["MESH"]:
            # TODO: Use convex hull of mesh as sensor mesh? Would not have remotely consistent spacing
            size = self.m_model.geom_rbound[geom_id]
            mesh = mesh_sphere(scale, size)
        else:
            return None
        return mesh

    def convert_active_sensor_idx(self, body_id, submesh_idx, vertex_idx):
        """ Converts a submesh vertex index (submesh index with vertex index) to a sensor index (single index in output)
        """
        return self._vertex_to_sensor_idx[body_id][submesh_idx][vertex_idx]

    def get_nearest_sensor(self, contact_pos, body_id):
        """ Given a position and a body, return the sensor on the body closest to that position.
        Returns the sensor index and the distance between the position and sensor """
        # Get closest active vertex on the whole body mesh. Does this by getting clostest active subvertex on each
        # submesh and then returning the closest of those.
        active_sensors = {}
        for i, mesh in enumerate(self._submeshes[body_id]):
            # Get closest vertex on mesh
            proximity_query = trimesh.proximity.ProximityQuery(mesh)
            distance, sub_idx = proximity_query.vertex(contact_pos)
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
        return self.convert_active_sensor_idx(body_id, closest[0], closest[1]), closest_distance

    def get_k_nearest_sensors(self, contact_pos, body_id, k):
        """ Given a position and a body, return the k closest sensors on the body to that position.
        Returns the sensor indices and the distances between the position and sensors """
        # Use trimesh meshes to get nearest vertex on all submeshes, then get up to k extra candidates for each submesh,
        # then get the k closest from the candidates of all submeshes
        candidate_sensors_idx = []
        candidate_sensor_distances = []
        largest_distance_so_far = 0
        for i, mesh in enumerate(self._submeshes[body_id]):
            proximity_query = trimesh.proximity.ProximityQuery(mesh)
            distance, sub_idx = proximity_query.vertex(contact_pos)
            active_vertices_on_submesh = self._active_subvertices[body_id][i]

            graph = mesh.vertex_adjacency_graph
            candidates = deque()
            candidates.append(sub_idx)
            candidates.extend(graph[sub_idx])
            checked = set()
            mesh_distances = self._sensor_distances(contact_pos, mesh)
            while len(candidates) > 0:
                candidate = candidates.pop()
                if candidate in checked:
                    continue
                checked.add(candidate)
                distance = mesh_distances[candidate]
                # If the sensor is an output sensor, we still need more candidates, or it is closer than another:
                #   Grab this sensor as a candidate
                if len(candidate_sensors_idx) >= k and distance > largest_distance_so_far:
                    continue
                else:
                    if active_vertices_on_submesh[candidate]:
                        candidate_sensors_idx.append(self.convert_active_sensor_idx(body_id, i, candidate))
                        candidate_sensor_distances.append(distance)
                        if len(candidate_sensors_idx) < k:
                            largest_distance_so_far = distance
                    candidates.extend(set(graph[candidate]) - checked)
        sensor_idx = np.asarray(candidate_sensors_idx)
        distances = np.asanyarray(candidate_sensor_distances)
        # Get k closest from all of these
        sorted_idxs = np.argpartition(distances, k)
        return sensor_idx[sorted_idxs[:k]], distances[sorted_idxs[:k]]

    def get_sensors_within_distance(self, contact_pos, body_id, distance_limit):
        """ Given a position, a body and a distance limit, returns all sensors on the body whose distance to the
        position is less than the distance limits.
        Returns the sensor indices and the distances between the position and sensors """
        # Use trimesh to get nearest vertex on each submesh and then inspect neighbours from there. Have to check
        # submeshes since we dont have edges between submeshes
        candidate_sensors_idx = []
        candidate_sensor_distances = []
        for i, mesh in enumerate(self._submeshes[body_id]):
            proximity_query = trimesh.proximity.ProximityQuery(mesh)
            distance, sub_idx = proximity_query.vertex(contact_pos)
            if distance > distance_limit:
                continue
            active_vertices_on_submesh = self._active_subvertices[body_id][i]

            graph = mesh.vertex_adjacency_graph
            candidates = deque()
            candidates.append(sub_idx)
            candidates.extend(graph[sub_idx])
            checked = set()
            mesh_distances = self._sensor_distances(contact_pos, mesh)
            within_distance = mesh_distances < distance_limit
            while len(candidates) > 0:
                candidate = candidates.pop()
                if candidate in checked:
                    continue
                checked.add(candidate)
                # If the sensor is an output sensor, we still need more candidates, or it is closer than another:
                #   Grab this sensor as a candidate
                if not within_distance[candidate]:
                    continue
                else:
                    if active_vertices_on_submesh[candidate]:
                        # Check that idx are correct
                        candidate_sensors_idx.append(self.convert_active_sensor_idx(body_id, i, candidate))
                        candidate_sensor_distances.append(mesh_distances[candidate])
                    candidates.extend(set(graph[candidate]) - checked)

        sensor_idx = np.asarray(candidate_sensors_idx)
        distances = np.asanyarray(candidate_sensor_distances)
        return sensor_idx, distances

    def _sensor_distances(self, point, mesh):
        """ Returns the distances between a point and all sensor points on the mesh. Optimally this would be the exact
        geodesic distance, currently this is direct euclidean distance """
        return np.linalg.norm(mesh.vertices - point, axis=-1, ord=2)

    # ======================== Positions and rotations ================================
    # =================================================================================

    def get_contact_position_world(self, contact_id):
        """ Get the position of a contact in world frame """
        return self.m_data.contact[contact_id].pos

    def get_contact_position_relative(self, contact_id, body_id: int):
        """ Get the position of a contact in the geom frame """
        return env_utils.world_pos_to_body(self.m_data, self.get_contact_position_world(contact_id), body_id)

    # =============== Raw force and contact normal ====================================
    # =================================================================================

    def get_raw_force(self, contact_id, body_id):
        """ Returns the full contact force in mujocos own contact frame. Output is a 3-d vector"""
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
        """ Returns the normal vector (unit vector in direction of normal) in body frame"""
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
        """ Returns full contact force in world frame. Output is a 3-d vector"""
        contact = self.m_data.contact[contact_id]
        forces = self.get_raw_force(contact_id, body_id)
        force_rot = np.reshape(contact.frame, (3, 3))
        global_forces = env_utils.rotate_vector_transpose(forces, force_rot)
        return global_forces

    def force_vector(self, contact_id, body_id):
        """ Returns full contact force in the frame of the body that geom_id belongs to. Output is a 3-d vector"""
        global_forces = self.force_vector_global(contact_id, body_id)
        relative_forces = env_utils.rotate_vector_transpose(global_forces, env_utils.get_body_rotation(self.m_data, body_id))
        return relative_forces

    # =============== Output related functions ========================================
    # =================================================================================

    def get_contacts(self):
        """ Returns a tuple containing (contact_id, geom_id, forces) for each active contact with a sensing geom,
        where contact_id is the index of the contact in the mujoco arrays, geom_id is the index of the geom and forces
        is a numpy array of the raw output force, determined by self.touch_type"""
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
        """ Returns a dictionary with empty sensor outputs. Keys are geom ids, corresponding values are the output
        arrays. For every geom with sensors, returns an empty numpy array of shape (n_sensors, size)"""
        sensor_outputs = {}
        for body_id in self.meshes:
            sensor_outputs[body_id] = np.zeros((self.get_sensor_count(body_id), size), dtype=np.float32)
        return sensor_outputs

    def flatten_sensor_dict(self, sensor_dict):
        """ Flattens a sensor dict, such as from get_empty_sensor_dict into a single large array in a deterministic
        fashion. Geoms with lower id come earlier, sensor outputs correspond to sensor_positions"""
        sensor_arrays = []
        for body_id in sorted(self.meshes):
            sensor_arrays.append(sensor_dict[body_id])
        return np.concatenate(sensor_arrays)

    def get_touch_obs(self) -> np.ndarray:
        """ Does the full contact getting-processing process, such that we get the forces, as determined by the touch
        type and the adjustments, for each sensor.
        """
        contact_tuples = self.get_contacts()
        self.sensor_outputs = self.get_empty_sensor_dict(self.touch_size)  # Initialize output dictionary

        for contact_id, body_id, forces in contact_tuples:
            # At this point we already have the forces for each contact, now we must attach/spread them to sensor
            # points, based on the adjustment function
            self.adjustment_function(contact_id, body_id, forces)

        sensor_obs = self.flatten_sensor_dict(self.sensor_outputs)
        return sensor_obs

    # =============== Force adjustment functions ======================================
    # =================================================================================

    def spread_linear(self, contact_id, body_id, force):
        # Get all sensors within distance (distance here is just double the sensor scale)
        scale = self.sensor_scales[body_id]
        contact_pos = self.get_contact_position_relative(contact_id=contact_id, body_id=body_id)
        nearest_sensors, sensor_distances = self.get_sensors_within_distance(contact_pos, body_id, 2*scale)

        adjusted_forces = {}
        force_total = np.zeros(force.shape)
        for sensor_id, distance in zip(nearest_sensors, sensor_distances):
            sensor_adjusted_force = scale_linear(force, distance, scale=2*scale)
            force_total += sensor_adjusted_force
            adjusted_forces[sensor_id] = sensor_adjusted_force

        # TODO: Check that this makes sense
        factors = force / (force_total + EPS)  # Add very small value to avoid divide by zero errors
        for sensor_id in adjusted_forces:
            rescaled_sensor_adjusted_force = adjusted_forces[sensor_id] * factors
            self.sensor_outputs[body_id][sensor_id] += rescaled_sensor_adjusted_force

    def nearest(self, contact_id, body_id, force):
        # Get the nearest sensor to this contact, add the force to it
        contact_pos = self.get_contact_position_relative(contact_id=contact_id, body_id=body_id)
        nearest_sensor, distance = self.get_nearest_sensor(contact_pos, body_id)
        self.sensor_outputs[body_id][nearest_sensor] += force

    # =============== Visualizations ==================================================
    # =================================================================================

    # Plot sensor points for single geom
    def plot_sensors_body(self, body_id: int = None, body_name: str = None):
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)

        points = self.meshes[body_id].vertices
        limit = self.plotting_limits[body_id]
        title = self.m_model.body_id2name(body_id)
        env_utils.plot_points(points, limit=limit, title=title)

    # Plot forces for single body
    def plot_force_body(self, body_id: int = None, body_name: str = None):
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
        """ Plots touch forces for a list of bodies. The bodies can be provided either as a list of ids or a list of
        names. The two parameters should not be mixed. If body_ids is provided it overrides body_names.
        The parameter focus determines how the coordinates are centered. Two options exist:
            1. world:   In this setting all the coordinates are translated into global coordinates
            2. first:   In this setting all the coordinates are translated into the frame of the first body in the list
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
        body_id = env_utils.get_body_id(self.m_model, body_id=body_id, body_name=body_name)
        # Go through all bodies and note their child bodies
        subtree = env_utils.get_child_bodies(self.m_model, body_id)
        self.plot_force_bodies(body_ids=subtree, title=title, focus="first")
