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
        "normal": 1,
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

        self._submeshes: Dict[int, List[trimesh.Trimesh]] = {}
        self._active_subvertices = {}

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
        active_vertices = []
        # Cleanup, removing all sensor points inside other sensors
        # TODO: More filtering to remove vertices too close to one another near geom intersections
        for mesh in meshes:
            mask = np.ones((mesh.vertices.shape[0],), dtype=bool)
            for other_mesh in meshes:
                if other_mesh == mesh:
                    continue
                # If our vertices are contained in the other mesh, make them inactive
                contained = other_mesh.contains(mesh.vertices)
                mask = np.logical_and(mask, np.invert(contained))
            active_vertices.append(mask)
        self._active_subvertices[body_id] = active_vertices
        print("{} total vertices, {} active sensors on body {}".format(self.meshes[body_id].vertices.shape[0],
                                                                       np.count_nonzero(self.active_vertices[body_id]),
                                                                       self.m_model.body_id2name(body_id)))

    @property
    def meshes(self):
        meshes = {}
        for body_id in self._submeshes:
            if len(self._submeshes[body_id]) > 1:
                meshes[body_id] = trimesh.util.concatenate(self._submeshes[body_id])
            else:
                meshes[body_id] = self._submeshes[body_id][0]
        return meshes

    @property
    def active_vertices(self):
        active_vertices = {}
        for body_id in self._active_subvertices:
            active_vertices[body_id] = np.concatenate(self._active_subvertices[body_id])
        return active_vertices

    @property
    def sensor_positions(self):
        positions = {}
        meshes = self.meshes
        active_sensors = self.active_vertices
        for body_id in self._submeshes:
            positions[body_id] = meshes[body_id].vertices[active_sensors[body_id], :]
        return positions

    def has_sensors(self, body_id):
        """ Returns true if the geom has sensors """
        return body_id in self.meshes

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
            size = self.m_model.geom_rbound[geom_id]
            mesh = mesh_sphere(scale, size)
        else:
            return None
        return mesh

    def convert_active_sensor_idx(self, body_id, submesh_idx, vertex_idx):
        """ Converts a sensor index of the submesh type to a body type sensor index"""
        offset = 0
        active_vertices = self._active_subvertices[body_id]
        for i in range(submesh_idx):
            offset += np.count_nonzero(active_vertices[i])
        submesh_offsets = np.cumsum(active_vertices[submesh_idx]) - 1
        offset += submesh_offsets[vertex_idx]
        return offset

    def get_nearest_sensor(self, contact_id, body_id):
        """ Given a contact and a geom, return the sensor on the geom closest to the contact.
        Returns the sensor index and the distance between contact and sensor"""
        # Get closest active vertex on the whole body mesh. Does this by getting clostest active subvertex on each
        # submesh and then returning the closest of those.
        contact_pos = self.get_contact_position_relative(contact_id=contact_id, body_id=body_id)
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
                checked = {sub_idx}
                while len(candidates) > 0:
                    candidate = candidates.pop()
                    if active_vertices_on_submesh[candidate]:
                        # Still assume convex meshes, so distance must >= for further vertices
                        distance = self._sensor_distance(contact_pos, body_id, i, candidate)
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

    def get_k_nearest_sensors(self, contact_id, body_id, k):
        # Use trimesh meshes to get nearest vertex on all submeshes, then get up to k extra candidates for each submesh,
        # then get the k closest from the candidates of all submeshes
        candidate_sensors_idx = []
        candidate_sensor_distances = []
        largest_distance_so_far = 0
        contact_pos = self.get_contact_position_relative(contact_id=contact_id, body_id=body_id)
        for i, mesh in enumerate(self._submeshes[body_id]):
            proximity_query = trimesh.proximity.ProximityQuery(mesh)
            distance, sub_idx = proximity_query.vertex(contact_pos)
            active_vertices_on_submesh = self._active_subvertices[body_id][i]

            graph = mesh.vertex_adjacency_graph
            candidates = deque()
            candidates.append(sub_idx)
            candidates.extend(graph[sub_idx])
            checked = set()
            while len(candidates) > 0:
                candidate = candidates.pop()
                if candidate in checked:
                    continue
                checked.add(candidate)
                distance = self._sensor_distance(contact_pos, body_id, i, candidate)
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

    def get_sensors_within_distance(self, contact_id, body_id, distance_limit):
        # TODO: All of it. Must take into account that mesh may not have faces
        # Use trimesh meshes to get all vertices on nearest edge and then inspect neighbours from there
        candidate_sensors_idx = []
        candidate_sensor_distances = []
        contact_pos = self.get_contact_position_relative(contact_id=contact_id, body_id=body_id)
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
            while len(candidates) > 0:
                candidate = candidates.pop()
                if candidate in checked:
                    continue
                checked.add(candidate)
                distance = self._sensor_distance(contact_pos, body_id, i, candidate)
                # If the sensor is an output sensor, we still need more candidates, or it is closer than another:
                #   Grab this sensor as a candidate
                if distance > distance_limit:
                    continue
                else:
                    if active_vertices_on_submesh[candidate]:
                        candidate_sensors_idx.append(self.convert_active_sensor_idx(body_id, i, candidate))
                        candidate_sensor_distances.append(distance)
                    candidates.extend(set(graph[candidate]) - checked)
        sensor_idx = np.asarray(candidate_sensors_idx)
        distances = np.asanyarray(candidate_sensor_distances)
        return sensor_idx, distances

    def _sensor_distance(self, point, body_id, submesh_idx, vertex_idx):
        """ Returns the distance between a point and a sensor. The point should be a (3,) numpy array while the sensor
        is defined by the body_id, submesh id and mesh vertex id. Optimally this would be the exact geodesic distance,
        currently this is direct euclidean distance """
        sensor_position = self._submeshes[body_id][submesh_idx].vertices[vertex_idx]
        return np.linalg.norm(point - sensor_position, ord=2)

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
        normal_vector = env_utils.mulRot(normal_vector, env_utils.get_body_rotation(self.m_data, body_id))
        return normal_vector

    # =============== Valid touch force functions =====================================
    # =================================================================================

    def normal(self, contact_id, body_id) -> float:
        """ Returns the normal force as a scalar"""
        return self.get_raw_force(contact_id, body_id)[0]

    def force_vector_global(self, contact_id, body_id):
        """ Returns full contact force in world frame. Output is a 3-d vector"""
        contact = self.m_data.contact[contact_id]
        forces = self.get_raw_force(contact_id, body_id)
        force_rot = np.reshape(contact.frame, (3, 3))
        global_forces = env_utils.mulRotT(forces, force_rot)
        return global_forces

    def force_vector(self, contact_id, body_id):
        """ Returns full contact force in the frame of the body that geom_id belongs to. Output is a 3-d vector"""
        global_forces = self.force_vector_global(contact_id, body_id)
        relative_forces = env_utils.mulRotT(global_forces, env_utils.get_body_rotation(self.m_data, body_id))
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
        nearest_sensors, sensor_distances = self.get_sensors_within_distance(contact_id, body_id, 2*scale)

        print(nearest_sensors)
        print(sensor_distances)

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
            print(self.sensor_outputs[body_id].shape)
            self.sensor_outputs[body_id][sensor_id] += rescaled_sensor_adjusted_force

    def nearest(self, contact_id, body_id, force):
        # Get the nearest sensor to this contact, add the force to it
        nearest_sensor, distance = self.get_nearest_sensor(contact_id, body_id)
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
        force_vectors = self.sensor_outputs[body_id]
        title = self.m_model.body_id2name(body_id) + " forces"
        if force_vectors.shape[1] == 1:
            # TODO: Need proper sensor normals, can't do this until trimesh rework
            raise RuntimeWarning("Plotting of scalar forces not implemented!")
        else:
            env_utils.plot_forces(sensor_points, force_vectors, limit=np.max(sensor_points) + 0.5, title=title)
