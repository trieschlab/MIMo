import numpy as np
from matplotlib import pyplot as plt
import mujoco
from typing import List, Dict, Tuple

EPS = 1e-10


MUJOCO_JOINT_SIZES = {
    mujoco.mjtJoint.mjJNT_FREE: 7,
    mujoco.mjtJoint.mjJNT_BALL: 4,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}
""" Size of qpos entries for each joint type; free, ball, slide, hinge.

:meta hide-value:
"""


MUJOCO_DOF_SIZES = {
    mujoco.mjtJoint.mjJNT_FREE: 6,
    mujoco.mjtJoint.mjJNT_BALL: 3,
    mujoco.mjtJoint.mjJNT_SLIDE: 1,
    mujoco.mjtJoint.mjJNT_HINGE: 1,
}
""" Size of qvel entries for each joint type; free, ball, slide, hinge. 

:meta hide-value:
"""


def rotate_vector(vector: np.ndarray, rot_matrix: np.ndarray):
    """ Rotates the vectors with the rotation matrix.

    The vector can be a 1d vector or a multidimensional array of vectors, as long as the final dimension has length 3.
    Convention for mujoco matrices: Use this function to convert from special frame to global and
    :func:`~mimoEnv.utils.rotate_vector_transpose` for the inverse rotation. The exception are the contact frames,
    which are transposed.

    Args:
        vector (numpy.ndarray): The vector(s). Must have shapes (3,) or (.., 3).
        rot_matrix (numpy.ndarray): The rotation matrix that will be applied to the vectors. Should be a (3,3) array.

    Returns:
        numpy.ndarray: The rotated vector(s).
    """
    return rot_matrix.dot(vector)


def rotate_vector_transpose(vector, rot_matrix):
    """ Rotates the vectors with the transpose of the rotation matrix.

    Works identical to rotate_vector, but transposes the rotation matrix first.

    Args:
        vector (numpy.ndarray): The vector(s). Must have shapes (3,) or (.., 3).
        rot_matrix (numpy.ndarray): The rotation matrix that will be applied to the vectors. Should be a (3,3) array.

    Returns:
        numpy.ndarray: The rotated vector(s).
    """
    return np.transpose(rot_matrix).dot(vector)


def weighted_sum_vectors(vector1, vector2, weight1, weight2):
    """ Adds two vectors with weights.

    Args:
        vector1 (numpy.ndarray): The first vector.
        vector2 (numpy.ndarray): The second vector.
        weight1 (float): Weight for the first vector.
        weight2 (float): Weight for the second vector.

    Returns:
        numpy.ndarray: (vector1 * weight1 + vector2 * weight2) / (weight1 + weight2)
    """
    return (vector1 * weight1 + vector2 * weight2) / (weight1 + weight2)


def normalize_vectors(vectors):
    """ Normalizes an array of vectors, such that each vector has unit length.

    Args:
        vectors (numpy.ndarray): The array of vectors. The last dimension should iterate over the elements of the
            vectors.

    Returns:
        numpy.ndarray: The normalized vectors. Same shape as the input, but the length of each vector is reduced to 1.
    """
    mag = np.linalg.norm(vectors, axis=-1, ord=2)
    return vectors / np.expand_dims(mag, -1)


# ======================== Mujoco access utils ====================================
# =================================================================================


def get_geom_id(mujoco_model, geom_id=None, geom_name=None):
    """ Convenience function to get geom ids.

    MuJoCo geoms can be referred to by either an id or a name. This function wraps this ambiguity and always returns
    the id of a geom when either is specified. If both an id and a name are specified the name is ignored!

    Args:
        mujoco_model (mujoco.MjModel): The MuJoCo model object.
        geom_id (int): The id of the geom. Default ``None``.
        geom_name (str): The name of the geom. Default ``None``.

    Returns:
        int: The id of the geom referred to by either the name or the id above.
    """
    if geom_id is None and geom_name is None:
        raise RuntimeError("Need either name or geom id")

    if geom_id is None:
        geom_id = mujoco_model.geom(geom_name).id

    return geom_id


def get_body_id(mujoco_model, body_id=None, body_name=None):
    """ Convenience function to get body ids.
    
    Works identical to :func:`~mimoEnv.utils.get_geom_id`
    
    Args:
        mujoco_model (mujoco.MjModel): The MuJoCo model object.
        body_id (int): The id of the body. Default ``None``.
        body_name (str): The name of the body. Default ``None``.
        
    Returns:
        int: The id of the geom referred to by either the name or the id above.
    """
    if body_id is None and body_name is None:
        raise RuntimeError("Need either name or body id")

    if body_id is None:
        body_id = mujoco_model.body(body_name).id

    return body_id


def get_geoms_for_body(sim_model, body_id):
    """ Returns all geom ids belonging to a given body.

    Args:
        sim_model (mujoco.MjModel): The MuJoCo model object.
        body_id (int): The id of the body.

    Returns:
        List[int]: A list of the ids of the geoms belonging to the given body.
    """
    geom_start = sim_model.body_geomadr[body_id]
    geom_end = geom_start + sim_model.body_geomnum[body_id]
    return range(geom_start, geom_end)


def get_child_bodies(sim_model, body_id):
    """ Returns the subtree of the body structure that has the provided body as its root.

    The body structure is defined in the MuJoCo XMLs. This function returns a list containing the ids of all descendant
    bodies of a given body, including the given body.

    Args:
        sim_model (mujoco.MjModel): The MuJoCo model object.
        body_id (int): The id of the root body.

    Returns:
        List[int]: The ids of the bodies in the subtree.
    """
    children_dict = {}
    # Built a dictionary listing the children for each node
    for i in range(sim_model.nbody):
        parent = sim_model.body_parentid[i]
        if parent in children_dict:
            children_dict[parent].append(i)
        else:
            children_dict[parent] = [i]
    # Collect all the children in the subtree that has body_id as its root.
    children = []
    to_process = [body_id]
    while len(to_process) > 0:
        child = to_process.pop()
        children.append(child)
        # If this node has children: add them as well
        if child in children_dict:
            to_process.extend(children_dict[child])
    return children

def get_data_for_sensor(mujoco_model, mujoco_data, sensor_name):
    """ Get sensor data from the sensor with the provided name.

    Args:
        mujoco_model (mujoco.MjModel): The MuJoCo model object.
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        sensor_name (str): The name of the sensor.

    Returns:
        numpy.ndarray: The output values of the sensor. The shape will depend on the sensor type.
    """
    sensor_id = mujoco_model.sensor(sensor_name).id
    start = mujoco_model.sensor_adr[sensor_id]
    end = start + mujoco_model.sensor_dim[sensor_id]
    return mujoco_data.sensordata[start:end]


def get_sensor_addr(mujoco_model, sensor_id):
    """ Get the indices in the sensordata array corresponding to the given sensor.

    Args:
        mujoco_model (mujoco.MjModel): The MuJoCo model object.
        sensor_id (int): The ID of the sensor.

    Returns:
        List[int]: The array indices.
    """
    start = mujoco_model.sensor_adr[sensor_id]
    end = start + mujoco_model.sensor_dim[sensor_id]
    return range(start, end)


# ======================== Joint manipulation utils ===============================
# =================================================================================


def set_joint_qpos(mujoco_model, mujoco_data, joint_name, qpos):
    """ Sets the joint position for the joint with name joint_name.

    Directly sets the joint to the position provided by `qpos`. Note that the shape of `qpos` must match the joint! A
    free joint for example has length 7. The sizes for all types can be found in :data:`MUJOCO_JOINT_SIZES`.

    Args:
        mujoco_model (mujoco.MjModel): The MuJoCo model object.
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        joint_name (str): The name of the joint.
        qpos (numpy.ndarray|float): The new joint position. The shape of the array must match the joint!
    """
    joint_id = mujoco_model.joint(joint_name).id
    joint_qpos_addr = mujoco_model.jnt_qposadr[joint_id]
    joint_type = mujoco_model.jnt_type[joint_id]
    n_qpos = MUJOCO_JOINT_SIZES[joint_type]
    mujoco_data.qpos[joint_qpos_addr:joint_qpos_addr + n_qpos] = qpos


def get_joint_qpos_addr(mujoco_model, joint_id):
    """ Get the indices in the qpos array corresponding to the given joint.

    Args:
        mujoco_model (mujoco.MjModel): The MuJoCo model object.
        joint_id (int): The ID of the joint.

    Returns:
        List[int]: The array indices.
    """
    joint_qpos_addr = mujoco_model.jnt_qposadr[joint_id]
    joint_type = mujoco_model.jnt_type[joint_id]
    n_qpos = MUJOCO_JOINT_SIZES[joint_type]
    return range(joint_qpos_addr, joint_qpos_addr + n_qpos)


def get_joint_qvel_addr(mujoco_model, joint_id):
    """ Get the indices in the qvel array corresponding to the given joint.

    Args:
        mujoco_model (mujoco.MjModel): The MuJoCo model object.
        joint_id (int): The ID of the joint.

    Returns:
        List[int]: The array indices.
    """
    joint_qvel_addr = mujoco_model.jnt_dofadr[joint_id]
    joint_type = mujoco_model.jnt_type[joint_id]
    n_qvel = MUJOCO_DOF_SIZES[joint_type]
    return range(joint_qvel_addr, joint_qvel_addr + n_qvel)


def set_joint_locking_angle(mujoco_model, joint_name, angle, constraint_id=None):
    """ Sets the angle from default at which the joint will be locked.

    The angle is in radians, and can be positive or negative. This function does not lock or unlock a joint, merely
    changes the angle.
    This function requires that there be a constraint already existing the scene XML. This is the case for MIMo by
    default, with each joint having a constraint of the same name that is disabled at initialization.

    Args:
        mujoco_model (mujoco.MjModel): The MuJoCo model object.
        joint_name (str): The name of the joint.
        angle (float|ndarray): The locking angle(s) in radians, as a delta from the model starting value.
        constraint_id (int|ndarray): If the ID(s) of the constraint is already known the id lookup can be bypassed by
            passing it here.
    """
    if constraint_id is None:
        constraint_id = mujoco_model.equality(joint_name).id
    mujoco_model.eq_data[constraint_id, 0] = angle


def lock_joint(mujoco_model, joint_name, joint_angle=None):
    """ Locks a joint to a fixed angle.

    This function utilizes MuJoCos equality constraints to achieve the locking effect, requiring that there be a
    constraint already existing the scene XML. This is the case for MIMo by default, with each joint having a constraint
    of the same name that is disabled at initialization.
    In effect this function enables the equality constraint with same name as the argument.

    Args:
        mujoco_model (mujoco.MjModel): The MuJoCo model object.
        joint_name (str): The name of the joint.
        joint_angle (float): The locking angle in radians, as a delta from the model starting value. The angle that the
            joint will be locked to can be set separately using :func:`~mimoEnv.utils.set_joint_locking_angle`. By
            default, joints are locked into the value they have in the scene xml.
    """
    constraint_id = mujoco_model.equality(joint_name).id
    if joint_angle is not None:
        set_joint_locking_angle(mujoco_model, joint_name, joint_angle, constraint_id=constraint_id)
    mujoco_model.eq_active[constraint_id] = True


def unlock_joint(mujoco_model, joint_name):
    """ Unlocks a given joint.

    See :func:`~mimoEnv.utils.lock_joint`.

    Args:
        mujoco_model (mujoco.MjModel): The MuJoCo model object.
        joint_name (str): The name of the joint.
    """
    constraint_id = mujoco_model.equality(joint_name).id
    mujoco_model.eq_active[constraint_id] = False


# ======================== Mujoco frame utils =====================================
# =================================================================================


def get_geom_position(mujoco_data, geom_id):
    """ Returns the position of geom in the world frame.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        geom_id (int): The id of the geom.

    Returns:
        numpy.ndarray: The position of the geom in the world frame. Shape (3,).
    """
    return mujoco_data.geom_xpos[geom_id]


def get_body_position(mujoco_data, body_id):
    """ Returns the position of body in the world frame.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        body_id (int): The id of the body.

    Returns:
        numpy.ndarray: The position of the body in the world frame. Shape (3,).
    """
    return mujoco_data.xpos[body_id]


def get_geom_rotation(mujoco_data, geom_id):
    """ Returns the rotation matrix that rotates the geoms frame to the world frame.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        geom_id (int): The id of the geom.

    Returns:
          numpy.ndarray: A (3,3) array containing the rotation matrix.
    """
    return np.reshape(mujoco_data.geom_xmat[geom_id], (3, 3))


def get_body_rotation(mujoco_data, body_id):
    """ Returns the rotation matrix that rotates the bodies frame to the world frame.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        body_id (int): The id of the body.

    Returns:
          numpy.ndarray: A (3,3) array containing the rotation matrix.
    """
    return np.reshape(mujoco_data.xmat[body_id], (3, 3))


def world_pos_to_geom(mujoco_data, position, geom_id):
    """ Converts position from the world coordinate frame to a geom specific frame.

    Position can be a vector or an array of vectors such that the last dimension has size 3.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        position (numpy.ndarray): Array containing position(s). Its shape should be either (3,) or (.., 3).
        geom_id (int): The id of the geom.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the converted coordinates.
    """
    rel_pos = position - get_geom_position(mujoco_data, geom_id)
    rel_pos = world_rot_to_geom(mujoco_data, rel_pos, geom_id)
    return rel_pos


def world_pos_to_body(mujoco_data, position, body_id):
    """ Converts position from the world coordinate frame to a body specific frame.

    Position can be a vector or an array of vectors such that the last dimension has size 3.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        position (numpy.ndarray): Array containing position(s). Its shape should be either (3,) or (.., 3).
        body_id (int): The id of the geom.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the converted coordinates.
    """
    rel_pos = position - get_body_position(mujoco_data, body_id)
    rel_pos = world_rot_to_body(mujoco_data, rel_pos, body_id)
    return rel_pos


def geom_pos_to_world(mujoco_data, position, geom_id):
    """ Converts position from the geom specific coordinate frame to the world frame.

    Position can be a vector or an array of vectors such that the last dimension has size 3.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        position (numpy.ndarray): Array containing position(s). Its shape should be either (3,) or (.., 3).
        geom_id (int): The id of the geom.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the converted coordinates.
    """
    global_pos = geom_rot_to_world(mujoco_data, position, geom_id)
    return global_pos + get_geom_position(mujoco_data, geom_id)


def body_pos_to_world(mujoco_data, position, body_id):
    """ Converts position from the body specific coordinate frame to the world frame.

    Position can be a vector or an array of vectors such that the last dimension has size 3.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        position (numpy.ndarray): Array containing position(s). Its shape should be either (3,) or (.., 3).
        body_id (int): The id of the body.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the converted coordinates.
    """
    global_pos = body_rot_to_world(mujoco_data, position, body_id)
    global_pos = global_pos + get_body_position(mujoco_data, body_id)
    return global_pos


def geom_pos_to_body(mujoco_data, position, geom_id, body_id):
    """ Converts position from the geom specific coordinate frame to the frame of a specific body.

    Position can be a vector or an array of vectors such that the last dimension has size 3.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        position (numpy.ndarray): Array containing position(s). Its shape should be either (3,) or (.., 3).
        geom_id (int): The id of the geom.
        body_id (int): The id of the body.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the converted coordinates.
    """
    world_pos = geom_pos_to_world(mujoco_data, position, geom_id)
    body_pos = world_pos_to_body(mujoco_data, world_pos, body_id)
    return body_pos


def body_pos_to_geom(mujoco_data, position, body_id, geom_id):
    """ Converts position from the body specific coordinate frame to the frame of a specific geom.

    Position can be a vector or an array of vectors such that the last dimension has size 3.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        position (numpy.ndarray): Array containing position(s). Its shape should be either (3,) or (.., 3).
        body_id (int): The id of the body.
        geom_id (int): The id of the geom.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the converted coordinates.
    """
    world_pos = body_pos_to_world(mujoco_data, position, body_id)
    return world_pos_to_geom(mujoco_data, world_pos, geom_id)


def geom_pos_to_geom(mujoco_data, position, geom_id_source, geom_id_target):
    """ Converts position from one geoms coordinate frame to another.

    Position can be a vector or an array of vectors such that the last dimension has size 3.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        position (numpy.ndarray): Array containing position(s). Its shape should be either (3,) or (.., 3).
        geom_id_source (int): The id of the geom for the initial coordinate frame.
        geom_id_target (int): The id of the geom for the output coordinate frame.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the converted coordinates.
    """
    world_pos = geom_pos_to_world(mujoco_data, position, geom_id_source)
    return world_pos_to_geom(mujoco_data, world_pos, geom_id_target)


def body_pos_to_body(mujoco_data, position, body_id_source, body_id_target):
    """ Converts position from one bodies coordinate frame to another.

    Position can be a vector or an array of vectors such that the last dimension has size 3.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        position (numpy.ndarray): Array containing position(s). Its shape should be either (3,) or (.., 3).
        body_id_source (int): The id of the body for the initial coordinate frame.
        body_id_target (int): The id of the body for the output coordinate frame.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the converted coordinates.
    """
    world_pos = body_pos_to_world(mujoco_data, position, body_id_source)
    return world_pos_to_body(mujoco_data, world_pos, body_id_target)


def geom_rot_to_world(mujoco_data, vector, geom_id):
    """ Converts a vectors direction from a geoms specific coordinate frame to the world frame.

    Unlike the functions for the positions, this only converts the direction.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        vector (numpy.ndarray): A vector or array of vectors. Shape must be either (3,) or (.., 3).
        geom_id (int): The id of the geom.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the rotated vector.
    """
    return np.transpose(rotate_vector(np.transpose(vector), get_geom_rotation(mujoco_data, geom_id)))


def body_rot_to_world(mujoco_data, vector, body_id):
    """ Converts a vectors direction from a bodies specific coordinate frame to the world frame.

    Unlike the functions for the positions, this only converts the direction.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        vector (numpy.ndarray): A vector or array of vectors. Shape must be either (3,) or (.., 3).
        body_id (int): The id of the body.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the rotated vector.
    """
    return np.transpose(rotate_vector(np.transpose(vector), get_body_rotation(mujoco_data, body_id)))


def world_rot_to_geom(mujoco_data, vector, geom_id):
    """ Converts a vectors direction from the world coordinate frame to a geoms specific frame.

    Unlike the functions for the positions, this only converts the direction.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        vector (numpy.ndarray): A vector or array of vectors. Shape must be either (3,) or (.., 3).
        geom_id (int): The id of the geom.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the rotated vector.
    """
    return np.transpose(rotate_vector_transpose(np.transpose(vector), get_geom_rotation(mujoco_data, geom_id)))


def world_rot_to_body(mujoco_data, vector, body_id):
    """ Converts a vectors direction from the world coordinate frame to a bodies specific frame.

    Unlike the functions for the positions, this only converts the direction.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        vector (numpy.ndarray): A vector or array of vectors. Shape must be either (3,) or (.., 3).
        body_id (int): The id of the body.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the rotated vector.
    """
    return np.transpose(rotate_vector_transpose(np.transpose(vector), get_body_rotation(mujoco_data, body_id)))


def geom_rot_to_body(mujoco_data, vector, geom_id, body_id):
    """ Converts a vectors direction from a geoms coordinate frame to a bodies frame.

    Unlike the functions for the positions, this only converts the direction.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        vector (numpy.ndarray): A vector or array of vectors. Shape must be either (3,) or (.., 3).
        geom_id (int): The id of the geom.
        body_id (int): The id of the body.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the rotated vector.
    """
    world_rot = geom_rot_to_world(mujoco_data, vector, geom_id)
    return world_rot_to_body(mujoco_data, world_rot, body_id)


def body_rot_to_geom(mujoco_data, vector, body_id, geom_id):
    """ Converts a vectors direction from a bodies coordinate frame to a geoms frame.

    Unlike the functions for the positions, this only converts the direction.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        vector (numpy.ndarray): A vector or array of vectors. Shape must be either (3,) or (.., 3).
        body_id (int): The id of the body.
        geom_id (int): The id of the geom.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the rotated vector.
    """
    world_rot = body_rot_to_world(mujoco_data, vector, body_id)
    return world_rot_to_geom(mujoco_data, world_rot, geom_id)


def geom_rot_to_geom(mujoco_data, vector, geom_id_source, geom_id_target):
    """ Converts a vectors direction from one geoms coordinate frame to another.

    Unlike the functions for the positions, this only converts the direction.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        vector (numpy.ndarray): A vector or array of vectors. Shape must be either (3,) or (.., 3).
        geom_id_source (int): The id of the geom for the initial coordinate frame.
        geom_id_target (int): The id of the geom for the output coordinate frame.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the rotated vector.
    """
    world_rot = geom_rot_to_world(mujoco_data, vector, geom_id_source)
    return world_rot_to_geom(mujoco_data, world_rot, geom_id_target)


def body_rot_to_body(mujoco_data, vector, body_id_source, body_id_target):
    """ Converts a vectors direction from one bodies coordinate frame to another.

    Unlike the functions for the positions, this only converts the direction.

    Args:
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        vector (numpy.ndarray): A vector or array of vectors. Shape must be either (3,) or (.., 3).
        body_id_source (int): The id of the body for the initial coordinate frame.
        body_id_target (int): The id of the body for the output coordinate frame.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the rotated vector.
    """
    world_rot = body_rot_to_world(mujoco_data, vector, body_id_source)
    return world_rot_to_body(mujoco_data, world_rot, body_id_target)


# ======================== Plotting utils =========================================
# =================================================================================

def plot_points(points, limit=1.0, title="", show=True):
    """ Plots an array of points.

    Args:
        points (numpy.ndarray): An array containing points. Shape should be (n, 3) for n points.
        limit (float): The limit that is applied to the axis. Default 1.
        title (str): The title for the plot. Empty by default.
        show (bool): If ``True`` the plot is rendered to a window, if ``False`` the figure and axis objects are returned
            instead.

    Returns:
        Tuple[plt.Figure, plt.Axes]|None: A tuple (fig, ax) containing the pyplot figure and axis objects if `show` is
        ``False``, ``None`` otherwise.
    """
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, color="k", s=20)
    ax.set_title(title)
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig, ax


def plot_forces(points, vectors, limit=1.0, title="", show=True):
    """ Plots an array of points and vectors pointing from those points.

    The arrays `points` and `vectors` must have the same shape. For each point there is a vector, plotted as arrows,
    the direction and size of which is determined by the `vectors` argument, starting from that point.

    Args:
        points (numpy.ndarray): An array containing the points. Shape should be (n, 3) for n points.
        vectors (numpy.ndarray): An array of vectors with one for each point. Shape should be (n, 3) for n points.
        limit (float): The limit that is applied to the axis. Default 1.
        title (str): The title for the plot. Empty by default.
        show (bool): If ``True`` the plot is rendered to a window, if ``False`` the figure and axis objects are returned
            instead.

    Returns:
        Tuple[plt.Figure, plt.Axes]|None: A tuple (fig, ax) containing the pyplot figure and axis objects if `show` is
        ``False``, ``None`` otherwise.
    """
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]

    us = vectors[:, 0]
    vs = vectors[:, 1]
    ws = vectors[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(xs, ys, zs, us, vs, ws)
    ax.scatter(xs, ys, zs, color="k", s=10, depthshade=True, alpha=0.4)
    ax.set_title(title)
    ax.set_xlim([-limit, limit])
    ax.set_ylim([-limit, limit])
    ax.set_zlim([-limit, limit])
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    if show:
        plt.show()
    else:
        return fig, ax


# ======================== Mass utils =============================================
# =================================================================================

def determine_geom_masses(mujoco_model, mujoco_data, body_ids, target_mass, print_out=False):
    """ Distribute a target mass over multiple bodies.

    Given a list of bodies and a desired target mass, calculate the mass of component geoms assuming identical
    density such that the total mass of the bodies matches the target mass. This function takes account of overlap
    between geoms within each body, but not between bodies.

    Args:
        mujoco_model (mujoco.MjModel): The MuJoCo model object.
        mujoco_data (mujoco.MjData): The MuJoCo data object.
        body_ids (List[int]): A list of bodies by ID over which the mass will be distributed.
        target_mass (float): The target mass.
        print_out (bool): If ``True``, target masses and body names are printed to console.

    Returns:
        Dict[str, List[float]]: A dictionary with body names as keys and a list of geom masses as values.
    """
    from mimoTouch.sensormeshes import mesh_box, mesh_sphere, mesh_capsule, mesh_cylinder, mesh_ellipsoid
    mesh_distance = 0.001  # The distance between points on the mesh we use to calculate volume
    mass = 0
    volume = 0  # Total volume of the bodies, accounting for overlap
    volumes = {}  # Overlap of each individual body, accounting for overlap
    volumes_with_overlap = {}  # Overlap of each individual body, not accounting for overlap, i.e. volume that is a part
    # of multiple geoms will be counted multiple times.
    meshes = {}
    for body_id in body_ids:
        mass += mujoco_model.body_mass[body_id]
        body_meshes = []
        meshes[body_id] = body_meshes
        for geom_id in get_geoms_for_body(mujoco_model, body_id):
            geom_type = mujoco_model.geom_type[geom_id]
            size = mujoco_model.geom_size[geom_id]

            if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                mesh = mesh_box(mesh_distance, size)
            elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
                mesh = mesh_sphere(mesh_distance, size[0])
            elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
                mesh = mesh_capsule(mesh_distance, 2 * size[1], size[0])
            elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
                # Cylinder size 0 is radius, size 1 is half the length
                mesh = mesh_cylinder(mesh_distance, 2 * size[1], size[0])
            elif geom_type == mujoco.mjtGeom.mjGEOM_PLANE:
                RuntimeWarning("Cannot add sensors to plane geoms!")
                return None
            elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                mesh = mesh_ellipsoid(mesh_distance, size)
            elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                size = mujoco_model.geom_rbound[geom_id]
                mesh = mesh_sphere(mesh_distance, size)
            body_meshes.append(mesh)
            mesh.vertices = geom_pos_to_body(mujoco_data, mesh.vertices.copy(), geom_id, body_id)

        if len(body_meshes) > 1:
            volumes[body_id] = body_meshes[0].union(body_meshes[1:]).volume
            volumes_with_overlap[body_id] = sum([mesh.volume for mesh in body_meshes])
        else:
            volumes[body_id] = body_meshes[0].volume
            volumes_with_overlap[body_id] = body_meshes[0].volume

        volume += volumes[body_id]

    output_dict = {}
    for body_id in body_ids:
        body_name = mujoco_model.body_id2name(body_id)
        this_body_volume_contribution_ratio = volumes[body_id] / volume
        this_body_target_mass = this_body_volume_contribution_ratio * target_mass
        this_body_target_density = this_body_target_mass / volumes_with_overlap[body_id]
        masses = [mesh.volume * this_body_target_density for mesh in meshes[body_id]]
        output_dict[body_name] = masses

    if print_out:
        print("Current total mass: {}\n".format(mass))
        print("Target mass: {}\n".format(target_mass))
        print("Volume: {}\n".format(volume))
        print("Final overall density: {}".format(target_mass / volume))
        print("Body Name\tTarget masses for constituent geoms")
        for body_name in output_dict:
            print(body_name, ":", ", ".join(["{:.4e}".format(mass) for mass in output_dict[body_name]]))

    return output_dict
