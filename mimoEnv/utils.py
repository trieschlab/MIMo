import numpy as np
from matplotlib import pyplot as plt
from mujoco_py.generated import const

EPS = 1e-10


MUJOCO_JOINT_SIZES = {
    const.JNT_FREE: 7,
    const.JNT_BALL: 4,
    const.JNT_SLIDE: 1,
    const.JNT_HINGE: 1,
}
""" Size of qpos entries for each joint type; free, ball, slide, hinge. 

:meta hide-value:
"""


MUJOCO_DOF_SIZES = {
    const.JNT_FREE: 6,
    const.JNT_BALL: 3,
    const.JNT_SLIDE: 1,
    const.JNT_HINGE: 1,
}
""" Size of qvel entries for each joint type; free, ball, slide, hinge. 

:meta hide-value:
"""


def rotate_vector(vector, rot_matrix):
    """ Rotates the vectors with the the rotation matrix.

    The vector can be a 1d vector or a multidimensional array of vectors, as long as the final dimension has length 3.
    Convention for mujoco matrices: Use rotate_vector to convert from special frame to global, rotate_vector_transpose
    for the inverse rotation. The exception are the contact frames, which are transposed.

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
        mujoco_model (sim.model): The MuJoCo model object.
        geom_id (int): The id of the geom.
        geom_name (str): The name of the geom.

    Returns:
        int: The id of the geom referred to by either the name or the id above.
    """
    if geom_id is None and geom_name is None:
        raise RuntimeError("Need either name or geom id")

    if geom_id is None:
        geom_id = mujoco_model.geom_name2id(geom_name)

    return geom_id


def get_body_id(mujoco_model, body_id=None, body_name=None):
    """ Convenience function to get body ids.
    
    Works identical to :func:`~mimoEnv.utils.get_geom_id`
    
    Args:
        mujoco_model (sim.model): The MuJoCo model object.
        body_id (int): The id of the body.
        body_name (str): The name of the body.
        
    Returns:
        int: The id of the geom referred to by either the name or the id above.
    """
    if body_id is None and body_name is None:
        raise RuntimeError("Need either name or body id")

    if body_id is None:
        body_id = mujoco_model.body_name2id(body_name)

    return body_id


def get_geoms_for_body(sim_model, body_id):
    """ Returns all geom ids belonging to a given body.

    Args:
        sim_model (sim.model): The MuJoCo model object.
        body_id (int): The id of the body.

    Returns:
        list of int: A list of the ids of the geoms belonging to the given body.
    """
    geom_start = sim_model.body_geomadr[body_id]
    geom_end = geom_start + sim_model.body_geomnum[body_id]
    return range(geom_start, geom_end)


def get_child_bodies(sim_model, body_id):
    """ Returns the subtree of the body structure that has the provided body as its root.

    The body structure is defined in the MuJoCo XMLs. This function returns a list containing the ids of all descendant
    bodies of a given body, including the given body.

    Args:
        sim_model (sim.model): The MuJoCo model object.
        body_id (int): The id of the root body.

    Returns:
        list of int: The ids of the bodies in the subtree.
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
        mujoco_model (sim.model): The MuJoCo model object.
        mujoco_data (sim.data): The MuJoCo data object.
        sensor_name (str): The name of the sensor.

    Returns:
        numpy.ndarray: The output values of the sensor. The shape will depend on the sensor type.
    """
    sensor_id = mujoco_model.sensor_name2id(sensor_name)
    start = mujoco_model.sensor_adr[sensor_id]
    end = start + mujoco_model.sensor_dim[sensor_id]
    return mujoco_data.sensordata[start:end]


def get_sensor_addr(mujoco_model, sensor_id):
    """ Get the indices in the sensordata array corresponding to the given sensor.

    Args:
        mujoco_model (sim.model): The MuJoCo model object.
        sensor_id (int): The ID of the sensor.

    Returns:
        A list of indices.
    """
    start = mujoco_model.sensor_adr[sensor_id]
    end = start + mujoco_model.sensor_dim[sensor_id]
    return range(start, end)


def _decode_name(mujoco_model, name_adr):
    """ Decode the name given a name array address.

    mujoco-py unfortunately does not properly wrap all of mujocos data structures/functions, so we have to get some
    names (such as textures and materials) manually. This is a very tedious process in python.

    Args:
        mujoco_model (sim.model): The MuJoCo model object.
        name_adr (int): Array address in the MuJoCo name array.

    Returns:
        str: The name located at the address.
    """
    # TODO: Figure out cython so we don't have to do this
    # TODO: Alternatively at least cache the name-id relationship somewhere
    i = 0
    while mujoco_model.names[name_adr + i].decode() != "":
        i += 1
    if i == 0:
        return None
    str_array = mujoco_model.names[name_adr: name_adr + i].astype(str)
    return "".join(str_array)


def texture_name2id(mujoco_model, texture_name):
    """ Returns the id for the texture with the given name.

    Textures in mujoco can be named, but we need the id to be able to do almost anything. This function allows grabbing
    the id of a texture with a given name. It uses :func:`~mimoEnv.utils._decode_name` to do this, which is not
    optimized, so do not use this often.

    Args:
        mujoco_model (sim.model): The MuJoCo model object.
        texture_name (str): The name of the texture.

    Returns:
        int: The id of the texture.
    """
    tex_id = None
    for i, name_adr in enumerate(mujoco_model.name_texadr):
        name = _decode_name(mujoco_model, name_adr)
        if name == texture_name:
            tex_id = i
            break
    if tex_id is None:
        raise RuntimeError("Could not find texture with name {}".format(texture_name))
    return tex_id


def material_name2id(mujoco_model, material_name):
    """ Returns the id for the material with the given name.

    Materials in mujoco can be named, but we need the id to be able to do almost anything. This function allows grabbing
    the id of a material with a given name. It uses :func:`~mimoEnv.utils._decode_name` to do this, which is not
    optimized, so do not use this often.

    Args:
        mujoco_model (sim.model): The MuJoCo model object.
        material_name (str): The name of the material.

    Returns:
        int: The id of the material.
    """
    mat_id = None
    for i, name_adr in enumerate(mujoco_model.name_matadr):
        name = _decode_name(mujoco_model, name_adr)
        if name == material_name:
            mat_id = i
            break
    if mat_id is None:
        raise RuntimeError("Could not find material with name {}".format(material_name))
    return mat_id


def equality_name2id(mujoco_model, equality_constraint_name):
    """ Returns the id for the equality constraint with the given name.

    Constraints in mujoco can be named, but we need the id to be able to do almost anything. This function allows
    grabbing the id of a constraint with a given name. It uses :func:`~mimoEnv.utils._decode_name` to do this, which is
    not optimized, so do not use this often.

    Args:
        mujoco_model (sim.model): The MuJoCo model object.
        equality_constraint_name (str): The name of the constraint.

    Returns:
        int: The id of the constraint.
    """
    const_id = None
    for i, name_adr in enumerate(mujoco_model.name_eqadr):
        name = _decode_name(mujoco_model, name_adr)
        if name == equality_constraint_name:
            const_id = i
            break
    if const_id is None:
        raise RuntimeError("Could not find equality constraint with name {}".format(equality_constraint_name))
    return const_id


# ======================== Joint manipulation utils ===============================
# =================================================================================


def set_joint_qpos(mujoco_model, mujoco_data, joint_name, qpos):
    """ Sets the joint position for the joint with name joint_name.

    Directly sets the joint to the position provided by `qpos`. Note that the shape of `qpos` must match the joint! A
    free joint for example has length 7. The sizes for all types can be found in :data:`MUJOCO_JOINT_SIZES`

    Args:
        mujoco_model (sim.model): The MuJoCo model object.
        mujoco_data (sim.data): The MuJoCo data object.
        joint_name (str): The name of the joint.
        qpos (numpy.ndarray): The new joint position. The shape of the array must match the joint!
    """
    joint_id = mujoco_model.joint_name2id(joint_name)
    joint_qpos_addr = mujoco_model.jnt_qposadr[joint_id]
    joint_type = mujoco_model.jnt_type[joint_id]
    n_qpos = MUJOCO_JOINT_SIZES[joint_type]
    mujoco_data.qpos[joint_qpos_addr:joint_qpos_addr + n_qpos] = qpos


def get_joint_qpos_addr(mujoco_model, joint_id):
    """ Get the indices in the qpos array corresponding to the given joint.

    Args:
        mujoco_model (sim.model): The MuJoCo model object.
        joint_id (int): The ID of the joint.

    Returns:
        A list of indices.
    """
    joint_qpos_addr = mujoco_model.jnt_qposadr[joint_id]
    joint_type = mujoco_model.jnt_type[joint_id]
    n_qpos = MUJOCO_JOINT_SIZES[joint_type]
    return range(joint_qpos_addr, joint_qpos_addr + n_qpos)


def get_joint_qvel_addr(mujoco_model, joint_id):
    """ Get the indices in the qvel array corresponding to the given joint.

    Args:
        mujoco_model (sim.model): The MuJoCo model object.
        joint_id (int): The ID of the joint.

    Returns:
        A list of indices.
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
        mujoco_model (sim.model): The MuJoCo model object.
        joint_name (str): The name of the joint.
        angle (float): The locking angle in radians, as a delta from the model starting value.
        constraint_id (int): If the ID of the constraint is already known the name search can be bypassed by passing
            it here.
    """
    if constraint_id is None:
        constraint_id = equality_name2id(mujoco_model, joint_name)
    mujoco_model.eq_data[constraint_id, 0] = angle


def lock_joint(mujoco_model, joint_name, joint_angle=None):
    """ Locks a joint to a fixed angle.

    This function utilizes MuJoCos equality constraints to achieve the locking effect, requiring that there be a
    constraint already existing the scene XML. This is the case for MIMo by default, with each joint having a constraint
    of the same name that is disabled at initialization.
    In effect this function enables the equality constraint with same name as the argument.

    Args:
        mujoco_model (sim.model): The MuJoCo model object.
        joint_name (str): The name of the joint.
        joint_angle (float): The locking angle in radians, as a delta from the model starting value. The angle that the
            joint will be locked to can be set separately using :func:`~mimoEnv.utils.set_joint_locking_angle`. By
            default joints are locked into the value they have in the scene xml.
    """
    constraint_id = equality_name2id(mujoco_model, joint_name)
    if joint_angle is not None:
        set_joint_locking_angle(mujoco_model, joint_name, joint_angle, constraint_id=constraint_id)
    mujoco_model.eq_active[constraint_id] = True


def unlock_joint(mujoco_model, joint_name):
    """ Unlocks a given joint.

    See :func:`~mimoEnv.utils.lock_joint`.

    Args:
        mujoco_model (sim.model): The MuJoCo model object.
        joint_name (str): The name of the joint.
    """
    constraint_id = equality_name2id(mujoco_model, joint_name)
    mujoco_model.eq_active[constraint_id] = False


# ======================== Mujoco frame utils =====================================
# =================================================================================


def get_geom_position(mujoco_data, geom_id):
    """ Returns the position of geom in the world frame.

    Args:
        mujoco_data (sim.data): The MuJoCo data object.
        geom_id (int): The id of the geom.

    Returns:
        numpy.ndarray: The position of the geom in the world frame. Shape (3,).
    """
    return mujoco_data.geom_xpos[geom_id]


def get_body_position(mujoco_data, body_id):
    """ Returns the position of body in the world frame.

    Args:
        mujoco_data (sim.data): The MuJoCo data object.
        body_id (int): The id of the body.

    Returns:
        numpy.ndarray: The position of the body in the world frame. Shape (3,).
    """
    return mujoco_data.body_xpos[body_id]


def get_geom_rotation(mujoco_data, geom_id):
    """ Returns the rotation matrix that rotates the geoms frame to the world frame.

    Args:
        mujoco_data (sim.data): The MuJoCo data object.
        geom_id (int): The id of the geom.

    Returns:
          numpy.ndarray: A (3,3) array containing the rotation matrix.
    """
    return np.reshape(mujoco_data.geom_xmat[geom_id], (3, 3))


def get_body_rotation(mujoco_data, body_id):
    """ Returns the rotation matrix that rotates the bodies frame to the world frame.

    Args:
        mujoco_data (sim.data): The MuJoCo data object.
        body_id (int): The id of the body.

    Returns:
          numpy.ndarray: A (3,3) array containing the rotation matrix.
    """
    return np.reshape(mujoco_data.body_xmat[body_id], (3, 3))


def world_pos_to_geom(mujoco_data, position, geom_id):
    """ Converts position from the world coordinate frame to a geom specific frame.

    Position can be a vector or an array of vectors such that the last dimension has size 3.

    Args:
        mujoco_data (sim.data): The MuJoCo data object.
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
        mujoco_data (sim.data): The MuJoCo data object.
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
        mujoco_data (sim.data): The MuJoCo data object.
        position (numpy.ndarray): Array containing position(s). Its shape should be either (3,) or (.., 3).
        geom_id (int): The id of the geom.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the converted coordinates.
    """
    global_pos = geom_rot_to_world(mujoco_data, position, geom_id)
    global_pos = global_pos + get_geom_position(mujoco_data, geom_id)
    return global_pos


def body_pos_to_world(mujoco_data, position, body_id):
    """ Converts position from the body specific coordinate frame to the world frame.

    Position can be a vector or an array of vectors such that the last dimension has size 3.

    Args:
        mujoco_data (sim.data): The MuJoCo data object.
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
        mujoco_data (sim.data): The MuJoCo data object.
        position (numpy.ndarray): Array containing position(s). Its shape should be either (3,) or (.., 3).
        geom_id (int): The id of the geom.
        body_id (int): The id of the body.

    Returns:
        numpy.ndarray: Array of the same shape as the input array with the converted coordinates.
    """
    world_pos = geom_pos_to_world(mujoco_data, position, geom_id)
    return world_pos_to_body(mujoco_data, world_pos, body_id)


def body_pos_to_geom(mujoco_data, position, body_id, geom_id):
    """ Converts position from the body specific coordinate frame to the frame of a specific geom.

    Position can be a vector or an array of vectors such that the last dimension has size 3.

    Args:
        mujoco_data (sim.data): The MuJoCo data object.
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
        mujoco_data (sim.data): The MuJoCo data object.
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
        mujoco_data (sim.data): The MuJoCo data object.
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
        mujoco_data (sim.data): The MuJoCo data object.
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
        mujoco_data (sim.data): The MuJoCo data object.
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
        mujoco_data (sim.data): The MuJoCo data object.
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
        mujoco_data (sim.data): The MuJoCo data object.
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
        mujoco_data (sim.data): The MuJoCo data object.
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
        mujoco_data (sim.data): The MuJoCo data object.
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
        mujoco_data (sim.data): The MuJoCo data object.
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
        mujoco_data (sim.data): The MuJoCo data object.
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

def plot_points(points, limit: float = 1.0, title="", show=True):
    """ Plots an array of points.

    Args:
        points (numpy.ndarray): An array containing points. Shape should be (n, 3) for n points.
        limit (float): The limit that is applied to the axis. Default 1.
        title (str): The title for the plot. Empty by default.
        show (bool): If `True` the plot is rendered to a window, if `False` the figure and axis objects are returned
            instead.

    Returns:
        The figure and axis objects if `show` is `False`, ``None`` otherwise.
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


def plot_forces(points, vectors, limit: float = 1.0, title="", show=True):
    """ Plots an array of points and vectors pointing from those points.

    points and vectors must have the same shape. For each point there is a vector, the direction and size of which is
    determined by the `vectors` argument, starting from that point.

    Args:
        points (numpy.ndarray): An array containing the points. Shape should be (n, 3) for n points.
        vectors (numpy.ndarray): An array of vectors with one for each point. Shape should be (n, 3) for n points.
        limit (float): The limit that is applied to the axis. Default 1.
        title (str): The title for the plot. Empty by default.
        show (bool): If `True` the plot is rendered to a window, if `False` the figure and axis objects are returned
            instead.

    Returns:
        The figure and axis objects if `show` is `False`, ``None`` otherwise.
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
