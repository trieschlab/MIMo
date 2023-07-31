Utility functions
=================

.. contents::
   :depth: 4

Function summaries
------------------
   
Vector operations
+++++++++++++++++

.. autosummary::
   :toctree: _autosummary
   :recursive:
   :nosignatures:

   mimoEnv.utils.rotate_vector
   mimoEnv.utils.rotate_vector_transpose
   mimoEnv.utils.weighted_sum_vectors
   mimoEnv.utils.normalize_vectors
   
MuJoCo access utilities
+++++++++++++++++++++++
 
 .. autosummary::
   :toctree: _autosummary
   :recursive:
   :nosignatures:
   
   mimoEnv.utils.get_geom_id
   mimoEnv.utils.get_body_id
   mimoEnv.utils.get_geoms_for_body
   mimoEnv.utils.get_child_bodies
   mimoEnv.utils.get_data_for_sensor
   mimoEnv.utils.get_sensor_addr
   mimoEnv.utils.texture_name2id
   mimoEnv.utils.material_name2id
   mimoEnv.utils.equality_name2id

Joint Manipulation utilities
++++++++++++++++++++++++++++

 .. autosummary::
   :toctree: _autosummary
   :recursive:
   :nosignatures:

   mimoEnv.utils.set_joint_qpos
   mimoEnv.utils.get_joint_qpos_addr
   mimoEnv.utils.get_joint_qvel_addr
   mimoEnv.utils.set_joint_locking_angle
   mimoEnv.utils.lock_joint
   mimoEnv.utils.unlock_joint
   
MuJoCo coordinate frame utilities
+++++++++++++++++++++++++++++++++
 
 .. autosummary::
   :toctree: _autosummary
   :recursive:
   :nosignatures:
   
   mimoEnv.utils.get_geom_position
   mimoEnv.utils.get_body_position
   mimoEnv.utils.get_geom_rotation
   mimoEnv.utils.get_body_rotation
   mimoEnv.utils.world_pos_to_geom
   mimoEnv.utils.world_pos_to_body
   mimoEnv.utils.geom_pos_to_world
   mimoEnv.utils.body_pos_to_world
   mimoEnv.utils.geom_pos_to_body
   mimoEnv.utils.body_pos_to_geom
   mimoEnv.utils.geom_pos_to_geom
   mimoEnv.utils.body_pos_to_body
   mimoEnv.utils.geom_rot_to_world
   mimoEnv.utils.body_rot_to_world
   mimoEnv.utils.world_rot_to_geom
   mimoEnv.utils.world_rot_to_body
   mimoEnv.utils.geom_rot_to_body
   mimoEnv.utils.body_rot_to_geom
   mimoEnv.utils.geom_rot_to_geom
   mimoEnv.utils.body_rot_to_body
   
Plotting utilities
++++++++++++++++++

 .. autosummary::
   :toctree: _autosummary
   :recursive:
   :nosignatures:
   
   mimoEnv.utils.plot_points
   mimoEnv.utils.plot_forces

Assorted functions
++++++++++++++++++

 .. autosummary::
   :toctree: _autosummary
   :recursive:
   :nosignatures:

   mimoEnv.utils.determine_geom_masses

Data fields
-----------

.. autodata:: mimoEnv.utils.MUJOCO_JOINT_SIZES
   :no-value:

.. autodata:: mimoEnv.utils.MUJOCO_DOF_SIZES
   :no-value:

Detail documentation
----------------------

.. automodule:: mimoEnv.utils
   :members:
   :undoc-members:
   :show-inheritance:
