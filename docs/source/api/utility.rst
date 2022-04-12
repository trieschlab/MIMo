Utility functions
=================

.. contents::
   :depth: 4

Vector operations
-----------------

.. autosummary::
   :toctree: _autosummary
   :recursive:
   :nosignatures:

   mimoEnv.utils.rotate_vector
   mimoEnv.utils.rotate_vector_transpose
   mimoEnv.utils.weighted_sum_vectors
   mimoEnv.utils.normalize_vectors
   
MuJoCo access utilities
-----------------------
 
 .. autosummary::
   :toctree: _autosummary
   :recursive:
   :nosignatures:
   
   mimoEnv.utils.get_geom_id
   mimoEnv.utils.get_body_id
   mimoEnv.utils.get_geoms_for_body
   mimoEnv.utils.get_child_bodies
   mimoEnv.utils.set_joint_qpos
   mimoEnv.utils.get_data_for_sensor
   mimoEnv.utils.texture_name2id
   mimoEnv.utils.material_name2id
   
MuJoCo coordinate frame utilities
---------------------------------
 
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
------------------

 .. autosummary::
   :toctree: _autosummary
   :recursive:
   :nosignatures:
   
   mimoEnv.utils.plot_points
   mimoEnv.utils.plot_forces
   
mimoEnv.utils
-------------

.. automodule:: mimoEnv.utils
   :members:
   :undoc-members:
   :show-inheritance:
