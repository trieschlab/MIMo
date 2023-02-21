MIMoEnv base class
==================

This module defines the base MIMo environment.

The abstract base class is :class:`~mimoEnv.envs.mimo_env.MIMoEnv`. Default parameters for all the sensory modalities
are provided as well.
The muscle model involves a drop-in replacement class :class:`~mimoEnv.envs.mimo_muscle_env.MIMoMusclenv`. Replacing
:class:`~mimoEnv.envs.mimo_env.MIMoEnv` with :class:`~mimoEnv.envs.mimo_muscle_env.MIMoMusclenv` as the parent class
is sufficient to swap between the two.

.. contents::
   :depth: 4


MIMoEnv
-------
   
.. autoclass:: mimoEnv.envs.mimo_env.MIMoEnv
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

MIMoMuscleEnv
-------

.. autoclass:: mimoEnv.envs.mimo_muscle_env.MIMoMuscleEnv
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

.. _sec default_data:

Default data fields
-------------------

.. autodata:: mimoEnv.envs.mimo_env.SCENE_DIRECTORY
   :no-value:

.. autodata:: mimoEnv.envs.mimo_env.EMOTES
   :no-value:
   
.. autodata:: mimoEnv.envs.mimo_env.DEFAULT_PROPRIOCEPTION_PARAMS
   :no-value:
   
.. autodata:: mimoEnv.envs.mimo_env.DEFAULT_TOUCH_PARAMS
   :no-value:

.. autodata:: mimoEnv.envs.mimo_env.DEFAULT_TOUCH_PARAMS_V2
   :no-value:

.. autodata:: mimoEnv.envs.mimo_env.DEFAULT_VISION_PARAMS
   :no-value:

.. autodata:: mimoEnv.envs.mimo_env.DEFAULT_VESTIBULAR_PARAMS
   :no-value:
