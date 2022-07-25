""" This module defines the base MIMo environment.

The abstract base class is :class:`~mimoEnv.envs.mimo_env.MIMoEnv`. Default parameters for all the sensory modalities
are provided as well.
"""
import os
import numpy as np
import mujoco_py
import copy
import glfw
import sys

from gym import spaces, utils

from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as mimo_utils


# TODO: rework

class MIMoMuscleEnv(MIMoEnv):
    """ This is the abstract base class for all MIMo experiments.

    This class meets the interface requirements for basic gym classes and adds some additional features. The
    observation space is of dictionary type.

    Sensory modules are configured by a parameter dictionary. Default configuration dictionaries are included in the
    same module as this class, :data:`DEFAULT_PROPRIOCEPTION_PARAMS`, :data:`DEFAULT_TOUCH_PARAMS`
    :data:`DEFAULT_VISION_PARAMS`, :data:`DEFAULT_VESTIBULAR_PARAMS`. Passing these to the constructor will enable the
    relevant sensory module.
    Not passing a dictionary disables the relevant module.
    By default all sensory modalities are disabled and the only sensor outputs are the relative joint positions.

    Implementing subclasses will have to override the following functions:

    - :meth:`._is_success`, to determine when an episode completes successfully.
    - :meth:`._is_failure`, to determine when an episode has conclusively failed.
    - :meth:`.compute_reward`, to compute the reward for at each step.
    - :meth:`._sample_goal`, which should determine the desired end state.
    - :meth:`._get_achieved_goal`, which should return the achieved end state.

    Depending on the requirements of your experiment any of these functions may be implemented as dummy functions
    returning fixed values.
    Additional functions that may be overridden optionally are:

    - :meth:`._is_done`, which determines the `done` return value after each step.
    - :meth:`._proprio_setup`, :meth:`._touch_setup`, :meth:`._vision_setup`, :meth:`._vestibular_setup`, these
      functions initialize the associated sensor modality. These should be overridden if you want to replace the default
      implementation. Default implementations are :class:`~mimoProprioception.proprio.SimpleProprioception`,
      :class:`~mimoTouch.touch.DiscreteTouch`, :class:`~mimoVision.vision.SimpleVision`,
      :class:`~mimoVestibular.vestibular.SimpleVestibular`.
    - :meth:`._get_proprio_obs`, :meth:`._get_touch_obs`, :meth:`._get_vision_obs`, :meth:`._get_vestibular_obs`, these
      functions collect the observations of the associated sensor modality. These allow you to do post processing on
      the output without having to alter the base implementations.
    - :meth:`._reset_sim`, which resets the physical simulation. If you have special conditions on the initial position
      this function should implement/ensure them.
    - :meth:`._step_callback`, which is called every step after the simulation step but before the observations are
      collected.

    These functions come with default implementations that should handle most scenarios.

    The constructor takes the following arguments:

    - `model_path`: The path to the scene xml. Required.
    - `initial_qpos`: A dictionary of the initial joint positions. Keys are the joint names. Only required if the
      initial position varies from that defined the XMLs.
    - `n_substeps`: The number of physics substeps for each simulation step. The duration of each physics step is set
      in the scene XML.
    - `proprio_params`: The configuration dictionary for the proprioceptive system. If `None` the module is disabled.
      Default `None`.
    - `touch_params`: The configuration dictionary for the touch system. If `None` the module is disabled.
      Default `None`.
    - `vision_params`: The configuration dictionary for the vision system. If `None` the module is disabled.
      Default `None`.
    - `vestibular_params`: The configuration dictionary for the vestibular system. If `None` the module is disabled.
      Default `None`.
    - `goals_in_observation`: If `True` the desired and achieved goals are included in the observation dictionary.
      Default `True`.
    - `done_active`: If `True`, :meth:`._is_done` returns `True` if the simulation reaches a success or failure state.
      If `False`, :meth:`._is_done` always returns `False` and the function calling :meth:`.step` has to figure out when
      to stop or reset the simulation on its own.

    Attributes:
        sim: The MuJoCo simulation object.
        initial_state: A copy of the simulation state at the start of the simulation.
        goal: The desired goal.
        action_space: The action space. See Gym documentation for more.
        observation_space: The observation space. See Gym documentation for more.
        proprio_params: The configuration dictionary for the proprioceptive system.
        proprioception: A reference to the proprioception instance.
        touch_params: The configuration dictionary for the touch system.
        touch: A reference to the touch instance.
        vision_params: The configuration dictionary for the vision system.
        vision: A reference to the vision instance.
        vestibular_params: The configuration dictionary for the vestibular system.
        vestibular: A reference to the vestibular instance.
        facial_expressions: A dictionary linking emotions with their associated facial textures. The keys of this
            dictionary are valid inputs for :meth:`.swap_facial_expression`
        goals_in_observation: If `True` the desired and achieved goals are included in the observation dictionary.
            Default `True`.
        done_active: If `True`, :meth:`._is_done` returns `True` if the simulation reaches a success or failure state.
            If `False`, :meth:`._is_done` always returns `False` and the function calling :meth:`.step` has to figure
            out when to stop or reset the simulation on its own.

    """

    def __init__(self,
                 model_path,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=None,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=None,
                 goals_in_observation=True,
                 done_active=False):

        super().__init__(model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)

    # TODO: Override all the relevant functions
