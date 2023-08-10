""" This module defines a dummy implementation for MIMo, to allow easy testing of modules.

The main class is :class:`~mimoEnv.envs.dummy.MIMoDummyEnv` which implements all methods from the base class as dummy
functions that returned fixed values. This allows for testing the model without the full gym bureaucracy.
The second class :class:`~mimoEnv.envs.dummy.MIMoShowroomEnv` is identical to the first, but changes the default
parameters to load the showroom scene instead.

Finally, there is a demo class for the v2 version of MIMo using five-fingered hands and feet with two toes each in
:class:`~mimoEnv.envs.dummy.MIMoV2DummyEnv`.
"""

import os
import numpy as np
import mujoco

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, DEFAULT_VISION_PARAMS, DEFAULT_VESTIBULAR_PARAMS, \
    DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_TOUCH_PARAMS, DEFAULT_TOUCH_PARAMS_V2
from mimoTouch.touch import TrimeshTouch
from mimoActuation.muscle import MuscleModel
from mimoActuation.actuation import SpringDamperModel
import mimoEnv.utils as env_utils


DEMO_XML = os.path.join(SCENE_DIRECTORY, "showroom.xml")
""" Path to the demo scene.

:meta hide-value:
"""

BENCHMARK_XML = os.path.join(SCENE_DIRECTORY, "benchmark_scene.xml")
""" Path to the benchmarking scene.

:meta hide-value:
"""

BENCHMARK_XML_V2 = os.path.join(SCENE_DIRECTORY, "benchmarkv2_scene.xml")
""" Path to the benchmarking scene using MIMo v2.

:meta hide-value:
"""

TEST_XML = os.path.join(SCENE_DIRECTORY, "test_scene.xml")
""" Path to the benchmarking scene using MIMo v2.

:meta hide-value:
"""


class MIMoDummyEnv(MIMoEnv):
    """ Dummy implementation for :class:`~mimoEnv.envs.mimo_env.MIMoEnv`.

    This class is meant for testing and demonstrating parts of the base class. All abstract methods are implemented as
    dummy functions that return fixed values. No meaningful goal or reward is specified. The default parameters use the
    default sensor configurations in a bare scene consisting of MIMo and two objects on an infinite plane.
    For testing and validation there are two additional parameters compared to the base class:

    Args:
        show_sensors: If ``True``, plot the sensor point distribution for the touch system during initialization.
            Default ``False``.
        print_space_sizes: If ``True``, the shape of the action space and all entries in the observation dictionary is
            printed during initialization. Default ``False``.

    Finally, there are two extra attributes:

    Attributes:
        steps (int): A step counter.
        show_sensors (bool): If ``True``, plot the sensor point distribution for the touch system during initialization.
    """
    def __init__(self,
                 model_path=BENCHMARK_XML,
                 frame_skip=2,
                 initial_qpos=None,
                 render_mode=None,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=DEFAULT_TOUCH_PARAMS,
                 vision_params=DEFAULT_VISION_PARAMS,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 actuation_model=SpringDamperModel,
                 goals_in_observation=False,
                 done_active=True,
                 show_sensors=False,
                 print_space_sizes=False,
                 **kwargs):

        self.steps = 0
        self.show_sensors = show_sensors

        super().__init__(model_path=model_path,
                         frame_skip=frame_skip,
                         initial_qpos=initial_qpos,
                         render_mode=render_mode,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active,
                         **kwargs)

        if print_space_sizes:
            print("Observation space:")
            for key in self.observation_space:
                print(key, self.observation_space[key].shape)
            print("\nAction space: ", self.action_space.shape)

    def touch_setup(self, touch_params):
        """ Perform the setup and initialization of the touch system.

        Uses the more complicated Trimesh implementation. Also plots the sensor points if :attr:`.show_sensors` is
        ``True``.

        Args:
            touch_params (dict): The parameter dictionary.
        """
        self.touch = TrimeshTouch(self, touch_params=touch_params)

        # Count and print the number of sensor points on each body
        count_touch_sensors = 0
        if self.show_sensors:
            print("Number of sensor points for each body: ")
        for body_id in self.touch.sensor_positions:
            if self.show_sensors:
                print(self.model.body(body_id).name, self.touch.sensor_positions[body_id].shape[0])
            count_touch_sensors += self.touch.get_sensor_count(body_id)
        print("Total number of sensor points: ", count_touch_sensors)

        # Plot the sensor points for each body once
        if self.show_sensors:
            for body_id in self.touch.sensor_positions:
                body_name = self.model.body(body_id).name
                env_utils.plot_points(self.touch.sensor_positions[body_id], limit=1., title=body_name)

    def _obs_callback(self):
        """ Simply increments the step counter. """
        self.steps += 1

    def reset_model(self):
        """ Resets to the initial simulation state"""
        self.set_state(self.init_qpos, self.init_qvel)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def is_success(self, achieved_goal, desired_goal):
        """ Dummy function that always returns ``False``.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``False``.
        """
        return False

    def is_failure(self, achieved_goal, desired_goal):
        """ Dummy function that always returns ``False``.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``False``.
        """
        return False

    def is_truncated(self):
        """ Dummy function. Always returns ``False``.

        Returns:
            bool: ``False``.
        """
        return False

    def sample_goal(self):
        """ A dummy function returning an empty array of shape (0,).

        Returns:
            numpy.ndarray: An empty size 0 array.
        """
        return np.zeros((0,))

    def get_achieved_goal(self):
        """Dummy function returning an empty array with the same shape as the goal.

        Returns:
            numpy.ndarray: An empty size 0 array.
        """
        return np.zeros(self.goal.shape)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Dummy function that always returns a dummy value of 0.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.
            info (dict): This parameter is ignored.

        Returns:
            float: 0
        """
        return 0


class MIMoV2DummyEnv(MIMoDummyEnv):
    """ Same as :class:`~mimoEnv.envs.dummy.MIMoDummyEnv`, but using the full hand version of MIMo which has hands with
    five fingers and feet with two toes.
    """
    def __init__(self,
                 model_path=BENCHMARK_XML_V2,
                 touch_params=DEFAULT_TOUCH_PARAMS_V2,
                 **kwargs):

        super().__init__(model_path=model_path,
                         touch_params=touch_params,
                         **kwargs)


class MIMoMuscleDummyEnv(MIMoDummyEnv):
    """Same as :class:`~mimoEnv.envs.dummy.MIMoDummyEnv`, but using the muscle actuation model. Uses the full hand
    version of MIMo by default.
    """
    def __init__(self,
                 model_path=BENCHMARK_XML_V2,
                 touch_params=DEFAULT_TOUCH_PARAMS_V2,
                 actuation_model=MuscleModel,
                 **kwargs):

        super().__init__(model_path=model_path,
                         touch_params=touch_params,
                         actuation_model=actuation_model,
                         **kwargs)
