""" This module defines a dummy implementation for MIMo, to allow easy testing of modules.

The main class is :class:`~mimoEnv.envs.dummy.MIMoDummyEnv` which implements all methods from the base class as dummy
functions that returned fixed values. This allows for testing the model without the full gym bureaucracy.
The second class :class:`~mimoEnv.envs.dummy.MIMoShowroomEnv` is identical to the first, but changes the default
parameters to load the showroom scene instead.

Finally there is a demo class for the v2 version of MIMo using five-fingered hands and feet with two toes each in
:class:`~mimoEnv.envs.dummy.MIMoV2DemoEnv`.
"""

import numpy as np
import os

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, DEFAULT_VISION_PARAMS, DEFAULT_VESTIBULAR_PARAMS, \
    DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_TOUCH_PARAMS, DEFAULT_TOUCH_PARAMS_V2
from mimoEnv.envs.mimo_muscle_env import MIMoMuscleEnv
from mimoTouch.touch import TrimeshTouch
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

TEST_XML = os.path.join(SCENE_DIRECTORY, "muscle_static_test.xml")
""" Path to the benchmarking scene using MIMo v2.

:meta hide-value:
"""


class MIMoDummyEnv(MIMoEnv):
    """ Dummy implementation for :class:`~mimoEnv.envs.mimo_env.MIMoEnv`.

    This class is meant for testing and demonstrating parts of the base class. All abstract methods are implemented as
    dummy functions that return fixed values. No meaningful goal or reward is specified. The default parameters use the
    default sensor configurations in a bare scene consisting of MIMo and two objects on an infinite plane.
    For testing and validation there are two additional parameters compared to the base class.

    - `show_sensors`: If `True`, plot the sensor point distribution for the touch system during initialization.
      Default `False`.
    - `print_space_sizes`: If `True`, the shape of the action space and all entries in the observation dictionary is
      printed during initialization. Default `False`.

    Finally there are two extra attributes:

    Attributes:
        steps: A step counter.
        show_sensors: If `True`, plot the sensor point distribution for the touch system during initialization.
    """
    def __init__(self,
                 model_path=BENCHMARK_XML,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=DEFAULT_TOUCH_PARAMS,
                 vision_params=DEFAULT_VISION_PARAMS,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 goals_in_observation=False,
                 done_active=True,
                 show_sensors=False,
                 print_space_sizes=False,):

        self.steps = 0
        self.show_sensors = show_sensors

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)

        if print_space_sizes:
            print("Observation space:")
            for key in self.observation_space:
                print(key, self.observation_space[key].shape)
            print("\nAction space: ", self.action_space.shape)

    def touch_setup(self, touch_params):
        """ Perform the setup and initialization of the touch system.

        Uses the more complicated Trimesh implementation. Also plots the sensor points if :attr:`.show_sensors` is
        `True`.

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
                print(self.sim.model.body_id2name(body_id), self.touch.sensor_positions[body_id].shape[0])
            count_touch_sensors += self.touch.get_sensor_count(body_id)
        print("Total number of sensor points: ", count_touch_sensors)

        # Plot the sensor points for each body once
        if self.show_sensors:
            for body_id in self.touch.sensor_positions:
                body_name = self.sim.model.body_id2name(body_id)
                env_utils.plot_points(self.touch.sensor_positions[body_id], limit=1., title=body_name)

    def _step_callback(self):
        """ Simply increments the step counter. """
        self.steps += 1

    def _is_success(self, achieved_goal, desired_goal):
        """ Dummy function that always returns `False`.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: `False`
        """
        return False

    def _is_failure(self, achieved_goal, desired_goal):
        """ Dummy function that always returns `False`.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: `False`
        """
        return False

    def _sample_goal(self):
        """ A dummy function returning an empty array of shape (0,).

        Returns:
            numpy.ndarray: An empty size 0 array.
        """
        return np.zeros((0,))

    def _get_achieved_goal(self):
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


class MIMoShowroomEnv(MIMoDummyEnv):
    """ Same as :class:`~mimoEnv.envs.dummy.MIMoDummyEnv`, but with a different scene.

    Unlike :class:`~mimoEnv.envs.dummy.MIMoDummyEnv` this uses the Showroom scene, in which MIMo is located in a
    enclosed room with a number of toy blocks and balls of various sizes and colors. This is also intended as a dummy
    class.
    """
    def __init__(self,
                 model_path=DEMO_XML,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=DEFAULT_TOUCH_PARAMS,
                 vision_params=DEFAULT_VISION_PARAMS,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 goals_in_observation=False,
                 done_active=True,
                 show_sensors=False,
                 print_space_sizes=False,):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active,
                         show_sensors=show_sensors,
                         print_space_sizes=print_space_sizes)


class MIMoV2DemoEnv(MIMoDummyEnv):
    """ Same as :class:`~mimoEnv.envs.dummy.MIMoDummyEnv`, but using the v2 Version of MIMo which has hands with five
    fingers and feet with two toes.
    """
    def __init__(self,
                 model_path=BENCHMARK_XML_V2,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=DEFAULT_TOUCH_PARAMS_V2,
                 vision_params=DEFAULT_VISION_PARAMS,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 goals_in_observation=False,
                 done_active=True,
                 show_sensors=False,
                 print_space_sizes=False,):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active,
                         show_sensors=show_sensors,
                         print_space_sizes=print_space_sizes)


class MIMoMuscleDemoEnv(MIMoMuscleEnv):

    def __init__(self,
                 model_path=BENCHMARK_XML_V2,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=DEFAULT_TOUCH_PARAMS_V2,
                 vision_params=DEFAULT_VISION_PARAMS,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 goals_in_observation=False,
                 done_active=True,
                 print_space_sizes=False, ):

        self.steps = 0

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)

        if print_space_sizes:
            print("Observation space:")
            for key in self.observation_space:
                print(key, self.observation_space[key].shape)
            print("\nAction space: ", self.action_space.shape)

    def _step_callback(self):
        """ Simply increments the step counter. """
        self.steps += 1

    def _is_success(self, achieved_goal, desired_goal):
        """ Dummy function that always returns `False`.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: `False`
        """
        return False

    def _is_failure(self, achieved_goal, desired_goal):
        """ Dummy function that always returns `False`.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: `False`
        """
        return False

    def _sample_goal(self):
        """ A dummy function returning an empty array of shape (0,).

        Returns:
            numpy.ndarray: An empty size 0 array.
        """
        return np.zeros((0,))

    def _get_achieved_goal(self):
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

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass
        #self.viewer.cam.trackbodyid = 0  # id of the body to track
        #self.viewer.cam.distance = 1.5  # how much you "zoom in", smaller is closer
        #self.viewer.cam.lookat[0] = 0  # x,y,z offset from the object (works if trackbodyid=-1)
        #self.viewer.cam.lookat[1] = 0
        #self.viewer.cam.lookat[2] = 0.5  # 0.24 -0.04 .8
        #self.viewer.cam.elevation = -20
        #self.viewer.cam.azimuth = 180
