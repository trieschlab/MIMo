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
import mujoco_py

from mimoEnv.envs.mimo_env import SCENE_DIRECTORY, DEFAULT_VESTIBULAR_PARAMS, DEFAULT_PROPRIOCEPTION_PARAMS, MIMoEnv
from mimoEnv.envs.mimo_muscle_env import MIMoMuscleEnv
from mimoTouch.touch import TrimeshTouch
import mimoEnv.utils as env_utils

STATIC_TEST_XML = os.path.join(SCENE_DIRECTORY, "muscle_static_test.xml")
""" Path to the benchmarking scene using MIMo v2.

:meta hide-value:
"""

STATIC_TEST_XML_V2 = os.path.join(SCENE_DIRECTORY, "muscle_static_test_v2.xml")
""" Path to the benchmarking scene using MIMo v2.

:meta hide-value:
"""

VELOCITY_TEST_XML = os.path.join(SCENE_DIRECTORY, "muscle_velocity_test.xml")
""" Path to the benchmarking scene using MIMo v2.

:meta hide-value:
"""

VELOCITY_TEST_XML_V2 = os.path.join(SCENE_DIRECTORY, "muscle_velocity_test_v2.xml")
""" Path to the benchmarking scene using MIMo v2.

:meta hide-value:
"""

COMPLIANCE_XML = os.path.join(SCENE_DIRECTORY, "compliance_test_scene.xml")
COMPLIANCE_INIT_POSITION = {"robot:right_shoulder_ad_ab": np.array([1.35]), }


class MIMoStaticMuscleTestEnv(MIMoMuscleEnv):
    def __init__(self,
                 model_path=STATIC_TEST_XML,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
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

    def _viewer_setup(self):
        super()._viewer_setup()
        self.viewer.cam.azimuth = 135


class MIMoStaticMuscleTestV2Env(MIMoStaticMuscleTestEnv):
    def __init__(self,
                 model_path=STATIC_TEST_XML_V2,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
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
                         print_space_sizes=print_space_sizes,)


class MIMoVelocityMuscleTestEnv(MIMoStaticMuscleTestEnv):
    def __init__(self,
                 model_path=VELOCITY_TEST_XML,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
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
                         print_space_sizes=print_space_sizes,)


class MIMoVelocityMuscleTestV2Env(MIMoStaticMuscleTestEnv):
    def __init__(self,
                 model_path=VELOCITY_TEST_XML_V2,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
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
                         print_space_sizes=print_space_sizes,)


class MIMoComplianceEnv(MIMoEnv):
    """ Test environment for muscle adaptive compliance.
    """

    def __init__(self,
                 model_path=COMPLIANCE_XML,
                 initial_qpos=COMPLIANCE_INIT_POSITION,
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=None,
                 ):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=False,
                         done_active=False)

        joint_names = [self.sim.model.joint_id2name(joint_id) for joint_id in self.mimo_joints]
        for joint_name in joint_names:
            env_utils.lock_joint(self.sim.model, joint_name)
        env_utils.unlock_joint(self.sim.model, "robot:right_shoulder_ad_ab")
        env_utils.lock_joint(self.sim.model, "robot:right_hand1", joint_angle=-0.25)
        # TODO: Determine two sets of muscle action inputs, one set of motor inputs.
        #   Inputs should keep arm approximately horizontal
        #   Then we drop ball on the arm and plot relevant units over time: qpos, qvel, torque
        # Let sim settle for a few timesteps to allow weld and locks to settle
        gravity = self.sim.model.opt.gravity[2]
        self.sim.model.opt.gravity[2] = 0
        self.do_simulation(np.zeros(self.action_space.shape), 2)
        self.sim.model.opt.gravity[2] = gravity
        self.init_qpos = self.sim.data.qpos.copy()

    def _sample_goal(self):
        """ Dummy function.
        """
        return np.zeros((0,))

    def _is_success(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns False.
        """
        return False

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Dummy function. Always returns 0.
        """
        return 0

    def _reset_sim(self):
        """ Reset to the initial sitting position.

        Returns:
            bool: `True`
        """
        # set qpos as new initial position and velocity as zero
        qpos = self.init_qpos
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            self.initial_state.time, qpos, qvel, self.initial_state.act, self.initial_state.udd_state
        )

        self.sim.set_state(new_state)
        self.sim.forward()

        return True

    def _is_failure(self, achieved_goal, desired_goal):
        """ Dummy function that always returns False.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: `False`
        """
        return False

    def _get_achieved_goal(self):
        """ Dummy function that returns an empty array.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.zeros(self.goal.shape)

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        #self.viewer.cam.trackbodyid = 0  # id of the body to track
        self.viewer.cam.distance = 1.5  # how much you "zoom in", smaller is closer
        self.viewer.cam.lookat[0] = 0  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] = -0.3
        self.viewer.cam.lookat[2] = 0.5  # 0.24 -0.04 .8
        self.viewer.cam.elevation = 0
        self.viewer.cam.azimuth = 180


class MIMoComplianceMuscleEnv(MIMoMuscleEnv):
    """ Test environment for muscle adaptive compliance.
    """
    def __init__(self,
                 model_path=COMPLIANCE_XML,
                 initial_qpos=COMPLIANCE_INIT_POSITION,
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=None,
                 ):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=False,
                         done_active=False)

        joint_names = [self.sim.model.joint_id2name(joint_id) for joint_id in self.mimo_joints]
        for joint_name in joint_names:
            env_utils.lock_joint(self.sim.model, joint_name)
        env_utils.unlock_joint(self.sim.model, "robot:right_shoulder_ad_ab")
        env_utils.lock_joint(self.sim.model, "robot:right_hand1", joint_angle=-0.25)
        # TODO: Determine two sets of muscle action inputs, one set of motor inputs.
        #   Inputs should keep arm approximately horizontal
        #   Then we drop ball on the arm and plot relevant units over time: qpos, qvel, torque
        # Let sim settle for a few timesteps to allow weld and locks to settle
        gravity = self.sim.model.opt.gravity[2]
        self.sim.model.opt.gravity[2] = 0
        self.do_simulation(np.zeros(self.action_space.shape), 2)
        self.sim.model.opt.gravity[2] = gravity
        self.init_qpos = self.sim.data.qpos.copy()

    def _sample_goal(self):
        """ Dummy function.
        """
        return np.zeros((0,))

    def _is_success(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns False.
        """
        return False

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Dummy function. Always returns 0.
        """
        return 0

    def _reset_sim(self):
        """ Reset to the initial sitting position.

        Returns:
            bool: `True`
        """
        # set qpos as new initial position and velocity as zero
        qpos = self.init_qpos
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            self.initial_state.time, qpos, qvel, self.initial_state.act, self.initial_state.udd_state
        )

        self.sim.set_state(new_state)
        self.sim.forward()

        return True

    def _is_failure(self, achieved_goal, desired_goal):
        """ Dummy function that always returns False.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: `False`
        """
        return False

    def _get_achieved_goal(self):
        """ Dummy function that returns an empty array.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.zeros(self.goal.shape)

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        #self.viewer.cam.trackbodyid = 0  # id of the body to track
        self.viewer.cam.distance = 1.5  # how much you "zoom in", smaller is closer
        self.viewer.cam.lookat[0] = 0  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] = -0.3
        self.viewer.cam.lookat[2] = 0.5  # 0.24 -0.04 .8
        self.viewer.cam.elevation = 0
        self.viewer.cam.azimuth = 180