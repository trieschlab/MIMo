""" This module contains the classes and functions used to determine the muscle parameters for the muscle actuation
model.

These include static scenes in which MIMo is locked into fixed position to determine MIMos maximum voluntary
isometric torque, :class:`~mimoEnv.envs.muscle_test.MIMoStaticMuscleTestEnv` and
:class:`~mimoEnv.envs.muscle_test.MIMoStaticMuscleTestV2Env`, and unlocked scenes where MIMo is floating but can move
his joints, :class:`~mimoEnv.envs.muscle_test.MIMoVelocityMuscleTestEnv` and
:class:`~mimoEnv.envs.muscle_test.MIMoVelocityMuscleTestV2Env`, to determine stable VMAX parameters.
All four of these environments are identical to the dummy environment, they exist for the sake of convenience and to
allow a different default camera position for easy videos.

Finally, there is an environment to demonstrate the adaptive compliance enabled by the muscle actuation model.
"""

import numpy as np
import os
import mujoco_py

from mimoEnv.envs.mimo_env import SCENE_DIRECTORY, MIMoEnv
from mimoEnv.envs.dummy import MIMoDummyEnv
import mimoEnv.utils as env_utils
from mimoActuation.actuation import TorqueMotorModel
from mimoActuation.muscle import MuscleModel

STATIC_TEST_XML = os.path.join(SCENE_DIRECTORY, "muscle_static_test.xml")
""" Path to the static muscle test scene for the mitten version of MIMo.

:meta hide-value:
"""

STATIC_TEST_XML_V2 = os.path.join(SCENE_DIRECTORY, "muscle_static_test_v2.xml")
""" Path to the static muscle test scene for the full hand version of MIMo.

:meta hide-value:
"""

VELOCITY_TEST_XML = os.path.join(SCENE_DIRECTORY, "muscle_velocity_test.xml")
""" Path to the velocity test scene for the mitten version of MIMo.

:meta hide-value:
"""

VELOCITY_TEST_XML_V2 = os.path.join(SCENE_DIRECTORY, "muscle_velocity_test_v2.xml")
""" Path to the velocity test scene for the full hand version of MIMo.

:meta hide-value:
"""

COMPLIANCE_XML = os.path.join(SCENE_DIRECTORY, "compliance_test_scene.xml")
""" Path to the compliance scene.

:meta hide-value:
"""

COMPLIANCE_INIT_POSITION = {"robot:right_shoulder_ad_ab": np.array([1.35]), }
""" Initial arm position for the compliance scene.

:meta hide-value:
"""


class MIMoStaticMuscleTestEnv(MIMoDummyEnv):
    """ Environment for static muscle tests using the mitten hand version MIMo.

    All special aspects of this scenario are defined in the scene XML, so this class consists of dummy functions.
    """
    def __init__(self,
                 model_path=STATIC_TEST_XML,
                 initial_qpos={},
                 n_substeps=2,
                 show_sensors=False,
                 print_space_sizes=False,):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=None,
                         touch_params=None,
                         vision_params=None,
                         vestibular_params=None,
                         actuation_model=MuscleModel,
                         show_sensors=show_sensors,
                         print_space_sizes=print_space_sizes,
                         goals_in_observation=False,
                         done_active=False)

    def _viewer_setup(self):
        super()._viewer_setup()
        self.viewer.cam.azimuth = 135


class MIMoStaticMuscleTestV2Env(MIMoStaticMuscleTestEnv):
    """ Environment for static muscle tests using the full hand version MIMo.
    """
    def __init__(self,
                 model_path=STATIC_TEST_XML_V2,
                 initial_qpos={},
                 n_substeps=2,
                 show_sensors=False,
                 print_space_sizes=False,):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         show_sensors=show_sensors,
                         print_space_sizes=print_space_sizes,)


class MIMoVelocityMuscleTestEnv(MIMoStaticMuscleTestEnv):
    """ Environment for velocity muscle tests using the mitten hand version MIMo.
    """
    def __init__(self,
                 model_path=VELOCITY_TEST_XML,
                 initial_qpos={},
                 n_substeps=2,
                 show_sensors=False,
                 print_space_sizes=False,):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         show_sensors=show_sensors,
                         print_space_sizes=print_space_sizes,)


class MIMoVelocityMuscleTestV2Env(MIMoStaticMuscleTestEnv):
    """ Environment for velocity muscle tests using the full hand version MIMo.
    """
    def __init__(self,
                 model_path=VELOCITY_TEST_XML_V2,
                 initial_qpos={},
                 n_substeps=2,
                 show_sensors=False,
                 print_space_sizes=False,):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         show_sensors=show_sensors,
                         print_space_sizes=print_space_sizes,)


class MIMoComplianceEnv(MIMoEnv):
    """ Test environment for adaptive compliance.

    In this environment we test the compliance behaviour of MIMos actuators by dropping a ball onto his outstretched
    arm and measuring the deflection. This is done by locking all of MIMos joints into set positions, except for the
    shoulder ab-/adduction joint.

    This particular class uses the spring-damper actuation model.
    """
    def __init__(self,
                 model_path=COMPLIANCE_XML,
                 initial_qpos=COMPLIANCE_INIT_POSITION,
                 n_substeps=2,
                 actuation_model=TorqueMotorModel,
                 ):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=None,
                         touch_params=None,
                         vision_params=None,
                         vestibular_params=None,
                         actuation_model=actuation_model,
                         goals_in_observation=False,
                         done_active=False)

        joint_names = [self.sim.model.joint_id2name(joint_id) for joint_id in self.mimo_joints]
        for joint_name in joint_names:
            env_utils.lock_joint(self.sim.model, joint_name)
        env_utils.unlock_joint(self.sim.model, "robot:right_shoulder_ad_ab")
        env_utils.lock_joint(self.sim.model, "robot:right_hand1", joint_angle=-0.25)
        # Let sim settle for a few timesteps to allow weld and locks to settle
        gravity = self.sim.model.opt.gravity[2]
        self.sim.model.opt.gravity[2] = 0
        self.do_simulation(np.zeros(self.action_space.shape), 2)
        self.sim.model.opt.gravity[2] = gravity
        self.init_qpos = self.sim.data.qpos.copy()

    def _sample_goal(self):
        """ Dummy function.

        Returns:
            np.ndarray: A size 0 dummy array.
        """
        return np.zeros((0,))

    def _is_success(self, achieved_goal, desired_goal):
        """ Dummy function.

        Args:
            achieved_goal (object): This argument is ignored.
            desired_goal (object): This argument is ignored.

        Returns:
            bool: Always returns ``False``.
        """
        return False

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Dummy function.

        Args:
            achieved_goal (object): This argument is ignored.
            desired_goal (object): This argument is ignored.
            info (dict): This argument is ignored.

        Returns:
            int: Always returns 0.
        """
        return 0

    def _reset_sim(self):
        """ Reset to the initial position.

        Returns:
            bool: ``True``.
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
        """ Dummy function that always returns ``False``.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``False``.
        """
        return False

    def _get_achieved_goal(self):
        """ Dummy function that returns an empty array.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.zeros(self.goal.shape)

    def _viewer_setup(self):
        """Initial configuration of the viewer. Setup to have a nice view of the ball dropping onto MIMos arm.
        """
        #self.viewer.cam.trackbodyid = 0  # id of the body to track
        self.viewer.cam.distance = 1.5  # how much you "zoom in", smaller is closer
        self.viewer.cam.lookat[0] = 0  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] = -0.3
        self.viewer.cam.lookat[2] = 0.5  # 0.24 -0.04 .8
        self.viewer.cam.elevation = 0
        self.viewer.cam.azimuth = 180


class MIMoComplianceMuscleEnv(MIMoComplianceEnv):
    """ Test environment for adaptive compliance.

    Same as :class:`.MIMoComplianceEnv`, but uses the muscle actuation model by default.
    """
    def __init__(self,
                 model_path=COMPLIANCE_XML,
                 initial_qpos=COMPLIANCE_INIT_POSITION,
                 n_substeps=2,
                 ):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         actuation_model=MuscleModel)
