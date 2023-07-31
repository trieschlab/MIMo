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

from mimoEnv.envs.mimo_env import SCENE_DIRECTORY, MIMoEnv
from mimoEnv.envs.dummy import MIMoDummyEnv
import mimoEnv.utils as env_utils
from mimoActuation.actuation import SpringDamperModel
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

COMPLIANCE_CAMERA_CONFIG={
    "trackbodyid": 0,
    "distance": 1.5,
    "lookat": np.asarray([0, -0.3, 0.5]),
    "elevation": 0,
    "azimuth": 180,
}
""" Camera configuration looking straight at MIMo such that the shoulder position is clearly visible.

:meta hide-value:
"""

class MIMoStaticMuscleTestEnv(MIMoDummyEnv):
    """ Environment for static muscle tests using the mitten hand version MIMo.

    All special aspects of this scenario are defined in the scene XML, so this class consists of dummy functions.
    """
    def __init__(self,
                 model_path=STATIC_TEST_XML,
                 default_camera_config={"azimuth": 135,},
                 **kwargs
                 ):

        super().__init__(model_path=model_path,
                         proprio_params=None,
                         touch_params=None,
                         vision_params=None,
                         vestibular_params=None,
                         actuation_model=MuscleModel,
                         goals_in_observation=False,
                         done_active=False,
                         default_camera_config=default_camera_config,
                         **kwargs)

class MIMoStaticMuscleTestV2Env(MIMoStaticMuscleTestEnv):
    """ Environment for static muscle tests using the full hand version MIMo.
    """
    def __init__(self,
                 model_path=STATIC_TEST_XML_V2,
                 **kwargs):

        super().__init__(model_path=model_path,
                         **kwargs)


class MIMoVelocityMuscleTestEnv(MIMoStaticMuscleTestEnv):
    """ Environment for velocity muscle tests using the mitten hand version MIMo.
    """
    def __init__(self,
                 model_path=VELOCITY_TEST_XML,
                 **kwargs):

        super().__init__(model_path=model_path,
                         **kwargs)


class MIMoVelocityMuscleTestV2Env(MIMoStaticMuscleTestEnv):
    """ Environment for velocity muscle tests using the full hand version MIMo.
    """
    def __init__(self,
                 model_path=VELOCITY_TEST_XML_V2,
                 **kwargs):

        super().__init__(model_path=model_path,
                         **kwargs)


class MIMoComplianceEnv(MIMoDummyEnv):
    """ Test environment for adaptive compliance.

    In this environment we test the compliance behaviour of MIMos actuators by dropping a ball onto his outstretched
    arm and measuring the deflection. This is done by locking all of MIMos joints into set positions, except for the
    shoulder ab-/adduction joint.

    This particular class uses the spring-damper actuation model.
    """
    def __init__(self,
                 model_path=COMPLIANCE_XML,
                 initial_qpos=COMPLIANCE_INIT_POSITION,
                 actuation_model=SpringDamperModel,
                 default_camera_config=COMPLIANCE_CAMERA_CONFIG,
                 **kwargs
                 ):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         proprio_params=None,
                         touch_params=None,
                         vision_params=None,
                         vestibular_params=None,
                         actuation_model=actuation_model,
                         default_camera_config=default_camera_config,
                         **kwargs)

        joint_names = [self.model.joint(joint_id).name for joint_id in self.mimo_joints]
        for joint_name in joint_names:
            env_utils.lock_joint(self.model, joint_name)
        env_utils.unlock_joint(self.model, "robot:right_shoulder_ad_ab")
        env_utils.lock_joint(self.model, "robot:right_hand1", joint_angle=-0.25)
        # Let sim settle for a few timesteps to allow weld and locks to settle
        gravity = self.model.opt.gravity[2]
        self.model.opt.gravity[2] = 0
        self.do_simulation(np.zeros(self.action_space.shape), 2)
        self.model.opt.gravity[2] = gravity
        self.init_qpos = self.data.qpos.copy()

class MIMoComplianceMuscleEnv(MIMoComplianceEnv):
    """ Test environment for adaptive compliance.

    Same as :class:`.MIMoComplianceEnv`, but uses the muscle actuation model by default.
    """
    def __init__(self,
                 actuation_model=MuscleModel,
                 **kwargs,
                 ):

        super().__init__(actuation_model=actuation_model,
                         **kwargs,)
