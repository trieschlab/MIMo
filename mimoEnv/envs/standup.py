""" This module contains a simple reaching experiment in which MIMo tries to stand up.

The scene consists of MIMo and some railings representing a crib. MIMo starts sitting on the ground with his hands
on the railings. The task is to stand up.
MIMos feet and hands are welded to the ground and railings, respectively. He can move all joints in his arms, legs and
torso. His head is fixed.
Sensory input consists of proprioceptive and vestibular inputs, using the default configurations for both.

MIMo initial position is determined by slightly randomizing all joint positions from a standing position and then
letting the simulation settle. This leads to MIMo sagging into a slightly random crouching or sitting position each
episode. All episodes have a fixed length, there are no goal or failure states.

Reward shaping is employed, such that MIMo is penalised for using muscle inputs and large inputs in particular.
Additionally, he is rewarded each step for the current height of his head.

The class with the environment is :class:`~mimoEnv.envs.standup.MIMoStandupEnv` while the path to the scene XML is
defined in :data:`STANDUP_XML`.
"""
import os
import numpy as np
import mujoco

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VESTIBULAR_PARAMS
from mimoActuation.actuation import SpringDamperModel

STANDUP_XML = os.path.join(SCENE_DIRECTORY, "standup_scene.xml")
""" Path to the stand up scene.

:meta hide-value:
"""

SITTING_POSITION = {
    "mimo_location": np.array([-0.103287, 0.00444494, 0.0672742, 0.965518, -0.00942109, -0.207444, 0.157016]),
    "robot:hip_lean1": np.array([-0.0134586]), "robot:hip_rot1": np.array([-0.259285]),
    "robot:hip_bend1": np.array([0.407198]), "robot:hip_lean2": np.array([-0.0565839]),
    "robot:hip_rot2": np.array([-0.248653]), "robot:hip_bend2": np.array([0.38224]),
    "robot:head_swivel": np.array([0]), "robot:head_tilt": np.array([0]), "robot:head_tilt_side": np.array([0]),
    "robot:left_eye_horizontal": np.array([0]), "robot:left_eye_vertical": np.array([0]),
    "robot:left_eye_torsional": np.array([0]), "robot:right_eye_horizontal": np.array([0]),
    "robot:right_eye_vertical": np.array([0]), "robot:right_eye_torsional": np.array([0]),
    "robot:right_shoulder_horizontal": np.array([1.59608]), "robot:right_shoulder_ad_ab": np.array([2.57899]),
    "robot:right_shoulder_rotation": np.array([0.259329]), "robot:right_elbow": np.array([-0.188292]),
    "robot:right_hand1": np.array([-0.429857]), "robot:right_hand2": np.array([-0.99162]),
    "robot:right_hand3": np.array([-0.0568468]), "robot:right_fingers": np.array([-1.4206]),
    "robot:left_shoulder_horizontal": np.array([0.778157]), "robot:left_shoulder_ad_ab": np.array([2.9349]),
    "robot:left_shoulder_rotation": np.array([1.16941]), "robot:left_elbow": np.array([-0.547872]),
    "robot:left_hand1": np.array([-1.54373]), "robot:left_hand2": np.array([-0.98379]),
    "robot:left_hand3": np.array([0.225526]), "robot:left_fingers": np.array([-1.27117]),
    "robot:right_hip1": np.array([-2.26831]), "robot:right_hip2": np.array([-0.295142]),
    "robot:right_hip3": np.array([-0.313409]), "robot:right_knee": np.array([-2.53125]),
    "robot:right_foot1": np.array([-0.109924]), "robot:right_foot2": np.array([-0.0352949]),
    "robot:right_foot3": np.array([0.106372]), "robot:right_toes": np.array([0.0205777]),
    "robot:left_hip1": np.array([-2.118]), "robot:left_hip2": np.array([-0.233242]),
    "robot:left_hip3": np.array([0.369615]), "robot:left_knee": np.array([-2.34683]),
    "robot:left_foot1": np.array([-0.279261]), "robot:left_foot2": np.array([-0.477783]),
    "robot:left_foot3": np.array([0.111583]), "robot:left_toes": np.array([0.0182025]),
}
""" Initial position of MIMo. Specifies initial values for all joints.
We grabbed these values by posing MIMo using the MuJoCo simulate executable and the positional actuator file.
We need these not just for the initial position but also resetting the position each step.

:meta hide-value:
"""


class MIMoStandupEnv(MIMoEnv):
    """ MIMo stands up using crib railings as an aid.

    Attributes and parameters are the same as in the base class, but the default arguments are adapted for the scenario.
    Specifically we have :attr:`.done_active` and :attr:`.goals_in_observation` as ``False`` and touch and vision
    sensors disabled.

    Even though we define a success condition in :meth:`~mimoEnv.envs.standup.MIMoStandupEnv._is_success`, it is
    disabled since :attr:`.done_active` is set to ``False``. The purpose of this is to enable extra information for
    the logging features of stable baselines.

    Attributes:
        init_crouch_position (numpy.ndarray): The initial position.
    """
    def __init__(self,
                 model_path=STANDUP_XML,
                 initial_qpos=SITTING_POSITION,
                 frame_skip=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 actuation_model=SpringDamperModel,
                 ):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         frame_skip=frame_skip,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         goals_in_observation=False,
                         done_active=False)

        self.init_crouch_position = self.data.qpos.copy()

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Computes the reward.

        The reward consists of the current height of MIMos head with a penalty of the square of the control signal.
        Args:
            achieved_goal (float): The achieved head height.
            desired_goal (float): This parameter is ignored.
            info (dict): This parameter is ignored.

        Returns:
            float: The reward as described above.
        """
        quad_ctrl_cost = 0.01 * np.square(self.data.ctrl).sum()
        reward = achieved_goal - 0.2 - quad_ctrl_cost
        return reward

    def is_success(self, achieved_goal, desired_goal):
        """ Did we reach our goal height.

        Args:
            achieved_goal (float): The achieved head height.
            desired_goal (float): This target head height.

        Returns:
            bool: If the achieved head height exceeds the desired height.
        """
        success = (achieved_goal >= desired_goal)
        return success

    def reset_model(self):
        """ Resets the simulation.

        Return the simulation to the XML state, then slightly randomize all joint positions. Afterwards we let the
        simulation settle for a fixed number of steps. This leads to MIMo settling into a slightly random sitting or
        crouching position.

        Returns:
            Dict: Observations after reset.
        """

        self.set_state(self.init_qpos, self.init_qvel)
        qpos = self.init_crouch_position.copy()

        # set initial positions stochastically
        qpos[7:] = qpos[7:] + self.np_random.uniform(low=-0.01, high=0.01, size=len(qpos[7:]))

        # set initial velocities to zero
        qvel = np.zeros(self.data.qvel.shape)

        self.set_state(qpos, qvel)

        # perform 100 steps with no actions to stabilize initial position
        actions = np.zeros(self.action_space.shape)
        self._set_action(actions)
        mujoco.mj_step(self.model, self.data, nstep=100)

        return self._get_obs()

    def is_failure(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns ``False``.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``False``
        """
        return False

    def is_truncated(self):
        """ Dummy function. Always returns ``False``.

        Returns:
            bool: ``False``.
        """
        return False

    def sample_goal(self):
        """ Returns the goal height.

        We use a fixed goal height of 0.5.

        Returns:
            float: 0.5
        """
        return 0.5

    def get_achieved_goal(self):
        """ Get the height of MIMos head.

        Returns:
            float: The height of MIMos head.
        """
        return self.data.body('head').xpos[2]
