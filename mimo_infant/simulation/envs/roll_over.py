"""
This module contains a simple experiment in which MIMo tries to roll over.

MIMo starts either in prone or supine position. This can be adjusted below.
The task is to roll over to the opposite position.

The scene consists only of MIMo. His head is fixed.
Sensory input consists of proprioceptive and vestibular inputs,
using the default configurations for both.

MIMo initial position is determined by slightly randomizing all joint
positions from a standing position and then letting the simulation settle.
This leads to MIMo being in a slightly random prone or supine position each
episode. All episodes have a fixed length, there are no goal or failure states.

Reward shaping is employed, such that MIMo is penalized for using muscle
inputs and large inputs in particular. Additionally, he is rewarded each step
for the current rotation of his hip.

The class with the env is :class:`~mimo_infant.simulation.envs.standup.MIMoRollOverEnv` while
the path to the scene XML is defined in :data:`ROLL_OVER_XML`.
"""

from mimo_infant.simulation.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, \
    DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VESTIBULAR_PARAMS
from mimo_infant.actuation.actuation import SpringDamperModel
import mujoco
import numpy as np
import os

STARTING_POSITION = "supine"
""" Initial position of MIMo. Can be 'prone' or 'supine'.

:meta hide-value:
"""

ROLL_OVER_XML = os.path.join(SCENE_DIRECTORY, "roll_over_scene.xml")
""" Path to the roll over scene.

:meta hide-value:
"""


class MIMoRollOverEnv(MIMoEnv):
    """
    MIMo learns to roll over from prone or supine position.

    Attributes and parameters are the same as in the base class, but the
    default arguments are adapted for the scenario. Specifically we have
    :attr:`.done_active` and :attr:`.goals_in_observation` as ``False`` and
    touch and vision sensors disabled.

    Even though we define a success condition in :meth:
    `~mimo_infant.simulation.envs.standup.MIMoStandupEnv._is_success`, it is disabled since
    :attr:`.done_active` is set to ``False``. The purpose of this is to enable
    extra information for the logging features of stable baselines.

    Attributes:
        init_position (numpy.ndarray): The initial position.
    """

    def __init__(self,
                 model_path=ROLL_OVER_XML,
                 initial_qpos=None,
                 frame_skip=2,
                 age=None,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 actuation_model=SpringDamperModel,
                 **kwargs):

        if STARTING_POSITION not in ["prone", "supine"]:
            msg = f"Unknown starting position '{STARTING_POSITION}'. "
            msg += "Needs to be 'prone' or 'supine'."
            raise ValueError(msg)

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         frame_skip=frame_skip,
                         age=age,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         goals_in_observation=False,
                         done_active=False,
                         **kwargs)

        # Set the floating-base pose
        free_joint_id = self.model.joint("mimo_location").id
        free_qpos_adr = int(self.model.jnt_qposadr[free_joint_id])
        qpos = self.init_qpos.copy()
        qvel = np.zeros_like(self.init_qvel)

        quat = np.array([0.0, -0.7071068, 0.0, 0.7071068], dtype=float)

        if STARTING_POSITION == "supine":
            quat *= np.array([1.0, -1.0, 1.0, 1.0])

        quat = quat / np.linalg.norm(quat)

        qpos[free_qpos_adr : free_qpos_adr + 3] = np.array([0.0, 0.0, 0.1])
        qpos[free_qpos_adr + 3 : free_qpos_adr + 7] = quat

        self.set_state(qpos, qvel)
        mujoco.mj_forward(self.model, self.data)
        qpos = self.data.qpos.copy()

        self.data.ctrl[:] = 0
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)

        self.init_position = self.data.qpos.copy()

    def is_success(self, achieved_goal, desired_goal):
        """ Did we reach our goal rotation.

        Arguments:
            achieved_goal (float): The achieved hip rotation.
            desired_goal (float): This target hip rotation.

        Returns:
            bool: If the achieved hip rotation exceeds the desired rotation.
        """

        success = (achieved_goal >= desired_goal)

        return success

    def is_failure(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns ``False``.

        Arguments:
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

    def reset_model(self):
        """ Resets the simulation.

        Return the simulation to the XML state, then slightly randomize all
        joint positions. Afterwards we let the simulation settle for a fixed
        number of steps. This leads to MIMo settling into a slightly random
        prone or supine position.

        Returns:
            Dict: Observations after reset.
        """

        self.set_state(self.init_qpos, self.init_qvel)
        qpos = self.init_position.copy()

        # Set initial positions stochastically.
        free_joint_id = self.model.joint("mimo_location").id
        free_qpos_adr = int(self.model.jnt_qposadr[free_joint_id])

        random = self.np_random.uniform(
            low=-0.01,
            high=0.01,
            size=len(qpos[free_qpos_adr + 7:]),
        )
        qpos[free_qpos_adr+7:] = qpos[free_qpos_adr+7:] + random

        # Set initial velocities to zero.
        qvel = np.zeros(self.data.qvel.shape)

        self.set_state(qpos, qvel)

        # Perform 100 steps with no actions to stabilize initial position.
        actions = np.zeros(self.action_space.shape)
        self._set_action(actions)
        mujoco.mj_step(self.model, self.data, nstep=100)

        return self._get_obs()

    def sample_goal(self):
        """ Returns the goal rotation.

        We use a fixed goal rotation of 0.8.

        Returns:
            float: 0.8
        """
        return 0.8

    def get_achieved_goal(self):
        """ Get the standardized hip rotation of MIMo.

        Returns:
            float: The standardized hip rotation of MIMo.
        """

        # Get the rotation matrix of the hip.
        xmat = self.data.body("hip").xmat.reshape(3, 3)

        # Calculate the euler angle of the y-axis.
        angle = np.arctan2(
            -xmat[2, 0],
            np.sqrt(xmat[2, 1] ** 2 + xmat[2, 2] ** 2)
        )
        angle *= (180 / np.pi)

        # Normalize the angle to [0, 1].
        angle_norm = (angle - (-90)) / (90 - (-90))

        # Invert the angle depending on the starting position.
        if STARTING_POSITION == "prone":
            angle_norm = 1 - angle_norm

        return angle_norm

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Computes the reward.

        The reward consists of the standardized hip rotation with a
        penalty of the square of the control signal.

        Arguments:
            achieved_goal (float): The achieved hip rotation.
            desired_goal (float): This parameter is ignored.
            info (dict): This parameter is ignored.

        Returns:
            float: The reward as described above.
        """

        # Use the hip rotation as the main reward.
        reward = achieved_goal  # [0, 1]

        # Penalize excessive use of force.
        quad_ctrl_cost = 0.01 * np.square(self.data.ctrl).sum()  # [0, 0.44]
        reward -= quad_ctrl_cost

        return reward
