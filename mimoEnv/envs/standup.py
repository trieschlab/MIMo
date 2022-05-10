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
Additionally he is rewarded each step for the current height of his head.

The class with the environment is :class:`~mimoEnv.envs.standup.MIMoStandupEnv` while the path to the scene XML is
defined in :data:`STANDUP_XML`.
"""
import os
import numpy as np
import mujoco_py

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VESTIBULAR_PARAMS


STANDUP_XML = os.path.join(SCENE_DIRECTORY, "standup_scene.xml")
""" Path to the stand up scene.

:meta hide-value:
"""


class MIMoStandupEnv(MIMoEnv):
    """ MIMo stands up using crib railings as an aid.

    Attributes and parameters are the same as in the base class, but the default arguments are adapted for the scenario.
    Specifically we have :attr:`.done_active` and :attr:`.goals_in_observation` as `False` and touch and vision sensors
    disabled.

    Even though we define a success condition in :meth:`~mimoEnv.envs.standup.MIMoStandupEnv._is_success`, it is
    disabled since :attr:`.done_active` is set to `False`. The purpose of this is to enable extra information for
    the logging features of stable baselines.

    """
    def __init__(self,
                 model_path=STANDUP_XML,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
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
        quad_ctrl_cost = 0.01 * np.square(self.sim.data.ctrl).sum()
        reward = achieved_goal - 0.2 - quad_ctrl_cost
        return reward

    def _is_success(self, achieved_goal, desired_goal):
        """ Did we reach our goal height.

        Args:
            achieved_goal (float): The achieved head height.
            desired_goal (float): This target head height.

        Returns:
            bool: If the achieved head height exceeds the desired height.
        """
        success = (achieved_goal >= desired_goal)
        return success

    def _reset_sim(self):
        """ Resets the simulation.

        Return the simulation to the XML state, then slightly randomize all joint positions. Afterwards we let the
        simulation settle for a fixed number of steps. This leads to MIMo settling into a slightly random sitting or
        crouching position.

        Returns:
            bool: `True`
        """

        self.sim.set_state(self.initial_state)
        default_state = self.sim.get_state()
        qpos = np.array([-0.103287, 0.00444494, 0.0672742, 0.965518, -0.00942109, -0.207444, 0.157016,
            -0.0134586, -0.259285, 0.407198, -0.0565839, -0.248653, 0.38224, -0.000150182, -8.92769e-05,
            0.000142948, 1.71655e-08, -1.541e-08, -6.07517e-09, -1.66566e-08, -1.48753e-08, -3.9641e-09,
            1.59608, 2.57899, 0.259329, -0.188292, -0.429857, -0.99162, -0.0568468, -1.4206, 0.778157,
            2.9349, 1.16941, -0.547872, -1.54373, -0.98379, 0.225526, -1.27117, -2.26831, -0.295142,
            -0.313409, -2.53125, -0.109924, -0.0352949, 0.106372, 0.0205777, -2.118, -0.233242, 0.369615,
            -2.34683, -0.279261, -0.477783, 0.111583, 0.0182025])

        # set initial positions stochastically
        qpos[6:] = qpos[6:] + self.np_random.uniform(low=-0.01, high=0.01, size=len(qpos[6:]))

        # set initial velocities to zero
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            default_state.time, qpos, qvel, default_state.act, default_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

        # perform 100 steps with no actions to stabilize initial position
        actions = np.zeros(self.action_space.shape)
        self._set_action(actions)
        for _ in range(100):
            self.sim.step()

        return True

    def _is_failure(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns `False`.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: `False`
        """
        return False

    def _sample_goal(self):
        """ Returns the goal height.

        We use a fixed goal height of 0.5.

        Returns:
            float: `0.5`
        """
        return 0.5

    def _get_achieved_goal(self):
        """ Get the height of MIMos head.

        Returns:
            float: The height of MIMos head.
        """
        return self.sim.data.get_body_xpos('head')[2]
