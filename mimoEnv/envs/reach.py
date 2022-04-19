""" This module contains a simple reaching experiment in which MIMo tries to touch a hovering ball.

The scene consists of MIMo and a hovering ball located within reach of MIMos right arm. The task is for MIMo to
touch the ball.
MIMo is fixed in position and can only move his right arm. His head automatically tracks the location of the ball,
i.e. the visual search for the ball is assumed.
Sensory input consists of the full proprioceptive inputs. All other modalities are disabled.

The ball hovers stationary. An episode is completed successfully if MIMo touches the ball, knocking it out of
position. There are no failure states. The position of the ball is slightly randomized each trial.

Reward shaping is employed, with a negative reward based on the distance between MIMos hand and the ball. A large fixed
reward is given when he touches the ball.

The class with the environment is :class:`~mimoEnv.envs.reach.MIMoReachEnv` while the path to the scene XML is defined
in :data:`REACH_XML`.

"""
import os
import numpy as np
import copy
import mujoco_py

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, DEFAULT_PROPRIOCEPTION_PARAMS

#: Path to the reach scene.
REACH_XML = os.path.join(SCENE_DIRECTORY, "reach_scene.xml")


class MIMoReachEnv(MIMoEnv):
    """ MIMo reaches for an object.

    Due to the goal condition we do not use the :attr:`.goal` attribute or the interfaces associated with it. Instead,
    the reward and success conditions are computed directly from the model state, while
    :meth:`~mimoEnv.envs.reach.MIMoReachEnv._sample_goal` and
    :meth:`~mimoEnv.envs.reach.MIMoReachEnv._get_achieved_goal` are dummy functions.
    """
    def __init__(self,
                 model_path=REACH_XML,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=None,
                 goals_in_observation=False,
                 done_active=True):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Computes the reward.

        A negative reward is given based on the distance between MIMos fingers and the ball.
        If contact is made a fixed positive reward of 100 is granted. The achieved and desired goal parameters are
        ignored.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.
            info (dict): This parameter is ignored.

        Returns:
            float: The reward as described above.
        """
        contact = self._is_success(achieved_goal, desired_goal)
        fingers_pos = self.sim.data.get_body_xpos('right_fingers')
        target_pos = self.sim.data.get_body_xpos('target')
        distance = np.linalg.norm(fingers_pos - target_pos)
        reward = - distance + 100 * contact
        return reward

    def _is_success(self, achieved_goal, desired_goal):
        """ Determines the goal states.

        Returns:
            bool: `True` if the ball is knocked out of position.
        """
        target_pos = self.sim.data.get_body_xpos('target')
        success = (np.linalg.norm(target_pos - self.target_init_pos) > 0.01)
        return success

    def _is_failure(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns `False`.

        Returns:
            bool: `False`
        """
        return False

    def _sample_goal(self):
        """ Dummy function. Returns an empty array.

        Returns:
            ndarray: An empty array.
        """
        return np.zeros((0,))

    def _get_achieved_goal(self):
        """ Dummy function. Returns an empty array.

        Returns:
            ndarray: An empty array.
        """
        return np.zeros((0,))

    def _reset_sim(self):
        """ Resets the simulation.

        We reset the simulation and then slightly move both MIMos arm and the ball randomly. The randomization is
        limited such that MIMo can always reach the ball.

        Returns:
            bool: `True`
        """

        self.sim.set_state(self.initial_state)
        self.sim.forward()

        # perform 10 random actions
        for _ in range(10):
            action = self.action_space.sample()
            self._set_action(action)
            self.sim.step()
            self._step_callback()

        # reset target in random initial position and velocities as zero
        qpos = self.sim.data.qpos
        qpos[[-7, -6, -5]] = np.array([
            self.initial_state.qpos[-7] + self.np_random.uniform(low=-0.1, high=0, size=1)[0],
            self.initial_state.qpos[-6] + self.np_random.uniform(low=-0.2, high=0.1, size=1)[0],
            self.initial_state.qpos[-5] + self.np_random.uniform(low=-0.1, high=0, size=1)[0]
        ])
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            self.initial_state.time, qpos, qvel, self.initial_state.act, self.initial_state.udd_state
        )

        self.sim.set_state(new_state)
        self.sim.forward()
        self.target_init_pos = copy.deepcopy(self.sim.data.get_body_xpos('target'))
        return True

    def _step_callback(self):
        """ Adjusts the head and eye positions to track the target.

        Manually computes the joint positions required for the head and eyes to look at the target objects.
        """
        # manually set head and eye positions to look at target
        target_pos = self.sim.data.get_body_xpos('target')
        head_pos = self.sim.data.get_body_xpos('head')
        head_target_dif = target_pos - head_pos
        head_target_dist = np.linalg.norm(head_target_dif)
        head_target_dif[2] = head_target_dif[2] - 0.067375  # extra difference to eyes height in head
        half_eyes_dist = 0.0245  # horizontal distance between eyes / 2
        eyes_target_dist = head_target_dist - 0.07  # remove distance from head center to eyes
        self.sim.data.qpos[13] = np.arctan(head_target_dif[1] / head_target_dif[0])  # head - horizontal
        self.sim.data.qpos[14] = np.arctan(-head_target_dif[2] / head_target_dif[0])  # head - vertical
        self.sim.data.qpos[15] = 0  # head - side tild
        self.sim.data.qpos[16] = np.arctan(-half_eyes_dist / eyes_target_dist)  # left eye -  horizontal
        self.sim.data.qpos[17] = 0  # left eye - vertical
        self.sim.data.qpos[17] = 0  # left eye - torsional
        self.sim.data.qpos[19] = np.arctan(-half_eyes_dist / eyes_target_dist)  # right eye - horizontal
        self.sim.data.qpos[20] = 0  # right eye - vertical
        self.sim.data.qpos[21] = 0  # right eye - torsional
