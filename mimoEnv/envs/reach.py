import os
import numpy as np
import copy
import mujoco_py

from mimoEnv.envs.mimo_env import MIMoEnv, DEFAULT_PROPRIOCEPTION_PARAMS

REACH_XML = os.path.abspath(os.path.join(__file__, "..", "..", "assets", "reach_scene.xml"))


class MIMoReachEnv(MIMoEnv):

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

    def compute_reward(self, achieved_goal, desired_goal, info):
        contact = self._is_success(achieved_goal, desired_goal)
        fingers_pos = self.sim.data.get_body_xpos('right_fingers')
        target_pos = self.sim.data.get_body_xpos('target')
        distance = np.linalg.norm(fingers_pos - target_pos)
        reward = - distance + 100 * contact
        return reward

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        target_pos = self.sim.data.get_body_xpos('target')
        success = (np.linalg.norm(target_pos - self.target_init_pos) > 0.01)
        return success

    def _is_failure(self, achieved_goal, desired_goal):
        """ Dummy function """
        return False

    def _sample_goal(self):
        """ Dummy function """
        return np.zeros(self._get_proprio_obs().shape)

    def _get_achieved_goal(self):
        """ Dummy function """
        return np.zeros(self._get_proprio_obs().shape)

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
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
