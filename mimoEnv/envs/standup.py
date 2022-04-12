import os
import numpy as np
import mujoco_py

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VESTIBULAR_PARAMS

STANDUP_XML = os.path.join(SCENE_DIRECTORY, "standup_scene.xml")


class MIMoStandupEnv(MIMoEnv):

    def __init__(self,
                 model_path=STANDUP_XML,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 goals_in_observation=False,
                 done_active=False):

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
        head_height = self.sim.data.get_body_xpos('head')[2]
        quad_ctrl_cost = 0.01 * np.square(self.sim.data.ctrl).sum()
        reward = head_height - 0.2 - quad_ctrl_cost
        return reward

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        head_height = self.sim.data.get_body_xpos('head')[2]
        success = (head_height >= 0.5)
        return success

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """

        self.sim.set_state(self.initial_state)
        default_state = self.sim.get_state()
        qpos = self.sim.data.qpos

        # set initial positions stochastically
        qpos[6:] = qpos[6:] + self.np_random.uniform(low=-0.1, high=0.1, size=len(qpos[6:]))

        # set initial velocities to zero
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            default_state.time, qpos, qvel, default_state.act, default_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

        # perform 100 steps with no actions to stabilize initial position
        for _ in range(100):
            self.step(np.zeros(self.action_space.shape))

        return True

    def _is_failure(self, achieved_goal, desired_goal):
        """ Dummy function """
        return False

    def _sample_goal(self):
        """ Dummy function """
        return np.zeros(self._get_proprio_obs().shape)

    def _get_achieved_goal(self):
        """ Dummy function """
        return np.zeros(self._get_proprio_obs().shape)
