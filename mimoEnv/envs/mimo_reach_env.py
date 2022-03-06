import os
import numpy as np
import copy 
from gym import utils
import mujoco_py

from mimoEnv.envs.mimo_env import MIMoEnv


VISION_PARAMS = {
    "eye_left": {"width": 32, "height": 32},
    "eye_right": {"width": 32, "height": 32}
}

VESTIBULAR_PARAMS = {
    "sensors": ["vestibular_acc", "vestibular_gyro"]
}

MIMO_XML = os.path.abspath(os.path.join(__file__, "..", "..", "assets", "reach_scene.xml"))

class MIMoEnvDummy(MIMoEnv):

    def __init__(self,
                 model_path=MIMO_XML,
                 initial_qpos={},
                 n_actions=40,  # Currently hardcoded
                 n_substeps=2,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=None,
                 goals_in_observation=True,
                 done_active=False):

        self.steps = 0

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_actions=n_actions,
                         n_substeps=n_substeps,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)

    def _get_obs(self):
        """Returns the observations."""
        obs = super()._get_obs()

        #if self.vision_params:
        #    self.vision.save_obs_to_file(directory="imgs", suffix="_" + str(self.steps))
        self.steps += 1

        return obs

    def _set_action(self, action):
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        self.sim.data.ctrl[:] = actuation_center + action * actuation_range
        self.sim.data.ctrl[:] = np.clip(
            self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1]
        )

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        # TODO: All of it
        return False

    def _is_failure(self, achieved_goal, desired_goal):
        return False

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        # TODO: Actually sample a goal
        return np.zeros(self._get_proprio_obs().shape)

    def _get_achieved_goal(self):
        """Get the goal state actually achieved in this episode/timeframe."""
        # TODO: All of it
        return np.zeros(self._get_proprio_obs().shape)

    def compute_reward(self, achieved_goal, desired_goal, info):
        # TODO: Actually compute a reward
        return 0

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        pass

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation. Can be used
        to enforce additional constraints on the simulation state.
        """
        pass


class MIMoReachEnv(MIMoEnvDummy, utils.EzPickle):
    def __init__(
        self,
    ):
        utils.EzPickle.__init__(
            self
        )

        MIMoEnvDummy.__init__(
            self,
            model_path=MIMO_XML,
            n_actions=8,
            touch_params=None,
            vision_params=None,
            vestibular_params=None,
            goals_in_observation=False,
            done_active=True,
        )

    def compute_reward(self, achieved_goal, desired_goal, info):
        contact = self._is_success(achieved_goal, desired_goal)
        fingers_pos = self.sim.data.get_body_xpos('right_fingers')
        target_pos = self.sim.data.get_body_xpos('target')
        distance = np.linalg.norm(fingers_pos-target_pos)
        reward = - distance + 100*(contact==True)
        return reward

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        target_pos = self.sim.data.get_body_xpos('target')
        success = (np.linalg.norm(target_pos-self.target_init_pos) > 0.01)
        return success

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
        target_pos_error = True
        while target_pos_error:
            new_target_pos = self.initial_state.qpos[[-7,-6,-5]] + self.np_random.uniform(low=-0.1, high=0.1, size=3)
            right_arm_pos = self.sim.data.get_body_xpos('left_upper_arm')
            target_dist = np.linalg.norm(new_target_pos - right_arm_pos)
            target_pos_error = target_dist > 0.25
        qpos[[-7,-6, -5]] = new_target_pos
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            self.initial_state.time, qpos, qvel, self.initial_state.act, self.initial_state.udd_state
        )

        self.sim.set_state(new_state)
        self.sim.forward()
        self.target_init_pos = copy.deepcopy(self.sim.data.get_body_xpos('target'))
        return True

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()

        # manually set head and eye positions to look at target
        target_pos = self.sim.data.get_body_xpos('target')
        head_pos = self.sim.data.get_body_xpos('head')
        head_target_dif = target_pos - head_pos
        head_target_dist = np.linalg.norm(head_target_dif)
        half_eyes_dist = 0.0245 # horizontal distance between eyes / 2
        eyes_target_dist = head_target_dist - 0.07
        self.sim.data.qpos[13] = np.arctan(head_target_dif[1]/head_target_dif[0]) # head - horizontal
        self.sim.data.qpos[14] = np.arctan(-head_target_dif[2]/head_target_dif[0]) # head - vertical
        self.sim.data.qpos[16] = np.arctan(half_eyes_dist/eyes_target_dist)    # left eye -  horizontal
        self.sim.data.qpos[18] = np.arctan(-half_eyes_dist/eyes_target_dist)    # right eye - horizontal

        obs = self._get_obs()

        achieved_goal = self._get_achieved_goal()

        # Done always false if not done_active, else either of is_success or is_failure must be true
        is_success = self._is_success(achieved_goal, self.goal)
        is_failure = self._is_failure(achieved_goal, self.goal)

        info = {
            "is_success": is_success,
            "is_failure": is_failure,
        }

        if not self.goals_in_observation:
            info["achieved_goal"] = achieved_goal.copy()
            info["desired_goal"] = self.goal.copy()

        done = self._is_done(achieved_goal, self.goal, info)

        reward = self.compute_reward(achieved_goal, self.goal, info)
        return obs, reward, done, info