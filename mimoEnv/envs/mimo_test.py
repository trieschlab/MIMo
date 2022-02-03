import numpy as np

from gym import utils, spaces

from mimoEnv.envs.mimo_env import MIMoEnv


# Dictionary with body_names as keys,
TOUCH_PARAMS = {
    "left_toes": 0.055,
    "left_foot": 0.055,
    "left_lleg": 0.15,
    "left_uleg": 0.15,
    "right_toes": 0.055,
    "right_foot": 0.055,
    "right_lleg": 0.15,
    "right_uleg": 0.15,
}

VISION_PARAMS = {
    "eye_left": {"width": 400, "height": 300},
    "eye_right": {"width": 400, "height": 300}
}


class MIMoEnvDummy(MIMoEnv):

    def __init__(self,
                 initial_qpos={},
                 n_actions=41,  # Currently hardcoded
                 n_substeps=2,
                 touch_params=None,
                 vision_params=None):

        super().__init__(initial_qpos=initial_qpos,
                         n_actions=n_actions,
                         n_substeps=n_substeps,
                         touch_params=touch_params,
                         vision_params=vision_params)

    def _get_obs(self):
        """Returns the observation."""
        # robot proprioception:
        proprio_obs = self._get_proprio_obs()

        # robot touch sensors:
        if self.touch:
            touch_obs = self._get_touch_obs().ravel()

        # robot vision:
        if self.vision:
            vision_obs = self._get_vision_obs().ravel()

        self.vision.save_obs_to_file(directory="imgs", suffix="_" + str(self.steps))
        self.steps += 1
        # Others:
        # TODO

        obs = [proprio_obs]

        # dummy goal for now
        achieved_goal = np.zeros(proprio_obs.shape)
        goal = np.zeros(proprio_obs.shape)

        obs.append(achieved_goal)

        observation = np.concatenate(
               obs
            )
        return {
            "observation": observation.copy(),
            "achieved_goal": achieved_goal.copy(),
            "desired_goal": goal.copy(),
        }

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
        return True

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        # TODO: Actually sample a goal
        return np.zeros(self._get_obs()["observation"].shape)

    def _get_achieved_goal(self):
        """Get the goal state actually achieved in this episode/timeframe."""
        # TODO: All of it
        return np.zeros(self._get_obs()["observation"].shape)

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


class MIMoTestEnv(MIMoEnvDummy, utils.EzPickle):
    def __init__(
        self,
    ):
        utils.EzPickle.__init__(
            self
        )
        MIMoEnvDummy.__init__(
            self,
            touch_params=TOUCH_PARAMS,
            vision_params=VISION_PARAMS
        )
