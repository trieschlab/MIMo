import os
import numpy as np

from gym import utils, spaces
from gym.envs.robotics import robot_env
from gym.envs.robotics.utils import robot_get_obs

from gymTouch.touch import DiscreteTouch, scale_linear
from gymTouch.utils import plot_points

from mimoVision.vision import SimpleVision, Vision


# Ensure we get the path separator correct on windows
MIMO_XML = os.path.abspath(os.path.join(__file__, "..", "..", "assets", "MIMo3.1.xml"))

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


class MIMoEnv(robot_env.RobotEnv):

    def __init__(self,
                 model_path,
                 initial_qpos={},
                 n_actions=41,  # Currently hardcoded
                 n_substeps=5,
                 touch_params=None,
                 vision_params=None):

        self.touch_params = touch_params
        self.vision_params = vision_params

        if self.touch_params is not None:
            self.touch: DiscreteTouch = None

        if self.vision_params is not None:
            self.vision: Vision = None

        super().__init__(
            model_path,
            initial_qpos=initial_qpos,
            n_actions=n_actions,
            n_substeps=n_substeps)
        # super().__init__ calls _env_setup, which is where we put our own init
        # TODO: Make sure spaces are appropriate:
        # Observation space: Vision should probably be treated differently from proprioception
        # Action space: Box with n_actions dims from -1 to +1

    def _env_setup(self, initial_qpos):
        # Our init goes here. At this stage the mujoco model is already loaded, but most of the gym attributes, such as
        # observation space and goals are not set yet

        # Do setups
        if self.touch_params is not None:
            self._touch_setup(self.touch_params)
        if self.vision_params is not None:
            self._vision_setup(self.vision_params)

        # Do proprio setup
        # Do sound setup
        # Do whatever actuation setup
        # Should be able to get all types of sensor outputs here
        # Should be able to produce all control inputs here
        pass

    def _touch_setup(self, touch_params):
        self.touch = DiscreteTouch(self)
        for body_name in touch_params:
            body_id = self.sim.model.body_name2id(body_name)
            self.touch.add_body(body_id, scale=touch_params[body_name])

        # Get touch obs once to ensure all output arrays are initialized
        self._get_touch_obs()

    def _vision_setup(self, vision_params):
        self.vision = SimpleVision(self, vision_params)  # This fixes the GLEW initialization error

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_proprio_obs(self):
        # Naive implementation: Joint positions and velocities
        robot_qpos, robot_qvel = robot_get_obs(self.sim)
        # Torque/forces will require sensors in mujoco
        return np.concatenate([robot_qpos, robot_qvel])

    def _get_touch_obs(self):
        touch_obs = self.touch.get_touch_obs(DiscreteTouch.get_force_relative, 3, scale_linear)
        return touch_obs

    def _get_vision_obs(self):
        vision_obs = self.vision.get_vision_obs()
        return np.concatenate([vision_obs[cam] for cam in vision_obs])

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

        # Others:
        # TODO

        obs = [proprio_obs]

        # dummy goal
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

    def compute_reward(self, achieved_goal, desired_goal, info):
        # TODO: Actually compute a reward
        return 0

    def _render_callback(self):
        """A custom callback that is called before rendering. Can be used
        to implement custom visualizations.
        """
        # self.touch.plot_force_body(body_name="left_foot")
        pass


class MIMoTestEnv(MIMoEnv, utils.EzPickle):
    def __init__(
        self,
    ):
        utils.EzPickle.__init__(
            self
        )
        MIMoEnv.__init__(
            self,
            model_path=MIMO_XML,
            touch_params=TOUCH_PARAMS,
            vision_params=VISION_PARAMS
        )
