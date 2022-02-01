import os
import numpy as np

from gym import utils, spaces
from gym.envs.robotics import robot_env
from gym.envs.robotics.utils import robot_get_obs

from gymTouch.touch import DiscreteTouch, scale_linear
from gymTouch.utils import plot_points

# Ensure we get the path separator correct on windows
MIMO_XML = os.path.abspath(os.path.join(__file__, "..", "..", "assets", "MIMo3.1.xml"))

# Dictionary with geom_names as keys,
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


class MIMoEnv(robot_env.RobotEnv):

    def __init__(self,
                 model_path,
                 initial_qpos={},
                 n_actions=41,
                 n_substeps=20):
        self.touch: DiscreteTouch = None
        super().__init__(
            model_path,
            initial_qpos=initial_qpos,
            n_actions=n_actions,
            n_substeps=n_substeps)
        # super().__init__ called _env_setup, which is where we put our own init
        # Make sure spaces are appropriate:
        # Observation space
        # Action space

    def _env_setup(self, initial_qpos):
        # Our init goes here. At this stage the mujoco model is already loaded, but most of the gym attributes, such as
        # observation space and goals are not set yet

        # Do touch setup
        self._touch_setup()

        # Do proprio setup
        # Do sound setup
        # Do whatever actuation setup
        # Should be able to get all types of sensor outputs here
        # Should be able to produce all control inputs here
        pass

    def _touch_setup(self):
        self.touch = DiscreteTouch(self)
        for body_name in TOUCH_PARAMS:
            body_id = self.sim.model.body_name2id(body_name)
            self.touch.add_body(body_id, scale=TOUCH_PARAMS[body_name])

        # Plot points for every geom, just to check
        #for geom_id in self.touch.sensing_geoms:
        #    plot_points(self.touch.sensor_positions[geom_id],
        #                self.touch.plotting_limits[geom_id],
        #                title="")

        # Get touch obs once to ensure all output arrays are initialized
        self._get_touch_obs()

        # Add sensor animation
        #self.touch.add_plot(body_name="left_foot")

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
        # TODO: Torque about joints?
        return np.concatenate([robot_qpos, robot_qvel])

    def _get_touch_obs(self):
        touch_obs = self.touch.get_touch_obs(DiscreteTouch.get_force_relative, 3, scale_linear)
        return touch_obs

    def _get_obs(self):
        """Returns the observation."""
        # robot proprioception:
        proprio_obs = self._get_proprio_obs()

        # robot touch sensors:
        touch_obs = self._get_touch_obs().ravel()

        # Others:
        # TODO

        obs = [proprio_obs, touch_obs]

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
        # TODO: Visualize touch outputs for body / whole model
        #self.touch.update_plots()
        self.touch.plot_force_body(body_name="left_foot")


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
        )
