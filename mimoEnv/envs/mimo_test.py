import numpy as np

from gym import utils

from mimoEnv.envs.mimo_env import MIMoEnv, MIMO_XML
from mimoTouch.touch_trimesh import TrimeshTouch
import mimoEnv.utils as env_utils


# Dictionary with body_names as keys,
TOUCH_PARAMS = {
    "scales": {
        "left_toes": 0.010,
        "right_toes": 0.010,
        "left_foot": 0.015,
        "right_foot": 0.015,
        "left_lower_leg": 0.038,
        "right_lower_leg": 0.038,
        "left_upper_leg": 0.027,
        "right_upper_leg": 0.027,
        "hip": 0.025,
        "lower_body": 0.025,
        "upper_body": 0.030,
        "head": 0.013,
        "left_eye": 1.0,
        "right_eye": 1.0,
        "left_upper_arm": 0.024,
        "right_upper_arm": 0.024,
        "left_lower_arm": 0.024,
        "right_lower_arm": 0.024,
        "left_hand": 0.007,
        "right_hand": 0.007,
        "left_fingers": 0.002,
        "right_fingers": 0.002,
    },
    "touch_function": "force_vector",
    "adjustment_function": "spread_linear",
}

VISION_PARAMS = {
    "eye_left": {"width": 400, "height": 300},
    "eye_right": {"width": 400, "height": 300}
}

VESTIBULAR_PARAMS = {
    "sensors": ["vestibular_acc", "vestibular_gyro"]
}


class MIMoEnvDummy(MIMoEnv):

    def __init__(self,
                 model_path=MIMO_XML,
                 initial_qpos={},
                 n_substeps=2,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=None,
                 goals_in_observation=True,
                 done_active=False):

        self.steps = 0

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)

    def _touch_setup(self, touch_params):
        self.touch = TrimeshTouch(self, touch_params=touch_params)

        # Count and print the number of sensor points on each body
        count_touch_sensors = 0
        print("Number of sensor points for each body: ")
        for body_id in self.touch.sensor_positions:
            print(self.sim.model.body_id2name(body_id), self.touch.sensor_positions[body_id].shape[0])
            count_touch_sensors += self.touch.sensor_positions[body_id].shape[0]
        print("Total number of sensor points: ", count_touch_sensors)

        # Plot the sensor points for each body once
        #for body_id in self.touch.sensor_positions:
        #    body_name = self.sim.model.body_id2name(body_id)
        #    env_utils.plot_points(self.touch.sensor_positions[body_id], limit=1., title=body_name)

    def _get_obs(self):
        """Returns the observations."""
        obs = super()._get_obs()

        if self.vision_params:
            self.vision.save_obs_to_file(directory="imgs", suffix="_" + str(self.steps))

        #for body_name in self.touch_params["scales"]:
        #    self.touch.plot_force_body(body_name=body_name)
        #self.touch.plot_force_body(body_name="left_lower_leg")
        #self.touch.plot_force_body_subtree(body_name="left_lower_leg")

        # Toggle through all of the emotions
        emotes = sorted(self.facial_expressions.keys())
        if self.steps % 20 == 0:
            new_emotion = emotes[(self.steps // 20) % 4]
            self.swap_facial_expression(new_emotion)

        return obs

    def _step_callback(self):
        self.steps += 1

    def _set_action(self, action):
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        self.sim.data.ctrl[:] = actuation_center + action * actuation_range
        self.sim.data.ctrl[:] = np.clip(
            self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1]
        )

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal. Since this class is just
        a demo environment to test sensor modalities, we do not care about this! """
        return False

    def _is_failure(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal is a failure state. Since this class is just a demo environment
        to test sensor modalities, we do not care about this! """
        return False

    def _sample_goal(self):
        """Samples a new goal and returns it. Again just a dummy return."""
        return np.zeros(self._get_proprio_obs().shape)

    def _get_achieved_goal(self):
        """Get the goal state actually achieved in this episode/timeframe. Again just a dummy return."""
        return np.zeros(self._get_proprio_obs().shape)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Computes the reward given the current state (achieved goal) and the desired state (desired_goal). Returns a
        dummy value for this test environment"""
        return 0


class MIMoTestEnv(MIMoEnvDummy, utils.EzPickle):
    def __init__(
        self,
    ):
        utils.EzPickle.__init__(
            self,
            model_path=MIMO_XML,
            touch_params=TOUCH_PARAMS,
            vision_params=VISION_PARAMS,
            vestibular_params=VESTIBULAR_PARAMS,
            goals_in_observation=False,
            done_active=True
        )
        MIMoEnvDummy.__init__(
            self,
            model_path=MIMO_XML,
            touch_params=TOUCH_PARAMS,
            vision_params=VISION_PARAMS,
            vestibular_params=VESTIBULAR_PARAMS,
            goals_in_observation=False,
            done_active=True
        )
