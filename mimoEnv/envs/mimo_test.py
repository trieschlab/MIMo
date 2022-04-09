import numpy as np

from gym import utils

from mimoEnv.envs.mimo_env import MIMoEnv, MIMO_XML, \
    DEFAULT_VISION_PARAMS, DEFAULT_VESTIBULAR_PARAMS, DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_TOUCH_PARAMS
from touch import TrimeshTouch
import mimoEnv.utils as env_utils


class MIMoEnvDummy(MIMoEnv):

    def __init__(self,
                 model_path=MIMO_XML,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=None,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=None,
                 goals_in_observation=True,
                 done_active=False,
                 show_sensors=False,
                 print_space_sizes=False):

        self.steps = 0
        self.show_sensors = show_sensors

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)

        if print_space_sizes:
            print("Observation space:")
            for key in self.observation_space:
                print(key, self.observation_space[key].shape)
            print("\nAction space: ", self.action_space.shape)

    def _touch_setup(self, touch_params):
        self.touch = TrimeshTouch(self, touch_params=touch_params)

        # Count and print the number of sensor points on each body
        count_touch_sensors = 0
        if self.show_sensors:
            print("Number of sensor points for each body: ")
        for body_id in self.touch.sensor_positions:
            if self.show_sensors:
                print(self.sim.model.body_id2name(body_id), self.touch.sensor_positions[body_id].shape[0])
            count_touch_sensors += self.touch.get_sensor_count(body_id)
        print("Total number of sensor points: ", count_touch_sensors)

        # Plot the sensor points for each body once
        if self.show_sensors:
            for body_id in self.touch.sensor_positions:
                body_name = self.sim.model.body_id2name(body_id)
                env_utils.plot_points(self.touch.sensor_positions[body_id], limit=1., title=body_name)

#    def _get_obs(self):
#        """Returns the observations."""
#        obs = super()._get_obs()
#
#        if self.vision_params:
#            self.vision.save_obs_to_file(directory="imgs", suffix="_" + str(self.steps))
#
#        #for body_name in self.touch_params["scales"]:
#        #    self.touch.plot_force_body(body_name=body_name)
#        #self.touch.plot_force_body(body_name="left_lower_leg")
#        self.touch.plot_force_body_subtree(body_name="left_lower_leg")
#
#        # Toggle through all of the emotions
#        #emotes = sorted(self.facial_expressions.keys())
#        #if self.steps % 20 == 0:
#        #    new_emotion = emotes[(self.steps // 20) % 4]
#        #    self.swap_facial_expression(new_emotion)
#
#        return obs

    def _step_callback(self):
        self.steps += 1

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
        return np.zeros(self.goal.shape)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Computes the reward given the current state (achieved goal) and the desired state (desired_goal). Returns a
        dummy value for this test environment"""
        return 0


class MIMoTestEnv(MIMoEnvDummy):
    def __init__(
        self,
        model_path=MIMO_XML,
        n_substeps=2,
        proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
        touch_params=DEFAULT_TOUCH_PARAMS,
        vision_params=DEFAULT_VISION_PARAMS,
        vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
        goals_in_observation=False,
        done_active=True,
        show_sensors=False,
        print_space_sizes=False,
    ):
        MIMoEnvDummy.__init__(
            self,
            model_path=model_path,
            n_substeps=n_substeps,
            proprio_params=proprio_params,
            touch_params=touch_params,
            vision_params=vision_params,
            vestibular_params=vestibular_params,
            goals_in_observation=goals_in_observation,
            done_active=done_active,
            show_sensors=show_sensors,
            print_space_sizes=print_space_sizes
        )
