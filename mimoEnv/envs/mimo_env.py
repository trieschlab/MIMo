import os
import numpy as np
import mujoco_py
import copy

from gym import spaces
from gym.envs.robotics import robot_env

from mimoTouch.touch import DiscreteTouch
from mimoVision.vision import SimpleVision
from mimoVestibular.vestibular import SimpleVestibular
from mimoProprioception.proprio import SimpleProprioception
import mimoEnv.utils as utils

# Ensure we get the path separator correct on windows
MIMO_XML = os.path.abspath(os.path.join(__file__, "..", "..", "assets", "Sample_Scene.xml"))

EMOTES = {
    "default": "tex_head_default",
    "happy": "tex_head_happy",
    "sad": "tex_head_sad",
    "surprised": "tex_head_surprised",
}


class MIMoEnv(robot_env.RobotEnv):

    def __init__(self,
                 model_path=MIMO_XML,
                 initial_qpos={},
                 n_substeps=2,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=None,
                 goals_in_observation=True,
                 done_active=False):

        self.touch_params = touch_params
        self.vision_params = vision_params
        self.vestibular_params = vestibular_params

        self.touch = None
        self.vision = None
        self.vestibular = None

        self.goals_in_observation = goals_in_observation
        self.done_active = done_active

        fullpath = os.path.abspath(model_path)
        if not os.path.exists(fullpath):
            raise IOError("File {} does not exist".format(fullpath))

        model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(model, nsubsteps=n_substeps)
        self.sim.forward()
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": int(np.round(1.0 / self.dt)),
        }

        self.seed()
        self._env_setup(initial_qpos=initial_qpos)
        self.initial_state = copy.deepcopy(self.sim.get_state())

        # Face emotions:
        self.facial_expressions = {}
        for emote in EMOTES:
            tex_name = EMOTES[emote]
            tex_id = utils.texture_name2id(self.sim, tex_name)
            self.facial_expressions[emote] = tex_id
        head_material_name = "head"
        self._head_material_id = utils.material_name2id(self.sim, head_material_name)

        self.goal = self._sample_goal()
        n_actions = len([name for name in self.sim.model.actuator_names if name.startswith("act:")])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
        obs = self._get_obs()
        # Observation spaces
        spaces_dict = {
            "observation": spaces.Box(
                -np.inf, np.inf, shape=obs["observation"].shape, dtype="float32"
            ),
            "desired_goal": spaces.Box(
                -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
            ),
            "achieved_goal": spaces.Box(
                -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32"
            ),
        }
        if self.touch:
            spaces_dict["touch"] = spaces.Box(
                    -np.inf, np.inf, shape=obs["touch"].shape, dtype="float32"
                )
        if self.vision:
            for sensor in self.vision_params:
                spaces_dict[sensor] = spaces.Box(
                        0, 256, shape=obs[sensor].shape, dtype="uint8"
                    )
        if self.vestibular:
            spaces_dict["vestibular"] = spaces.Box(
                    -np.inf, np.inf, shape=obs["vestibular"].shape, dtype="float32"
                )

        self.observation_space = spaces.Dict(spaces_dict)

    def _env_setup(self, initial_qpos):
        # Our init goes here. At this stage the mujoco model is already loaded, but most of the gym attributes, such as
        # observation space and goals are not set yet

        # Always do proprioception
        self.proprioception = SimpleProprioception(self, {})

        # Do setups
        if self.touch_params is not None:
            self._touch_setup(self.touch_params)
        if self.vision_params is not None:
            self._vision_setup(self.vision_params)
        if self.vestibular_params is not None:
            self._vestibular_setup(self.vestibular_params)

        # Do sound setup
        # Do whatever actuation setup
        # Should be able to get all types of sensor outputs here
        # Should be able to produce all control inputs here
        pass

    def _touch_setup(self, touch_params):
        self.touch = DiscreteTouch(self, touch_params=touch_params)

    def _vision_setup(self, vision_params):
        self.vision = SimpleVision(self, vision_params)  # This fixes the GLEW initialization error

    def _vestibular_setup(self, vestibular_params):
        self.vestibular = SimpleVestibular(self, vestibular_params)

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
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
        return self.proprioception.get_proprioception_obs()

    def _get_touch_obs(self):
        touch_obs = self.touch.get_touch_obs()
        return touch_obs

    def _get_vision_obs(self):
        """ Output renders from the camera. Multiple cameras are concatenated along the first axis"""
        vision_obs = self.vision.get_vision_obs()
        return vision_obs

    def _get_vestibular_obs(self):
        vestibular_obs = self.vestibular.get_vestibular_obs()
        return vestibular_obs

    def _get_obs(self):
        """Returns the observation."""
        # robot proprioception:
        proprio_obs = self._get_proprio_obs()

        observation_dict = {
            "observation": proprio_obs,
            "achieved_goal": np.empty(shape=(0,)),
            "desired_goal": np.empty(shape=(0,))
        }

        # robot touch sensors:
        if self.touch:
            touch_obs = self._get_touch_obs().ravel()
            observation_dict["touch"] = touch_obs
        # robot vision:
        if self.vision:
            vision_obs = self._get_vision_obs()
            for sensor in vision_obs:
                observation_dict[sensor] = vision_obs[sensor]

        # vestibular
        if self.vestibular:
            vestibular_obs = self._get_vestibular_obs()
            observation_dict["vestibular"] = vestibular_obs

        if self.goals_in_observation:
            achieved_goal = self._get_achieved_goal()
            observation_dict["achieved_goal"] = achieved_goal.copy()
            observation_dict["desired_goal"] = self.goal.copy()

        return observation_dict

    def _set_action(self, action):
        raise NotImplementedError

    def swap_facial_expression(self, emotion):
        """ Changes MIMos facial texture. Valid emotion names are in self.facial_expression, which links readable
        emotion names to their associated texture ids """
        assert emotion in self.facial_expressions, "{} is not a valid facial expression!".format(emotion)
        new_tex_id = self.facial_expressions[emotion]
        self.sim.model.mat_texid[self._head_material_id] = new_tex_id

    def _is_success(self, achieved_goal, desired_goal):
        """Indicates whether or not the achieved goal successfully achieved the desired goal."""
        raise NotImplementedError

    def _is_failure(self, achieved_goal, desired_goal):
        """Indicates that we reached a failure state."""
        raise NotImplementedError

    def _is_done(self, achieved_goal, desired_goal, info):
        return self.done_active and (info["is_success"] or info["is_failure"])

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        raise NotImplementedError

    def _get_achieved_goal(self):
        raise NotImplementedError

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward. This externalizes the reward function and makes
        it dependent on a desired goal and the one that was achieved. If you wish to include
        additional rewards that are independent of the goal, you can include the necessary values
        to derive it in 'info' and compute it accordingly.

        Args:
            achieved_goal (object): the goal that was achieved during execution
            desired_goal (object): the desired goal that we asked the agent to attempt to achieve
            info (dict): an info dictionary with additional information

        Returns:
            float: The reward that corresponds to the provided achieved goal w.r.t. to the desired
            goal. Note that the following should always hold true:

                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """
        raise NotImplementedError
