""" This module defines the base MIMo environment.

The abstract base class is :class:`~mimoEnv.envs.mimo_env.MIMoEnv`. Default parameters for all the sensory modalities
are provided as well.
"""
import os
import sys
import glfw
import numpy as np
import mujoco_py
import copy

from gym import spaces, utils
from gym.envs.robotics import robot_env

from mimoTouch.touch import DiscreteTouch
from mimoVision.vision import SimpleVision
from mimoVestibular.vestibular import SimpleVestibular
from mimoProprioception.proprio import SimpleProprioception
import mimoEnv.utils as mimo_utils


SCENE_DIRECTORY = os.path.abspath(os.path.join(__file__, "..", "..", "assets"))
""" Path to the scene directory.

:meta hide-value:
"""


EMOTES = {
    "default": "tex_head_default",
    "happy": "tex_head_happy",
    "sad": "tex_head_sad",
    "surprised": "tex_head_surprised",
    "angry": "tex_head_angry",
    "disgusted": "tex_head_disgusted",
    "scared": "tex_head_scared",
}
""" Valid facial expressions.

:meta hide-value:
"""


DEFAULT_TOUCH_PARAMS = {
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
    "response_function": "spread_linear",
}
""" Default touch parameters.

:meta hide-value:
"""


DEFAULT_TOUCH_PARAMS_V2 = {
    "scales": {
        "left_big_toe": 0.010,
        "right_big_toe": 0.010,
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
        "left_ffdistal": 0.002,
        "left_mfdistal": 0.002,
        "left_rfdistal": 0.002,
        "left_lfdistal": 0.002,
        "left_thdistal": 0.002,
        "left_ffmiddle": 0.004,
        "left_mfmiddle": 0.004,
        "left_rfmiddle": 0.004,
        "left_lfmiddle": 0.004,
        "left_thhub": 0.004,
        "left_ffknuckle": 0.004,
        "left_mfknuckle": 0.004,
        "left_rfknuckle": 0.004,
        "left_lfknuckle": 0.004,
        "left_thbase": 0.004,
        "left_lfmetacarpal": 0.007,
        "right_ffdistal": 0.002,
        "right_mfdistal": 0.002,
        "right_rfdistal": 0.002,
        "right_lfdistal": 0.002,
        "right_thdistal": 0.002,
        "right_ffmiddle": 0.004,
        "right_mfmiddle": 0.004,
        "right_rfmiddle": 0.004,
        "right_lfmiddle": 0.004,
        "right_thhub": 0.004,
        "right_ffknuckle": 0.004,
        "right_mfknuckle": 0.004,
        "right_rfknuckle": 0.004,
        "right_lfknuckle": 0.004,
        "right_thbase": 0.004,
        "right_lfmetacarpal": 0.007,
    },
    "touch_function": "force_vector",
    "response_function": "spread_linear",
}
""" Default touch parameters for the v2 version of MIMo with five fingers and two toes.

:meta hide-value:
"""


DEFAULT_VISION_PARAMS = {
    "eye_left": {"width": 256, "height": 256},
    "eye_right": {"width": 256, "height": 256},
}
""" Default vision parameters.

:meta hide-value:
"""


DEFAULT_VESTIBULAR_PARAMS = {
    "sensors": ["vestibular_acc", "vestibular_gyro"],
}
""" Default vestibular parameters.

:meta hide-value:
"""


DEFAULT_PROPRIOCEPTION_PARAMS = {
    "components": ["velocity", "torque", "limits"],
    "threshold": .035,
}
""" Default parameters for proprioception. Relative joint positions are always included.

:meta hide-value:
"""

DEFAULT_SIZE = 500
""" Default window size for gym rendering functions.
"""


class MIMoEnv(robot_env.RobotEnv, utils.EzPickle):
    """ This is the abstract base class for all MIMo experiments.

    This class meets the interface requirements for basic gym classes and adds some additional features. The
    observation space is of dictionary type.

    Sensory modules are configured by a parameter dictionary. Default configuration dictionaries are included in the
    same module as this class, :data:`DEFAULT_PROPRIOCEPTION_PARAMS`, :data:`DEFAULT_TOUCH_PARAMS`
    :data:`DEFAULT_VISION_PARAMS`, :data:`DEFAULT_VESTIBULAR_PARAMS`. Passing these to the constructor will enable the
    relevant sensory module.
    Not passing a dictionary disables the relevant module.
    By default all sensory modalities are disabled and the only sensor outputs are the relative joint positions.

    Implementing subclasses will have to override the following functions:

    - :meth:`._is_success`, to determine when an episode completes successfully.
    - :meth:`._is_failure`, to determine when an episode has conclusively failed.
    - :meth:`.compute_reward`, to compute the reward for at each step.
    - :meth:`._sample_goal`, which should determine the desired end state.
    - :meth:`._get_achieved_goal`, which should return the achieved end state.

    Depending on the requirements of your experiment any of these functions may be implemented as dummy functions
    returning fixed values.
    Additional functions that may be overridden optionally are:

    - :meth:`._is_done`, which determines the `done` return value after each step.
    - :meth:`._proprio_setup`, :meth:`._touch_setup`, :meth:`._vision_setup`, :meth:`._vestibular_setup`, these
      functions initialize the associated sensor modality. These should be overridden if you want to replace the default
      implementation. Default implementations are :class:`~mimoProprioception.proprio.SimpleProprioception`,
      :class:`~mimoTouch.touch.DiscreteTouch`, :class:`~mimoVision.vision.SimpleVision`,
      :class:`~mimoVestibular.vestibular.SimpleVestibular`.
    - :meth:`._get_proprio_obs`, :meth:`._get_touch_obs`, :meth:`._get_vision_obs`, :meth:`._get_vestibular_obs`, these
      functions collect the observations of the associated sensor modality. These allow you to do post processing on
      the output without having to alter the base implementations.
    - :meth:`._reset_sim`, which resets the physical simulation. If you have special conditions on the initial position
      this function should implement/ensure them.
    - :meth:`._step_callback`, which is called every step after the simulation step but before the observations are
      collected.

    These functions come with default implementations that should handle most scenarios.

    The constructor takes the following arguments:

    - `model_path`: The path to the scene xml. Required.
    - `initial_qpos`: A dictionary of the initial joint positions. Keys are the joint names. Only required if the
      initial position varies from that defined the XMLs.
    - `n_substeps`: The number of physics substeps for each simulation step. The duration of each physics step is set
      in the scene XML.
    - `proprio_params`: The configuration dictionary for the proprioceptive system. If `None` the module is disabled.
      Default `None`.
    - `touch_params`: The configuration dictionary for the touch system. If `None` the module is disabled.
      Default `None`.
    - `vision_params`: The configuration dictionary for the vision system. If `None` the module is disabled.
      Default `None`.
    - `vestibular_params`: The configuration dictionary for the vestibular system. If `None` the module is disabled.
      Default `None`.
    - `goals_in_observation`: If `True` the desired and achieved goals are included in the observation dictionary.
      Default `True`.
    - `done_active`: If `True`, :meth:`._is_done` returns `True` if the simulation reaches a success or failure state.
      If `False`, :meth:`._is_done` always returns `False` and the function calling :meth:`.step` has to figure out when
      to stop or reset the simulation on its own.

    Attributes:
        sim: The MuJoCo simulation object.
        initial_state: A copy of the simulation state at the start of the simulation.
        goal: The desired goal.
        action_space: The action space. See Gym documentation for more.
        observation_space: The observation space. See Gym documentation for more.
        proprio_params: The configuration dictionary for the proprioceptive system.
        proprioception: A reference to the proprioception instance.
        touch_params: The configuration dictionary for the touch system.
        touch: A reference to the touch instance.
        vision_params: The configuration dictionary for the vision system.
        vision: A reference to the vision instance.
        vestibular_params: The configuration dictionary for the vestibular system.
        vestibular: A reference to the vestibular instance.
        facial_expressions: A dictionary linking emotions with their associated facial textures. The keys of this
            dictionary are valid inputs for :meth:`.swap_facial_expression`
        goals_in_observation: If `True` the desired and achieved goals are included in the observation dictionary.
            Default `True`.
        done_active: If `True`, :meth:`._is_done` returns `True` if the simulation reaches a success or failure state.
            If `False`, :meth:`._is_done` always returns `False` and the function calling :meth:`.step` has to figure
            out when to stop or reset the simulation on its own.

    """

    def __init__(self,
                 model_path,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=None,
                 touch_params=None,
                 vision_params=None,
                 vestibular_params=None,
                 goals_in_observation=True,
                 done_active=False):

        utils.EzPickle.__init__(**locals())

        self.proprio_params = proprio_params
        self.touch_params = touch_params
        self.vision_params = vision_params
        self.vestibular_params = vestibular_params

        self.proprioception = None
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
        self.offscreen_context = None
        self._current_mode = None

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
            tex_id = mimo_utils.texture_name2id(self.sim.model, tex_name)
            self.facial_expressions[emote] = tex_id
        head_material_name = "head"
        self._head_material_id = mimo_utils.material_name2id(self.sim.model, head_material_name)

        self.goal = self._sample_goal()
        n_actions = len([name for name in self.sim.model.actuator_names if name.startswith("act:")])
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_actions,), dtype="float32")
        obs = self._get_obs()
        # Observation spaces
        spaces_dict = {
            "observation": spaces.Box(-np.inf, np.inf, shape=obs["observation"].shape, dtype="float32")
        }
        if self.touch:
            spaces_dict["touch"] = spaces.Box(-np.inf, np.inf, shape=obs["touch"].shape, dtype="float32")
        if self.vision:
            for sensor in self.vision_params:
                spaces_dict[sensor] = spaces.Box(0, 256, shape=obs[sensor].shape, dtype="uint8")
        if self.vestibular:
            spaces_dict["vestibular"] = spaces.Box(-np.inf, np.inf, shape=obs["vestibular"].shape, dtype="float32")
        if self.goals_in_observation:
            spaces_dict["desired_goal"] = spaces.Box(
                -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32")
            spaces_dict["achieved_goal"] = spaces.Box(
                -np.inf, np.inf, shape=obs["achieved_goal"].shape, dtype="float32")

        self.observation_space = spaces.Dict(spaces_dict)

    def _env_setup(self, initial_qpos):
        """ This function initializes all the sensory components of the model.

        Calls the setup functions for all the sensory components and sets the initial positions of the joints according
        to the qpos dictionary.

        Args:
            initial_qpos (dict): A dictionary with the intial joint position for each joint. Keys are the joint names.

        """
        # Our init goes here. At this stage the mujoco model is already loaded, but most of the gym attributes, such as
        # observation space and goals are not set yet

        # Do setups
        self._proprio_setup(self.proprio_params)
        if self.touch_params is not None:
            self._touch_setup(self.touch_params)
        if self.vision_params is not None:
            self._vision_setup(self.vision_params)
        if self.vestibular_params is not None:
            self._vestibular_setup(self.vestibular_params)
        # Should be able to get all types of sensor outputs here
        # Should be able to produce all control inputs here

        # Implement qpos:
        for joint_name in initial_qpos:
            mimo_utils.set_joint_qpos(self.sim.model, self.sim.data, joint_name, initial_qpos[joint_name])

    def _proprio_setup(self, proprio_params):
        """ Perform the setup and initialization of the proprioceptive system.

        This should be overridden if you want to use another implementation!

        Args:
            proprio_params (dict): The parameter dictionary.
        """
        self.proprioception = SimpleProprioception(self, proprio_params)

    def _touch_setup(self, touch_params):
        """ Perform the setup and initialization of the touch system.

        This should be overridden if you want to use another implementation!

        Args:
            touch_params (dict): The parameter dictionary.
        """
        self.touch = DiscreteTouch(self, touch_params)

    def _vision_setup(self, vision_params):
        """ Perform the setup and initialization of the vision system.

        This should be overridden if you want to use another implementation!

        Args:
            vision_params (dict): The parameter dictionary.
        """
        self.vision = SimpleVision(self, vision_params)

    def _vestibular_setup(self, vestibular_params):
        """ Perform the setup and initialization of the vestibular system.

        This should be overridden if you want to use another implementation!

        Args:
            vestibular_params (dict): The parameter dictionary.
        """
        self.vestibular = SimpleVestibular(self, vestibular_params)

    def step(self, action):
        """ The step function for the simulation.

        This function takes a simulation step, collects the observations, computes the reward and finally determines if
        we are done with this episode or not. :meth:`._get_obs` collects the observations, :meth:`.compute_reward`
        calculates the reward.`:meth:`._is_done` is called to determine if we are done with the episode and
        :meth:`._step_callback` can be used for extra functions each step, such as incrementing a step counter.

        Args:
            action (numpy.ndarray): A numpy array with the control inputs for this step. The shape must match the action
                space!

        Returns:
            A tuple `(observations, reward, done, info)` as described above, with info containing extra information,
            such as whether we reached a success state specifically.
        """
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
            info["achieved_goal"] = copy.deepcopy(achieved_goal)
            info["desired_goal"] = copy.deepcopy(self.goal)

        done = self._is_done(achieved_goal, self.goal, info)

        reward = self.compute_reward(achieved_goal, self.goal, info)
        return obs, reward, done, info

    def _step_callback(self):
        """A custom callback that is called after stepping the simulation.

        Can be used to enforce additional constraints on the simulation state.
        """
        pass

    def reset(self):
        """ Attempt to reset the simulator and sample a new goal.

        Resets the simulation state, samples a new goal and collects an initial set of observations.
        This function calls :meth:`._reset_sim` until it returns `True`. This is useful if your resetting function has
        a randomized component that can end up in an illegal state. In this case this function will try again until a
        valid state is reached.

        Returns:
            dict: The observations after reset.
        """
        #
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()
        self.goal = self._sample_goal()
        obs = self._get_obs()
        return obs

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.

        Resets the simulation state and returns whether or not the reset was successful. This is useful if your
        resetting function has a randomized component that can end up in an illegal state. In this case this function
        will be called again until a valid state is reached.

        Returns:
            bool: Whether we reset into a valid state.
        """
        self.sim.set_state(self.initial_state)
        self.sim.forward()
        return True

    def _get_proprio_obs(self):
        """ Collects and returns the outputs of the proprioceptive system.

        Override this function if you want to make some simple post-processing!

        Returns:
            numpy.ndarray: A numpy array containing the proprioceptive output.
        """
        return self.proprioception.get_proprioception_obs()

    def _get_touch_obs(self):
        """ Collects and returns the outputs of the touch system.

        Override this function if you want to make some simple post-processing!

        Returns:
            numpy.ndarray: A numpy array containing the touch output.
        """
        touch_obs = self.touch.get_touch_obs()
        return touch_obs

    def _get_vision_obs(self):
        """ Collects and returns the outputs of the vision system.

        Override this function if you want to make some simple post-processing!

        Returns:
            dict: A dictionary with one entry for each separate image. In the default implementation each eye renders
            one image, so each eye gets one entry.
        """
        vision_obs = self.vision.get_vision_obs()
        return vision_obs

    def _get_vestibular_obs(self):
        """ Collects and returns the outputs of the vestibular system.

        Override this function if you want to make some simple post-processing!

        Returns:
            numpy.ndarray: A numpy array with the vestibular data.
        """
        vestibular_obs = self.vestibular.get_vestibular_obs()
        return vestibular_obs

    def _get_obs(self):
        """Returns the observation.

        This function should return all simulation outputs relevant to whatever learning algorithm you wish to use. We
        always return proprioceptive information in the 'observation' entry, and this information always includes
        relative joint positions. Other sensory modalities get their own entries, if they are enabled. If
        :attr:`.goals_in_observation` is set to `True`, the achieved and desired goal are also included.

        Returns:
            dict: A dictionary containing simulation outputs with separate entries for each sensor modality.
        """
        # robot proprioception:
        proprio_obs = self._get_proprio_obs()
        observation_dict = {
            "observation": proprio_obs,
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
            observation_dict["achieved_goal"] = copy.deepcopy(achieved_goal)
            observation_dict["desired_goal"] = copy.deepcopy(self.goal)

        return observation_dict

    def _set_action(self, action):
        """ Set the control inputs for the next step.

        Control values are clipped to the control range limits defined the MuJoCo xmls and normalized to be even in
        both directions, i.e. an input of 0 corresponds to the center of the control range, rather than the default or
        neutral control position. The control ranges for the MIMo xmls are set up to be symmetrical, such that an input
        of 0 corresponds to no motor torque.

        Args:
            action (numpy.ndarray): A numpy array with control values.
        """
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        self.sim.data.ctrl[:] = actuation_center + action * actuation_range
        self.sim.data.ctrl[:] = np.clip(
            self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1]
        )

    def swap_facial_expression(self, emotion):
        """ Changes MIMos facial texture.

        Valid emotion names are in :attr:`.facial_expression`, which links readable emotion names to their associated
        texture ids

        Args:
            emotion (str): A valid emotion name.
        """
        assert emotion in self.facial_expressions, "{} is not a valid facial expression!".format(emotion)
        new_tex_id = self.facial_expressions[emotion]
        self.sim.model.mat_texid[self._head_material_id] = new_tex_id

    def _is_success(self, achieved_goal, desired_goal):
        """ Indicates whether or not the achieved goal successfully achieved the desired goal.

        Args:
            achieved_goal (object): The goal that was achieved during execution.
            desired_goal (object): The desired goal that we asked the agent to attempt to achieve.

        Returns:
            bool: If we successfully reached the desired goal state.

        """
        raise NotImplementedError

    def _is_failure(self, achieved_goal, desired_goal):
        """Indicates that we reached a failure state.

        Args:
            achieved_goal (object): The goal that was achieved during execution.
            desired_goal (object): The desired goal that we asked the agent to attempt to achieve.

        Returns:
            bool: If we reached an unrecoverable failure state.
        """
        raise NotImplementedError

    def _is_done(self, achieved_goal, desired_goal, info):
        """ This function should return `True` if we have reached a success or failure state.

        By default this function always returns `False`. If :attr:`.done_active` is set to `True`, ignores both goal
        parameters and instead returns `True` if either :meth:`._is_success` or :meth:`._is_failure` return True.
        The goal parameters are there to allow this class to be more easily overridden by subclasses, should this be
        required.

        Args:
            achieved_goal (object): The goal that was achieved during execution.
            desired_goal (object): The desired goal that we asked the agent to attempt to achieve.
            info (dict): An info dictionary with additional information.

        Return:
            bool: Whether the current episode is done.

        """
        return self.done_active and (info["is_success"] or info["is_failure"])

    def _sample_goal(self):
        """ Should sample a new goal and return it.

        Returns:
            object: The desired end state.
        """
        raise NotImplementedError

    def _get_achieved_goal(self):
        """ Should return the goal that was achieved during the simulation.

        Returns:
            object: The achieved end state.
        """
        raise NotImplementedError

    def compute_reward(self, achieved_goal, desired_goal, info):
        """Compute the step reward.

        This externalizes the reward function and makes it dependent on a desired goal and the one that was achieved.
        If you wish to include additional rewards that are independent of the goal, you can include the necessary values
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

    # ====================== gym rendering =======================================================

    def _get_viewer(self, mode):
        """ Handles render contexts.

        Args:
            mode (str): One of "human" or "rgb_array". If "rgb_array" an offscreen render context is used, otherwise we render
            to an interactive viewer window.

        """
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array":
                if sys.platform != "darwin":
                    self.offscreen_context = mujoco_py.GlfwContext(offscreen=True)
                else:
                    self.offscreen_context = self._get_viewer('rgb_array').opengl_context
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, device_id=-1)
            self._viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def _swap_context(self, window):
        """ Swaps the current render context to 'window'

        Args:
            window: The new render context.
        """
        glfw.make_context_current(window)

    def close(self):
        """ Removes all references to render contexts, etc..."""
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}
            self.offscreen_context = None

    def render(self, mode="human", width=DEFAULT_SIZE, height=DEFAULT_SIZE, camera_name=None, camera_id=None):
        """ General rendering function for cameras or interactive environment.

        There are two modes, 'human' and 'rgb_array'. In 'human' we render to an interactive window, ignoring all other
        parameters. Width and size are determined by the size of the window (which can be resized).
        In mode 'rgb_array' we return the rendered image as a numpy array. The size of the image is determined by the
        `width` and `height` parameters. A specific camera can be rendered by providing either its name or its ID. By
        default the standard Mujoco free cam is used. The vertical field of view for each camera is defined in the
        scene xml, with the horizontal field of view determined by the rendering resolution.

        Args:
            mode (str): One of either 'human' or 'rgb_array'.
            width (int): The width of the output image
            height (int): The height of the output image
            camera_name (str): The name of the camera that will be rendered. Default None.
            camera_id (int): The ID of the camera that will be rendered. Default None.

        Returns:
            A numpy array with the output image or None if mode is 'human'.
        """
        self._render_callback()

        assert camera_name is None or camera_id is None, "Only one of camera_name or camera_id can be supplied!"
        if camera_name is not None:
            camera_id = self.sim.model.camera_name2id(camera_name)

        # Make sure viewers and contexts are setup before we try to swap to/from nonexisting contexts
        self._get_viewer(mode)

        if mode == "rgb_array":
            # Swap to offscreen context if necessary
            if self._current_mode != "rgb_array":
                self._swap_context(self.offscreen_context.window)
            self._current_mode = "rgb_array"

            self._get_viewer(mode).render(width, height, camera_id)
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]

        elif mode == "human":
            # Swap to onscreen context if necessary
            if self._current_mode != "human":
                self._swap_context(self.sim._render_context_window.window)
            self._current_mode = "human"
            self._get_viewer(mode).render()

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        pass
