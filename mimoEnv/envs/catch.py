""" This module contains a simple reaching experiment in which MIMo tries to catch a falling ball.

The scene consists of MIMo with his right arm outstretched and his palm open. A ball is located just above MIMos palm.
The task is for him to catch the falling ball.
MIMo is fixed in position and can only move his right hand.
Sensory input consists of the full proprioceptive inputs and touch input.

An episode is completed successfully if MIMo holds onto the ball continuously for 1 second. An episode fails when the
ball drops some distance below MIMos hand or is bounced into the distance.

There is a small negative reward for each step without touching the ball, a larger positive reward for each step in
contact with the ball and then a large fixed reward on success.
"""
import os
import random

import mujoco
import numpy as np

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, DEFAULT_PROPRIOCEPTION_PARAMS
from mimoActuation.muscle import MuscleModel
import mimoEnv.utils as env_utils


CATCH_XML = os.path.join(SCENE_DIRECTORY, "catch_scene.xml")
""" Path to the reach scene.

:meta hide-value:
"""


TOUCH_PARAMS = {
    "scales": {
        "right_upper_arm": 0.024,
        "right_lower_arm": 0.024,
        "right_hand": 0.007,
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
""" Touch parameters for the catch environment. Only the right arm is equipped with sensors.

:meta hide-value:
"""

CATCH_CAMERA_CONFIG={
    "trackbodyid": 0,
    "distance": 0.5,
    "lookat": np.asarray([0.15, -0.04, 0.6]),
    "elevation": -30,
    "azimuth": 120,
}
""" Camera configuration so it looks straight at the hand.

:meta hide-value:
"""

class MIMoCatchEnv(MIMoEnv):
    """ MIMo tries to catch a falling ball.

    MIMo is tasked with catching a falling ball and holding onto it for one second. MIMo's head and eyes automatically
    track the ball. The position of the ball is slightly randomized each episode.
    The constructor takes three additional arguments over the base environment.

    Args:
        action_penalty (bool): If ``True``, an action penalty based on the cost function of the actuation model is
            applied to the reward. Default ``True``.
        jitter (bool): If ``True``, the input actions are multiplied with a perturbation array which is randomized
            every 10-50 time steps. Default ``False``.
        position_inaccurate (bool): If ``True``, the position tracked by the head is offset by a small random distance
            from the true position of the ball. Default ``False``.

    Attributes:
        action_penalty (bool): If ``True``, an action penalty based on the cost function of the actuation model is
            applied to the reward. Default ``True``.
        jitter (bool): If ``True``, the input actions are multiplied with a perturbation array which is randomized
            every 10-50 time steps. Default ``False``.
        use_position_inaccuracy (bool): If ``True``, the position tracked by the head is offset by a small random
            distance from the true position of the ball. Default ``False``.
        position_limits (np.ndarray): Maximum distances away from the default ball position for the randomization.
        position_inaccuracy_limits (np.ndarray): Maximum distances for the head tracking offset.
        position_offset (np.ndarray): The actual inaccuracy of the head tracking. This is randomized each episode.
        size_limits (Tuple[float, float]): Minimum and maximum size of the ball.
        ball_size (float): Current ball size. Changes each episode.
        mass_limits (Tuple[float, float]): Minimum and maximum mass of the ball.
        ball_mass (float): Current ball mass. Changes each episode.
        jitter_array (np.ndarray): Control inputs are multiplied by this array before being passed to MuJoCo. This is
            randomized every so often.
        jitter_period (int): The number of steps the current jitter array is used for before being randomized again.
        steps_in_contact_for_success (int): For how many steps MIMo must hold onto the ball.
        in_contact_past (List[bool]): A list storing which past steps we were in contact for. This list works by
            modulo, i.e. to determine if MIMo held the ball on step `i`, do
            ``in_contact_past[i % steps_in_contact_for_success]``.
    """
    def __init__(self,
                 model_path=CATCH_XML,
                 initial_qpos=None,
                 frame_skip=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=TOUCH_PARAMS,
                 vision_params=None,
                 vestibular_params=None,
                 actuation_model=MuscleModel,
                 goals_in_observation=False,
                 done_active=True,
                 action_penalty=True,
                 jitter=False,
                 position_inaccurate=False,
                 default_camera_config=CATCH_CAMERA_CONFIG,
                 **kwargs):

        self.jitter = jitter
        self.use_position_inaccuracy = position_inaccurate

        # Target ball randomization limits.
        self.position_limits = np.array([0.01, 0.01, 0.08, 0, 0, 0, 0])
        self.position_inaccuracy_limits = np.asarray([0.005, 0.005, 0.005])
        self.position_offset = np.zeros((3,))
        self.size_limits = (0.005, 0.025)
        self.ball_size = 0.025
        self.mass_limits = (0.05, 0.5)
        self.ball_mass = 0.5

        self.jitter_period = 0
        self.jitter_array = 1.0

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         frame_skip=frame_skip,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active,
                         default_camera_config=default_camera_config,
                         **kwargs)

        self.steps_in_contact_for_success = 100
        self.in_contact_past = [False for _ in range(self.steps_in_contact_for_success)]
        self.steps = 0
        self.target_id = self.model.body("target").id
        self.hand_id = self.model.body("right_hand").id
        self.head_id = self.model.body("head").id
        self.target_geoms = env_utils.get_geoms_for_body(self.model, self.target_id)
        self.own_geoms = []
        for body_name in touch_params["scales"]:
            body_id = self.model.body(body_name).id
            self.own_geoms.extend(env_utils.get_geoms_for_body(self.model, body_id))
        self.action_penalty = action_penalty
        print("Action penalty: ", self.action_penalty)

        # Info required to randomize ball position
        target_joint = "target_joint"
        self.target_joint_id = self.model.joint(target_joint).id
        self.target_joint_qpos = env_utils.get_joint_qpos_addr(self.model, self.target_joint_id)
        self.target_joint_qvel = env_utils.get_joint_qvel_addr(self.model, self.target_joint_id)


    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Computes the reward.

        MIMo is rewarded for each time step in contact with the target. Completing an episode successfully awards +100,
        while failing leads to a -100 penalty. Additionally, there is an action penalty based on the cost function of
        the actuation model.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.
            info (dict): This parameter is ignored.

        Returns:
            float: The reward as described above.
        """
        reward = 0

        # Positive reward for contact with the target
        if self._currently_in_contact():
            reward += 1

        if self.action_penalty:
            cost = self.actuation_model.cost()
            reward -= 40*cost

        if info['is_failure']:
            reward -= 100
        if info['is_success']:
            reward += 100

        return reward

    def do_simulation(self, action, n_frames):
        """ Implementation that adds jitter to the actions. """
        # Implement the jitter
        action = np.clip(action * self.jitter_array, 0, 1)
        super().do_simulation(action, n_frames)

    def _get_obs(self):
        """ Adds the size of the ball to the observations.

        Returns:
            Dict: The altered observation dictionary.
        """
        obs = super()._get_obs()
        obs["observation"] = np.append(obs["observation"], self.ball_size)
        return obs

    def is_success(self, achieved_goal, desired_goal):
        """ Returns true if MIMo touches the object continuously for 1 second.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``True`` if MIMo has been touching the ball for the last second, ``False`` otherwise.
        """

        return all(self.in_contact_past)

    def is_failure(self, achieved_goal, desired_goal):
        """ Returns ``True`` if the ball drops below MIMo's hand.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: ``True`` if the ball drops below MIMo's hand, ``False`` otherwise.
        """
        hand_pos = env_utils.get_body_position(self.data, self.hand_id)
        target_pos = env_utils.get_body_position(self.data, self.target_id)
        return (target_pos[2] < (hand_pos[2] - 0.1)) or (np.linalg.norm(target_pos-hand_pos) > 0.5)

    def is_truncated(self):
        """Dummy function.

        Returns:
            bool: Always returns ``False``.
        """
        return False

    def sample_goal(self):
        """ Dummy function. Returns an empty array.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.zeros((0,))

    def get_achieved_goal(self):
        """ Dummy function. Returns an empty array.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.zeros((0,))

    def reset_model(self):
        """ Resets the simulation.

        We reset the simulation and then slightly move both MIMos arm and the ball randomly. The randomization is
        limited such that MIMo can always reach the ball.

        Returns:
            bool: Always returns ``True``.
        """
        self.set_state(self.init_qpos, self.init_qvel)

        # Randomize ball position
        random_shift = np.random.uniform(low=-self.position_limits, high=self.position_limits)
        self.data.qpos[self.target_joint_qpos] += random_shift
        self.data.qvel[self.target_joint_qvel] = np.zeros(self.data.qvel[self.target_joint_qvel].shape)

        # Randomize ball size
        target_geom = self.target_geoms[0]  # Target_geoms is a list with a single entry
        self.ball_size = random.uniform(self.size_limits[0], self.size_limits[1])
        self.model.geom_size[target_geom][0] = self.ball_size
        self.model.geom_rbound[target_geom] = self.ball_size

        # Randomize ball mass
        self.ball_mass = random.uniform(self.mass_limits[0], self.mass_limits[1])
        self.model.body_mass[self.target_id] = self.ball_mass
        inertia = 2 * self.ball_mass * self.ball_size * self.ball_size / 5
        self.model.body_inertia[self.target_id] = np.asarray([inertia, inertia, inertia])
        mujoco.mj_setConst(self.model, self.data) # Recompute derived mujoco quantities

        # Randomize ball offset
        if self.use_position_inaccuracy:
            self.position_offset = np.random.uniform(low=-self.position_inaccuracy_limits,
                                                     high=self.position_inaccuracy_limits)

        mujoco.mj_forward(self.model, self.data)

        # perform 50 steps (.5 secs) with gravity off to settle arm
        gravity = self.model.opt.gravity[2]
        self.model.opt.gravity[2] = 0
        self.actuation_model.reset()
        action = np.zeros(self.action_space.shape)
        self._set_action(action)
        for _ in range(50):
            self._single_mujoco_step()

        # Reset gravity
        self.model.opt.gravity[2] = gravity

        self._step_callback()
        self.steps = 0

        # reset target in random initial position and velocities as zero
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _step_callback(self):
        """ Checks if MIMo is touching the ball and performs head tracking.
        """
        self.steps += 1

        self.in_contact_past[self.steps % self.steps_in_contact_for_success] = self._in_contact()

        # manually set head and eye positions to look at target
        target_pos = env_utils.get_body_position(self.data, self.target_id) + self.position_offset
        head_pos = env_utils.get_body_position(self.data, self.head_id)
        head_target_dif = target_pos - head_pos
        head_target_dist = np.linalg.norm(head_target_dif)
        head_target_dif[2] = head_target_dif[2] - 0.067375  # extra difference to eye height in head
        half_eyes_dist = 0.0245  # horizontal distance between eyes / 2
        eyes_target_dist = head_target_dist - 0.07  # remove distance from head center to eyes
        env_utils.set_joint_qpos(self.model, self.data, "robot:head_swivel",
                                 np.arctan(head_target_dif[1] / head_target_dif[0]))
        env_utils.set_joint_qpos(self.model, self.data, "robot:head_tilt",
                                 np.arctan(-head_target_dif[2] / head_target_dif[0]))
        env_utils.set_joint_qpos(self.model, self.data, "robot:head_tilt_side", 0)

        env_utils.set_joint_qpos(self.model, self.data, "robot:left_eye_horizontal",
                                 np.arctan(-half_eyes_dist / eyes_target_dist))
        env_utils.set_joint_qpos(self.model, self.data, "robot:left_eye_vertical", 0)
        env_utils.set_joint_qpos(self.model, self.data, "robot:left_eye_torsional", 0)

        env_utils.set_joint_qpos(self.model, self.data, "robot:right_eye_horizontal",
                                 np.arctan(-half_eyes_dist / eyes_target_dist))
        env_utils.set_joint_qpos(self.model, self.data, "robot:right_eye_vertical", 0)
        env_utils.set_joint_qpos(self.model, self.data, "robot:right_eye_torsional", 0)

        if self.jitter:
            if self.jitter_period <= 0:
                self.jitter_period = int(random.uniform(10, 51))
                self.jitter_array = np.random.uniform(0.8, 1.2, self.action_space.shape)
            else:
                self.jitter_period -= 1

    def _in_contact(self):
        """ Check if MIMo is currently touching the target ball.

        This function performs the actual contact check and is called during :meth:`.step_callback`.

        Returns:
            bool: ``True`` if MIMo is currently touching the ball, ``False`` otherwise..
        """
        in_contact = False
        # Go over all contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Is this a contact between us and the target object?
            if (contact.geom1 in self.target_geoms or contact.geom2 in self.target_geoms) \
                    and (contact.geom1 in self.own_geoms or contact.geom2 in self.own_geoms):

                # Check that contact is active
                forces = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, forces)
                if abs(forces[0]) < 1e-9:  # Contact probably inactive
                    continue
                else:
                    in_contact = True
                    break
        return in_contact

    def body_contact_reward(self):
        """ Reward function that provides higher rewards the more geoms are touching the target.

        Returns:
            float: The reward component as described above.
        """
        reward = 0
        # Positive reward for contact with the target
        for i in range(self.data.ncon):
            # Max seems to be 9
            contact = self.data.contact[i]
            # Is this a contact between us and the target object?
            if (contact.geom1 in self.target_geoms or contact.geom2 in self.target_geoms) \
                    and (contact.geom1 in self.own_geoms or contact.geom2 in self.own_geoms):

                # Check that contact is active
                forces = np.zeros(6, dtype=np.float64)
                mujoco.mj_contactForce(self.model, self.data, i, forces)
                if abs(forces[0]) > 1e-9:  # Contact active
                    reward += 1
        return reward / 10

    def _currently_in_contact(self):
        """ Check if MIMo is currently touching the ball.

        Unlike :meth:`._in_contact` this function does not perform the check itself, instead checking the array of past
        contacts for the current time step. The output of this function will not be accurate if called before
        :meth:`._in_contact`!

        Returns:
            bool: ``True`` if MIMo is currently touching the ball, ``False`` otherwise."""
        return self.in_contact_past[self.steps % self.steps_in_contact_for_success]
