""" This module contains a simple reaching experiment in which MIMo tries to catch a falling ball.

The scene consists of MIMo with his right arm outstretched and his palm open. A ball is located just above MIMos palm.
The task is for him to catch the falling ball.
MIMo is fixed in position and can only move his right hand.
Sensory input consists of the full proprioceptive inputs and touch input.

An episode is completed successfully if MIMo holds onto the ball continously for 1 second. There are no failure states.

There is a small negative reward for each step without touching the ball, a larger positive reward for each step in
contact with the ball and then a large fixed reward on success.

The class with the environment is :class:`~mimoEnv.envs.reach.MIMoReachEnv` while the path to the scene XML is defined
in :data:`REACH_XML`.
"""
import os
import numpy as np
import mujoco_py

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, DEFAULT_PROPRIOCEPTION_PARAMS
from mimoEnv.envs.mimo_muscle_env import MIMoMuscleEnv
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


class MIMoCatchEnv(MIMoMuscleEnv):
    """ MIMo reaches for an object.

    Attributes and parameters are the same as in the base class, but the default arguments are adapted for the scenario.

    Due to the goal condition we do not use the :attr:`.goal` attribute or the interfaces associated with it. Instead,
    the reward and success conditions are computed directly from the model state, while
    :meth:`~mimoEnv.envs.reach.MIMoReachEnv._sample_goal` and
    :meth:`~mimoEnv.envs.reach.MIMoReachEnv._get_achieved_goal` are dummy functions.

    """
    def __init__(self,
                 model_path=CATCH_XML,
                 initial_qpos={},
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=TOUCH_PARAMS,
                 vision_params=None,
                 vestibular_params=None,
                 goals_in_observation=False,
                 done_active=True,
                 action_penalty=False):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)

        self.steps_in_contact_for_success = 100
        self.in_contact_past = [False for _ in range(self.steps_in_contact_for_success)]
        self.steps = 0
        self.target_body = self.sim.model.body_name2id("target")
        self.target_geoms = env_utils.get_geoms_for_body(self.sim.model, self.target_body)
        self.own_geoms = []
        for body_name in touch_params["scales"]:
            body_id = self.sim.model.body_name2id(body_name)
            self.own_geoms.extend(env_utils.get_geoms_for_body(self.sim.model, body_id))
        self.action_penalty = action_penalty
        print("Action penalty: ", self.action_penalty)

        # Info required to randomize ball position
        self.random_limits = np.array([0.01, 0.01, 0.08, 0, 0, 0, 0])
        target_joint = "target_joint"
        self.target_joint_id = self.sim.model.joint_name2id(target_joint)
        self.target_joint_qpos = env_utils.get_joint_qpos_addr(self.sim.model, self.target_joint_id)
        self.target_joint_qvel = env_utils.get_joint_qvel_addr(self.sim.model, self.target_joint_id)

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Computes the reward.

        Fixed negative reward for each step not in contact, fixed positive for each step in contact?
        Maybe add success and a large fixed rewards for continuous contact for n steps?

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.
            info (dict): This parameter is ignored.

        Returns:
            float: The reward as described above.
        """
        if self._currently_in_contact():
            reward = 0
        else:
            reward = -1

        if self.action_penalty:
            # TODO: Use motor torque instead of just control input
            reward -= 0.5 * np.square(self.sim.data.ctrl).sum() / self.action_space.shape[0]

        if self._is_success(achieved_goal, desired_goal):
            reward = 500
        return reward

    def _is_success(self, achieved_goal, desired_goal):
        """ Returns true if MIMo touches the object continuously for 1 second.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: `True` if the ball is knocked out of position.
        """

        return all(self.in_contact_past)

    def _is_failure(self, achieved_goal, desired_goal):
        """ Dummy function. Always returns `False`.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: `False`
        """
        return False

    def _sample_goal(self):
        """ Dummy function. Returns an empty array.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.zeros((0,))

    def _get_achieved_goal(self):
        """ Dummy function. Returns an empty array.

        Returns:
            numpy.ndarray: An empty array.
        """
        return np.zeros((0,))

    def _reset_sim(self):
        """ Resets the simulation.

        We reset the simulation and then slightly move both MIMos arm and the ball randomly. The randomization is
        limited such that MIMo can always reach the ball.

        Returns:
            bool: `True`
        """

        self.sim.set_state(self.initial_state)

        # Randomize ball position
        random_shift = np.random.uniform(low=-self.random_limits, high=self.random_limits)
        self.sim.data.qpos[self.target_joint_qpos] += random_shift
        self.sim.data.qvel[self.target_joint_qvel] = np.zeros(self.sim.data.qvel[self.target_joint_qvel].shape)

        self.sim.forward()

        # perform 50 steps (.5 secs) with gravity off to settle arm
        gravity = self.sim.model.opt.gravity[2]
        self.sim.model.opt.gravity[2] = 0
        for _ in range(50):
            action = np.zeros(self.action_space.shape)
            self._set_action(action)
            self.sim.step()
            self._step_callback()

        # Reset gravity
        self.sim.model.opt.gravity[2] = gravity

        # reset target in random initial position and velocities as zero
        self.sim.forward()
        return True

    def _step_callback(self):
        self.steps += 1
        self.in_contact_past[self.steps % 100] = self._in_contact()
        pass

    def _in_contact(self):
        in_contact = False
        # Go over all contacts
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            # Is this a contact between us and the target object?
            if (contact.geom1 in self.target_geoms or contact.geom2 in self.target_geoms) \
                    and (contact.geom1 in self.own_geoms or contact.geom2 in self.own_geoms):

                # Check that contact is active
                forces = np.zeros(6, dtype=np.float64)
                mujoco_py.functions.mj_contactForce(self.sim.model, self.sim.data, i, forces)
                if abs(forces[0]) < 1e-9:  # Contact probably inactive
                    continue
                else:
                    in_contact = True
                    break
        return in_contact

    def _currently_in_contact(self):
        return self.in_contact_past[self.steps % 100]

    def _viewer_setup(self):
        """Initial configuration of the viewer. Can be used to set the camera position,
        for example.
        """
        self.viewer.cam.trackbodyid = 0  # id of the body to track
        self.viewer.cam.distance = .5  # how much you "zoom in", model.stat.extent is the max limits of the arena smaller is closer
        self.viewer.cam.lookat[0] = 0.15  # x,y,z offset from the object (works if trackbodyid=-1)
        self.viewer.cam.lookat[1] = -0.04
        self.viewer.cam.lookat[2] = 0.6  # 0.24 -0.04 .8
        self.viewer.cam.elevation = -30
        self.viewer.cam.azimuth = 120
