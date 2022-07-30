import os
import numpy as np
import mujoco_py

from mimoEnv.envs.mimo_env import MIMoEnv, SCENE_DIRECTORY, DEFAULT_PROPRIOCEPTION_PARAMS, DEFAULT_VESTIBULAR_PARAMS, \
    DEFAULT_PAIN_PARAMS, DEFAULT_TOUCH_PARAMS


EXPERIMENT_XML = os.path.join(SCENE_DIRECTORY, "experiment.xml")
""" Path to the stand up scene.

:meta hide-value:
"""

LYING_POSITION = {
    "robot:hip_lean1": np.array([-4.55e-7]), "robot:hip_rot1": np.array([-1.85e-7]),
    "robot:hip_bend1": np.array([0.0383]), "robot:hip_lean2": np.array([-4.43e-7]), "robot:hip_rot2": np.array([-1.61e-7]),
    "robot:hip_bend2": np.array([0.0487]),
    "robot:head_swivel": np.array([-1.48e-6]), "robot:head_tilt": np.array([0.274]), "robot:head_tilt_side": np.array([-1.02e-6]),
    "robot:left_eye_horizontal": np.array([0]), "robot:left_eye_vertical": np.array([0]),
    "robot:left_eye_torsional": np.array([0]), "robot:right_eye_horizontal": np.array([0]),
    "robot:right_eye_vertical": np.array([0]), "robot:right_eye_torsional": np.array([0]),
    "robot:right_shoulder_horizontal": np.array([-0.271]), "robot:right_shoulder_ad_ab": np.array([0.11]),
    "robot:right_shoulder_rotation": np.array([0.00244]), "robot:right_elbow": np.array([-0.195]),
    "robot:right_hand1": np.array([-0.347]), "robot:right_hand2": np.array([0.0081]),
    "robot:right_hand3": np.array([-0.0106]), "robot:right_fingers": np.array([-0.666]),
    "robot:left_shoulder_horizontal": np.array([-0.274]), "robot:left_shoulder_ad_ab": np.array([0.105]),
    "robot:left_shoulder_rotation": np.array([-0.0111]), "robot:left_elbow": np.array([-0.196]),
    "robot:left_hand1": np.array([-0.347]), "robot:left_hand2": np.array([0.0083]), "robot:left_hand3": np.array([-0.0104]),
    "robot:left_fingers": np.array([-0.666]),
    "robot:right_hip1": np.array([-0.168]), "robot:right_hip2": np.array([4.71e-8]),
    "robot:right_hip3": np.array([3.55e-8]), "robot:right_knee": np.array([-0.502]),
    "robot:right_foot1": np.array([-0.175]), "robot:right_foot2": np.array([1.41e-8]),
    "robot:right_foot3": np.array([-1.9e-8]), "robot:right_toes": np.array([-4.35e-5]),
    "robot:left_hip1": np.array([-0.168]), "robot:left_hip2": np.array([-3.8e-8]),
    "robot:left_hip3": np.array([-2.85e-8]), "robot:left_knee": np.array([-0.502]),
    "robot:left_foot1": np.array([-0.175]), "robot:left_foot2": np.array([-1.25e-8]),
    "robot:left_foot3": np.array([1.68e-8]), "robot:left_toes": np.array([-6.35e-5]),
}
""" Initial position of MIMo. Specifies initial values for all joints.
We grabbed these values by posing MIMo using the MuJoCo simulate executable and the positional actuator file.
We need these not just for the initial position but also resetting the position each step.

:meta hide-value:
"""

class MIMoExperimentEnv(MIMoEnv):
    """ MIMo stands up using crib railings as an aid.

    Attributes and parameters are the same as in the base class, but the default arguments are adapted for the scenario.
    Specifically we have :attr:`.done_active` and :attr:`.goals_in_observation` as `False` and touch and vision sensors
    disabled.

    Even though we define a success condition in :meth:`~mimoEnv.envs.standup.MIMoStandupEnv._is_success`, it is
    disabled since :attr:`.done_active` is set to `False`. The purpose of this is to enable extra information for
    the logging features of stable baselines.

    """
    def __init__(self,
                 model_path=EXPERIMENT_XML,
                 initial_qpos=LYING_POSITION,
                 n_substeps=2,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 touch_params=DEFAULT_TOUCH_PARAMS,
                 pain_params=DEFAULT_PAIN_PARAMS,
                 vision_params=None,
                 vestibular_params=DEFAULT_VESTIBULAR_PARAMS,
                 ):

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         pain_params=pain_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=False,
                         done_active=False)

        self.init_lying_position = self.sim.data.qpos.copy()
        self.steps = 0

    def _step_callback(self):
        self.steps += 1

    def _is_success(self, achieved_goal, desired_goal):

        return False

    def _is_failure(self, achieved_goal, desired_goal):

        return False

    def _sample_goal(self):

        return np.zeros((0,))

    def _get_achieved_goal(self):

        return np.zeros(self.goal.shape)

    def compute_reward(self, achieved_goal, desired_goal, info):
        return 0

    def _reset_sim(self):
        """ Resets the simulation.

        Return the simulation to the XML state, then slightly randomize all joint positions. Afterwards we let the
        simulation settle for a fixed number of steps. This leads to MIMo settling into a slightly random sitting or
        crouching position.

        Returns:
            bool: `True`
        """

        self.sim.set_state(self.initial_state)
        default_state = self.sim.get_state()
        qpos = self.init_lying_position

        # set initial positions stochastically
        qpos[7:] = qpos[7:] + self.np_random.uniform(low=-0.01, high=0.01, size=len(qpos[7:]))

        # set initial velocities to zero
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            default_state.time, qpos, qvel, default_state.act, default_state.udd_state
        )
        self.sim.set_state(new_state)
        self.reset_colours()
        self.sim.forward()

        # perform 100 steps with no actions to stabilize initial position
        actions = np.zeros(self.action_space.shape)
        self._set_action(actions)
        for _ in range(100):
            self.sim.step()

        return True

    def reset_colours(self):
        original = [[0.5, 0.5, 0.5, 1.], [0.,  0.,  0.9, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.],
                    [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.],
                    [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.],
                    [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.],
                    [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.],
                    [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.],
                    [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.],
                    [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.], [0.5, 0.5, 0.5, 1.]]

        colors = self.sim.model.geom_rgba
        for i, original_colour in enumerate(original):
            colors[i] = original_colour