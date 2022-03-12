import os
import numpy as np
import copy
from gym import utils
from gym import spaces
import mujoco_py

from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as env_utils

TOUCH_PARAMS = {
    "scales": {
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
        "left_upper_arm": 0.024,
        "left_lower_arm": 0.024,
    },
    "touch_function": "force_vector",
    "adjustment_function": "spread_linear",
}

MIMO_XML = os.path.abspath(os.path.join(__file__, "..", "..", "assets", "selfbody_scene.xml"))

class MIMoSelfBodyEnv(MIMoEnv):

    def __init__(self,
                 model_path=MIMO_XML,
                 initial_qpos={},
                 n_actions=8,  # Currently hardcoded
                 n_substeps=1,
                 touch_params=TOUCH_PARAMS,
                 vision_params=None,
                 vestibular_params=None,
                 goals_in_observation=True,
                 done_active=False):

        self.steps = 0
        self.init_sitting_qpos = np.array([
            0.0579584, -0.00157173, 0.0566738, 0.892294, -0.0284863, -0.450353, -0.0135029, 0.039088, 0.113112, 0.5323,
            0, 0, 0.5323, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.683242, 0.3747, -0.62714, -0.756016, 0.28278, 
            0, 0, -0.461583, -1.51997, -0.397578, 0.0976615, -1.85479, -0.585865, -0.358165, 0, 0, -1.23961, -0.8901,
            0.7156, -2.531, -0.63562, 0.5411, 0.366514, 0.24424
        ])
        self.target_geom = 0

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_actions=n_actions,
                         n_substeps=n_substeps,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)

        # New observation space
        obs = self._get_obs()
        
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
        spaces_dict["touch"] = spaces.Box(
                    -np.inf, np.inf, shape=obs["touch"].shape, dtype="float32"
            )
        spaces_dict["target_geom"] = spaces.Box(
                    -np.inf, np.inf, shape=obs['target_geom'].shape, dtype="float32"
            )

        self.observation_space = spaces.Dict(spaces_dict)


    def _get_obs(self):
        """Returns the observations."""
        # robot proprioception:
        proprio_obs = self._get_proprio_obs()

        observation_dict = {
            "observation": proprio_obs,
            "achieved_goal": np.array([0]),
            "desired_goal": np.array([0])
        }

        # robot touch sensors:
        if self.touch:
            touch_obs = self._get_touch_obs().ravel()
            observation_dict["touch"] = touch_obs

        target_geom = np.zeros(37)  # 36 geoms in MIMo
        if isinstance(self.target_geom,int):
            target_geom[self.target_geom] = 1
        observation_dict['target_geom'] = target_geom

        self.steps += 1
        return observation_dict

    def _set_action(self, action):
        ctrlrange = self.sim.model.actuator_ctrlrange
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        self.sim.data.ctrl[:] = actuation_center + action * actuation_range
        self.sim.data.ctrl[:] = np.clip(
            self.sim.data.ctrl, ctrlrange[:, 0], ctrlrange[:, 1]
        )

    def _is_failure(self, achieved_goal, desired_goal):
        return False

    def _sample_goal(self):
        """Samples a new goal and returns it."""
        # TODO: Actually sample a goal
        return np.zeros(1)

    def _get_achieved_goal(self):
        """Get the goal state actually achieved in this episode/timeframe."""
        # TODO: All of it
        return np.zeros(1)

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

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """

        self.sim.set_state(self.initial_state)
        self.sim.forward()
        
        # perform 100 random actions 
        for _ in range(100):
            action = self.action_space.sample()
            self._set_action(action)
            self.sim.step()
            self._step_callback()

        # set qpos as new initial position and velocity as zero
        qpos = self.init_sitting_qpos
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            self.initial_state.time, qpos, qvel, self.initial_state.act, self.initial_state.udd_state
        )

        self.sim.set_state(new_state)
        self.sim.forward()

        # randomly select body part as target
        active_geom_codes = list(self.touch.sensor_outputs.keys())
        target_geom_idx = np.random.randint(len(active_geom_codes))
        
        self.target_geom = active_geom_codes[int(target_geom_idx)]
        for body_id in self.touch.sensor_scales:
            body_geoms = env_utils.get_geoms_for_body(self.sim.model, body_id)
            if self.target_geom in body_geoms:
                self.target_body = self.sim.model.body_id2name(body_id)
        print('Target body: ', self.target_body)

        return True

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()

        obs = self._get_obs()
        #print(obs)

        # check if contact with target sensor:
        target_geom_touch_max = np.max(self.touch.sensor_outputs[self.target_geom])
        contact_with_target_geom = (target_geom_touch_max > 0)
        done = contact_with_target_geom
        
        # compute reward:
        target_body_pos = self.sim.data.get_body_xpos(self.target_body)
        fingers_pos = self.sim.data.get_body_xpos("right_fingers")
        distance = np.linalg.norm(fingers_pos - target_body_pos)
        reward = -2*distance + 500*contact_with_target_geom
        
        info = {
            'is_success': done,
            'target_geom': self.target_geom,
            'target_body': self.target_body,
            'target_touch': target_geom_touch_max
        }

        if contact_with_target_geom:
            print(info)

        # manually set body to sitting position (except for the right arm joints)
        self.sim.data.qpos[:19] = self.init_sitting_qpos[:19]
        self.sim.data.qpos[27:] = self.init_sitting_qpos[27:]

        return obs, reward, done, info