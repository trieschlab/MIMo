import os
import numpy as np
import copy
import mujoco_py

import mimoEnv.utils as mimo_utils
from mimoEnv.envs.mimo_env import MIMoEnv, DEFAULT_PROPRIOCEPTION_PARAMS

SELFBODY_XML = os.path.abspath(os.path.join(__file__, "..", "..", "assets", "selfbody_scene.xml"))

TOUCH_PARAMS = {
    "scales": {
        "left_foot": 0.03,
        "right_foot": 0.03,
        "left_lower_leg": 0.05,
        "right_lower_leg": 0.05,
        "left_upper_leg": 0.05,
        "right_upper_leg": 0.05,
        "left_upper_arm": 0.05,
        "left_lower_arm": 0.05,
    },
    "touch_function": "force_vector",
    "response_function": "spread_linear",
}

PROPRIOCEPTION_PARAMS = {
    "components": [],
}

class MIMoSelfbodyEnv(MIMoEnv):

    def __init__(self,
                 model_path=SELFBODY_XML,
                 initial_qpos={},
                 n_substeps=1,
                 proprio_params=PROPRIOCEPTION_PARAMS,
                 touch_params=TOUCH_PARAMS,
                 vision_params=None,
                 vestibular_params=None,
                 goals_in_observation=False,
                 done_active=False):

        self.steps = 0

        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         n_substeps=n_substeps,
                         proprio_params=proprio_params,
                         touch_params=touch_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active)

    def _sample_goal(self):
        """ Dummy function """
        return np.zeros(self._get_proprio_obs().shape)

    def _get_achieved_goal(self):
        """ Dummy function """
        return np.zeros(self._get_proprio_obs().shape)

    def _reset_sim(self):
        """Resets a simulation and indicates whether or not it was successful.
        If a reset was unsuccessful (e.g. if a randomized state caused an error in the
        simulation), this method should indicate such a failure by returning False.
        In such a case, this method will be called again to attempt a the reset again.
        """

        qpos = np.array([-0.0542433, -0.0130054, 0.0484481, 0.999891, -0.00672267, 0.0112609, 0.00679211, 0.0136196, -0.00865072, 0.088785, 0.00723084, -0.0109759, 0.02661, 0, 0.51776, -0.01222, -5.14089e-08, -1.30871e-07, -3.28495e-09, -0.0063788, 0.00766752, 0.00977246, -0.132022, 2.2387, -1.23551, -1.38848, 0.03142, 0.012222, 0.035726, 0.1396, -0.4887, -0.0214, 0.488205, -0.0444935, -1.06828, -0.130175, -0.025972, -0.520235, -2.321, -0.8901, 0.620043, -1.64672, -0.444892, -0.00708538, 0.0123599, 7.45989e-05, -2.321, -0.8901, 0.7156, -1.84179, -0.162948, 0.004892, -0.3491, 9.51667e-05])
        qvel = np.zeros(self.sim.data.qvel.shape)

        new_state = mujoco_py.MjSimState(
            self.initial_state.time, qpos, qvel, self.initial_state.act, self.initial_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

        # perform 10 random actions
        for _ in range(10):
            action = self.action_space.sample()
            self._set_action(action)
            self.sim.step()
            self._step_callback()
        
        # randomly select body as target
        active_geom_codes = list(self.touch.sensor_outputs.keys())
        target_geom_idx = np.random.randint(len(active_geom_codes))
        target_geom = active_geom_codes[int(target_geom_idx)]
        self.target_geom = target_geom

        for body_id in self.touch.sensor_scales:
            body_geoms = mimo_utils.get_geoms_for_body(self.sim.model, body_id)
            if target_geom in body_geoms:
                self.target_body = self.sim.model.body_id2name(body_id)
                self.target_geoms =  mimo_utils.get_geoms_for_body(self.sim.model, body_id)
        print('Target body: ', self.target_body)

        return True

    def _step_callback(self):
        # reset body positions (except torso and right arm)
        self.sim.data.qpos[:10] = np.array([-0.0542433, -0.0130054, 0.0484481, 0.999891, -0.00672267, 0.0112609, 0.00679211, 0.0136196, -0.00865072, 0.088785])
        self.sim.data.qpos[13:22] = np.array([0, 0.51776, -0.01222, -5.14089e-08, -1.30871e-07, -3.28495e-09, -0.0063788, 0.00766752, 0.00977246])
        self.sim.data.qpos[-24:] = np.array([-0.4887, -0.0214, 0.488205, -0.0444935, -1.06828, -0.130175, -0.025972, -0.520235, -2.321, -0.8901, 0.620043, -1.64672, -0.444892, -0.00708538, 0.0123599, 7.45989e-05, -2.321, -0.8901, 0.7156, -1.84179, -0.162948, 0.004892, -0.3491, 9.51667e-05])

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        
        # check if contact with target sensor:
        obs = self._get_obs()
        target_geom_touch_max = 0
        #for geom in self.target_geoms:
        #    target_geom_touch_max = max(target_geom_touch_max, np.max(self.touch.sensor_outputs[geom]))
        target_geom_touch_max = np.max(self.touch.sensor_outputs[self.target_geom])
        contact_with_target_geom = (target_geom_touch_max > 0)
        done = contact_with_target_geom

        # get observation with fake touch in target sensor:
        touch_obs = self.touch.sensor_outputs
        #for geom in self.target_geoms:
        #    touch_obs[geom] += 999 * np.ones(touch_obs[geom].shape)
        touch_obs[self.target_geom] += 999 * np.ones(touch_obs[self.target_geom].shape)
        touch_obs = self.touch.flatten_sensor_dict(touch_obs)
        obs["touch"] = touch_obs.ravel()

        # compute reward based on distance to target:
        target_pos = self.sim.data.geom_xpos[self.target_geom]
        #target_pos = self.sim.data.get_body_xpos(self.target_body)
        fingers_pos = self.sim.data.get_body_xpos("right_fingers")
        distance = np.linalg.norm(fingers_pos - target_pos)
        if contact_with_target_geom:
            reward = 200
        else:
            reward = - distance

        # fill out information dictionary:
        info = {
            'target_body': self.target_body,
            'target_geom': self.target_geom,
            'target_touch': target_geom_touch_max
        }
        if contact_with_target_geom:
            print(info)

        return obs, reward, done, info