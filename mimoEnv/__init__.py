from gym.envs.registration import register

register(id='MIMo-v0',
         entry_point='mimoEnv.envs:MIMoTestEnv',
         )

register(id='MIMoDemo-v0',
         entry_point='mimoEnv.envs:MIMoDemoEnv',
         )