from gym.envs.registration import register

register(id='MIMo-v0',
         entry_point='mimoEnv.envs:MIMoTestEnv',
         )

register(id='MIMoDemo-v0',
         entry_point='mimoEnv.envs:MIMoDemoEnv',
         )
         
register(id='MIMoGrasp-v0',
         entry_point='mimoEnv.envs:MIMoGraspEnv',
         )
