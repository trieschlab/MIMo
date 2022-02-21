from gym.envs.registration import register

register(id='MIMo-v0',
         #entry_point='mimoEnv.envs.mimo_test:MIMoTestEnv',
         entry_point='envs.mimo_test:MIMoTestEnv',
         )
