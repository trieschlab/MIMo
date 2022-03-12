from gym.envs.registration import register

register(id='MIMo-v0',
         entry_point='mimoEnv.envs:MIMoTestEnv',
         )

register(id='MIMoSelfBody-v0',
         entry_point='mimoEnv.envs:MIMoSelfBodyEnv',
         max_episode_steps=500, 
         )