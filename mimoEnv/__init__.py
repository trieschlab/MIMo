from gym.envs.registration import register


register(id='MIMoBench-v0',
         entry_point='mimoEnv.envs:MIMoDummyEnv',
         max_episode_steps=6000,
         )

register(id='MIMoBenchV2-v0',
         entry_point='mimoEnv.envs:MIMoV2DemoEnv',
         max_episode_steps=6000,
         )

register(id='MIMoV2Demo-v0',
         entry_point='mimoEnv.envs:MIMoV2DemoEnv',
         max_episode_steps=500,
         )

register(id='MIMoShowroom-v0',
         entry_point='mimoEnv.envs:MIMoShowroomEnv',
         max_episode_steps=500,
         )

register(id='MIMoReach-v0',
         entry_point='mimoEnv.envs:MIMoReachEnv',
         max_episode_steps=1000,
         )

register(id='MIMoStandup-v0',
         entry_point='mimoEnv.envs:MIMoStandupEnv',
         max_episode_steps=500, 
         )

register(id='MIMoSelfBody-v0',
         entry_point='mimoEnv.envs:MIMoSelfBodyEnv',
         max_episode_steps=500, 
         )

register(id='MIMoMuscle-v0',
         entry_point='mimoEnv.envs:MIMoMuscleDemoEnv',
         max_episode_steps=6000,
         )

register(id='MIMoMuscleStaticTest-v0',
         entry_point='mimoEnv.envs:MIMoStaticMuscleTestEnv',
         max_episode_steps=5000,
         )

register(id='MIMoVelocityMuscleTest-v0',
         entry_point='mimoEnv.envs:MIMoVelocityMuscleTestEnv',
         max_episode_steps=3000,
         )

register(id='MIMoMuscleStaticTestV2-v0',
         entry_point='mimoEnv.envs:MIMoStaticMuscleTestV2Env',
         max_episode_steps=5000,
         )

register(id='MIMoVelocityMuscleTestV2-v0',
         entry_point='mimoEnv.envs:MIMoVelocityMuscleTestV2Env',
         max_episode_steps=3000,
         )

register(id='MIMoComplianceTest-v0',
         entry_point='mimoEnv.envs:MIMoComplianceEnv',
         )

register(id='MIMoComplianceMuscleTest-v0',
         entry_point='mimoEnv.envs:MIMoComplianceMuscleEnv',
         )


