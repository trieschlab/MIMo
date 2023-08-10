from gymnasium.envs.registration import register
from mimoEnv.envs.dummy import DEMO_XML

register(id='MIMoBench-v0',
         entry_point='mimoEnv.envs:MIMoDummyEnv',
         max_episode_steps=6000,
         )

register(id='MIMoBenchV2-v0',
         entry_point='mimoEnv.envs:MIMoV2DummyEnv',
         max_episode_steps=6000,
         )

register(id='MIMoShowroom-v0',
         entry_point='mimoEnv.envs:MIMoV2DummyEnv',
         max_episode_steps=500,
         kwargs={"model_path": DEMO_XML,
                 "render_mode": "human",
                 },
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

register(id='MIMoCatch-v0',
         entry_point='mimoEnv.envs:MIMoCatchEnv',
         max_episode_steps=800,
         )

register(id='MIMoMuscle-v0',
         entry_point='mimoEnv.envs:MIMoMuscleDummyEnv',
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
