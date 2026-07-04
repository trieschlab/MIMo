from gymnasium.envs.registration import register, registry
from mimo_infant.simulation.envs.dummy import DEMO_XML

def safe_register(id: str, **kwargs):
    if id not in registry:
        register(id=id, **kwargs)

def register_mimo_envs():
    safe_register(
        id='MIMoBench-v0',
        entry_point='mimo_infant.simulation.envs:MIMoDummyEnv',
        max_episode_steps=6000,
    )

    safe_register(
        id='MIMoBenchV2-v0',
        entry_point='mimo_infant.simulation.envs:MIMoV2DummyEnv',
        max_episode_steps=6000,
    )

    safe_register(
        id='MIMoShowroom-v0',
        entry_point='mimo_infant.simulation.envs:MIMoV2DummyEnv',
        max_episode_steps=500,
        kwargs={"model_path": DEMO_XML,
                "render_mode": "human",
        },
    )

    safe_register(
        id='MIMoReach-v0',
        entry_point='mimo_infant.simulation.envs:MIMoReachEnv',
        max_episode_steps=1000,
    )

    safe_register(
        id='MIMoStandup-v0',
        entry_point='mimo_infant.simulation.envs:MIMoStandupEnv',
        max_episode_steps=500, 
    )

    safe_register(
        id='MIMoSelfBody-v0',
        entry_point='mimo_infant.simulation.envs:MIMoSelfBodyEnv',
        max_episode_steps=500, 
    )

    safe_register(
        id='MIMoCatch-v0',
        entry_point='mimo_infant.simulation.envs:MIMoCatchEnv',
        max_episode_steps=800,
    )

    safe_register(
        id='MIMoMuscle-v0',
        entry_point='mimo_infant.simulation.envs:MIMoMuscleDummyEnv',
        max_episode_steps=6000,
    )

    safe_register(
        id='MIMoMuscleStaticTest-v0',
        entry_point='mimo_infant.simulation.envs:MIMoStaticMuscleTestEnv',
        max_episode_steps=5000,
    )

    safe_register(
        id='MIMoVelocityMuscleTest-v0',
        entry_point='mimo_infant.simulation.envs:MIMoVelocityMuscleTestEnv',
        max_episode_steps=3000,
    )

    safe_register(
        id='MIMoMuscleStaticTestV2-v0',
        entry_point='mimo_infant.simulation.envs:MIMoStaticMuscleTestV2Env',
        max_episode_steps=5000,
    )

    safe_register(
        id='MIMoVelocityMuscleTestV2-v0',
        entry_point='mimo_infant.simulation.envs:MIMoVelocityMuscleTestV2Env',
        max_episode_steps=3000,
    )

    safe_register(
        id='MIMoComplianceTest-v0',
        entry_point='mimo_infant.simulation.envs:MIMoComplianceEnv',
    )

    safe_register(
        id='MIMoComplianceMuscleTest-v0',
        entry_point='mimo_infant.simulation.envs:MIMoComplianceMuscleEnv',
    )

    safe_register(
        id='MIMoRollOver-v0',
        entry_point='mimo_infant.simulation.envs:MIMoRollOverEnv',
        max_episode_steps=500,
    )
