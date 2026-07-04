"""MIMo, the multimodal infant model."""

__version__ = "3.0.0"

# Environments
from mimo_infant.simulation.envs.mimo_env import MIMoEnv
from mimo_infant.simulation.envs.reach import MIMoReachEnv
from mimo_infant.simulation.envs.standup import MIMoStandupEnv
from mimo_infant.simulation.envs.selfbody import MIMoSelfBodyEnv
from mimo_infant.simulation.envs.catch import MIMoCatchEnv
from mimo_infant.simulation.envs.roll_over import MIMoRollOverEnv

# Actuation models
from mimo_infant.actuation.actuation import SpringDamperModel
from mimo_infant.actuation.actuation import PositionalModel
from mimo_infant.actuation.muscle import MuscleModel

# Other
import mimo_infant.simulation.simulate as simulate
from mimo_infant.growth.growth import adjust_mimo_to_age #growth
from mimo_infant.growth.scene import delete_growth_scene #growth
import mimo_infant.simulation.utils as mimo_utils #utils

__all__ = [
    "__version__",
    "MIMoEnv",
    "MIMoReachEnv",
    "MIMoStandupEnv",
    "MIMoSelfBodyEnv",
    "MIMoCatchEnv",
    "MIMoRollOverEnv",
    "SpringDamperModel",
    "PositionalModel",
    "MuscleModel",
    "simulate",
    "adjust_mimo_to_age",
    "delete_growth_scene",
    "mimo_utils",
]