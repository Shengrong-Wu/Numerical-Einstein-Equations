from nee.experiments.base import ExperimentDefinition
from nee.experiments.campaigns import schwarzschild_horizon

DEFINITION = ExperimentDefinition("exp02", "Schwarzschild horizon test", "vacuum", schwarzschild_horizon)
