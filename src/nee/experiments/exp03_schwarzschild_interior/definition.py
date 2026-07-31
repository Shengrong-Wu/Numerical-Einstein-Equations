from nee.experiments.base import ExperimentDefinition
from nee.experiments.campaigns import schwarzschild_interior

DEFINITION = ExperimentDefinition("exp03", "Schwarzschild interior", "vacuum", schwarzschild_interior)
