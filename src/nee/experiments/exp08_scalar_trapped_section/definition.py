from nee.experiments.base import ExperimentDefinition
from nee.experiments.campaigns import trapped_scalar

DEFINITION = ExperimentDefinition(
    "exp08",
    "Scalar-pulse trapped region and apparent horizon",
    "scalar_field",
    trapped_scalar,
)
