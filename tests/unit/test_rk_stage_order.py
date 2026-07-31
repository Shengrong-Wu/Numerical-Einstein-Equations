import numpy as np

from nee.numerics.coordinate_quadrature import midpoint_values, stage_value
from nee.numerics.vacuum_iteration import _quadratic_stage_value


def test_classical_rk_nodes_preserve_stage_operation_order() -> None:
    rng = np.random.default_rng(17)
    nodes = np.linspace(-1.0, -0.5, 9)
    values = rng.normal(size=(11, len(nodes), 7))
    midpoints = midpoint_values(values, nodes, axis=1)
    for index in range(len(nodes) - 1):
        for alpha in (0.0, 0.5, 1.0):
            established = stage_value(values, midpoints, index, alpha, axis=1)
            subdividable = _quadratic_stage_value(
                values, midpoints, index, alpha, axis=1
            )
            np.testing.assert_array_equal(subdividable, established)
