import numpy as np

from nee.geometry.curved_sphere import solve


def test_curved_picard_backtracks_before_losing_positive_radius() -> None:
    solution = solve(2.0**-6, 17, 17, iterations=3)

    assert np.all(solution.radius > 0.0)
    assert any(record["relaxation"] < 1.0 for record in solution.records)
