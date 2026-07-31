import numpy as np

from nee import BoundaryData, picard_sweep
from nee.discretization.double_null_mesh import Discretization
from nee.equations.vacuum import VacuumEquations
from nee.exact_solutions import vacuum_benchmarks
from nee.solver.backend import from_numerical


def test_picard_sweep_preserves_inputs_and_returns_complete_state() -> None:
    grid = vacuum_benchmarks._grid(30, 3)
    u = np.linspace(-1.0, -0.5, 5)
    v = np.linspace(0.0, 0.2, 5)
    exact, _ = vacuum_benchmarks.regular_schwarzschild_state(grid, u, v, 1.0)
    outgoing, incoming = vacuum_benchmarks.characteristic_data(grid, exact)
    boundary = BoundaryData.create(outgoing, incoming)
    state = from_numerical(vacuum_benchmarks.face_compatible_seed(exact, u), grid)
    state_bytes = {name: value.tobytes() for name, value in state.arrays().items()}
    boundary_hash = boundary.content_hash
    result = picard_sweep(
        state,
        boundary,
        Discretization(grid, u, v, metric_substeps=1),
        VacuumEquations(),
    )
    assert set(result.state.arrays()) == set(state.arrays())
    assert result.weighted_update > 0.0
    assert result.maximum_update_map > 0.0
    for name, value in state.arrays().items():
        assert value.tobytes() == state_bytes[name]
    assert boundary.content_hash == boundary_hash
    boundary.verify_unchanged()
