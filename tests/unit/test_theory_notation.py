import numpy as np

from nee.initial_data.nonspherical_scalar import _config
from nee.exact_solutions import vacuum_benchmarks
from nee.experiments.exp08_scalar_trapped_section.campaign import (
    _finite_prefix_config,
    numerical_config,
)
from nee.numerics.coordinate_differentiation import high_order_differentiate
from nee.numerics.scalar_coordinates import mesh_from_config
from nee.numerics.scalar_initial_data import construct_initial_data
from nee.numerics.sphere import scalar_gradient


def test_incoming_lapse_coefficient_uses_Omega_e3() -> None:
    config = _config(
        name="theory-notation-unit",
        cap=0.01,
        lambda_omega=1.0,
        lambda_b=1.0,
        lambda_chi=1.0,
    )
    boundary, grid, _, mesh = construct_initial_data(config)
    incoming = boundary.incoming
    log_Omega = np.log(incoming["Omega"])
    Omega_e3_log_Omega = mesh.differentiate_u(log_Omega, axis=1)
    Omega_e3_log_Omega += np.einsum(
        "nui,nui->nu",
        incoming["b"],
        scalar_gradient(grid, log_Omega),
    )

    np.testing.assert_allclose(
        incoming["Omega_omegab"],
        -0.5 * Omega_e3_log_Omega,
        rtol=0.0,
        atol=1.0e-7,
    )
    assert np.max(
        np.abs(
            incoming["Omega_omegab"]
            + 0.5 * Omega_e3_log_Omega / incoming["Omega"]
        )
    ) > 1.0e-3


def test_outgoing_raychaudhuri_evolves_Omega_trchi() -> None:
    grid = vacuum_benchmarks._grid(30, 3)
    u = np.linspace(-4.0, -3.0, 31)
    v = np.linspace(0.0, 0.5, 101)
    state, _ = vacuum_benchmarks.regular_schwarzschild_state(
        grid, u, v, 1.0
    )

    Omega_e4_Omega_trchi = high_order_differentiate(
        state.Omega_trchi, v, axis=2
    )
    raychaudhuri = (
        Omega_e4_Omega_trchi
        + 0.5 * state.Omega_trchi**2
        + 4.0 * state.Omega_omega * state.Omega_trchi
    )
    metric_v = high_order_differentiate(state.g, v, axis=2)
    metric_rhs = (
        state.Omega_trchi[..., None, None] * state.g
        + 2.0 * state.Omega_chih
    )

    np.testing.assert_allclose(
        raychaudhuri[:, :, 5:-5], 0.0, rtol=0.0, atol=2.0e-12
    )
    np.testing.assert_allclose(
        metric_v[:, :, 5:-5],
        metric_rhs[:, :, 5:-5],
        rtol=0.0,
        atol=2.0e-10,
    )


def test_exp08_finite_prefix_preserves_whole_base_elements() -> None:
    base = numerical_config(control=False, continued=False, quick=False)
    prefix = _finite_prefix_config(base)
    base_mesh = mesh_from_config(base.scalar_coordinates)
    prefix_mesh = mesh_from_config(prefix.scalar_coordinates)

    assert prefix.scalar_coordinates.tau_elements == 4
    assert prefix.scalar_coordinates.u_right == base_mesh.u[32]
    np.testing.assert_array_equal(prefix_mesh.u, base_mesh.u[:33])
    np.testing.assert_array_equal(prefix_mesh.v, base_mesh.v)
