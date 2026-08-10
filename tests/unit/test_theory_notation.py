import numpy as np

from nee.initial_data.nonspherical_scalar import _config
from nee.exact_solutions import vacuum_benchmarks
from nee.experiments.exp08_scalar_trapped_section.campaign import (
    _mots_control_specs,
    _production_specs,
)
from nee.config import load_config
from nee.numerics.coordinate_differentiation import high_order_differentiate
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


def test_exp08_rectangles_lie_inside_configured_curved_region() -> None:
    config = load_config(
        "configs/experiments/exp08/standard.toml"
    )
    constant = float(config.physics["curved_constant"])
    exponent = float(config.physics["curved_exponent"])
    for spec in (*_production_specs(config), *_mots_control_specs(config)):
        assert spec.v_cap <= constant * (-spec.u_right) ** exponent + 1.0e-14
        assert spec.v_cap <= float(config.physics["v_max"])

    target_u = float(config.physics["target_u"])
    target_v = float(config.physics["target_v"])
    assert target_v < constant * (-target_u) ** exponent
