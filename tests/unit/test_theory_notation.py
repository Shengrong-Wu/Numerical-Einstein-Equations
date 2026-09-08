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


def test_weighted_raychaudhuri_coefficients_and_first_order_diagnostic() -> None:
    from nee.numerics.vacuum_residual import components
    grid = vacuum_benchmarks._grid(30, 3)
    u, v = np.linspace(-4.0, -3.0, 31), np.linspace(0.0, 0.5, 31)
    state, _ = vacuum_benchmarks.regular_schwarzschild_state(grid, u, v, 1.0)
    values = components(grid, state, u, v, mode='first_order', include_gauss_curvature=False)
    np.testing.assert_allclose(values['Ric44'][:, 4:-4, 4:-4], 0, atol=2e-10)
    np.testing.assert_allclose(values['Omega2_Ric33'][:, 4:-4, 4:-4], 0, atol=2e-9)
    import copy
    wrong = copy.deepcopy(state)
    wrong.Omega_omega *= 0.5
    changed = components(grid, wrong, u, v, mode='first_order', include_gauss_curvature=False)
    assert np.max(np.abs(changed['Ric44'][:, 4:-4, 4:-4])) > 1e-4


def test_harmonic_normalization_is_independent_of_other_sample_points() -> None:
    from nee.initial_data.nonspherical_scalar import _rotated_harmonic_fields
    points = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0.6, 0., 0.8]])
    small = _rotated_harmonic_fields(points[3:], np.eye(3))
    larger = _rotated_harmonic_fields(points, np.eye(3))
    for a, b in zip(small, larger):
        np.testing.assert_allclose(a, b[3:], atol=0, rtol=0)


def test_first_order_mode_never_differentiates_outgoing_shear_in_v(monkeypatch) -> None:
    from nee.numerics import vacuum_residual
    grid = vacuum_benchmarks._grid(30, 3)
    u, v = np.linspace(-1, -0.5, 9), np.linspace(0, 0.1, 9)
    state, _ = vacuum_benchmarks.regular_schwarzschild_state(grid, u, v, 1.0)
    differentiate = vacuum_residual._differentiate_v
    def checked(value, *args, **kwargs):
        assert value.ndim != 5, 'outgoing tensor derivative would require extra null regularity'
        return differentiate(value, *args, **kwargs)
    monkeypatch.setattr(vacuum_residual, '_differentiate_v', checked)
    result = vacuum_residual.components(grid, state, u, v, mode='first_order', include_gauss_curvature=False)
    assert np.all(np.isfinite(result['Ric44']))


def test_first_derivative_closures_detect_a_wrong_weighted_lapse():
    from copy import deepcopy
    from nee.solver.backend import from_numerical
    from nee.diagnostics.construction_residuals import evaluate
    grid = vacuum_benchmarks._grid(30, 3)
    u, v = np.linspace(-4., -3., 31), np.linspace(0., .5, 31)
    exact, _ = vacuum_benchmarks.regular_schwarzschild_state(grid, u, v, 1.)
    state = from_numerical(exact, grid)
    baseline = evaluate(grid, state, u, v, halo=4)
    assert baseline['outgoing_lapse']['masked_maximum'] < 1e-10
    assert baseline['incoming_lapse']['masked_maximum'] < 1e-10
    assert baseline['shift_torsion']['masked_maximum'] < 1e-12
    changed = deepcopy(state)
    changed.Omega_omega *= 0.5
    assert evaluate(grid, changed, u, v, halo=4)['outgoing_lapse']['masked_maximum'] > 1e-3
