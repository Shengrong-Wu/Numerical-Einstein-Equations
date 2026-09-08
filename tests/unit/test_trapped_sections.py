import numpy as np

from nee.diagnostics.trapped_sections import trapped_sections
from nee.initial_data.scalar_pulse import (
    TrappedSectionParameters,
    incoming_spherical_data,
    outgoing_scalar_data,
    prolong_exact_prefix,
    pulse_profile,
)


def test_planned_continuation_adds_one_equal_tau_element() -> None:
    parameters = TrappedSectionParameters()
    assert np.isclose(
        parameters.continued_tau_right - parameters.base_tau_right,
        parameters.tau_element_width,
    )
    assert np.isclose(parameters.continued_u_right, -0.0144360760729319)


def test_pulse_profile_has_exact_corner_and_is_increasing() -> None:
    values = pulse_profile(np.array([0.0, 1e-8, 1e-4, 0.04]), TrappedSectionParameters())
    assert values[0] == 0.0
    assert np.all(np.diff(values) > 0.0)


def test_outgoing_scalar_perturbation_is_negative() -> None:
    parameters = TrappedSectionParameters()
    v = np.array([0.0, 0.01, 0.04])
    values = outgoing_scalar_data(v, parameters)
    perturbation = values - values[0]
    expected = -2.0 * (v / (v + 0.01)) ** 0.1
    expected[0] = 0.0
    np.testing.assert_allclose(perturbation, expected, rtol=0.0, atol=1.0e-15)


def test_incoming_analytic_face_identities() -> None:
    u = np.array([-1.0, -0.5, -0.02])
    values = incoming_spherical_data(u, TrappedSectionParameters())
    np.testing.assert_allclose(values["radius"], -u)
    assert np.all(values["omega_squared"] > 0.0)
    np.testing.assert_allclose(
        values["outgoing_inverse_lapse_trace"],
        values["outgoing_weighted_trace"] / values["omega_squared"],
    )


def test_prolongation_preserves_prefix_and_endpoint_correction() -> None:
    source = np.array([[1.0, 2.0, 4.0]])
    seed = np.array([[10.0, 20.0, 30.0, 31.0, 32.0]])
    prolonged = prolong_exact_prefix(source, seed, 3, u_axis=1)
    np.testing.assert_array_equal(prolonged[:, :3], source)
    np.testing.assert_array_equal(prolonged[:, 3:], np.array([[5.0, 6.0]]))


def test_endpoint_halo_excludes_candidates() -> None:
    outgoing = -np.ones((9, 4))
    incoming = -np.ones((9, 4))
    trapped, protected = trapped_sections(outgoing, incoming, u_endpoint_halo=3)
    assert np.all(trapped)
    assert not np.any(protected[:3])
    assert not np.any(protected[-3:])
    assert np.all(protected[3:-3])


def test_mots_acceptance_checks_more_than_solver_success() -> None:
    import copy
    from nee.experiments.exp08_scalar_trapped_section.horizon import acceptance_errors
    valid = {'nonlinear_success': True, 'sign_check': {'bracketed': True},
             'surface': {'h_min': -0.6, 'h_max': -0.5, 'area': 3.0,
                         'distance_to_patch_right': 0.1,
                         'theta_out': {'linf': 1e-6}, 'theta_in': {'maximum': -1.0}}}
    assert not acceptance_errors(valid)
    for field, value in [('nonlinear_success', False), ('sign_check', {'bracketed': False})]:
        invalid = copy.deepcopy(valid)
        invalid[field] = value
        assert acceptance_errors(invalid)
    invalid = copy.deepcopy(valid)
    invalid['surface']['theta_out']['linf'] = 0.01
    assert acceptance_errors(invalid)
