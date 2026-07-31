import numpy as np

from nee.diagnostics.trapped_sections import (
    TrappedSectionParameters,
    incoming_spherical_data,
    prolong_exact_prefix,
    pulse_profile,
    trapped_sections,
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

