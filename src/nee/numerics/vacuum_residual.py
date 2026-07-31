"""First-derivative null Ricci diagnostics for the numerical state."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent

from .coordinate_differentiation import high_order_differentiate  # noqa: E402
from .vacuum_iteration import (  # noqa: E402
    FirstOrderState,
    section_geometry,
)
from .lgl import CharacteristicLGLMesh  # noqa: E402
from .ricci_residual import (  # noqa: E402
    one_form_lie_derivative,
)
from .sphere import (  # noqa: E402
    PointSphereGrid,
    scalar_gradient,
    tangent_inverse,
    tensor_divergence,
    tensor_norm_sq,
    tensor_trace,
    tensor_tracefree,
    vector_divergence,
)


Array = np.ndarray


def _context_array(
    context: dict[str, object],
    name: str,
    *aliases: str,
) -> Array:
    """Return one required construction array with a useful error message."""

    for candidate in (name, *aliases):
        if candidate in context:
            return np.asarray(context[candidate])
    choices = ", ".join(repr(candidate) for candidate in (name, *aliases))
    raise KeyError(f"construction context is missing {choices}")


def _assert_roundoff_match(
    name: str,
    actual: Array,
    expected: Array,
) -> None:
    """Reject a context/state pairing that is not the same Picard sweep."""

    if actual.shape != expected.shape:
        raise ValueError(
            f"{name} shape mismatch: {actual.shape} != {expected.shape}"
        )
    scale = max(
        1.0,
        float(np.max(np.abs(actual))),
        float(np.max(np.abs(expected))),
    )
    error = float(np.max(np.abs(actual - expected)))
    tolerance = 4096.0 * np.finfo(float).eps * scale
    if not np.isfinite(error) or error > tolerance:
        raise ValueError(
            f"{name} is inconsistent with the supplied S^N -> S^(N+1) "
            f"construction context: max error={error:.12g}, "
            f"roundoff tolerance={tolerance:.12g}"
        )


def _context_applied_angular_projection(
    context: dict[str, object], label: str
) -> bool:
    """Whether a stored source passed through the modal Galerkin projector."""

    tails = context.get("projection_tails")
    return isinstance(tails, dict) and label in tails


def _project_tangent_vector(grid: PointSphereGrid, vector: Array) -> Array:
    return np.einsum("nij,n...j->n...i", grid.projector, vector)


def _project_symmetric_tangent_tensor(
    grid: PointSphereGrid, tensor: Array
) -> Array:
    tangent = np.einsum(
        "nia,n...ab,nbj->n...ij",
        grid.projector,
        tensor,
        grid.projector,
    )
    return 0.5 * (tangent + np.swapaxes(tangent, -1, -2))


def _construction_omegab(
    grid: PointSphereGrid,
    state: FirstOrderState,
    previous_state: FirstOrderState,
    context: dict[str, object],
    grad_log_omega: Array,
) -> dict[str, Array]:
    """Recover the full integer-step incoming lapse coefficient.

    The context is produced by ``PicardStep(S^N)`` and therefore constructs
    the supplied ``state=S^(N+1)``.  Its half iterate is not a geometric Ricci
    coefficient.  We independently apply the shift correction here, then
    verify the duplicated full value/source stored by the sweep.  A modal
    Galerkin run projects the corrected products once more; in that case the
    stored projected values are authoritative and the raw-to-projected
    differences are returned as explicit truncation diagnostics.
    """

    half = _context_array(context, "weighted_omegab_half")
    half_source = _context_array(
        context, "omegab_half_source", "omegab_source"
    )
    shift_difference = state.shift - previous_state.shift
    full_raw = half - 0.5 * np.einsum(
        "n...i,n...i->n...", shift_difference, grad_log_omega
    )

    new_shift_source = -4.0 * state.omega[..., None] ** 2 * state.zeta_up
    old_shift_source = (
        -4.0
        * previous_state.omega[..., None] ** 2
        * previous_state.zeta_up
    )
    shift_source_difference = new_shift_source - old_shift_source
    full_source_raw = half_source - 0.5 * (
        np.einsum(
            "n...i,n...i->n...",
            shift_source_difference,
            grad_log_omega,
        )
        - 2.0
        * np.einsum(
            "n...i,n...i->n...",
            shift_difference,
            scalar_gradient(grid, state.weighted_omega),
        )
    )

    full = _context_array(context, "weighted_omegab_full")
    full_source = _context_array(context, "omegab_full_source")
    _assert_roundoff_match(
        "S^(N+1).weighted_omegab versus context full iterate",
        state.weighted_omegab,
        full,
    )
    if not _context_applied_angular_projection(
        context, "weighted_omegab_full"
    ):
        _assert_roundoff_match(
            "full weighted_omegab reconstructed from the half iterate",
            full,
            full_raw,
        )
    if not _context_applied_angular_projection(
        context, "weighted_omegab_full_source"
    ):
        _assert_roundoff_match(
            "full weighted_omegab source reconstructed from the half source",
            full_source,
            full_source_raw,
        )
    return {
        "value": full,
        "source": full_source,
        "raw_value": full_raw,
        "raw_source": full_source_raw,
        "value_projection_defect": full - full_raw,
        "source_projection_defect": full_source - full_source_raw,
    }


def _differentiate_u(
    value: Array,
    u: Array,
    scalar_coordinates: CharacteristicLGLMesh | None,
) -> Array:
    if scalar_coordinates is None:
        return high_order_differentiate(value, u, axis=1)
    return scalar_coordinates.differentiate_u(value, axis=1)


def _differentiate_v(
    value: Array,
    v: Array,
    scalar_coordinates: CharacteristicLGLMesh | None,
) -> Array:
    if scalar_coordinates is None:
        return high_order_differentiate(value, v, axis=2)
    return scalar_coordinates.differentiate_v(value, axis=2)


def _d3_scalar(
    grid: PointSphereGrid,
    scalar: Array,
    shift: Array,
    u: Array,
    scalar_coordinates: CharacteristicLGLMesh | None,
) -> Array:
    return _differentiate_u(scalar, u, scalar_coordinates) + np.einsum(
        "n...i,n...i->n...", shift, scalar_gradient(grid, scalar)
    )


def _d3_one_form(
    grid: PointSphereGrid,
    form: Array,
    shift: Array,
    weighted_chib_mixed: Array,
    u: Array,
    scalar_coordinates: CharacteristicLGLMesh | None,
) -> Array:
    coordinate_lie = _differentiate_u(form, u, scalar_coordinates)
    coordinate_lie += one_form_lie_derivative(grid, shift, form)
    return coordinate_lie - np.einsum(
        "n...ij,n...j->n...i", weighted_chib_mixed, form
    )


def _d3_covariant_tensor(
    grid: PointSphereGrid,
    tensor: Array,
    shift: Array,
    weighted_chib_mixed: Array,
    u: Array,
    scalar_coordinates: CharacteristicLGLMesh | None,
) -> Array:
    coordinate_lie = _differentiate_u(tensor, u, scalar_coordinates)
    derivative_tensor = grid.reference_derivative(tensor, tensor_rank=2)
    derivative_shift = grid.reference_derivative(shift, tensor_rank=1)
    coordinate_lie += np.einsum(
        "n...k,n...kij->n...ij", shift, derivative_tensor
    )
    coordinate_lie += np.einsum(
        "n...kj,n...ik->n...ij", tensor, derivative_shift
    )
    coordinate_lie += np.einsum(
        "n...ik,n...jk->n...ij", tensor, derivative_shift
    )
    coordinate_lie -= np.matmul(weighted_chib_mixed, tensor)
    coordinate_lie -= np.matmul(
        tensor, np.swapaxes(weighted_chib_mixed, -1, -2)
    )
    return coordinate_lie


def components(
    grid: PointSphereGrid,
    state: FirstOrderState,
    u: Array,
    v: Array,
    *,
    mode: str = "established",
    previous_state: FirstOrderState | None = None,
    construction_context: dict[str, object] | None = None,
    scalar_coordinates: CharacteristicLGLMesh | None = None,
    include_gauss_curvature: bool = True,
) -> dict[str, Array]:
    """Return the weighted null-Ricci component expressions.

    ``mode="construction"`` evaluates the Picard residual on
    ``state=S^(N+1)`` using the context returned by the *same* sweep
    ``PicardStep(S^N) -> (S^(N+1), context^N)``.  The supplied
    ``previous_state`` is therefore ``S^N``.  The half-step incoming lapse
    coefficient is converted to its full geometric value before it enters
    Ric33 or Ric3A.  Context/state invariants reject the common off-by-one
    mistake of evaluating ``S^N`` with ``context^N``.

    Construction sources replace only the v derivatives that their ODEs
    actually constructed.  All D3 and angular terms remain current-state
    calculations, so the result still measures Picard and compatibility
    defects.  Fresh v differentiation is retained in separately named
    closure diagnostics.
    ``mode="fresh"`` reconstructs both null lapse coefficients by
    differentiating ``log(Omega)``.  The
    ``mode="established"`` is retained so existing experiment summaries
    remain reproducible.
    """

    if mode not in {"established", "construction", "fresh"}:
        raise ValueError(f"unknown residual mode: {mode}")
    if mode == "construction" and (
        previous_state is None or construction_context is None
    ):
        raise ValueError(
            "construction mode requires previous_state and construction_context"
        )

    # The intrinsic Gauss curvature is needed only for the angular trace
    # sector.  Callers auditing the five first-order Ricci-coefficient
    # components can disable it and thereby avoid every second derivative of
    # the section metric.
    geometry = section_geometry(
        grid, state, include_curvature=include_gauss_curvature
    )
    inverse = geometry["inverse"]
    omega = state.omega
    omega_sq = omega**2
    weighted_tr_chi = omega_sq * state.q
    weighted_tr_chib = geometry["weighted_tr_chib"]
    weighted_hatchib = geometry["weighted_hatchib"]
    weighted_chib_mixed = np.matmul(state.weighted_chib, inverse)

    log_omega = np.log(omega)
    grad_log_omega = scalar_gradient(grid, log_omega)
    zeta = np.einsum("n...ij,n...j->n...i", state.metric, state.zeta_up)
    eta = geometry["eta"]
    etab = geometry["etab"]
    eta_up = np.einsum("n...ij,n...j->n...i", inverse, eta)
    etab_up = np.einsum("n...ij,n...j->n...i", inverse, etab)
    div_eta = vector_divergence(grid, eta_up, geometry["difference"])
    div_etab = vector_divergence(grid, etab_up, geometry["difference"])
    eta_norm = np.einsum("n...i,n...ij,n...j->n...", eta, inverse, eta)
    etab_norm = np.einsum("n...i,n...ij,n...j->n...", etab, inverse, etab)
    eta_etab = np.einsum("n...i,n...ij,n...j->n...", eta, inverse, etab)

    d3_weighted_tr_chi = _d3_scalar(
        grid, weighted_tr_chi, state.shift, u, scalar_coordinates
    )
    d3_weighted_tr_chib = _d3_scalar(
        grid, weighted_tr_chib, state.shift, u, scalar_coordinates
    )
    d3_shear = _d3_covariant_tensor(
        grid, state.shear, state.shift, weighted_chib_mixed, u, scalar_coordinates
    )
    d3_zeta = _d3_one_form(
        grid, zeta, state.shift, weighted_chib_mixed, u, scalar_coordinates
    )
    raw_metric_v = (
        weighted_tr_chi[..., None, None] * state.metric
        + 2.0 * state.shear
    )
    d4_zeta_fresh = _differentiate_v(zeta, v, scalar_coordinates)
    d4_weighted_omegab_fresh = _differentiate_v(
        state.weighted_omegab, v, scalar_coordinates
    )
    d4_weighted_tr_chib_fresh = _differentiate_v(
        weighted_tr_chib, v, scalar_coordinates
    )
    omegab_data: dict[str, Array] | None = None
    if mode == "construction":
        assert previous_state is not None
        assert construction_context is not None
        omegab_data = _construction_omegab(
            grid,
            state,
            previous_state,
            construction_context,
            grad_log_omega,
        )
        weighted_omegab = omegab_data["value"]
        d4_weighted_omegab = omegab_data["source"]
        weighted_omega = state.weighted_omega
        projected_zeta_source = _project_tangent_vector(
            grid, _context_array(construction_context, "zeta_source")
        )
        # Lower the newly constructed contravariant zeta with the current
        # metric.  The metric factor obeys its all-current kinematic equation;
        # a separately returned diagnostic records any Galerkin projection
        # defect in the stored metric source.
        d4_zeta_coordinate = np.einsum(
            "n...ij,n...j->n...i", raw_metric_v, state.zeta_up
        ) + np.einsum(
            "n...ij,n...j->n...i", state.metric, projected_zeta_source
        )
    elif mode == "fresh":
        weighted_omegab = -0.5 * _d3_scalar(
            grid, log_omega, state.shift, u, scalar_coordinates
        )
        weighted_omega = -0.5 * _differentiate_v(
            log_omega, v, scalar_coordinates
        )
        d4_weighted_omegab = _differentiate_v(
            weighted_omegab, v, scalar_coordinates
        )
        d4_zeta_coordinate = d4_zeta_fresh
    else:
        weighted_omegab = state.weighted_omegab
        weighted_omega = state.weighted_omega
        d4_weighted_omegab = d4_weighted_omegab_fresh
        d4_zeta_coordinate = d4_zeta_fresh

    div_shear = tensor_divergence(
        grid, state.shear, geometry["difference"], inverse
    )
    div_weighted_hatchib = tensor_divergence(
        grid, weighted_hatchib, geometry["difference"], inverse
    )

    weighted_ric33 = -(
        d3_weighted_tr_chib
        + 4.0 * weighted_omegab * weighted_tr_chib
        + 0.5 * weighted_tr_chib**2
        + tensor_norm_sq(weighted_hatchib, inverse)
    )
    weighted_ric3 = (
        d3_zeta
        + 1.5 * weighted_tr_chib[..., None] * zeta
        + np.einsum(
            "n...ij,n...j->n...i",
            np.matmul(weighted_hatchib, inverse),
            zeta,
        )
        + 2.0 * scalar_gradient(grid, weighted_omegab)
        + div_weighted_hatchib
        - 0.5 * scalar_gradient(grid, weighted_tr_chib)
        + weighted_tr_chib[..., None] * grad_log_omega
    )

    # The displayed covariant formula simplifies to -partial_v(zeta_A)
    # - (Omega tr chi) zeta_A after expanding nabla_4 on a one-form.
    weighted_ric4 = (
        -d4_zeta_coordinate
        - weighted_tr_chi[..., None] * zeta
        + 2.0 * scalar_gradient(grid, weighted_omega)
        + div_shear
        - 0.5 * scalar_gradient(grid, weighted_tr_chi)
        + weighted_tr_chi[..., None] * grad_log_omega
    )

    weighted_hat = (
        d3_shear
        + 0.5 * weighted_tr_chib[..., None, None] * state.shear
        - omega_sq[..., None, None]
        * (geometry["eta_grad_hat"] + geometry["eta_square_hat"])
        + 0.5
        * weighted_tr_chi[..., None, None]
        * weighted_hatchib
    )
    weighted_hat = tensor_tracefree(weighted_hat, state.metric, inverse)

    trace_combo = (
        d3_weighted_tr_chi
        + weighted_tr_chi * weighted_tr_chib
        - 2.0 * omega_sq * div_eta
        - 2.0 * omega_sq * eta_norm
        + 2.0 * omega_sq * geometry["curvature"]
    )
    if mode == "construction":
        assert construction_context is not None
        chib_v = _project_symmetric_tangent_tensor(
            grid, _context_array(construction_context, "chib_source")
        )
        inverse_v = -np.matmul(np.matmul(inverse, raw_metric_v), inverse)
        d4_weighted_tr_chib = tensor_trace(chib_v, inverse) + np.einsum(
            "n...ij,n...ij->n...",
            inverse_v,
            state.weighted_chib,
        )
    else:
        d4_weighted_tr_chib = d4_weighted_tr_chib_fresh
    trace_combo_from_4 = (
        d4_weighted_tr_chib
        + weighted_tr_chi * weighted_tr_chib
        - 2.0 * omega_sq * div_etab
        - 2.0 * omega_sq * etab_norm
        + 2.0 * omega_sq * geometry["curvature"]
    )
    shear_dot = np.einsum(
        "n...ik,n...jl,n...ij,n...kl->n...",
        inverse,
        inverse,
        state.shear,
        weighted_hatchib,
    )
    weighted_ric34 = (
        4.0 * d4_weighted_omegab
        - shear_dot
        - 0.5 * weighted_tr_chi * weighted_tr_chib
        + 4.0 * omega_sq * eta_etab
        - d3_weighted_tr_chi
        + 2.0 * omega_sq * div_eta
    )
    weighted_scalar = trace_combo - weighted_ric34
    shear_norm = tensor_norm_sq(state.shear, inverse)
    ric44_fresh_closure = (
        _differentiate_v(state.q, v, scalar_coordinates)
        + 0.5 * omega_sq * state.q**2
        + shear_norm / omega_sq
    )
    if mode == "construction":
        assert construction_context is not None
        ric44_construction_closure = (
            _context_array(construction_context, "raychaudhuri_source")
            + 0.5 * omega_sq * state.q**2
            + shear_norm / omega_sq
        )
    else:
        ric44_construction_closure = np.zeros_like(ric44_fresh_closure)
    if mode == "construction":
        assert construction_context is not None
        metric_source = _context_array(construction_context, "metric_source")
        metric_projection_defect = np.sqrt(
            np.maximum(
                tensor_norm_sq(
                    metric_source - raw_metric_v,
                    inverse,
                ),
                0.0,
            )
        )
        omegab_source_closure = (
            d4_weighted_omegab_fresh - d4_weighted_omegab
        )
        zeta_source_closure = d4_zeta_fresh - d4_zeta_coordinate
        chib_trace_source_closure = (
            d4_weighted_tr_chib_fresh - d4_weighted_tr_chib
        )
    else:
        metric_projection_defect = np.zeros_like(ric44_fresh_closure)
        omegab_source_closure = np.zeros_like(ric44_fresh_closure)
        zeta_source_closure = np.zeros_like(zeta)
        chib_trace_source_closure = np.zeros_like(ric44_fresh_closure)
    outgoing_lapse_closure = _differentiate_v(
        log_omega, v, scalar_coordinates
    ) + 2.0 * state.weighted_omega
    incoming_lapse_closure = (
        _d3_scalar(grid, log_omega, state.shift, u, scalar_coordinates)
        + 2.0 * state.weighted_omegab
    )
    return {
        "Omega2_Ric33": weighted_ric33,
        "Omega2_Ric34": weighted_ric34,
        "Omega_Ric3A": weighted_ric3,
        "Omega_Ric4A": weighted_ric4,
        "Omega2_hat_RicAB": weighted_hat,
        "Omega2_R_plus_Ric34": trace_combo,
        "Omega2_R_plus_Ric34_from_4": trace_combo_from_4,
        "Omega2_R": weighted_scalar,
        "Ric44": (
            np.zeros_like(ric44_fresh_closure)
            if mode == "construction"
            else (
                -ric44_fresh_closure
                if mode == "fresh"
                else np.zeros_like(ric44_fresh_closure)
            )
        ),
        "Ric44_construction_closure": ric44_construction_closure,
        # This is the angular semidiscretization defect that older diagnostics
        # conflated with the equation-enforced construction residual Ric44=0.
        "Ric44_semidiscrete_projection_defect": (
            -ric44_construction_closure
        ),
        "Ric44_fresh": -ric44_fresh_closure,
        "Ric44_fresh_closure": ric44_fresh_closure,
        "metric_projection_defect": metric_projection_defect,
        "outgoing_lapse_closure": outgoing_lapse_closure,
        "incoming_lapse_closure": incoming_lapse_closure,
        "omegab_source_closure": omegab_source_closure,
        "zeta_source_closure": zeta_source_closure,
        "chib_trace_source_closure": chib_trace_source_closure,
        "d4_zeta_coordinate_used": d4_zeta_coordinate,
        "d4_weighted_tr_chib_used": d4_weighted_tr_chib,
        "weighted_omega_used": weighted_omega,
        "weighted_omegab_used": weighted_omegab,
        "weighted_omegab_full_raw": (
            omegab_data["raw_value"]
            if omegab_data is not None
            else weighted_omegab
        ),
        "omegab_full_source_raw": (
            omegab_data["raw_source"]
            if omegab_data is not None
            else d4_weighted_omegab
        ),
        "weighted_omegab_projection_defect": (
            omegab_data["value_projection_defect"]
            if omegab_data is not None
            else np.zeros_like(weighted_omegab)
        ),
        "omegab_source_projection_defect": (
            omegab_data["source_projection_defect"]
            if omegab_data is not None
            else np.zeros_like(d4_weighted_omegab)
        ),
    }


def sphere_l2_maps(
    grid: PointSphereGrid,
    state: FirstOrderState,
    u: Array,
    values: dict[str, Array],
) -> dict[str, Array]:
    inverse = tangent_inverse(grid, state.metric)
    local = np.einsum(
        "nia,n...ij,njb->n...ab", grid.frames, state.metric, grid.frames
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local), 0.0))

    def scalar_density(value: Array) -> Array:
        return np.abs(value)

    def form_density(value: Array) -> Array:
        return np.sqrt(
            np.maximum(
                np.einsum("n...i,n...ij,n...j->n...", value, inverse, value),
                0.0,
            )
        )

    def tensor_density(value: Array) -> Array:
        return np.sqrt(np.maximum(tensor_norm_sq(value, inverse), 0.0))

    densities = {
        "r44": scalar_density(values["Ric44"]),
        "r33": scalar_density(values["Omega2_Ric33"]),
        "r34": scalar_density(values["Omega2_Ric34"]),
        "r3": form_density(values["Omega_Ric3A"]),
        "r4": form_density(values["Omega_Ric4A"]),
        # The displayed Ric4A identity naturally returns Omega*Ric4A.  The
        # attachment's later norm line asks for Omega^2*Ric4A, so retain both.
        "r4_Omega2": form_density(
            state.omega[..., None] * values["Omega_Ric4A"]
        ),
        "rhat": tensor_density(values["Omega2_hat_RicAB"]),
        "rR": scalar_density(values["Omega2_R"]),
        "trace_path_mismatch": scalar_density(
            values["Omega2_R_plus_Ric34"]
            - values["Omega2_R_plus_Ric34_from_4"]
        ),
    }
    maps = {}
    for name, density in densities.items():
        l2 = np.sqrt(
            np.maximum(
                4.0 * math.pi * np.mean(density**2 * area_ratio, axis=0),
                0.0,
            )
        )
        maps[name] = (-u[:, None]) * l2
    return maps


def physical_component_l2_maps(
    grid: PointSphereGrid,
    state: FirstOrderState,
    u: Array,
    values: dict[str, Array],
) -> dict[str, Array]:
    """Return ``(-u)`` times physical component ``L2(S)`` maps.

    Unlike :func:`sphere_l2_maps`, this routine removes every displayed
    ``Omega`` weight before taking a norm.  It is therefore directly
    comparable with ``f=(-u)||Ric||_L2``.
    """

    inverse = tangent_inverse(grid, state.metric)
    omega = state.omega
    omega_sq = omega**2
    ric33 = values["Omega2_Ric33"] / omega_sq
    ric44 = values["Ric44"]
    ric34 = values["Omega2_Ric34"] / omega_sq
    ric3 = values["Omega_Ric3A"] / omega[..., None]
    ric4 = values["Omega_Ric4A"] / omega[..., None]
    hat_ab = values["Omega2_hat_RicAB"] / omega_sq[..., None, None]
    trace_ab = values["Omega2_R_plus_Ric34"] / omega_sq
    scalar = trace_ab - ric34
    trace_path_mismatch = (
        values["Omega2_R_plus_Ric34"]
        - values["Omega2_R_plus_Ric34_from_4"]
    ) / omega_sq

    def form_density(value: Array) -> Array:
        return np.sqrt(
            np.maximum(
                np.einsum(
                    "n...i,n...ij,n...j->n...", value, inverse, value
                ),
                0.0,
            )
        )

    def tensor_density(value: Array) -> Array:
        return np.sqrt(np.maximum(tensor_norm_sq(value, inverse), 0.0))

    densities = {
        "Ric44": np.abs(ric44),
        "Ric33": np.abs(ric33),
        "Ric34": np.abs(ric34),
        "Ric3A": form_density(ric3),
        "Ric4A": form_density(ric4),
        "hat_RicAB": tensor_density(hat_ab),
        "trace_RicAB": np.abs(trace_ab),
        "scalar_R": np.abs(scalar),
        "trace_path_mismatch": np.abs(trace_path_mismatch),
    }
    local = np.einsum(
        "nia,n...ij,njb->n...ab", grid.frames, state.metric, grid.frames
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local), 0.0))
    maps = {}
    for name, density in densities.items():
        l2 = np.sqrt(
            np.maximum(
                4.0 * math.pi * np.mean(density**2 * area_ratio, axis=0),
                0.0,
            )
        )
        maps[name] = (-u[:, None]) * l2
    return maps


def safe_maxima(
    maps: dict[str, Array], u_halo: int = 4, v_halo: int = 5
) -> dict[str, float]:
    """Return maxima after excluding independent u/v boundary halos.

    A zero halo includes the complete axis.  It cannot be represented by
    ``halo:-halo`` because Python interprets ``-0`` as ``0``, producing an
    empty slice.
    """

    if not maps:
        raise ValueError("at least one residual map is required")
    if u_halo < 0 or v_halo < 0:
        raise ValueError("derivative halos must be nonnegative")
    first = np.asarray(next(iter(maps.values())))
    if first.ndim != 2:
        raise ValueError("residual maps must have shape (u, v)")
    for name, value in maps.items():
        if np.shape(value) != first.shape:
            raise ValueError(
                f"residual map {name!r} has shape {np.shape(value)}, "
                f"expected {first.shape}"
            )
    safe = np.zeros_like(first, dtype=bool)
    if first.shape[0] > 2 * u_halo and first.shape[1] > 2 * v_halo:
        u_stop = None if u_halo == 0 else -u_halo
        v_stop = None if v_halo == 0 else -v_halo
        safe[slice(u_halo, u_stop), slice(v_halo, v_stop)] = True
    if not np.any(safe):
        raise ValueError("the requested derivative halo leaves no safe cells")
    return {name: float(np.max(value[safe])) for name, value in maps.items()}
