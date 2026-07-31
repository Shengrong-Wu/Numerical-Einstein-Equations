"""Vacuum Ricci residual reconstructed from null Ricci coefficients.

The primary audit intentionally avoids differentiating the trace-free
``R_{A4B4}`` component.  In the iteration, ``Ric_44`` is zero by the
Raychaudhuri construction of ``Omega^{-1} tr(chi)``.  The remaining Ricci
components are obtained from the null propagation and Gauss--Codazzi
identities in ``doc/theoretical-research-ese/Sections/preliminaries.tex`` and
``approximation.tex``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .coordinate_differentiation import high_order_differentiate
from .vacuum_state import GlobalState, section_geometry
from .sphere import (
    PointSphereGrid,
    scalar_gradient,
    tangent_inverse,
    tensor_divergence,
    tensor_norm_sq,
    tensor_tracefree,
    vector_divergence,
)


Array = np.ndarray


@dataclass(frozen=True)
class ResidualCoefficients:
    """Coefficients exposed so deliberate mutations cannot pass silently."""

    raychaudhuri_square: float = 0.5
    incoming_raychaudhuri_square: float = 0.5
    incoming_raychaudhuri_omegab: float = 4.0
    hat_cross: float = 0.5
    trace_gauss: float = 2.0
    ric34_d3_trace: float = -0.25
    ric4_divergence: float = 1.0
    ric3_divergence: float = 1.0


def one_form_lie_derivative(
    grid: PointSphereGrid, vector: Array, form: Array
) -> Array:
    derivative_form = grid.reference_derivative(form, tensor_rank=1)
    derivative_vector = grid.reference_derivative(vector, tensor_rank=1)
    return np.einsum("n...k,n...ki->n...i", vector, derivative_form) + np.einsum(
        "n...k,n...ik->n...i", form, derivative_vector
    )


def d3_scalar(grid: PointSphereGrid, scalar: Array, shift: Array, u: Array) -> Array:
    return high_order_differentiate(scalar, u, axis=1) + np.einsum(
        "n...i,n...i->n...", shift, scalar_gradient(grid, scalar)
    )


def d3_one_form(
    grid: PointSphereGrid,
    form: Array,
    shift: Array,
    weighted_chib_mixed: Array,
    u: Array,
) -> Array:
    coordinate_lie = high_order_differentiate(form, u, axis=1)
    coordinate_lie += one_form_lie_derivative(grid, shift, form)
    return coordinate_lie - np.einsum(
        "n...ij,n...j->n...i", weighted_chib_mixed, form
    )


def d3_covariant_tensor(
    grid: PointSphereGrid,
    tensor: Array,
    shift: Array,
    weighted_chib_mixed: Array,
    u: Array,
) -> Array:
    coordinate_lie = high_order_differentiate(tensor, u, axis=1)
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


def ricci_coefficient_components(
    grid: PointSphereGrid,
    state: GlobalState,
    u: Array,
    v: Array,
    coefficients: ResidualCoefficients = ResidualCoefficients(),
    previous_state: GlobalState | None = None,
    construction_context: dict[str, Array] | None = None,
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Return physical null Ricci components and construction diagnostics."""

    geometry = section_geometry(grid, state, u)
    inverse = geometry["inverse"]
    omega = state.omega
    omega_sq = omega**2
    log_omega = np.log(omega)
    grad_log_omega = scalar_gradient(grid, log_omega)

    tr_chi = omega * state.q
    weighted_tr_chi = omega_sq * state.q
    weighted_tr_chib = omega * geometry["tr_chib"]
    weighted_hatchib = omega[..., None, None] * geometry["hatchib"]
    weighted_chib = weighted_hatchib + 0.5 * weighted_tr_chib[
        ..., None, None
    ] * state.metric
    weighted_chib_mixed = np.matmul(weighted_chib, inverse)

    zeta = np.einsum("n...ij,n...j->n...i", state.metric, state.zeta_up)
    eta = geometry["eta"]
    etab = geometry["etab"]
    eta_up = np.einsum("n...ij,n...j->n...i", inverse, eta)
    div_eta = vector_divergence(grid, eta_up, geometry["difference"])
    eta_norm = np.einsum("n...i,n...ij,n...j->n...", eta, inverse, eta)
    eta_etab = np.einsum("n...i,n...ij,n...j->n...", eta, inverse, etab)

    if construction_context is not None:
        if previous_state is None:
            raise ValueError("construction context requires the previous state")
        shift_difference = state.shift - previous_state.shift
        correction = -0.5 * np.einsum(
            "n...i,n...i->n...", shift_difference, grad_log_omega
        )
        weighted_omegab = (
            construction_context["weighted_omegab_half"] + correction
        )
        weighted_omega = construction_context["weighted_omega"]
        shift_v_difference = -4.0 * (
            omega_sq[..., None] * state.zeta_up
            - previous_state.omega[..., None] ** 2 * previous_state.zeta_up
        )
        correction_v = -0.5 * (
            np.einsum(
                "n...i,n...i->n...", shift_v_difference, grad_log_omega
            )
            - 2.0
            * np.einsum(
                "n...i,n...i->n...",
                shift_difference,
                scalar_gradient(grid, weighted_omega),
            )
        )
        weighted_omegab_v = construction_context["omegab_source"] + correction_v
        dzeta_v = (
            weighted_tr_chi[..., None] * zeta
            + 2.0
            * np.einsum("n...ij,n...j->n...i", state.shear, state.zeta_up)
            + np.einsum(
                "n...ij,n...j->n...i",
                state.metric,
                construction_context["zeta_source"],
            )
        )
    else:
        d3_log_omega = d3_scalar(grid, log_omega, state.shift, u)
        weighted_omegab = -0.5 * d3_log_omega
        weighted_omega = -0.5 * high_order_differentiate(log_omega, v, axis=2)
        weighted_omegab_v = high_order_differentiate(
            weighted_omegab, v, axis=2
        )
        dzeta_v = high_order_differentiate(zeta, v, axis=2)
    d3_weighted_tr_chi = d3_scalar(
        grid, weighted_tr_chi, state.shift, u
    )

    div_shear = tensor_divergence(
        grid, state.shear, geometry["difference"], inverse
    )
    div_weighted_hatchib = tensor_divergence(
        grid, weighted_hatchib, geometry["difference"], inverse
    )

    d3_shear = d3_covariant_tensor(
        grid, state.shear, state.shift, weighted_chib_mixed, u
    )
    weighted_hat = (
        d3_shear
        + 0.5 * weighted_tr_chib[..., None, None] * state.shear
        - omega_sq[..., None, None]
        * (geometry["eta_grad_hat"] + geometry["eta_square_hat"])
        + coefficients.hat_cross
        * weighted_tr_chi[..., None, None]
        * weighted_hatchib
    )
    weighted_hat = tensor_tracefree(weighted_hat, state.metric, inverse)

    trace_ab_weighted = (
        d3_weighted_tr_chi
        + weighted_tr_chi * weighted_tr_chib
        - 2.0 * omega_sq * div_eta
        - 2.0 * omega_sq * eta_norm
        + coefficients.trace_gauss * omega_sq * geometry["curvature"]
    )
    trace_ab = trace_ab_weighted / omega_sq
    ric_ab = weighted_hat / omega_sq[..., None, None]
    ric_ab += 0.5 * trace_ab[..., None, None] * state.metric

    shear_dot_hatchib = np.einsum(
        "n...ik,n...jl,n...ij,n...kl->n...",
        inverse,
        inverse,
        state.shear,
        weighted_hatchib,
    )
    weighted_ric34 = 4.0 * (
        weighted_omegab_v
        - 0.25 * shear_dot_hatchib
        + omega_sq * eta_etab
        + coefficients.ric34_d3_trace * d3_weighted_tr_chi
        - 0.125 * weighted_tr_chi * weighted_tr_chib
        + 0.5 * omega_sq * div_eta
    )

    grad_weighted_omega = scalar_gradient(grid, weighted_omega)
    grad_weighted_tr_chi = scalar_gradient(grid, weighted_tr_chi)
    weighted_ric4 = (
        2.0 * grad_weighted_omega
        + coefficients.ric4_divergence * div_shear
        - 0.5 * grad_weighted_tr_chi
        + weighted_tr_chi[..., None] * grad_log_omega
        - dzeta_v
        - weighted_tr_chi[..., None] * zeta
    )

    d3_zeta = d3_one_form(
        grid, zeta, state.shift, weighted_chib_mixed, u
    )
    weighted_hatchib_mixed = np.matmul(weighted_hatchib, inverse)
    weighted_ric3 = (
        d3_zeta
        + 1.5 * weighted_tr_chib[..., None] * zeta
        + np.einsum(
            "n...ij,n...j->n...i", weighted_hatchib_mixed, zeta
        )
        + 2.0 * scalar_gradient(grid, weighted_omegab)
        + coefficients.ric3_divergence * div_weighted_hatchib
        - 0.5 * scalar_gradient(grid, weighted_tr_chib)
        + weighted_tr_chib[..., None] * grad_log_omega
    )

    d3_weighted_tr_chib = d3_scalar(
        grid, weighted_tr_chib, state.shift, u
    )
    weighted_hatchib_norm = tensor_norm_sq(weighted_hatchib, inverse)
    weighted_ric33 = -(
        d3_weighted_tr_chib
        + coefficients.incoming_raychaudhuri_square * weighted_tr_chib**2
        + coefficients.incoming_raychaudhuri_omegab
        * weighted_omegab
        * weighted_tr_chib
        + weighted_hatchib_norm
    )

    shear_norm = tensor_norm_sq(state.shear, inverse)
    ric44_closure = (
        high_order_differentiate(state.q, v, axis=2)
        + coefficients.raychaudhuri_square * omega_sq * state.q**2
        + shear_norm / omega_sq
    )
    zero = np.zeros_like(state.q)
    components = {
        "Ric44": zero,
        "Ric33": weighted_ric33 / omega_sq,
        "Ric34": weighted_ric34 / omega_sq,
        "Ric4A": weighted_ric4 / omega[..., None],
        "Ric3A": weighted_ric3 / omega[..., None],
        "RicAB": ric_ab,
    }
    diagnostics = {
        "Ric44_construction_closure": ric44_closure,
        "Omega2_hat_RicAB": weighted_hat,
        "Omega2_trace_RicAB": trace_ab_weighted,
        "Omega2_Ric34": weighted_ric34,
        "Omega_Ric4A": weighted_ric4,
        "Omega_Ric3A": weighted_ric3,
        "Omega2_Ric33": weighted_ric33,
    }
    return components, diagnostics


def positive_null_ricci_norm(
    components: dict[str, Array], inverse: Array
) -> Array:
    value = components["Ric33"] ** 2 + components["Ric44"] ** 2
    value += 2.0 * components["Ric34"] ** 2
    for name in ("Ric3A", "Ric4A"):
        form = components[name]
        value += np.einsum("n...i,n...ij,n...j->n...", form, inverse, form)
    angular = components["RicAB"]
    value += np.einsum(
        "n...ik,n...jl,n...ij,n...kl->n...",
        inverse,
        inverse,
        angular,
        angular,
    )
    return np.sqrt(np.maximum(value, 0.0))


def ricci_component_densities(
    components: dict[str, Array], inverse: Array, metric: Array
) -> dict[str, Array]:
    """Return positive pointwise densities for individual null components.

    With ``g(e_3,e_4)=-2``, the spacetime scalar curvature is
    ``R = tr_g(RicAB) - Ric34``.  The ``Ric34`` density itself has no extra
    factor; its factor two enters only when reconstructing the complete
    positive null norm.
    """

    trace_ab = np.einsum("n...ij,n...ij->n...", inverse, components["RicAB"])
    hat_ab = components["RicAB"] - 0.5 * trace_ab[..., None, None] * metric

    def one_form_norm(name: str) -> Array:
        form = components[name]
        return np.sqrt(
            np.maximum(
                np.einsum("n...i,n...ij,n...j->n...", form, inverse, form),
                0.0,
            )
        )

    def tensor_norm(tensor: Array) -> Array:
        return np.sqrt(
            np.maximum(
                np.einsum(
                    "n...ik,n...jl,n...ij,n...kl->n...",
                    inverse,
                    inverse,
                    tensor,
                    tensor,
                ),
                0.0,
            )
        )

    scalar = trace_ab - components["Ric34"]
    return {
        "total": positive_null_ricci_norm(components, inverse),
        "Ric33": np.abs(components["Ric33"]),
        "Ric34": np.abs(components["Ric34"]),
        "RicAB": tensor_norm(components["RicAB"]),
        "Scalar": np.abs(scalar),
        "Ric3A": one_form_norm("Ric3A"),
        "Ric4A": one_form_norm("Ric4A"),
        "RicAB_hat": tensor_norm(hat_ab),
        # This is the tensor norm of (tr RicAB / 2) g in two dimensions.
        "RicAB_pure_trace": np.abs(trace_ab) / math.sqrt(2.0),
        "RicAB_trace": np.abs(trace_ab),
    }


def renormalized_component_l2_maps(
    grid: PointSphereGrid,
    state: GlobalState,
    u: Array,
    densities: dict[str, Array],
) -> dict[str, Array]:
    """Return ``(-u)||density||_L2(S_{u,v})`` for every supplied density."""

    local_metric = np.einsum(
        "nia,n...ij,njb->n...ab", grid.frames, state.metric, grid.frames
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local_metric), 0.0))
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


def coefficient_residual_map(
    grid: PointSphereGrid,
    state: GlobalState,
    u: Array,
    v: Array,
    coefficients: ResidualCoefficients = ResidualCoefficients(),
    previous_state: GlobalState | None = None,
    construction_context: dict[str, Array] | None = None,
) -> tuple[Array, Array, dict[str, Array]]:
    """Return ``(-u)||Ric||_L2(S)``, its density, and component diagnostics."""

    components, diagnostics = ricci_coefficient_components(
        grid,
        state,
        u,
        v,
        coefficients,
        previous_state,
        construction_context,
    )
    inverse = tangent_inverse(grid, state.metric)
    rho = positive_null_ricci_norm(components, inverse)
    local_metric = np.einsum(
        "nia,n...ij,njb->n...ab", grid.frames, state.metric, grid.frames
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local_metric), 0.0))
    l2 = np.sqrt(
        np.maximum(4.0 * math.pi * np.mean(rho**2 * area_ratio, axis=0), 0.0)
    )
    return (-u[:, None]) * l2, rho, diagnostics
