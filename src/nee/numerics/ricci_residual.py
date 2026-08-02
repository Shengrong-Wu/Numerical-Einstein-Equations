"""Vacuum Ricci residual reconstructed from null Ricci coefficients.

The primary audit intentionally avoids differentiating the trace-free
``R_{A4B4}`` component.  In the iteration, ``Ric_44`` is zero by the
Raychaudhuri construction of ``Omega tr(chi)``.  The remaining Ricci
components are obtained from the null propagation and Gauss--Codazzi
identities documented in ``docs/article/article.tex``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from nee.geometry.null_geometry import one_form_lie_derivative

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
    raychaudhuri_omega: float = 4.0
    incoming_raychaudhuri_square: float = 0.5
    incoming_raychaudhuri_omegab: float = 4.0
    hat_cross: float = 0.5
    trace_gauss: float = 2.0
    ric34_Omega_e3_Omega_trchib: float = -0.25
    ric4_divergence: float = 1.0
    ric3_divergence: float = 1.0


def Omega_e3_scalar(grid: PointSphereGrid, scalar: Array, b: Array, u: Array) -> Array:
    return high_order_differentiate(scalar, u, axis=1) + np.einsum(
        "n...i,n...i->n...", b, scalar_gradient(grid, scalar)
    )


def Omega_nabla3_one_form(
    grid: PointSphereGrid,
    form: Array,
    b: Array,
    weighted_chib_mixed: Array,
    u: Array,
) -> Array:
    coordinate_lie = high_order_differentiate(form, u, axis=1)
    coordinate_lie += one_form_lie_derivative(grid, b, form)
    return coordinate_lie - np.einsum(
        "n...ij,n...j->n...i", weighted_chib_mixed, form
    )


def Omega_nabla3_covariant_tensor(
    grid: PointSphereGrid,
    tensor: Array,
    b: Array,
    weighted_chib_mixed: Array,
    u: Array,
) -> Array:
    coordinate_lie = high_order_differentiate(tensor, u, axis=1)
    derivative_tensor = grid.reference_derivative(tensor, tensor_rank=2)
    derivative_shift = grid.reference_derivative(b, tensor_rank=1)
    coordinate_lie += np.einsum(
        "n...k,n...kij->n...ij", b, derivative_tensor
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
    inverse_g = geometry["inverse_g"]
    Omega = state.Omega
    omega_sq = Omega**2
    log_Omega = np.log(Omega)
    grad_log_omega = scalar_gradient(grid, log_Omega)

    tr_chi = state.Omega_trchi / Omega
    Omega_trchi = state.Omega_trchi
    Omega_trchib = Omega * geometry["tr_chib"]
    Omega_chibh = Omega[..., None, None] * geometry["hatchib"]
    Omega_chib = Omega_chibh + 0.5 * Omega_trchib[
        ..., None, None
    ] * state.g
    weighted_chib_mixed = np.matmul(Omega_chib, inverse_g)

    zeta = np.einsum("n...ij,n...j->n...i", state.g, state.zeta)
    eta = geometry["eta"]
    etab = geometry["etab"]
    eta_up = np.einsum("n...ij,n...j->n...i", inverse_g, eta)
    div_eta = vector_divergence(grid, eta_up, geometry["difference"])
    eta_norm = np.einsum("n...i,n...ij,n...j->n...", eta, inverse_g, eta)
    eta_etab = np.einsum("n...i,n...ij,n...j->n...", eta, inverse_g, etab)

    if construction_context is not None:
        if previous_state is None:
            raise ValueError("construction context requires the previous state")
        shift_difference = state.b - previous_state.b
        correction = -0.5 * np.einsum(
            "n...i,n...i->n...", shift_difference, grad_log_omega
        )
        Omega_omegab = (
            construction_context["weighted_omegab_half"] + correction
        )
        Omega_omega = construction_context["Omega_omega"]
        shift_v_difference = -4.0 * (
            omega_sq[..., None] * state.zeta
            - previous_state.Omega[..., None] ** 2 * previous_state.zeta
        )
        correction_v = -0.5 * (
            np.einsum(
                "n...i,n...i->n...", shift_v_difference, grad_log_omega
            )
            - 2.0
            * np.einsum(
                "n...i,n...i->n...",
                shift_difference,
                scalar_gradient(grid, Omega_omega),
            )
        )
        weighted_omegab_v = construction_context["omegab_source"] + correction_v
        dzeta_v = (
            Omega_trchi[..., None] * zeta
            + 2.0
            * np.einsum("n...ij,n...j->n...i", state.Omega_chih, state.zeta)
            + np.einsum(
                "n...ij,n...j->n...i",
                state.g,
                construction_context["zeta_source"],
            )
        )
    else:
        Omega_e3_log_Omega = Omega_e3_scalar(grid, log_Omega, state.b, u)
        Omega_omegab = -0.5 * Omega_e3_log_Omega
        Omega_omega = -0.5 * high_order_differentiate(log_Omega, v, axis=2)
        weighted_omegab_v = high_order_differentiate(
            Omega_omegab, v, axis=2
        )
        dzeta_v = high_order_differentiate(zeta, v, axis=2)
    Omega_e3_Omega_trchi = Omega_e3_scalar(
        grid, Omega_trchi, state.b, u
    )

    div_shear = tensor_divergence(
        grid, state.Omega_chih, geometry["difference"], inverse_g
    )
    div_weighted_hatchib = tensor_divergence(
        grid, Omega_chibh, geometry["difference"], inverse_g
    )

    Omega_nabla3_Omega_chih = Omega_nabla3_covariant_tensor(
        grid, state.Omega_chih, state.b, weighted_chib_mixed, u
    )
    weighted_hat = (
        Omega_nabla3_Omega_chih
        + 0.5 * Omega_trchib[..., None, None] * state.Omega_chih
        - omega_sq[..., None, None]
        * (geometry["eta_grad_hat"] + geometry["eta_square_hat"])
        + coefficients.hat_cross
        * Omega_trchi[..., None, None]
        * Omega_chibh
    )
    weighted_hat = tensor_tracefree(weighted_hat, state.g, inverse_g)

    trace_ab_weighted = (
        Omega_e3_Omega_trchi
        + Omega_trchi * Omega_trchib
        - 2.0 * omega_sq * div_eta
        - 2.0 * omega_sq * eta_norm
        + coefficients.trace_gauss * omega_sq * geometry["curvature"]
    )
    trace_ab = trace_ab_weighted / omega_sq
    ric_ab = weighted_hat / omega_sq[..., None, None]
    ric_ab += 0.5 * trace_ab[..., None, None] * state.g

    shear_dot_hatchib = np.einsum(
        "n...ik,n...jl,n...ij,n...kl->n...",
        inverse_g,
        inverse_g,
        state.Omega_chih,
        Omega_chibh,
    )
    weighted_ric34 = 4.0 * (
        weighted_omegab_v
        - 0.25 * shear_dot_hatchib
        + omega_sq * eta_etab
        + coefficients.ric34_Omega_e3_Omega_trchib * Omega_e3_Omega_trchi
        - 0.125 * Omega_trchi * Omega_trchib
        + 0.5 * omega_sq * div_eta
    )

    grad_weighted_omega = scalar_gradient(grid, Omega_omega)
    grad_weighted_tr_chi = scalar_gradient(grid, Omega_trchi)
    weighted_ric4 = (
        2.0 * grad_weighted_omega
        + coefficients.ric4_divergence * div_shear
        - 0.5 * grad_weighted_tr_chi
        + Omega_trchi[..., None] * grad_log_omega
        - dzeta_v
        - Omega_trchi[..., None] * zeta
    )

    Omega_nabla3_zeta = Omega_nabla3_one_form(
        grid, zeta, state.b, weighted_chib_mixed, u
    )
    weighted_hatchib_mixed = np.matmul(Omega_chibh, inverse_g)
    weighted_ric3 = (
        Omega_nabla3_zeta
        + 1.5 * Omega_trchib[..., None] * zeta
        + np.einsum(
            "n...ij,n...j->n...i", weighted_hatchib_mixed, zeta
        )
        + 2.0 * scalar_gradient(grid, Omega_omegab)
        + coefficients.ric3_divergence * div_weighted_hatchib
        - 0.5 * scalar_gradient(grid, Omega_trchib)
        + Omega_trchib[..., None] * grad_log_omega
    )

    Omega_e3_Omega_trchib = Omega_e3_scalar(
        grid, Omega_trchib, state.b, u
    )
    weighted_hatchib_norm = tensor_norm_sq(Omega_chibh, inverse_g)
    weighted_ric33 = -(
        Omega_e3_Omega_trchib
        + coefficients.incoming_raychaudhuri_square * Omega_trchib**2
        + coefficients.incoming_raychaudhuri_omegab
        * Omega_omegab
        * Omega_trchib
        + weighted_hatchib_norm
    )

    shear_norm = tensor_norm_sq(state.Omega_chih, inverse_g)
    ric44_closure = (
        high_order_differentiate(state.Omega_trchi, v, axis=2)
        + coefficients.raychaudhuri_square * state.Omega_trchi**2
        + coefficients.raychaudhuri_omega
        * Omega_omega
        * state.Omega_trchi
        + shear_norm
    ) / omega_sq
    zero = np.zeros_like(state.Omega_trchi)
    components = {
        "Ric44": zero,
        "Ric33": weighted_ric33 / omega_sq,
        "Ric34": weighted_ric34 / omega_sq,
        "Ric4A": weighted_ric4 / Omega[..., None],
        "Ric3A": weighted_ric3 / Omega[..., None],
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
    components: dict[str, Array], inverse_g: Array
) -> Array:
    value = components["Ric33"] ** 2 + components["Ric44"] ** 2
    value += 2.0 * components["Ric34"] ** 2
    for name in ("Ric3A", "Ric4A"):
        form = components[name]
        value += np.einsum("n...i,n...ij,n...j->n...", form, inverse_g, form)
    angular = components["RicAB"]
    value += np.einsum(
        "n...ik,n...jl,n...ij,n...kl->n...",
        inverse_g,
        inverse_g,
        angular,
        angular,
    )
    return np.sqrt(np.maximum(value, 0.0))


def ricci_component_densities(
    components: dict[str, Array], inverse_g: Array, g: Array
) -> dict[str, Array]:
    """Return positive pointwise densities for individual null components.

    With ``g(e_3,e_4)=-2``, the spacetime scalar curvature is
    ``R = tr_g(RicAB) - Ric34``.  The ``Ric34`` density itself has no extra
    factor; its factor two enters only when reconstructing the complete
    positive null norm.
    """

    trace_ab = np.einsum("n...ij,n...ij->n...", inverse_g, components["RicAB"])
    hat_ab = components["RicAB"] - 0.5 * trace_ab[..., None, None] * g

    def one_form_norm(name: str) -> Array:
        form = components[name]
        return np.sqrt(
            np.maximum(
                np.einsum("n...i,n...ij,n...j->n...", form, inverse_g, form),
                0.0,
            )
        )

    def tensor_norm(tensor: Array) -> Array:
        return np.sqrt(
            np.maximum(
                np.einsum(
                    "n...ik,n...jl,n...ij,n...kl->n...",
                    inverse_g,
                    inverse_g,
                    tensor,
                    tensor,
                ),
                0.0,
            )
        )

    scalar = trace_ab - components["Ric34"]
    return {
        "total": positive_null_ricci_norm(components, inverse_g),
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
        "nia,n...ij,njb->n...ab", grid.frames, state.g, grid.frames
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
    inverse_g = tangent_inverse(grid, state.g)
    rho = positive_null_ricci_norm(components, inverse_g)
    local_metric = np.einsum(
        "nia,n...ij,njb->n...ab", grid.frames, state.g, grid.frames
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local_metric), 0.0))
    l2 = np.sqrt(
        np.maximum(4.0 * math.pi * np.mean(rho**2 * area_ratio, axis=0), 0.0)
    )
    return (-u[:, None]) * l2, rho, diagnostics
