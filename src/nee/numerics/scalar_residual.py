"""Construction-aware Einstein--scalar and wave residuals."""

from __future__ import annotations

import math

import numpy as np

from .scalar_coordinates import CharacteristicPowerMesh
from . import vacuum_residual
from .sphere import (
    PointSphereGrid,
    scalar_gradient,
    tangent_inverse,
    tensor_norm_sq,
)
from .scalar_iteration import ESEState, section_geometry


Array = np.ndarray


def components(
    grid: PointSphereGrid,
    state: ESEState,
    previous_state: ESEState,
    context: dict[str, object],
    mesh: CharacteristicPowerMesh,
) -> dict[str, Array]:
    """Return weighted components of Ric-dphi^2 and ``-Omega^2 box phi``."""

    ricci = vacuum_residual.components(
        grid,
        state,
        mesh.u,
        mesh.v,
        mode="construction",
        previous_state=previous_state,
        construction_context=context,
        scalar_coordinates=mesh,
    )
    geometry = section_geometry(grid, state, include_curvature=True)
    omega = state.omega
    omega_sq = omega**2
    grad_phi = geometry["grad_phi"]
    grad_phi_norm = np.einsum(
        "n...i,n...ij,n...j->n...",
        grad_phi,
        geometry["inverse"],
        grad_phi,
    )

    raychaudhuri_source = np.asarray(context["raychaudhuri_source"])
    shear_norm = tensor_norm_sq(state.shear, geometry["inverse"])
    ric44_construction = -(
        raychaudhuri_source
        + 0.5 * omega_sq * state.q**2
        + shear_norm / omega_sq
    )

    d3_scalar_p = mesh.differentiate_u(state.scalar_p, axis=1)
    d3_scalar_p += np.einsum(
        "n...i,n...i->n...",
        state.shift,
        scalar_gradient(grid, state.scalar_p),
    )
    eta_grad_phi = np.einsum(
        "n...i,n...i->n...", geometry["eta"], geometry["grad_phi_up"]
    )
    weighted_tr_chi = omega_sq * state.q
    wave = (
        d3_scalar_p
        + 0.5 * weighted_tr_chi * state.incoming_scalar
        + 0.5 * geometry["weighted_tr_chib"] * state.scalar_p
        - omega_sq * geometry["lap_phi"]
        - 2.0 * omega_sq * eta_grad_phi
    )
    d3_phi = mesh.differentiate_u(state.phi, axis=1)
    d3_phi += np.einsum(
        "n...i,n...i->n...",
        state.shift,
        grad_phi,
    )
    phi_v = mesh.differentiate_v(state.phi, axis=2)

    return {
        "E44": ric44_construction - state.scalar_p**2 / omega_sq,
        "Omega2_E33": (
            ricci["Omega2_Ric33"] - state.incoming_scalar**2
        ),
        "Omega2_E34": (
            ricci["Omega2_Ric34"]
            - state.incoming_scalar * state.scalar_p
        ),
        "Omega_E3A": (
            ricci["Omega_Ric3A"]
            - state.incoming_scalar[..., None] * grad_phi
        ),
        "Omega_E4A": (
            ricci["Omega_Ric4A"]
            - state.scalar_p[..., None] * grad_phi
        ),
        "Omega2_hat_EAB": (
            ricci["Omega2_hat_RicAB"]
            - 0.5
            * omega_sq[..., None, None]
            * geometry["phi_square_hat"]
        ),
        "Omega2_trace_EAB": (
            ricci["Omega2_R_plus_Ric34"]
            - omega_sq * grad_phi_norm
        ),
        "minus_Omega2_box_phi": wave,
        "incoming_phi_definition": state.incoming_scalar - d3_phi,
        "outgoing_phi_definition": state.scalar_p - phi_v,
        "incoming_metric_closure": (
            np.asarray(context["incoming_metric"]) - state.metric
        ),
        "incoming_phi_closure": (
            np.asarray(context["incoming_phi"]) - state.phi
        ),
        "Ric44_semidiscrete_projection_defect": (
            ric44_construction - state.scalar_p**2 / omega_sq
        ),
        "omegab_source_closure": ricci["omegab_source_closure"],
        "zeta_source_closure": ricci["zeta_source_closure"],
        "chib_trace_source_closure": ricci["chib_trace_source_closure"],
        "outgoing_lapse_closure": ricci["outgoing_lapse_closure"],
        "incoming_lapse_closure": ricci["incoming_lapse_closure"],
    }


def l2_maps(
    grid: PointSphereGrid,
    state: ESEState,
    mesh: CharacteristicPowerMesh,
    values: dict[str, Array],
) -> dict[str, Array]:
    """Return physical ``(-u)L2(S)`` component and combined maps."""

    inverse = tangent_inverse(grid, state.metric)
    omega = state.omega
    omega_sq = omega**2

    def form_norm_sq(value: Array) -> Array:
        return np.maximum(
            np.einsum(
                "n...i,n...ij,n...j->n...", value, inverse, value
            ),
            0.0,
        )

    def tensor_norm(value: Array) -> Array:
        return np.maximum(tensor_norm_sq(value, inverse), 0.0)

    physical = {
        "E44": values["E44"],
        "E33": values["Omega2_E33"] / omega_sq,
        "E34": values["Omega2_E34"] / omega_sq,
        "E3A": values["Omega_E3A"] / omega[..., None],
        "E4A": values["Omega_E4A"] / omega[..., None],
        "hat_EAB": (
            values["Omega2_hat_EAB"] / omega_sq[..., None, None]
        ),
        "trace_EAB": values["Omega2_trace_EAB"] / omega_sq,
        "box_phi": values["minus_Omega2_box_phi"] / omega_sq,
    }
    densities_sq = {
        "E44": physical["E44"] ** 2,
        "E33": physical["E33"] ** 2,
        "E34": physical["E34"] ** 2,
        "E3A": form_norm_sq(physical["E3A"]),
        "E4A": form_norm_sq(physical["E4A"]),
        "hat_EAB": tensor_norm(physical["hat_EAB"]),
        "trace_EAB": physical["trace_EAB"] ** 2,
        "box_phi": physical["box_phi"] ** 2,
    }
    densities_sq["combined"] = sum(densities_sq.values())

    local_metric = np.einsum(
        "nia,n...ij,njb->n...ab",
        grid.frames,
        state.metric,
        grid.frames,
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local_metric), 0.0))
    maps: dict[str, Array] = {}
    for name, density_sq in densities_sq.items():
        integral = 4.0 * math.pi * np.mean(
            density_sq * area_ratio, axis=0
        )
        maps[name] = (-mesh.u[:, None]) * np.sqrt(
            np.maximum(integral, 0.0)
        )
    return maps


def closure_maxima(
    grid: PointSphereGrid,
    state: ESEState,
    values: dict[str, Array],
) -> dict[str, float]:
    inverse = tangent_inverse(grid, state.metric)
    metric_closure = np.sqrt(
        np.maximum(
            tensor_norm_sq(values["incoming_metric_closure"], inverse),
            0.0,
        )
    )
    zeta_closure = np.sqrt(
        np.maximum(
            np.einsum(
                "n...i,n...ij,n...j->n...",
                values["zeta_source_closure"],
                inverse,
                values["zeta_source_closure"],
            ),
            0.0,
        )
    )
    return {
        "incoming_metric": float(np.max(metric_closure)),
        "incoming_phi": float(
            np.max(np.abs(values["incoming_phi_closure"]))
        ),
        "incoming_phi_definition": float(
            np.max(np.abs(values["incoming_phi_definition"]))
        ),
        "outgoing_phi_definition": float(
            np.max(np.abs(values["outgoing_phi_definition"]))
        ),
        "outgoing_lapse": float(
            np.max(np.abs(values["outgoing_lapse_closure"]))
        ),
        "incoming_lapse": float(
            np.max(np.abs(values["incoming_lapse_closure"]))
        ),
        "omegab_source": float(
            np.max(np.abs(values["omegab_source_closure"]))
        ),
        "zeta_source": float(np.max(zeta_closure)),
        "chib_trace_source": float(
            np.max(np.abs(values["chib_trace_source_closure"]))
        ),
    }


def safe_summary(
    maps: dict[str, Array],
    u_halo: int,
    v_halo: int,
    mesh: CharacteristicPowerMesh | None = None,
) -> dict[str, dict[str, float]]:
    first = next(iter(maps.values()))
    safe = reliability_mask(first.shape, u_halo, v_halo, mesh)
    if not np.any(safe):
        raise ValueError("the derivative halo leaves no safe cells")
    return {
        name: {
            "maximum": float(np.max(value[safe])),
            "median": float(np.median(value[safe])),
            "minimum": float(np.min(value[safe])),
        }
        for name, value in maps.items()
    }


def reliability_mask(
    shape: tuple[int, int],
    u_halo: int,
    v_halo: int,
    mesh: CharacteristicPowerMesh | None = None,
) -> Array:
    """Exclude outer and composite-element differentiation halos."""

    if u_halo < 0 or v_halo < 0:
        raise ValueError("halo widths must be nonnegative")
    safe_u = np.ones(shape[0], dtype=bool)
    safe_v = np.ones(shape[1], dtype=bool)

    def exclude_near(
        safe: Array, centers: list[int], width: int
    ) -> None:
        if width == 0:
            return
        for center in centers:
            left = max(0, center - width + 1)
            right = min(len(safe), center + width)
            safe[left:right] = False

    u_centers = [0, shape[0] - 1]
    v_centers = [0, shape[1] - 1]
    if mesh is not None:
        u_centers.extend(
            int(indices[-1]) for indices in mesh.tau.indices[:-1]
        )
        v_centers.extend(
            int(indices[-1]) for indices in mesh.s.indices[:-1]
        )
    exclude_near(safe_u, u_centers, u_halo)
    exclude_near(safe_v, v_centers, v_halo)
    return safe_u[:, None] & safe_v[None, :]
