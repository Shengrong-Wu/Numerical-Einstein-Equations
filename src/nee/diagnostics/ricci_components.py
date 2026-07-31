"""All-current first-order null-identity residual, without construction context."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nee.state.iterate import PicardState as WeightedState


Array = np.ndarray

from nee.numerics import vacuum_residual  # noqa: E402
from nee.numerics.sphere import (  # noqa: E402
    scalar_gradient,
    tangent_inverse,
    tensor_norm_sq,
)


def components(
    grid: Any,
    state: WeightedState,
    u: Array,
    v: Array,
    *,
    coordinates: Any | None = None,
) -> dict[str, Array]:
    """Return fresh null residuals using only the current official iterate."""

    ricci = vacuum_residual.components(
        grid,
        state,
        u,
        v,
        mode="fresh",
        scalar_coordinates=coordinates,
    )
    if not state.is_scalar:
        return ricci

    from nee.numerics import scalar_iteration as ese

    geometry = ese.section_geometry(grid, state, include_curvature=True)
    omega = state.omega
    omega_sq = omega**2
    grad_phi = scalar_gradient(grid, state.phi)
    grad_norm = np.einsum(
        "n...i,n...ij,n...j->n...",
        grad_phi,
        geometry["inverse"],
        grad_phi,
    )
    if coordinates is None:
        from nee.numerics.coordinate_differentiation import high_order_differentiate

        d3_p4 = high_order_differentiate(state.p4, u, axis=1)
    else:
        d3_p4 = coordinates.differentiate_u(state.p4, axis=1)
    d3_p4 += np.einsum(
        "n...i,n...i->n...", state.shift, scalar_gradient(grid, state.p4)
    )
    eta_grad = np.einsum(
        "n...i,n...i->n...", geometry["eta"], geometry["grad_phi_up"]
    )
    wave = (
        d3_p4
        + 0.5 * state.a_out * state.p3
        + 0.5 * state.a_in * state.p4
        - omega_sq * geometry["lap_phi"]
        - 2.0 * omega_sq * eta_grad
    )
    return {
        "E44": ricci["Ric44"] - state.p4**2 / omega_sq,
        "Omega2_E33": ricci["Omega2_Ric33"] - state.p3**2,
        "Omega2_E34": ricci["Omega2_Ric34"] - state.p3 * state.p4,
        "Omega_E3A": (
            ricci["Omega_Ric3A"] - state.p3[..., None] * grad_phi
        ),
        "Omega_E4A": (
            ricci["Omega_Ric4A"] - state.p4[..., None] * grad_phi
        ),
        "Omega2_hat_EAB": (
            ricci["Omega2_hat_RicAB"]
            - 0.5
            * omega_sq[..., None, None]
            * geometry["phi_square_hat"]
        ),
        "Omega2_trace_EAB": (
            ricci["Omega2_R_plus_Ric34"] - omega_sq * grad_norm
        ),
        "minus_Omega2_box_phi": wave,
        "Ric44_fresh_closure": ricci["Ric44_fresh_closure"],
    }


def section_maps(
    grid: Any,
    state: WeightedState,
    u: Array,
    values: dict[str, Array],
) -> dict[str, Array]:
    inverse = tangent_inverse(grid, state.metric)
    omega = state.omega
    omega_sq = omega**2

    if not state.is_scalar:
        physical = {
            "Ric44": values["Ric44"],
            "Ric33": values["Omega2_Ric33"] / omega_sq,
            "Ric34": values["Omega2_Ric34"] / omega_sq,
            "Ric3A": values["Omega_Ric3A"] / omega[..., None],
            "Ric4A": values["Omega_Ric4A"] / omega[..., None],
            "hat_RicAB": values["Omega2_hat_RicAB"]
            / omega_sq[..., None, None],
            "trace_RicAB": values["Omega2_R_plus_Ric34"] / omega_sq,
        }
    else:
        physical = {
            "E44": values["E44"],
            "E33": values["Omega2_E33"] / omega_sq,
            "E34": values["Omega2_E34"] / omega_sq,
            "E3A": values["Omega_E3A"] / omega[..., None],
            "E4A": values["Omega_E4A"] / omega[..., None],
            "hat_EAB": values["Omega2_hat_EAB"] / omega_sq[..., None, None],
            "trace_EAB": values["Omega2_trace_EAB"] / omega_sq,
            "box_phi": values["minus_Omega2_box_phi"] / omega_sq,
        }
    density: dict[str, Array] = {}
    for name, value in physical.items():
        if value.ndim == state.metric.ndim:
            density[name] = tensor_norm_sq(value, inverse)
        elif value.ndim == state.shift.ndim:
            density[name] = np.einsum(
                "n...i,n...ij,n...j->n...", value, inverse, value
            )
        else:
            density[name] = value**2
    density["combined"] = sum(density.values())
    local = np.einsum(
        "nia,n...ij,njb->n...ab", grid.frames, state.metric, grid.frames
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local), 0.0))
    return {
        name: (-u[:, None])
        * np.sqrt(
            np.maximum(
                4.0 * math.pi * np.mean(value * area_ratio, axis=0),
                0.0,
            )
        )
        for name, value in density.items()
    }


def summarize(maps: dict[str, Array], halo: int = 2) -> dict[str, Any]:
    first = next(iter(maps.values()))
    mask = np.zeros(first.shape, dtype=bool)
    if 2 * halo < min(first.shape):
        mask[halo:-halo, halo:-halo] = True
    else:
        mask[:] = True
    return {
        name: {
            "raw_maximum": float(np.max(value)),
            "masked_maximum": float(np.max(value[mask])),
            "masked_median": float(np.median(value[mask])),
        }
        for name, value in maps.items()
    }
