"""Fresh reconstruction closures for the official weighted state."""

from __future__ import annotations

from typing import Any

import numpy as np

from nee.state.iterate import PicardState


Array = np.ndarray

from nee.numerics.coordinate_differentiation import high_order_differentiate  # noqa: E402
from nee.numerics.sphere import (  # noqa: E402
    lie_covariant_tensor,
    scalar_gradient,
)


def _summary(
    value: Array, halo: int, mask: Array | None = None
) -> dict[str, float]:
    magnitude = np.max(np.abs(value), axis=(0, *range(3, value.ndim)))
    if mask is not None:
        safe = magnitude[mask]
    elif 2 * halo < min(magnitude.shape):
        safe = magnitude[halo:-halo, halo:-halo]
    else:
        safe = magnitude
    return {
        "raw_maximum": float(np.max(magnitude)),
        "masked_maximum": float(np.max(safe)),
        "rms": float(np.sqrt(np.mean(value**2))),
    }


def evaluate(
    grid: Any,
    state: PicardState,
    u: Array,
    v: Array,
    *,
    stencil: int = 9,
    halo: int = 2,
    omit_lie_derivative: bool = False,
    coordinates: Any | None = None,
    protected_s_minimum: float | None = None,
) -> dict[str, Any]:
    metric_u = (
        high_order_differentiate(
            state.g, u, axis=1, stencil=min(stencil, len(u))
        )
        if coordinates is None
        else coordinates.differentiate_u(state.g, axis=1)
    )
    metric_v = (
        high_order_differentiate(
            state.g, v, axis=2, stencil=min(stencil, len(v))
        )
        if coordinates is None
        else coordinates.differentiate_v(state.g, axis=2)
    )
    if not omit_lie_derivative:
        metric_u = metric_u + lie_covariant_tensor(
            grid, state.b, state.g
        )
    c3 = metric_u - 2.0 * state.Omega_chib
    c4 = metric_v - 2.0 * state.Omega_chi
    quantities: dict[str, Array] = {
        "C3": c3,
        "C4": c4,
    }
    result: dict[str, Any] = {
        "method": "fresh derivatives of the stored primitives/full forms",
        "C3": _summary(c3, halo),
        "C4": _summary(c4, halo),
    }
    if state.is_scalar:
        phi_u = (
            high_order_differentiate(
                state.phi, u, axis=1, stencil=min(stencil, len(u))
            )
            if coordinates is None
            else coordinates.differentiate_u(state.phi, axis=1)
        )
        phi_v = (
            high_order_differentiate(
                state.phi, v, axis=2, stencil=min(stencil, len(v))
            )
            if coordinates is None
            else coordinates.differentiate_v(state.phi, axis=2)
        )
        nabla_phi = scalar_gradient(grid, state.phi)
        Omega_e3phi_from_phi = phi_u + np.einsum(
            "n...i,n...i->n...", state.b, nabla_phi
        )
        quantities.update(
            {
                "scalar_P3": Omega_e3phi_from_phi - state.Omega_e3phi,
                "scalar_P4": phi_v - state.Omega_e4phi,
                "scalar_gradient": nabla_phi - state.nabla_phi,
            }
        )
        result.update(
            {
                name: _summary(value, halo)
                for name, value in quantities.items()
                if name.startswith("scalar_")
            }
        )
    if protected_s_minimum is not None:
        if coordinates is None or not hasattr(coordinates, "s"):
            raise ValueError(
                "protected_s_minimum requires mapped coordinates"
            )
        safe_u = np.ones(len(u), dtype=bool)
        safe_v = np.asarray(coordinates.s.nodes) >= protected_s_minimum
        safe_u[:halo] = False
        safe_u[-halo:] = False
        safe_v[-halo:] = False
        for indices in coordinates.tau.indices[:-1]:
            center = int(indices[-1])
            safe_u[
                max(0, center - halo + 1) : min(
                    len(safe_u), center + halo
                )
            ] = False
        for indices in coordinates.s.indices[:-1]:
            center = int(indices[-1])
            safe_v[
                max(0, center - halo + 1) : min(
                    len(safe_v), center + halo
                )
            ] = False
        mask = safe_u[:, None] & safe_v[None, :]
        if not np.any(mask):
            raise ValueError("protected closure mask is empty")
        result["protected"] = {
            "minimum_s": protected_s_minimum,
            "cell_count": int(np.count_nonzero(mask)),
            **{
                name: _summary(value, halo, mask)
                for name, value in quantities.items()
            },
        }
    result["coordinate_derivatives"] = (
        "physical-coordinate local polynomial weights"
        if coordinates is None
        else "mapped computational-coordinate derivatives"
    )
    return result
