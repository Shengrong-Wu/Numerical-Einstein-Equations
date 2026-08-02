"""Lossless adapters between the public full-form state and numerical kernels."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nee.numerics import scalar_iteration, sphere, vacuum_iteration
from nee.state.iterate import PicardState
from nee.state.validation import validate_state


Array = np.ndarray


def from_numerical(state: Any, grid: Any | None = None) -> PicardState:
    Omega = np.asarray(state.Omega)
    outgoing_form = np.asarray(state.Omega_chih) + 0.5 * (
        np.asarray(state.Omega_trchi)
    )[..., None, None] * np.asarray(state.g)
    is_scalar = hasattr(state, "phi")
    scalar_gradient = None
    if is_scalar and grid is not None:
        scalar_gradient = sphere.scalar_gradient(grid, np.asarray(state.phi))
    result = PicardState(
        g=np.asarray(state.g).copy(),
        b=np.asarray(state.b).copy(),
        log_Omega=np.log(Omega).copy(),
        Omega_chi=outgoing_form.copy(),
        Omega_chib=np.asarray(state.Omega_chib).copy(),
        zeta=np.asarray(state.zeta).copy(),
        Omega_omega=np.asarray(state.Omega_omega).copy(),
        Omega_omegab=np.asarray(state.Omega_omegab).copy(),
        phi=np.asarray(state.phi).copy() if is_scalar else None,
        Omega_e3phi=np.asarray(state.Omega_e3phi).copy() if is_scalar else None,
        Omega_e4phi=np.asarray(state.Omega_e4phi).copy() if is_scalar else None,
        nabla_phi=scalar_gradient,
    )
    if grid is not None:
        validate_state(result, grid.frames)
    return result


def to_numerical(state: PicardState, *, scalar: bool = False) -> Any:
    common = dict(
        g=state.g.copy(),
        Omega=state.Omega.copy(),
        zeta=state.zeta.copy(),
        b=state.b.copy(),
        Omega_trchi=state.Omega_trchi.copy(),
        Omega_chih=state.Omega_chih.copy(),
        Omega_chib=state.Omega_chib.copy(),
        Omega_omega=state.Omega_omega.copy(),
        Omega_omegab=state.Omega_omegab.copy(),
    )
    if not scalar:
        return vacuum_iteration.FirstOrderState(**common)
    if not state.is_scalar:
        raise ValueError("scalar sweep requires scalar state fields")
    return scalar_iteration.ESEState(
        **common,
        phi=np.asarray(state.phi).copy(),
        Omega_e4phi=np.asarray(state.Omega_e4phi).copy(),
        Omega_e3phi=np.asarray(state.Omega_e3phi).copy(),
    )


def vacuum_step(
    grid: Any,
    state: PicardState,
    outgoing: dict[str, Array],
    incoming: dict[str, Array],
    u: Array,
    v: Array,
    **kwargs: Any,
) -> tuple[PicardState, dict[str, Any]]:
    validate_state(state, grid.frames)
    # The numerical kernel reads Omega_trchi and Omega_chih as derived views of the complete
    # weighted forms. Passing a copied public state preserves that operation
    # order and prevents a redundant decompose/recompose roundoff cycle.
    working = state.copy()
    candidate, context = vacuum_iteration.picard_step(
        grid,
        working,
        outgoing,
        u,
        v,
        incoming=incoming,
        **kwargs,
    )
    return from_numerical(candidate, grid), context


def scalar_step(
    grid: Any,
    angular: Any,
    mesh: Any,
    state: PicardState,
    outgoing: dict[str, Array],
    incoming: dict[str, Array],
    **kwargs: Any,
) -> tuple[PicardState, dict[str, Any]]:
    if not state.is_scalar:
        raise ValueError("scalar sweep requires scalar fields")
    validate_state(state, grid.frames)
    bundle = scalar_iteration.InitialDataBundle(
        incoming=incoming,
        outgoing=outgoing,
        raw={},
        metadata={},
    )
    working = state.copy()
    candidate, context = scalar_iteration.picard_step(
        grid,
        angular,
        mesh,
        working,
        bundle,
        **kwargs,
    )
    return from_numerical(candidate, grid), context


def weighted_update_norm(new: PicardState, old: PicardState) -> float:
    values: list[float] = []
    for name in new.__dataclass_fields__:
        current = getattr(new, name)
        previous = getattr(old, name)
        if current is None and previous is None:
            continue
        if current is None or previous is None:
            raise ValueError(f"state type changed in field {name}")
        scale = np.maximum(1.0, np.maximum(np.abs(current), np.abs(previous)))
        values.append(float(np.mean(((current - previous) / scale) ** 2)))
    return math.sqrt(sum(values))


def weighted_update_map(new: PicardState, old: PicardState) -> Array:
    result = np.zeros(new.g.shape[1:3], dtype=float)
    for name in new.__dataclass_fields__:
        current = getattr(new, name)
        previous = getattr(old, name)
        if current is None and previous is None:
            continue
        if current is None or previous is None:
            raise ValueError(f"state type changed in field {name}")
        scale = np.maximum(1.0, np.maximum(np.abs(current), np.abs(previous)))
        relative_sq = ((current - previous) / scale) ** 2
        axes = (0, *range(3, relative_sq.ndim))
        result += np.mean(relative_sq, axis=axes)
    return np.sqrt(result)


def vacuum_picard_step(
    grid: Any,
    state: PicardState,
    boundary: dict[str, Array],
    u: Array,
    v: Array,
    **kwargs: Any,
) -> tuple[PicardState, dict[str, Any]]:
    incoming = kwargs.pop("incoming", {})
    if "coordinates" in kwargs and "scalar_coordinates" not in kwargs:
        kwargs["scalar_coordinates"] = kwargs.pop("coordinates")
    return vacuum_step(grid, state, boundary, incoming, u, v, **kwargs)


def ese_picard_step(
    grid: Any,
    angular: Any,
    mesh: Any,
    state: PicardState,
    boundary: Any,
    **kwargs: Any,
) -> tuple[PicardState, dict[str, Any]]:
    return scalar_step(
        grid,
        angular,
        mesh,
        state,
        dict(boundary.outgoing),
        dict(boundary.incoming),
        **kwargs,
    )
