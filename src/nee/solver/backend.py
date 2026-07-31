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
    lapse = np.asarray(state.omega)
    outgoing_form = np.asarray(state.shear) + 0.5 * (
        lapse**2 * np.asarray(state.q)
    )[..., None, None] * np.asarray(state.metric)
    is_scalar = hasattr(state, "phi")
    scalar_gradient = None
    if is_scalar and grid is not None:
        scalar_gradient = sphere.scalar_gradient(grid, np.asarray(state.phi))
    result = PicardState(
        sphere_metric=np.asarray(state.metric).copy(),
        shift=np.asarray(state.shift).copy(),
        log_lapse=np.log(lapse).copy(),
        outgoing_null_form=outgoing_form.copy(),
        incoming_null_form=np.asarray(state.weighted_chib).copy(),
        torsion=np.asarray(state.zeta_up).copy(),
        outgoing_weighted_omega=np.asarray(state.weighted_omega).copy(),
        incoming_weighted_omega=np.asarray(state.weighted_omegab).copy(),
        scalar=np.asarray(state.phi).copy() if is_scalar else None,
        scalar_e3=np.asarray(state.incoming_scalar).copy() if is_scalar else None,
        scalar_e4=np.asarray(state.scalar_p).copy() if is_scalar else None,
        scalar_sphere_gradient=scalar_gradient,
    )
    if grid is not None:
        validate_state(result, grid.frames)
    return result


def to_numerical(state: PicardState, *, scalar: bool = False) -> Any:
    common = dict(
        metric=state.sphere_metric.copy(),
        omega=state.lapse.copy(),
        zeta_up=state.torsion.copy(),
        shift=state.shift.copy(),
        q=state.q.copy(),
        shear=state.outgoing_shear.copy(),
        weighted_chib=state.incoming_null_form.copy(),
        weighted_omega=state.outgoing_weighted_omega.copy(),
        weighted_omegab=state.incoming_weighted_omega.copy(),
    )
    if not scalar:
        return vacuum_iteration.FirstOrderState(**common)
    if not state.is_scalar:
        raise ValueError("scalar sweep requires scalar state fields")
    return scalar_iteration.ESEState(
        **common,
        phi=np.asarray(state.scalar).copy(),
        scalar_p=np.asarray(state.scalar_e4).copy(),
        incoming_scalar=np.asarray(state.scalar_e3).copy(),
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
    # The numerical kernel reads q and shear as derived views of the complete
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
    result = np.zeros(new.sphere_metric.shape[1:3], dtype=float)
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
