"""Coordinate-aware spectral deferred correction on one tau LGL element.

This module is an isolated pilot for the incoming ``tau=-log(-u)`` solves.
It deliberately does not import, wrap, or modify the production SDC kernel.
The important differences from that kernel are:

* a retraction receives the coordinate of every trial value;
* both collocation and independent overgrid acceptance are scalar global
  gates; and
* the overgrid differential defect can retain selected batch axes so a
  failure can be localized rather than represented only by one maximum.

``batch_axes`` are numbered relative to a value at one tau, not relative to
the leading collocation/overgrid-node axis.  For example, if one value has
shape ``(sphere, v, component)``, then ``batch_axes=(1,)`` returns one defect
per ``v`` after maximizing over tau, sphere, and component.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .lgl import LGLSegment


Array = np.ndarray
RightHandSide = Callable[[float, Array], Array]
CoordinateRetraction = Callable[[float, Array], Array]


@dataclass(frozen=True)
class TauSDCResult:
    """One element's solution and independently checked acceptance data."""

    values: Array
    corrections: int
    converged: bool
    resolved: bool
    accepted: bool
    collocation_defect: float
    collocation_defect_by_batch: Array
    overgrid_defect: float
    overgrid_defect_by_batch: Array
    batch_axes: tuple[int, ...]
    correction_history: tuple[float, ...]


def identity_retraction(_: float, value: Array) -> Array:
    """Return ``value`` unchanged while satisfying the coordinate-aware API."""

    return value


def _retract(
    retract: CoordinateRetraction,
    coordinate: float,
    value: Array,
    expected_shape: tuple[int, ...],
) -> Array:
    result = np.asarray(retract(float(coordinate), value), dtype=float)
    if result.shape != expected_shape:
        raise ValueError(
            "the coordinate retraction changed the state shape: "
            f"expected {expected_shape}, received {result.shape}"
        )
    return result


def _rhs(
    rhs: RightHandSide,
    coordinate: float,
    value: Array,
    expected_shape: tuple[int, ...],
) -> Array:
    result = np.asarray(rhs(float(coordinate), value), dtype=float)
    if result.shape != expected_shape:
        raise ValueError(
            "the right-hand side changed the state shape: "
            f"expected {expected_shape}, received {result.shape}"
        )
    return result


def _scaled_residual(residual: Array, reference: Array) -> Array:
    """Pointwise mixed absolute/relative residual used by all gates."""

    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = np.abs(residual) / np.maximum(1.0, np.abs(reference))
    return np.where(np.isfinite(scaled), scaled, np.inf)


def _scaled_maximum(residual: Array, reference: Array) -> float:
    return float(np.max(_scaled_residual(residual, reference)))


def _normalize_batch_axes(
    batch_axes: int | tuple[int, ...], state_ndim: int
) -> tuple[int, ...]:
    raw = (batch_axes,) if isinstance(batch_axes, int) else tuple(batch_axes)
    normalized: list[int] = []
    for axis in raw:
        value = int(axis)
        if value < 0:
            value += state_ndim
        if value < 0 or value >= state_ndim:
            raise ValueError(
                f"batch axis {axis} is invalid for a {state_ndim}-D state"
            )
        if value in normalized:
            raise ValueError("batch axes must be unique")
        normalized.append(value)
    # NumPy retains unreduced axes in their original order.  Returning sorted
    # axes makes the profile shape and its metadata unambiguous.
    return tuple(sorted(normalized))


def _retain_batch_maximum(
    scaled: Array, state_ndim: int, batch_axes: tuple[int, ...]
) -> Array:
    # ``scaled`` has one leading coordinate-node axis.  ``batch_axes`` refer
    # to the axes after removing that leading coordinate axis.
    retained = {axis + 1 for axis in batch_axes}
    reduction = tuple(axis for axis in range(scaled.ndim) if axis not in retained)
    if not reduction:
        return np.asarray(scaled).copy()
    return np.asarray(np.max(scaled, axis=reduction))


def rk4_predictor(
    segment: LGLSegment,
    initial: Array,
    rhs: RightHandSide,
    retract: CoordinateRetraction = identity_retraction,
) -> Array:
    """Build an element predictor, retracting at each trial coordinate."""

    initial_array = np.asarray(initial, dtype=float)
    shape = initial_array.shape
    result = np.empty((len(segment.nodes), *shape), dtype=float)
    result[0] = _retract(retract, segment.nodes[0], initial_array, shape)
    for index in range(len(segment.nodes) - 1):
        left = float(segment.nodes[index])
        right = float(segment.nodes[index + 1])
        midpoint = 0.5 * (left + right)
        step = right - left
        current = result[index]
        k1 = _rhs(rhs, left, current, shape)
        stage2 = _retract(
            retract, midpoint, current + 0.5 * step * k1, shape
        )
        k2 = _rhs(rhs, midpoint, stage2, shape)
        stage3 = _retract(
            retract, midpoint, current + 0.5 * step * k2, shape
        )
        k3 = _rhs(rhs, midpoint, stage3, shape)
        stage4 = _retract(retract, right, current + step * k3, shape)
        k4 = _rhs(rhs, right, stage4, shape)
        result[index + 1] = _retract(
            retract,
            right,
            current + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0,
            shape,
        )
    return result


def _collocation_evaluation(
    segment: LGLSegment,
    values: Array,
    initial: Array,
    rhs: RightHandSide,
    *,
    batch_axes: tuple[int, ...],
    forcing: Array | None = None,
) -> tuple[float, Array, Array]:
    """Return collocation defects and the forcing used to form them.

    The forcing at the final defect check is precisely the ``forcing_old``
    needed by the next SDC correction.  Returning it prevents a second full
    PDE right-hand-side pass over the same collocation values.
    """

    values_array = np.asarray(values, dtype=float)
    initial_array = np.asarray(initial, dtype=float)
    expected = (len(segment.nodes), *initial_array.shape)
    if values_array.shape != expected:
        raise ValueError(
            f"collocation values have shape {values_array.shape}, expected {expected}"
        )
    if forcing is None:
        forcing_array = np.stack(
            [
                _rhs(rhs, float(node), value, initial_array.shape)
                for node, value in zip(
                    segment.nodes, values_array, strict=True
                )
            ]
        )
    else:
        forcing_array = np.asarray(forcing, dtype=float)
        if forcing_array.shape != values_array.shape:
            raise ValueError(
                "collocation forcing must have the same shape as values"
            )
    integral = np.tensordot(segment.integral, forcing_array, axes=(1, 0))
    residual = values_array - initial_array[None, ...] - integral
    scaled = _scaled_residual(residual, values_array)
    profile = _retain_batch_maximum(
        scaled, initial_array.ndim, batch_axes
    )
    return float(np.max(scaled)), profile, forcing_array


def collocation_defect(
    segment: LGLSegment,
    values: Array,
    initial: Array,
    rhs: RightHandSide,
) -> float:
    """Return the scalar global integral collocation defect."""

    defect, _, _ = _collocation_evaluation(
        segment,
        values,
        initial,
        rhs,
        batch_axes=(),
    )
    return defect


def overgrid_defects(
    segment: LGLSegment,
    values: Array,
    rhs: RightHandSide,
    *,
    retract: CoordinateRetraction = identity_retraction,
    extra_degree: int = 3,
    batch_axes: int | tuple[int, ...] = (),
) -> tuple[float, Array, tuple[int, ...]]:
    """Return global and batch-local independent differential defects.

    The degree-``p`` solution polynomial is differentiated at degree
    ``p+extra_degree`` LGL nodes.  The RHS is freshly evaluated there after a
    coordinate-aware retraction of the interpolated value.  Thus a small
    collocation defect at the original nodes is not sufficient for acceptance.
    """

    if extra_degree < 1:
        raise ValueError("the overgrid extra degree must be positive")
    values_array = np.asarray(values, dtype=float)
    if values_array.ndim < 1 or values_array.shape[0] != len(segment.nodes):
        raise ValueError("the leading values axis must match the LGL element")
    state_shape = values_array.shape[1:]
    normalized_axes = _normalize_batch_axes(batch_axes, len(state_shape))
    over = LGLSegment.create(
        segment.left, segment.right, segment.degree + extra_degree
    )
    interpolation = segment.interpolation_matrix(over.nodes)
    derivative = segment.derivative_interpolation_matrix(over.nodes)
    over_values = np.tensordot(interpolation, values_array, axes=(1, 0))
    over_derivative = np.tensordot(derivative, values_array, axes=(1, 0))
    forcing = np.stack(
        [
            _rhs(
                rhs,
                float(node),
                _retract(retract, float(node), value, state_shape),
                state_shape,
            )
            for node, value in zip(over.nodes, over_values, strict=True)
        ]
    )
    scaled = _scaled_residual(over_derivative - forcing, forcing)
    global_defect = float(np.max(scaled))
    by_batch = _retain_batch_maximum(
        scaled, len(state_shape), normalized_axes
    )
    return global_defect, by_batch, normalized_axes


def solve_tau_sdc(
    segment: LGLSegment,
    initial: Array,
    rhs: RightHandSide,
    *,
    tolerance: float = 1.0e-11,
    overgrid_tolerance: float = np.inf,
    maximum_corrections: int = 12,
    retract: CoordinateRetraction = identity_retraction,
    predictor: Array | None = None,
    overgrid_extra_degree: int = 3,
    batch_axes: int | tuple[int, ...] = (),
) -> TauSDCResult:
    """Solve ``Y=Y_left+Q F(Y)`` with coordinate-aware explicit SDC."""

    if (
        maximum_corrections < 1
        or tolerance <= 0.0
        or overgrid_tolerance <= 0.0
        or overgrid_extra_degree < 1
    ):
        raise ValueError("invalid tau-SDC stopping parameters")

    supplied_initial = np.asarray(initial, dtype=float)
    state_shape = supplied_initial.shape
    initial_value = _retract(
        retract, float(segment.nodes[0]), supplied_initial, state_shape
    )
    normalized_axes = _normalize_batch_axes(batch_axes, len(state_shape))

    if predictor is None:
        values = rk4_predictor(
            segment, initial_value, rhs, retract=retract
        )
    else:
        predictor_array = np.asarray(predictor, dtype=float)
        expected = (len(segment.nodes), *state_shape)
        if predictor_array.shape != expected:
            raise ValueError(
                f"the predictor has shape {predictor_array.shape}, expected {expected}"
            )
        values = np.stack(
            [
                _retract(retract, float(node), value, state_shape)
                for node, value in zip(
                    segment.nodes, predictor_array, strict=True
                )
            ]
        )
        values[0] = initial_value

    history: list[float] = []
    converged = False
    defect, collocation_profile, forcing_old = _collocation_evaluation(
        segment,
        values,
        initial_value,
        rhs,
        batch_axes=normalized_axes,
    )
    completed = 0
    for correction in range(1, maximum_corrections + 1):
        old = values
        updated = np.empty_like(old)
        updated[0] = initial_value
        for interval in range(len(segment.nodes) - 1):
            left = float(segment.nodes[interval])
            right = float(segment.nodes[interval + 1])
            step = right - left
            high_order_increment = np.tensordot(
                segment.integral[interval + 1]
                - segment.integral[interval],
                forcing_old,
                axes=(0, 0),
            )
            low_order_correction = step * (
                _rhs(rhs, left, updated[interval], state_shape)
                - forcing_old[interval]
            )
            updated[interval + 1] = _retract(
                retract,
                right,
                updated[interval]
                + low_order_correction
                + high_order_increment,
                state_shape,
            )
        change = _scaled_maximum(updated - old, updated)
        history.append(change)
        values = updated
        defect, collocation_profile, forcing_old = _collocation_evaluation(
            segment,
            values,
            initial_value,
            rhs,
            batch_axes=normalized_axes,
        )
        completed = correction
        if max(change, defect) <= tolerance:
            converged = True
            break

    independent, by_batch, checked_axes = overgrid_defects(
        segment,
        values,
        rhs,
        retract=retract,
        extra_degree=overgrid_extra_degree,
        batch_axes=normalized_axes,
    )
    resolved = bool(independent <= overgrid_tolerance)
    return TauSDCResult(
        values=values,
        corrections=completed,
        converged=converged,
        resolved=resolved,
        accepted=bool(converged and resolved),
        collocation_defect=defect,
        collocation_defect_by_batch=collocation_profile,
        overgrid_defect=independent,
        overgrid_defect_by_batch=by_batch,
        batch_axes=checked_axes,
        correction_history=tuple(history),
    )
