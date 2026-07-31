"""Explicit spectral deferred correction on one LGL element."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .lgl import LGLSegment


Array = np.ndarray
RightHandSide = Callable[[float, Array], Array]
Retraction = Callable[[Array], Array]


@dataclass(frozen=True)
class SDCResult:
    values: Array
    corrections: int
    converged: bool
    resolved: bool
    accepted: bool
    collocation_defect: float
    overgrid_defect: float
    correction_history: tuple[float, ...]


def _scaled_maximum(value: Array, reference: Array) -> float:
    scale = np.maximum(1.0, np.abs(reference))
    return float(np.max(np.abs(value) / scale))


def rk4_predictor(
    segment: LGLSegment,
    initial: Array,
    rhs: RightHandSide,
    retract: Retraction,
) -> Array:
    result = np.empty((len(segment.nodes), *initial.shape), dtype=float)
    result[0] = retract(np.asarray(initial, dtype=float))
    for index in range(len(segment.nodes) - 1):
        left = float(segment.nodes[index])
        right = float(segment.nodes[index + 1])
        step = right - left
        current = result[index]
        k1 = rhs(left, current)
        k2 = rhs(left + 0.5 * step, retract(current + 0.5 * step * k1))
        k3 = rhs(left + 0.5 * step, retract(current + 0.5 * step * k2))
        k4 = rhs(right, retract(current + step * k3))
        result[index + 1] = retract(
            current + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        )
    return result


def collocation_defect(
    segment: LGLSegment,
    values: Array,
    initial: Array,
    rhs: RightHandSide,
) -> float:
    forcing = np.stack(
        [rhs(float(node), value) for node, value in zip(segment.nodes, values)]
    )
    integral = np.tensordot(segment.integral, forcing, axes=(1, 0))
    defect = values - initial[None, ...] - integral
    return _scaled_maximum(defect, values)


def overgrid_defect(
    segment: LGLSegment,
    values: Array,
    rhs: RightHandSide,
    extra_degree: int = 3,
) -> float:
    over = LGLSegment.create(
        segment.left, segment.right, segment.degree + extra_degree
    )
    interpolation = segment.interpolation_matrix(over.nodes)
    derivative = segment.derivative_interpolation_matrix(over.nodes)
    over_values = np.tensordot(interpolation, values, axes=(1, 0))
    over_derivative = np.tensordot(derivative, values, axes=(1, 0))
    forcing = np.stack(
        [rhs(float(node), value) for node, value in zip(over.nodes, over_values)]
    )
    return _scaled_maximum(over_derivative - forcing, forcing)


def solve_sdc(
    segment: LGLSegment,
    initial: Array,
    rhs: RightHandSide,
    *,
    tolerance: float = 1.0e-11,
    overgrid_tolerance: float = np.inf,
    maximum_corrections: int = 12,
    retract: Retraction | None = None,
    predictor: Array | None = None,
) -> SDCResult:
    """Solve ``Y=Y_left+Q F(Y)`` with an explicit-Euler SDC preconditioner."""

    if (
        maximum_corrections < 1
        or tolerance <= 0.0
        or overgrid_tolerance <= 0.0
    ):
        raise ValueError("invalid SDC stopping parameters")
    retraction = (lambda value: value) if retract is None else retract
    initial_value = np.asarray(initial, dtype=float)
    if predictor is None:
        values = rk4_predictor(segment, initial_value, rhs, retraction)
    else:
        values = np.asarray(predictor, dtype=float).copy()
        if values.shape != (len(segment.nodes), *initial_value.shape):
            raise ValueError("the SDC predictor shape does not match the element")
        values[0] = retraction(initial_value)

    history: list[float] = []
    converged = False
    defect = collocation_defect(segment, values, initial_value, rhs)
    completed = 0
    for correction in range(1, maximum_corrections + 1):
        old = values
        forcing_old = np.stack(
            [
                rhs(float(node), value)
                for node, value in zip(segment.nodes, old)
            ]
        )
        updated = np.empty_like(old)
        updated[0] = retraction(initial_value)
        for interval in range(len(segment.nodes) - 1):
            step = float(
                segment.nodes[interval + 1] - segment.nodes[interval]
            )
            high_order_increment = np.tensordot(
                segment.integral[interval + 1]
                - segment.integral[interval],
                forcing_old,
                axes=(0, 0),
            )
            low_order_correction = step * (
                rhs(float(segment.nodes[interval]), updated[interval])
                - forcing_old[interval]
            )
            updated[interval + 1] = retraction(
                updated[interval]
                + low_order_correction
                + high_order_increment
            )
        change = _scaled_maximum(updated - old, updated)
        history.append(change)
        values = updated
        defect = collocation_defect(segment, values, initial_value, rhs)
        completed = correction
        if max(change, defect) <= tolerance:
            converged = True
            break

    independent = overgrid_defect(segment, values, rhs)
    resolved = bool(independent <= overgrid_tolerance)
    return SDCResult(
        values=values,
        corrections=completed,
        converged=converged,
        resolved=resolved,
        accepted=bool(converged and resolved),
        collocation_defect=defect,
        overgrid_defect=independent,
        correction_history=tuple(history),
    )
