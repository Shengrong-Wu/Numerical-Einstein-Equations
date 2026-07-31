"""Smooth two-hemisphere outgoing shear datum for the vacuum experiment.

The two round-metric trace-free tensors are squares of stereographic
translation fields.  ``X_upper`` has its only zero at the south pole and is
uniformly nonzero on the closed upper hemisphere; ``X_lower`` is the reflected
construction.  Their time profiles have disjoint supports in ``[0,v1/2)`` and
``(v1/2,v1)`` and are flat at the separating point.  The first profile has the
allowed ``sqrt(v)`` behavior at the initial corner, while both profiles are
C-infinity for every ``v>0``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .coordinate_quadrature import cumulative_polynomial_quadrature
from .sphere import (
    PointSphereGrid,
    gaussian_curvature,
    scalar_gradient,
    tangent_inverse,
    tensor_divergence,
    tensor_norm_sq,
)


Array = np.ndarray
TENSOR_SCALE = 2.0
TENSOR_REGULARIZATION = 0.1
HEMISPHERE_TENSOR_LOWER_BOUND = (
    TENSOR_SCALE / math.sqrt(2.0) / (1.0 + TENSOR_REGULARIZATION)
)
PROFILE_TRANSITION_FRACTION = 0.08
CORNER_SCALE_FRACTION = 0.004
SUPPORT_MARGIN_FRACTION = 0.01
PROFILE_OSCILLATIONS = 8
LOWER_DIRECTION_ANGLE = 7.0 * math.pi / 18.0


@dataclass(frozen=True)
class HemisphereProfileCalibration:
    v1: float
    split: float
    c: float
    safety_factor: float
    first_shape_integral: float
    second_shape_integral: float
    first_amplitude: float
    second_amplitude: float
    hemisphere_tensor_lower_bound: float


class BoundaryDegeneracy(FloatingPointError):
    def __init__(self, last_regular_v: float, failed_v: float, minimum_eigenvalue: float):
        super().__init__(
            f"outgoing boundary lost positivity between v={last_regular_v:.8g} "
            f"and v={failed_v:.8g}"
        )
        self.last_regular_v = last_regular_v
        self.failed_v = failed_v
        self.minimum_eigenvalue = minimum_eigenvalue


def scaled_calibration(
    calibration: HemisphereProfileCalibration, divisor: float
) -> HemisphereProfileCalibration:
    """Return the datum ``hat(chi)0 / divisor`` without recalibrating shapes."""

    if not math.isfinite(divisor) or divisor <= 0.0:
        raise ValueError("the shear divisor must be positive and finite")
    return HemisphereProfileCalibration(
        v1=calibration.v1,
        split=calibration.split,
        c=calibration.c,
        safety_factor=calibration.safety_factor,
        first_shape_integral=calibration.first_shape_integral,
        second_shape_integral=calibration.second_shape_integral,
        first_amplitude=calibration.first_amplitude / divisor,
        second_amplitude=calibration.second_amplitude / divisor,
        hemisphere_tensor_lower_bound=calibration.hemisphere_tensor_lower_bound,
    )


def smooth_step(t: Array | float) -> Array:
    """C-infinity transition from zero to one on the unit interval."""

    value = np.asarray(t, dtype=float)
    result = np.zeros_like(value)
    result[value >= 1.0] = 1.0
    active = (value > 0.0) & (value < 1.0)
    ta = value[active]
    left = np.exp(-1.0 / ta)
    right = np.exp(-1.0 / (1.0 - ta))
    result[active] = left / (left + right)
    return result


def left_profile_shape(v: Array | float, split: float) -> Array:
    """sqrt(v) at zero, nearly constant inside, and flat at ``split``."""

    value = np.asarray(v, dtype=float)
    result = np.zeros_like(value)
    support_end = (1.0 - SUPPORT_MARGIN_FRACTION) * split
    active = (value > 0.0) & (value < support_end)
    va = value[active]
    transition = PROFILE_TRANSITION_FRACTION * support_end
    corner_scale = CORNER_SCALE_FRACTION * support_end
    corner = np.sqrt(va / (va + corner_scale))
    cutoff = 1.0 - smooth_step(
        (va - (support_end - transition)) / transition
    )
    oscillation = np.cos(
        2.0 * math.pi * PROFILE_OSCILLATIONS * va / support_end
    )
    result[active] = corner * cutoff * oscillation
    return result


def right_profile_shape(v: Array | float, split: float, v1: float) -> Array:
    """C-infinity bump supported strictly between ``split`` and ``v1``."""

    value = np.asarray(v, dtype=float)
    result = np.zeros_like(value)
    half_length = v1 - split
    support_start = split + SUPPORT_MARGIN_FRACTION * half_length
    support_end = v1 - SUPPORT_MARGIN_FRACTION * half_length
    active = (value > support_start) & (value < support_end)
    va = value[active]
    support_length = support_end - support_start
    transition = PROFILE_TRANSITION_FRACTION * support_length
    rise = smooth_step((va - support_start) / transition)
    fall = 1.0 - smooth_step(
        (va - (support_end - transition)) / transition
    )
    oscillation = np.cos(
        2.0
        * math.pi
        * PROFILE_OSCILLATIONS
        * (va - support_start)
        / support_length
    )
    result[active] = rise * fall * oscillation
    return result


def calibrate_profiles(
    v1: float,
    c: float,
    split: float | None = None,
    safety_factor: float = 1.01,
    integration_count: int = 65537,
) -> HemisphereProfileCalibration:
    if v1 <= 0.0 or c <= 0.0 or integration_count < 1025:
        raise ValueError("invalid smooth-hemisphere pulse parameters")
    value_split = 0.5 * v1 if split is None else float(split)
    if not 0.0 < value_split < v1:
        raise ValueError("the profile split must lie strictly inside (0,v1)")
    v = np.linspace(0.0, v1, integration_count)
    first_integral = float(
        np.trapezoid(np.abs(left_profile_shape(v, value_split)), v)
    )
    second_integral = float(
        np.trapezoid(np.abs(right_profile_shape(v, value_split, v1)), v)
    )
    lower_bound = HEMISPHERE_TENSOR_LOWER_BOUND
    return HemisphereProfileCalibration(
        v1=v1,
        split=value_split,
        c=c,
        safety_factor=safety_factor,
        first_shape_integral=first_integral,
        second_shape_integral=second_integral,
        first_amplitude=safety_factor * c / (lower_bound * first_integral),
        second_amplitude=safety_factor * c / (lower_bound * second_integral),
        hemisphere_tensor_lower_bound=lower_bound,
    )


def one_zero_tangent_field(points: Array, pole: Array, direction: Array) -> Array:
    """Stereographic translation field whose only zero is ``pole``."""

    pole = np.asarray(pole, dtype=float)
    direction = np.asarray(direction, dtype=float)
    pole /= np.linalg.norm(pole)
    direction -= np.dot(direction, pole) * pole
    direction /= np.linalg.norm(direction)
    point_pole = points @ pole
    point_direction = points @ direction
    return (
        (1.0 - point_pole)[:, None] * direction[None, :]
        - point_direction[:, None] * (points - pole[None, :])
    )


def square_tracefree_tensor(
    grid: PointSphereGrid, vector: Array, scale: float = TENSOR_SCALE
) -> Array:
    norm_sq = np.einsum("ni,ni->n", vector, vector)
    return scale * (
        np.einsum("ni,nj->nij", vector, vector)
        - 0.5 * norm_sq[:, None, None] * grid.projector
    )


def hemisphere_tensors(grid: PointSphereGrid) -> tuple[Array, Array]:
    """Return ``(X_upper, X_lower)`` with zeros at south/north respectively."""

    north = np.array([0.0, 0.0, 1.0])
    south = -north
    x_direction = np.array([1.0, 0.0, 0.0])
    lower_direction = np.array(
        [math.cos(LOWER_DIRECTION_ANGLE), math.sin(LOWER_DIRECTION_ANGLE), 0.0]
    )
    upper_vector = one_zero_tangent_field(grid.points, south, x_direction)
    lower_vector = one_zero_tangent_field(grid.points, north, lower_direction)
    z = grid.points[:, 2]
    upper_denominator = (
        (1.0 + z) ** 2 + TENSOR_REGULARIZATION * (1.0 - z) ** 2
    )
    lower_denominator = (
        (1.0 - z) ** 2 + TENSOR_REGULARIZATION * (1.0 + z) ** 2
    )
    return (
        square_tracefree_tensor(grid, upper_vector)
        / upper_denominator[:, None, None],
        square_tracefree_tensor(grid, lower_vector)
        / lower_denominator[:, None, None],
    )


def profile_values(
    v: Array | float, calibration: HemisphereProfileCalibration
) -> tuple[Array, Array]:
    return (
        calibration.first_amplitude
        * left_profile_shape(v, calibration.split),
        calibration.second_amplitude
        * right_profile_shape(v, calibration.split, calibration.v1),
    )


def reference_shear(
    tensors: tuple[Array, Array],
    value_v: float,
    calibration: HemisphereProfileCalibration,
) -> Array:
    first, second = profile_values(np.array([value_v]), calibration)
    return first[0] * tensors[0] + second[0] * tensors[1]


def transferred_shear(sphere: PointSphereGrid, chi0: Array, metric: Array) -> Array:
    mixed = np.matmul(np.matmul(chi0, sphere.projector), metric)
    return 0.5 * (mixed + np.swapaxes(mixed, -1, -2))


def minimum_tangent_eigenvalue(sphere: PointSphereGrid, metric: Array) -> float:
    local = np.einsum("nia,n...ij,njb->n...ab", sphere.frames, metric, sphere.frames)
    if not np.all(np.isfinite(local)):
        return -math.inf
    return float(np.min(np.linalg.eigvalsh(local)))


def solve_boundary_zeta(
    sphere: PointSphereGrid,
    metric: Array,
    expansion: Array,
    shear: Array,
    v: Array,
    tolerance: float = 1.0e-13,
    maximum_iterations: int = 48,
) -> tuple[Array, Array, Array, int, float]:
    """Solve the vacuum ``zeta`` constraint with fixed boundary ``g,chi``."""

    _, difference, inverse = gaussian_curvature(sphere, metric)
    div_shear = tensor_divergence(sphere, shear, difference, inverse)
    grad_expansion = scalar_gradient(sphere, expansion)
    free_source = (
        np.einsum("nvij,nvj->nvi", inverse, div_shear)
        - 0.5 * np.einsum("nvij,nvj->nvi", inverse, grad_expansion)
    )
    zeta = np.zeros(expansion.shape + (3,), dtype=float)
    source = free_source.copy()
    update = math.inf
    completed = 0
    for iteration in range(maximum_iterations):
        shear_zeta = np.einsum("nvij,nvj->nvi", shear, zeta)
        source = (
            -2.0 * expansion[..., None] * zeta
            - 2.0 * np.einsum("nvij,nvj->nvi", inverse, shear_zeta)
            + free_source
        )
        new_zeta = cumulative_polynomial_quadrature(source, v, axis=1)
        new_zeta = np.einsum("nij,nvj->nvi", sphere.projector, new_zeta)
        difference_zeta = new_zeta - zeta
        update = float(
            np.sqrt(
                np.mean(
                    np.maximum(
                        np.einsum(
                            "nvi,nvij,nvj->nv",
                            difference_zeta,
                            metric,
                            difference_zeta,
                        ),
                        0.0,
                    )
                )
            )
        )
        zeta = new_zeta
        completed = iteration + 1
        if update <= tolerance:
            break
    shear_zeta = np.einsum("nvij,nvj->nvi", shear, zeta)
    source = (
        -2.0 * expansion[..., None] * zeta
        - 2.0 * np.einsum("nvij,nvj->nvi", inverse, shear_zeta)
        + free_source
    )
    shift = cumulative_polynomial_quadrature(-4.0 * zeta, v, axis=1)
    shift = np.einsum("nij,nvj->nvi", sphere.projector, shift)
    return zeta, shift, source, completed, update


def solve_outgoing_boundary(
    sphere: PointSphereGrid,
    v: Array,
    calibration: HemisphereProfileCalibration,
    minimum_eigenvalue: float = 1.0e-7,
) -> dict[str, Array | float]:
    """Solve the H_-1 metric and Raychaudhuri constraints with RK4."""

    if len(v) < 9 or abs(float(v[0])) > 1.0e-15 or np.any(np.diff(v) <= 0.0):
        raise ValueError("v must be a strictly increasing grid beginning at zero")
    if float(v[-1]) > calibration.v1 + 1.0e-14:
        raise ValueError("the v grid extends beyond the calibrated pulse")
    tensors = hemisphere_tensors(sphere)
    count = len(v)
    metric = np.zeros((sphere.count, count, 3, 3), dtype=float)
    expansion = np.zeros((sphere.count, count), dtype=float)
    shear = np.zeros_like(metric)
    reference = np.zeros_like(metric)
    metric[:, 0] = sphere.projector
    expansion[:, 0] = 2.0

    def rhs(
        value: float, current_metric: Array, current_expansion: Array
    ) -> tuple[Array, Array]:
        chi0 = reference_shear(tensors, value, calibration)
        current_shear = transferred_shear(sphere, chi0, current_metric)
        inverse = tangent_inverse(sphere, current_metric)
        norm_sq = tensor_norm_sq(current_shear, inverse)
        return (
            current_expansion[:, None, None] * current_metric + 2.0 * current_shear,
            -0.5 * current_expansion**2 - norm_sq,
        )

    for j in range(count - 1):
        step = float(v[j + 1] - v[j])
        value = float(v[j])
        current_metric = metric[:, j]
        current_expansion = expansion[:, j]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                k1_metric, k1_expansion = rhs(value, current_metric, current_expansion)
                k2_metric, k2_expansion = rhs(
                    value + 0.5 * step,
                    current_metric + 0.5 * step * k1_metric,
                    current_expansion + 0.5 * step * k1_expansion,
                )
                k3_metric, k3_expansion = rhs(
                    value + 0.5 * step,
                    current_metric + 0.5 * step * k2_metric,
                    current_expansion + 0.5 * step * k2_expansion,
                )
                k4_metric, k4_expansion = rhs(
                    value + step,
                    current_metric + step * k3_metric,
                    current_expansion + step * k3_expansion,
                )
                next_metric = current_metric + step * (
                    k1_metric + 2.0 * k2_metric + 2.0 * k3_metric + k4_metric
                ) / 6.0
                next_expansion = current_expansion + step * (
                    k1_expansion
                    + 2.0 * k2_expansion
                    + 2.0 * k3_expansion
                    + k4_expansion
                ) / 6.0
            except (FloatingPointError, np.linalg.LinAlgError) as error:
                raise BoundaryDegeneracy(float(v[j]), float(v[j + 1]), -math.inf) from error
        eigenvalue = minimum_tangent_eigenvalue(sphere, next_metric)
        if eigenvalue <= minimum_eigenvalue:
            raise BoundaryDegeneracy(float(v[j]), float(v[j + 1]), eigenvalue)
        metric[:, j + 1] = next_metric
        expansion[:, j + 1] = next_expansion

    inverse = tangent_inverse(sphere, metric)
    for j, value in enumerate(v):
        reference[:, j] = reference_shear(tensors, float(value), calibration)
        shear[:, j] = transferred_shear(sphere, reference[:, j], metric[:, j])
    reference_norm = np.sqrt(
        np.maximum(np.einsum("nvij,nvij->nv", reference, reference), 0.0)
    )
    physical_norm = np.sqrt(np.maximum(tensor_norm_sq(shear, inverse), 0.0))
    reference_integral = np.trapezoid(reference_norm, v, axis=1)
    physical_integral = np.trapezoid(physical_norm, v, axis=1)
    upper = sphere.points[:, 2] >= 0.0
    lower = sphere.points[:, 2] <= 0.0
    first_profile, second_profile = profile_values(v, calibration)
    first_integral = np.trapezoid(
        np.abs(first_profile)[None, :] * np.sqrt(
            np.maximum(np.einsum("nij,nij->n", tensors[0], tensors[0]), 0.0)
        )[:, None],
        v,
        axis=1,
    )
    second_integral = np.trapezoid(
        np.abs(second_profile)[None, :] * np.sqrt(
            np.maximum(np.einsum("nij,nij->n", tensors[1], tensors[1]), 0.0)
        )[:, None],
        v,
        axis=1,
    )
    trace = np.einsum("nvij,nvji->nv", inverse, shear)
    reference_trace = np.einsum("nvij,nij->nv", reference, sphere.projector)
    zeta, shift, zeta_source, zeta_iterations, zeta_update = solve_boundary_zeta(
        sphere, metric, expansion, shear, v
    )
    return {
        "metric": metric,
        "inverse": inverse,
        "expansion": expansion,
        "shear": shear,
        "reference_shear": reference,
        "zeta_up": zeta,
        "shift": shift,
        "zeta_source": zeta_source,
        "minimum_eigenvalue": minimum_tangent_eigenvalue(sphere, metric),
        "minimum_reference_l1": float(np.min(reference_integral)),
        "minimum_physical_l1": float(np.min(physical_integral)),
        "minimum_upper_first_l1": float(np.min(first_integral[upper])),
        "minimum_lower_second_l1": float(np.min(second_integral[lower])),
        "maximum_reference_trace": float(np.max(np.abs(reference_trace))),
        "maximum_physical_trace": float(np.max(np.abs(trace))),
        "minimum_expansion": float(np.min(expansion)),
        "maximum_final_expansion": float(np.max(expansion[:, -1])),
        "zeta_iterations": zeta_iterations,
        "zeta_final_update": zeta_update,
    }
