"""Low-band two-hemisphere characteristic datum for near-cone tests.

This module keeps the geometric construction used by the smooth-hemisphere
experiment but removes the two features that made the datum needlessly hard
to resolve:

* no angular norm-flattening denominator is used; and
* no temporal sign oscillations are inserted.

The two reference tensors are polynomial ambient tangent tensors.  The first
has its only zero at the south pole and is bounded from below on the closed
upper hemisphere; the second has the reflected property.  The time profiles
have disjoint supports, flat interior endpoints, and the allowed ``sqrt(v)``
corner behavior.  The outgoing metric, expansion, and transferred shear are
advanced simultaneously at every RK4 stage.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent

from .sphere import (  # noqa: E402
    PointSphereGrid,
    connection_difference,
    scalar_gradient,
    tangent_inverse,
    tensor_divergence,
    tensor_norm_sq,
)
from .coordinate_quadrature import (  # noqa: E402
    cumulative_polynomial_quadrature,
)
from .spherical_harmonics import AngularGalerkin  # noqa: E402
from .lgl import CharacteristicLGLMesh  # noqa: E402
from .sdc import solve_sdc  # noqa: E402
from .smooth_pulse import (  # noqa: E402
    BoundaryDegeneracy,
    minimum_tangent_eigenvalue,
    one_zero_tangent_field,
    smooth_step,
    solve_boundary_zeta,
    square_tracefree_tensor,
    transferred_shear,
)


Array = np.ndarray
PROFILE_TRANSITION_FRACTION = 0.08
CORNER_SCALE_FRACTION = 0.004
SUPPORT_MARGIN_FRACTION = 0.01
LOWER_DIRECTION_ANGLE = 7.0 * math.pi / 18.0

# A pure ell=2 stereographic tensor has norm sqrt(2)(1+z)^2 on the upper
# hemisphere, hence a factor-four amplitude variation.  That variation made
# the full-Q1 outgoing constraint nearly degenerate even after the earlier
# divisor 3.2 was applied.  Multiply it by the degree-three Chebyshev
# interpolant of (1+z)^(-2) on [0,1].  The resulting tensor is still exactly
# band limited (ell<=5), while its norm varies by only 2.3% on the protected
# hemisphere.  The reflected polynomial is used for the lower tensor.
HEMISPHERE_FLATTENING_COEFFICIENTS = (
    0.99336495168639583,
    -1.7793103033980233,
    1.6605101988712332,
    -0.62744909575314978,
)
# The true minimum multiplier is >0.9884.  A slightly conservative analytic
# calibration bound leaves room for floating-point and sampling error.
POLYNOMIAL_HEMISPHERE_LOWER_BOUND = 0.98 * math.sqrt(2.0)


@dataclass(frozen=True)
class LowBandCalibration:
    v1: float
    split: float
    c: float
    divisor: float
    safety_factor: float
    first_shape_integral: float
    second_shape_integral: float
    first_amplitude: float
    second_amplitude: float
    hemisphere_tensor_lower_bound: float

    @property
    def target_l1(self) -> float:
        return self.c / self.divisor


def left_profile_shape(v: Array | float, split: float) -> Array:
    """Nonoscillatory ``sqrt(v)`` corner with a flat support endpoint."""

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
    result[active] = corner * cutoff
    return result


def right_profile_shape(v: Array | float, split: float, v1: float) -> Array:
    """Nonoscillatory C-infinity bump supported strictly after ``split``."""

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
    result[active] = rise * fall
    return result


def calibrate_low_band_profiles(
    v1: float,
    c: float,
    divisor: float = 3.2,
    split: float | None = None,
    safety_factor: float = 1.01,
    integration_count: int = 65537,
) -> LowBandCalibration:
    """Calibrate the continuum hemisphere L1 lower bound after division."""

    if (
        v1 <= 0.0
        or c <= 0.0
        or divisor <= 0.0
        or integration_count < 1025
    ):
        raise ValueError("invalid low-band pulse parameters")
    value_split = 0.5 * v1 if split is None else float(split)
    if not 0.0 < value_split < v1:
        raise ValueError("the profile split must lie strictly inside (0,v1)")
    values = np.linspace(0.0, v1, integration_count)
    first_integral = float(
        np.trapezoid(left_profile_shape(values, value_split), values)
    )
    second_integral = float(
        np.trapezoid(
            right_profile_shape(values, value_split, v1), values
        )
    )
    lower = POLYNOMIAL_HEMISPHERE_LOWER_BOUND
    target = safety_factor * c / divisor
    return LowBandCalibration(
        v1=v1,
        split=value_split,
        c=c,
        divisor=divisor,
        safety_factor=safety_factor,
        first_shape_integral=first_integral,
        second_shape_integral=second_integral,
        first_amplitude=target / (lower * first_integral),
        second_amplitude=target / (lower * second_integral),
        hemisphere_tensor_lower_bound=lower,
    )


def polynomial_hemisphere_tensors(
    sphere: PointSphereGrid,
) -> tuple[Array, Array]:
    """Return flat-norm, band-limited STF tensors with opposite zeros."""

    north = np.array([0.0, 0.0, 1.0])
    south = -north
    first_direction = np.array([1.0, 0.0, 0.0])
    second_direction = np.array(
        [
            math.cos(LOWER_DIRECTION_ANGLE),
            math.sin(LOWER_DIRECTION_ANGLE),
            0.0,
        ]
    )
    upper_vector = one_zero_tangent_field(
        sphere.points, south, first_direction
    )
    lower_vector = one_zero_tangent_field(
        sphere.points, north, second_direction
    )
    z = sphere.points[:, 2]

    def flattening_polynomial(value: Array) -> Array:
        result = np.zeros_like(value) + HEMISPHERE_FLATTENING_COEFFICIENTS[-1]
        for coefficient in reversed(
            HEMISPHERE_FLATTENING_COEFFICIENTS[:-1]
        ):
            result = coefficient + value * result
        return result

    upper_weight = flattening_polynomial(z)
    lower_weight = flattening_polynomial(-z)
    return (
        square_tracefree_tensor(sphere, upper_vector)
        * upper_weight[:, None, None],
        square_tracefree_tensor(sphere, lower_vector)
        * lower_weight[:, None, None],
    )


def profile_values(
    v: Array | float, calibration: LowBandCalibration
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
    calibration: LowBandCalibration,
) -> Array:
    first, second = profile_values(np.array([value_v]), calibration)
    return first[0] * tensors[0] + second[0] * tensors[1]


def solve_low_band_boundary(
    sphere: PointSphereGrid,
    v: Array,
    calibration: LowBandCalibration,
    minimum_eigenvalue: float = 1.0e-7,
    angular: AngularGalerkin | None = None,
    scalar_coordinates: CharacteristicLGLMesh | None = None,
    coordinate_integrator: str = "rk4",
    sdc_overgrid_tolerance: float = np.inf,
) -> dict[str, Array | float]:
    """Solve the coupled outgoing characteristic constraints with RK4."""

    if len(v) < 9 or abs(float(v[0])) > 1.0e-15 or np.any(np.diff(v) <= 0.0):
        raise ValueError("v must be a strictly increasing grid beginning at zero")
    if float(v[-1]) > calibration.v1 + 1.0e-14:
        raise ValueError("the v grid extends beyond the calibrated pulse")
    if scalar_coordinates is not None and not np.allclose(
        scalar_coordinates.v, v, rtol=0.0, atol=2.0e-14
    ):
        raise ValueError("the characteristic LGL v grid does not match the boundary")
    if coordinate_integrator not in {"rk4", "sdc"}:
        raise ValueError("coordinate_integrator must be 'rk4' or 'sdc'")
    if coordinate_integrator == "sdc" and scalar_coordinates is None:
        raise ValueError("the boundary SDC integrator requires an LGL mesh")

    def integrate_v(value: Array) -> Array:
        if scalar_coordinates is None:
            return cumulative_polynomial_quadrature(value, v, axis=1)
        return scalar_coordinates.integrate_v(value, axis=1)

    tensors = polynomial_hemisphere_tensors(sphere)
    count = len(v)
    metric = np.zeros((sphere.count, count, 3, 3), dtype=float)
    expansion = np.zeros((sphere.count, count), dtype=float)
    shear = np.zeros_like(metric)
    reference = np.zeros_like(metric)
    metric[:, 0] = sphere.projector
    expansion[:, 0] = 2.0

    boundary_tails: dict[str, float] = {}

    def record_tail(kind: str, value: Array, label: str) -> None:
        if angular is None:
            return
        family = getattr(angular, kind)
        coefficients = family.analyze(value)
        total = float(np.sum(np.abs(coefficients) ** 2))
        discarded = float(
            np.sum(np.abs(coefficients[~family.retained]) ** 2)
        )
        maximum = math.sqrt(discarded / max(total, 1.0e-300))
        boundary_tails[label] = max(
            boundary_tails.get(label, 0.0), maximum
        )

    def constrained_shear(chi0: Array, value_metric: Array) -> Array:
        raw = transferred_shear(sphere, chi0, value_metric)
        inverse = tangent_inverse(sphere, value_metric)
        # Pointwise trace removal is part of the nonlinear transfer.  The
        # retained coefficient constraint then prevents an invisible nodal
        # trace component from becoming an iteration variable.
        trace = np.einsum("n...ij,n...ij->n...", inverse, raw)
        raw = raw - 0.5 * trace[..., None, None] * value_metric
        if angular is None:
            return raw
        record_tail("sym2", raw, "boundary_shear_transfer")
        # The physical shear is an algebraic function of retained primitives,
        # just like g^{-1}.  Keep its exact pointwise trace constraint on the
        # work grid and project only each complete evolution RHS that uses it.
        return raw

    def rhs(
        value: float, current_metric: Array, current_expansion: Array
    ) -> tuple[Array, Array]:
        chi0 = reference_shear(tensors, value, calibration)
        if angular is not None:
            chi0 = angular.project_sym2(chi0)
            round_trace = np.einsum("nij,nij->n", sphere.projector, chi0)
            chi0 = chi0 - 0.5 * round_trace[:, None, None] * sphere.projector
        current_shear = constrained_shear(chi0, current_metric)
        inverse = tangent_inverse(sphere, current_metric)
        norm_sq = tensor_norm_sq(current_shear, inverse)
        metric_rhs = (
            current_expansion[:, None, None] * current_metric
            + 2.0 * current_shear
        )
        expansion_rhs = -0.5 * current_expansion**2 - norm_sq
        if angular is not None:
            record_tail("sym2", metric_rhs, "boundary_metric_rhs")
            record_tail("scalar", expansion_rhs, "boundary_expansion_rhs")
            metric_rhs = angular.project_sym2(metric_rhs)
            expansion_rhs = angular.project_scalar(expansion_rhs)
        return metric_rhs, expansion_rhs

    boundary_sdc_diagnostics: list[dict[str, float | int | bool]] = []
    if coordinate_integrator == "rk4":
        for j in range(count - 1):
            step = float(v[j + 1] - v[j])
            value = float(v[j])
            current_metric = metric[:, j]
            current_expansion = expansion[:, j]
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                try:
                    k1_metric, k1_expansion = rhs(
                        value, current_metric, current_expansion
                    )
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
                except (FloatingPointError, np.linalg.LinAlgError) as error:
                    raise BoundaryDegeneracy(
                        float(v[j]), float(v[j + 1]), -math.inf
                    ) from error
            next_metric = current_metric + step * (
                k1_metric + 2.0 * k2_metric + 2.0 * k3_metric + k4_metric
            ) / 6.0
            next_expansion = current_expansion + step * (
                k1_expansion
                + 2.0 * k2_expansion
                + 2.0 * k3_expansion
                + k4_expansion
            ) / 6.0
            eigenvalue = minimum_tangent_eigenvalue(sphere, next_metric)
            if eigenvalue <= minimum_eigenvalue:
                raise BoundaryDegeneracy(
                    float(v[j]), float(v[j + 1]), eigenvalue
                )
            metric[:, j + 1] = next_metric
            expansion[:, j + 1] = next_expansion
    else:
        assert scalar_coordinates is not None
        local_initial = np.einsum(
            "nia,nij,njb->nab",
            sphere.frames,
            metric[:, 0],
            sphere.frames,
        )
        factor = np.zeros((sphere.count, count, 2, 2), dtype=float)
        factor[:, 0] = np.linalg.cholesky(local_initial)

        def ambient_metric(value_factor: Array) -> Array:
            local_metric = np.matmul(
                value_factor, np.swapaxes(value_factor, -1, -2)
            )
            return np.einsum(
                "nia,n...ab,njb->n...ij",
                sphere.frames,
                local_metric,
                sphere.frames,
            )

        for element, (segment, index) in enumerate(
            zip(scalar_coordinates.s.segments, scalar_coordinates.s.indices, strict=True)
        ):
            initial_packed = np.concatenate(
                (
                    expansion[:, index[0], None],
                    factor[:, index[0]].reshape((sphere.count, 4)),
                ),
                axis=-1,
            )

            def rhs_s(value_s: float, packed: Array) -> Array:
                value_expansion = packed[:, 0]
                value_factor = packed[:, 1:].reshape((sphere.count, 2, 2))
                value_metric = ambient_metric(value_factor)
                metric_rhs, expansion_rhs = rhs(
                    calibration.v1 * value_s**2,
                    value_metric,
                    value_expansion,
                )
                local_rhs = np.einsum(
                    "nia,nij,njb->nab",
                    sphere.frames,
                    metric_rhs,
                    sphere.frames,
                )
                inverse_transpose = np.swapaxes(
                    np.linalg.inv(value_factor), -1, -2
                )
                factor_rhs = 0.5 * np.matmul(
                    local_rhs, inverse_transpose
                )
                jacobian = 2.0 * calibration.v1 * value_s
                return jacobian * np.concatenate(
                    (
                        expansion_rhs[:, None],
                        factor_rhs.reshape((sphere.count, 4)),
                    ),
                    axis=-1,
                )

            result = solve_sdc(
                segment,
                initial_packed,
                rhs_s,
                tolerance=1.0e-11,
                overgrid_tolerance=sdc_overgrid_tolerance,
                maximum_corrections=14,
            )
            boundary_sdc_diagnostics.append(
                {
                    "element": element + 1,
                    "v_left": float(calibration.v1 * segment.left**2),
                    "v_right": float(calibration.v1 * segment.right**2),
                    "corrections": result.corrections,
                    "converged": result.converged,
                    "resolved": result.resolved,
                    "accepted": result.accepted,
                    "collocation_defect": result.collocation_defect,
                    "overgrid_defect": result.overgrid_defect,
                }
            )
            if not result.converged:
                raise FloatingPointError(
                    "outgoing-boundary SDC failed on "
                    f"v=[{calibration.v1 * segment.left**2:.12g}, "
                    f"{calibration.v1 * segment.right**2:.12g}], "
                    f"defect={result.collocation_defect:.6g}"
                )
            if not result.resolved:
                raise FloatingPointError(
                    "outgoing-boundary SDC element is underresolved on "
                    f"v=[{calibration.v1 * segment.left**2:.12g}, "
                    f"{calibration.v1 * segment.right**2:.12g}]: "
                    f"independent overgrid defect={result.overgrid_defect:.6g} "
                    f"> {sdc_overgrid_tolerance:.6g}"
                )
            local_expansion = result.values[..., 0]
            local_factor = result.values[..., 1:].reshape(
                (len(segment.nodes), sphere.count, 2, 2)
            )
            expansion[:, index] = np.moveaxis(local_expansion, 0, 1)
            factor[:, index] = np.moveaxis(local_factor, 0, 1)
            metric[:, index] = ambient_metric(factor[:, index])

    inverse = tangent_inverse(sphere, metric)
    for j, value in enumerate(v):
        reference[:, j] = reference_shear(
            tensors, float(value), calibration
        )
        if angular is not None:
            reference[:, j] = angular.project_sym2(reference[:, j])
            round_trace = np.einsum(
                "nij,nij->n", sphere.projector, reference[:, j]
            )
            reference[:, j] -= (
                0.5 * round_trace[:, None, None] * sphere.projector
            )
        shear[:, j] = constrained_shear(reference[:, j], metric[:, j])

    reference_norm = np.sqrt(
        np.maximum(np.einsum("nvij,nvij->nv", reference, reference), 0.0)
    )
    physical_norm = np.sqrt(
        np.maximum(tensor_norm_sq(shear, inverse), 0.0)
    )
    reference_integral = np.trapezoid(reference_norm, v, axis=1)
    physical_integral = np.trapezoid(physical_norm, v, axis=1)
    upper = sphere.points[:, 2] >= 0.0
    lower = sphere.points[:, 2] <= 0.0
    first_profile, second_profile = profile_values(v, calibration)
    tensor_norms = [
        np.sqrt(
            np.maximum(np.einsum("nij,nij->n", tensor, tensor), 0.0)
        )
        for tensor in tensors
    ]
    first_integral = np.trapezoid(
        first_profile[None, :] * tensor_norms[0][:, None], v, axis=1
    )
    second_integral = np.trapezoid(
        second_profile[None, :] * tensor_norms[1][:, None], v, axis=1
    )
    trace = np.einsum("nvij,nvji->nv", inverse, shear)
    reference_trace = np.einsum(
        "nvij,nij->nv", reference, sphere.projector
    )
    if angular is None:
        zeta, shift, zeta_source, zeta_iterations, zeta_update = (
            solve_boundary_zeta(sphere, metric, expansion, shear, v)
        )
    else:
        difference, inverse = connection_difference(sphere, metric)
        div_shear = tensor_divergence(
            sphere, shear, difference, inverse
        )
        grad_expansion = scalar_gradient(sphere, expansion)
        free_source = (
            np.einsum("nvij,nvj->nvi", inverse, div_shear)
            - 0.5 * np.einsum("nvij,nvj->nvi", inverse, grad_expansion)
        )
        zeta = np.zeros(expansion.shape + (3,), dtype=float)
        zeta_source = free_source.copy()
        zeta_update = math.inf
        zeta_iterations = 0
        for iteration in range(48):
            shear_zeta = np.einsum("nvij,nvj->nvi", shear, zeta)
            zeta_source = (
                -2.0 * expansion[..., None] * zeta
                - 2.0
                * np.einsum("nvij,nvj->nvi", inverse, shear_zeta)
                + free_source
            )
            record_tail("vector", zeta_source, "boundary_zeta_rhs")
            zeta_source = angular.project_vector(zeta_source)
            new_zeta = integrate_v(zeta_source)
            new_zeta = angular.project_vector(new_zeta)
            difference_zeta = new_zeta - zeta
            zeta_update = float(
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
            zeta_iterations = iteration + 1
            if zeta_update <= 1.0e-13:
                break
        shear_zeta = np.einsum("nvij,nvj->nvi", shear, zeta)
        zeta_source = (
            -2.0 * expansion[..., None] * zeta
            - 2.0 * np.einsum("nvij,nvj->nvi", inverse, shear_zeta)
            + free_source
        )
        record_tail("vector", zeta_source, "boundary_zeta_rhs")
        zeta_source = angular.project_vector(zeta_source)
        shift_source = -4.0 * zeta
        record_tail("vector", shift_source, "boundary_shift_rhs")
        shift = integrate_v(angular.project_vector(shift_source))
        shift = angular.project_vector(shift)
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
        "continuum_target_l1": calibration.target_l1,
        "profile_first_amplitude": calibration.first_amplitude,
        "profile_second_amplitude": calibration.second_amplitude,
        "profile_corner_scale": (
            CORNER_SCALE_FRACTION
            * (1.0 - SUPPORT_MARGIN_FRACTION)
            * calibration.split
        ),
        "boundary_sdc_maximum_collocation_defect": (
            0.0
            if not boundary_sdc_diagnostics
            else max(
                float(item["collocation_defect"])
                for item in boundary_sdc_diagnostics
            )
        ),
        "boundary_sdc_maximum_overgrid_defect": (
            0.0
            if not boundary_sdc_diagnostics
            else max(
                float(item["overgrid_defect"])
                for item in boundary_sdc_diagnostics
            )
        ),
        # Preserve elementwise evidence.  A tiny collocation defect alone is
        # not a resolution certificate; the adaptive mesh builder consumes
        # these independent overgrid defects panel by panel.
        "boundary_sdc_diagnostics": boundary_sdc_diagnostics,
        **{
            f"galerkin_tail_{name}": value
            for name, value in boundary_tails.items()
        },
    }
