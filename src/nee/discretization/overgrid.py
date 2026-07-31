"""Independent coordinate/angular resampling of primitive fields."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline, PchipInterpolator

from nee.diagnostics.independent_audit import PrimitiveFields


Array = np.ndarray

from nee.numerics.sphere import (  # noqa: E402
    PointSphereGrid,
    spherical_harmonic_collocation,
)
from nee.numerics.coordinate_differentiation import high_order_differentiate  # noqa: E402


def chebyshev_lobatto(count: int, left: float, right: float) -> Array:
    if count < 5:
        raise ValueError("overgrid requires at least five coordinate nodes")
    values = np.cos(np.pi * np.arange(count) / (count - 1))[::-1]
    return left + 0.5 * (values + 1.0) * (right - left)


def _coordinate_resample(
    values: Array,
    source_u: Array,
    source_v: Array,
    target_u: Array,
    target_v: Array,
) -> Array:
    def interpolate(
        source: Array, data: Array, target: Array, axis: int
    ) -> Array:
        spacings = np.diff(source)
        grading = float(np.max(spacings) / np.min(spacings))
        # Experiment 6 has v=s^10 nodes separated by many orders of
        # magnitude near the corner.  In that case an unconstrained cubic
        # manufactures enormous overshoots, so use a shape-preserving fit.
        if grading > 1.0e5:
            return PchipInterpolator(source, data, axis=axis)(target)
        return CubicSpline(
            source, data, axis=axis, bc_type="not-a-knot"
        )(target)

    first = interpolate(source_u, values, target_u, axis=1)
    return interpolate(source_v, first, target_v, axis=2)


def _angular_resample(
    values: Array,
    source_grid: PointSphereGrid,
    target_grid: PointSphereGrid,
    degree: int,
) -> tuple[Array, float]:
    _, source_basis, condition = spherical_harmonic_collocation(
        source_grid.points, degree
    )
    _, target_basis, _ = spherical_harmonic_collocation(
        target_grid.points, degree
    )
    flat = values.reshape(source_grid.count, -1)
    coefficients = np.linalg.pinv(source_basis, rcond=1.0e-13) @ flat
    result = (target_basis @ coefficients).reshape(
        (target_grid.count, *values.shape[1:])
    )
    return result, condition


def _project_vector(grid: PointSphereGrid, vector: Array) -> Array:
    return np.einsum("nij,n...j->n...i", grid.projector, vector)


def _project_tensor(grid: PointSphereGrid, tensor: Array) -> Array:
    value = np.einsum(
        "nia,n...ab,njb->n...ij", grid.projector, tensor, grid.projector
    )
    return 0.5 * (value + np.swapaxes(value, -1, -2))


@dataclass(frozen=True)
class OvergridResult:
    grid: PointSphereGrid
    u: Array
    v: Array
    fields: PrimitiveFields
    diagnostics: dict[str, float | int | str]
    coordinates: "PowerAuditCoordinates | None" = None


@dataclass(frozen=True)
class PowerAuditCoordinates:
    """Independent nodes and chain-rule derivatives for ``v=v1*s^(1/delta)``."""

    tau: Array
    s: Array
    u: Array
    v: Array
    v1: float
    delta: float
    stencil: int = 7
    tau_operator: object | None = None
    s_operator: object | None = None

    def differentiate_u(self, values: Array, axis: int = 1) -> Array:
        derivative_tau = (
            high_order_differentiate(
                values,
                self.tau,
                axis=axis,
                stencil=min(self.stencil, len(self.tau)),
            )
            if self.tau_operator is None
            else self.tau_operator.differentiate(
                values, axis=axis, interface_rule="average"
            )
        )
        shape = [1] * values.ndim
        shape[axis] = len(self.u)
        return derivative_tau / (-self.u).reshape(shape)

    def differentiate_v(self, values: Array, axis: int = 2) -> Array:
        derivative_s = (
            high_order_differentiate(
                values,
                self.s,
                axis=axis,
                stencil=min(self.stencil, len(self.s)),
            )
            if self.s_operator is None
            else self.s_operator.differentiate(
                values, axis=axis, interface_rule="average"
            )
        )
        exponent = 1.0 / self.delta
        jacobian = (
            self.v1
            * exponent
            * self.s ** np.maximum(exponent - 1.0, 0.0)
        )
        if exponent == 1.0:
            jacobian[:] = self.v1
        moved = np.moveaxis(derivative_s, axis, 0)
        result = np.empty_like(moved)
        positive = jacobian > 64.0 * np.finfo(float).tiny
        result[positive] = moved[positive] / jacobian[positive].reshape(
            (int(np.count_nonzero(positive)),)
            + (1,) * (moved.ndim - 1)
        )
        first_positive = int(np.flatnonzero(positive)[0])
        result[~positive] = result[first_positive]
        return np.moveaxis(result, 0, axis)

    def subset(
        self, u_slice: slice, v_slice: slice
    ) -> "PowerAuditCoordinates":
        return PowerAuditCoordinates(
            tau=self.tau[u_slice],
            s=self.s[v_slice],
            u=self.u[u_slice],
            v=self.v[v_slice],
            v1=self.v1,
            delta=self.delta,
            stencil=self.stencil,
        )


def _composite_interpolate(
    mesh: object,
    values: Array,
    targets: Array,
    *,
    axis: int,
) -> Array:
    """Evaluate the stored piecewise LGL polynomial on independent nodes."""

    moved = np.moveaxis(values, axis, 0)
    result = np.empty((len(targets), *moved.shape[1:]), dtype=values.dtype)
    assigned = np.zeros(len(targets), dtype=bool)
    segments = tuple(getattr(mesh, "segments"))
    indices = tuple(getattr(mesh, "indices"))
    for element, (segment, index) in enumerate(
        zip(segments, indices, strict=True)
    ):
        tolerance = 32.0 * np.finfo(float).eps
        selected = (targets >= segment.left - tolerance) & (
            targets <= segment.right + tolerance
        )
        # Assign a shared interface from the element on its right, except at
        # the final endpoint. Both traces agree in value, but this rule makes
        # ownership deterministic.
        if element < len(segments) - 1:
            selected &= targets < segment.right - tolerance
        if not np.any(selected):
            continue
        local = segment.interpolate(
            moved[index], targets[selected], axis=0
        )
        result[selected] = local
        assigned[selected] = True
    if not np.all(assigned):
        missing = np.flatnonzero(~assigned)
        raise ValueError(
            f"composite interpolation left target indices {missing.tolist()}"
        )
    return np.moveaxis(result, 0, axis)


def resample_primitives(
    source_grid: PointSphereGrid,
    fields: PrimitiveFields,
    source_u: Array,
    source_v: Array,
    *,
    u_count: int,
    v_count: int,
    point_count: int,
    harmonic_degree: int,
) -> OvergridResult:
    """Fit primitives and evaluate them on nodes absent from construction."""

    minimum_points = (harmonic_degree + 1) ** 2 + 8
    if point_count < minimum_points:
        raise ValueError(
            f"audit point count {point_count} is below {minimum_points}"
        )
    target_grid = PointSphereGrid.create(
        point_count,
        neighbor_count=min(max(24, 3 * harmonic_degree), point_count - 1),
        degree=min(4, harmonic_degree),
        spectral_degree=harmonic_degree,
    )
    target_u = chebyshev_lobatto(
        u_count, float(source_u[0]), float(source_u[-1])
    )
    target_v = chebyshev_lobatto(
        v_count, float(source_v[0]), float(source_v[-1])
    )

    def transfer(value: Array) -> tuple[Array, float]:
        coordinate = _coordinate_resample(
            np.asarray(value),
            source_u,
            source_v,
            target_u,
            target_v,
        )
        return _angular_resample(
            coordinate, source_grid, target_grid, harmonic_degree
        )

    metric, condition = transfer(fields.metric)
    log_omega, _ = transfer(fields.log_omega)
    shift, _ = transfer(fields.shift)
    phi = None
    if fields.phi is not None:
        phi, _ = transfer(fields.phi)
    metric = _project_tensor(target_grid, metric)
    shift = _project_vector(target_grid, shift)
    local = np.einsum(
        "nia,n...ij,njb->n...ab",
        target_grid.frames,
        metric,
        target_grid.frames,
    )
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(local)))
    if minimum_eigenvalue <= 0.0:
        raise FloatingPointError(
            "overgrid metric interpolation left the positive cone: "
            f"{minimum_eigenvalue:.12g}"
        )
    return OvergridResult(
        grid=target_grid,
        u=target_u,
        v=target_v,
        fields=PrimitiveFields(
            metric=metric,
            log_omega=log_omega,
            shift=shift,
            phi=phi,
        ),
        diagnostics={
            "coordinate_interpolant": (
                "tensor-product cubic; shape-preserving PCHIP on axes "
                "whose spacing ratio exceeds 1e5"
            ),
            "angular_interpolant": "real scalar harmonics per ambient component",
            "u_count": u_count,
            "v_count": v_count,
            "point_count": point_count,
            "harmonic_degree": harmonic_degree,
            "source_harmonic_condition": float(condition),
            "minimum_metric_eigenvalue": minimum_eigenvalue,
        },
    )


def resample_primitives_power(
    source_grid: PointSphereGrid,
    fields: PrimitiveFields,
    source_coordinates: object,
    *,
    u_count: int,
    v_count: int,
    point_count: int,
    harmonic_degree: int,
    stencil: int = 7,
    spectral_degree_increment: int | None = None,
) -> OvergridResult:
    """Resample in ``(tau,s)`` and retain the exact physical chain rule.

    The target nodes are absent Chebyshev--Lobatto nodes in the computational
    coordinates. The source field is evaluated from its piecewise LGL
    polynomial, while all derivatives on the target grid are reconstructed
    independently by local finite-difference weights in ``tau`` and ``s``.
    """

    minimum_points = (harmonic_degree + 1) ** 2 + 8
    if point_count < minimum_points:
        raise ValueError(
            f"audit point count {point_count} is below {minimum_points}"
        )
    tau_mesh = getattr(source_coordinates, "tau")
    s_mesh = getattr(source_coordinates, "s")
    v1 = float(getattr(source_coordinates, "v1"))
    delta = float(getattr(source_coordinates, "delta"))
    target_tau_operator = None
    target_s_operator = None
    if spectral_degree_increment is None:
        target_tau = chebyshev_lobatto(
            u_count, float(tau_mesh.nodes[0]), float(tau_mesh.nodes[-1])
        )
        target_s = chebyshev_lobatto(
            v_count, float(s_mesh.nodes[0]), float(s_mesh.nodes[-1])
        )
    else:
        if spectral_degree_increment < 1:
            raise ValueError("spectral degree increment must be positive")
        tau_breakpoints = np.asarray(
            [
                tau_mesh.segments[0].left,
                *[segment.right for segment in tau_mesh.segments],
            ]
        )
        s_breakpoints = np.asarray(
            [
                s_mesh.segments[0].left,
                *[segment.right for segment in s_mesh.segments],
            ]
        )
        target_tau_operator = type(tau_mesh).create(
            tau_breakpoints,
            np.asarray(
                [
                    segment.degree + spectral_degree_increment
                    for segment in tau_mesh.segments
                ]
            ),
        )
        target_s_operator = type(s_mesh).create(
            s_breakpoints,
            np.asarray(
                [
                    segment.degree + spectral_degree_increment
                    for segment in s_mesh.segments
                ]
            ),
        )
        target_tau = target_tau_operator.nodes
        target_s = target_s_operator.nodes
    coordinates = PowerAuditCoordinates(
        tau=target_tau,
        s=target_s,
        u=-np.exp(-target_tau),
        v=v1 * target_s ** (1.0 / delta),
        v1=v1,
        delta=delta,
        stencil=stencil,
        tau_operator=target_tau_operator,
        s_operator=target_s_operator,
    )
    target_grid = PointSphereGrid.create(
        point_count,
        neighbor_count=min(max(24, 3 * harmonic_degree), point_count - 1),
        degree=min(4, harmonic_degree),
        spectral_degree=harmonic_degree,
    )

    def transfer(value: Array) -> tuple[Array, float]:
        in_tau = _composite_interpolate(
            tau_mesh, np.asarray(value), target_tau, axis=1
        )
        in_coordinates = _composite_interpolate(
            s_mesh, in_tau, target_s, axis=2
        )
        return _angular_resample(
            in_coordinates,
            source_grid,
            target_grid,
            harmonic_degree,
        )

    metric, condition = transfer(fields.metric)
    log_omega, _ = transfer(fields.log_omega)
    shift, _ = transfer(fields.shift)
    phi = None if fields.phi is None else transfer(fields.phi)[0]
    metric = _project_tensor(target_grid, metric)
    shift = _project_vector(target_grid, shift)
    local = np.einsum(
        "nia,n...ij,njb->n...ab",
        target_grid.frames,
        metric,
        target_grid.frames,
    )
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(local)))
    if minimum_eigenvalue <= 0.0:
        raise FloatingPointError(
            "mapped overgrid metric interpolation left the positive cone: "
            f"{minimum_eigenvalue:.12g}"
        )
    return OvergridResult(
        grid=target_grid,
        u=coordinates.u,
        v=coordinates.v,
        fields=PrimitiveFields(
            metric=metric,
            log_omega=log_omega,
            shift=shift,
            phi=phi,
        ),
        diagnostics={
            "coordinate_interpolant": (
                "piecewise LGL evaluation in tau=-log(-u) and "
                f"s=(v/v_max)^{delta:.12g}"
            ),
            "coordinate_derivatives": (
                (
                    "independent local polynomial weights in tau,s"
                    if spectral_degree_increment is None
                    else "independent higher-degree LGL overgrid operators"
                )
                + " plus exact physical chain rule"
            ),
            "angular_interpolant": "real scalar harmonics per ambient component",
            "u_count": len(coordinates.u),
            "v_count": len(coordinates.v),
            "point_count": point_count,
            "harmonic_degree": harmonic_degree,
            "source_harmonic_condition": float(condition),
            "minimum_metric_eigenvalue": minimum_eigenvalue,
        },
        coordinates=coordinates,
    )
