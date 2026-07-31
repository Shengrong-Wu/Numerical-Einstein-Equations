"""Crossed characteristic vacuum shear data and constraint solves."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from nee.discretization.double_null_mesh import DoubleSqrtLGLMesh
from nee.geometry.null_geometry import one_form_lie_derivative
from nee.geometry.sphere import (
    PointSphereGrid,
    connection_difference,
    gaussian_curvature,
    lie_covariant_tensor,
    scalar_gradient,
    tangent_inverse,
    tensor_divergence,
    tensor_norm_sq,
    tensor_tracefree,
    vector_divergence,
)
from nee.state.boundary import BoundaryData


Array = np.ndarray


def rotation_shift(points: Array) -> Array:
    """Return ``0.1 partial_phi`` as an ambient tangent vector."""

    x, y, _ = points.T
    return 0.1 * np.column_stack((-y, x, np.zeros_like(x)))


def quadratic_hessian_tensors(grid: PointSphereGrid) -> tuple[Array, Array]:
    """Return the two explicitly normalized trace-free Hessians.

    For ``f(n)=n^T A n`` with ``tr(A)=0`` on the unit sphere,
    ``tf Hess(f)=2 P A P+fP``.  The exact normalizers below make the
    pointwise tensor norm at most one.
    """

    projector = grid.projector
    points = grid.points
    matrices = (
        np.diag([-0.5, -0.5, 1.0]),
        np.diag([1.0, -1.0, 0.0]),
    )
    normalizers = (math.sqrt(2.0) / 3.0, 1.0 / (2.0 * math.sqrt(2.0)))
    tensors = []
    for matrix, normalizer in zip(matrices, normalizers, strict=True):
        function = np.einsum("ni,ij,nj->n", points, matrix, points)
        raw = (
            2.0
            * np.einsum(
                "nik,kl,nlj->nij",
                projector,
                matrix,
                projector,
            )
            + function[:, None, None] * projector
        )
        tensors.append(normalizer * raw)
    return tensors[0], tensors[1]


def tensor_norm_certificates(
    grid: PointSphereGrid, tensors: tuple[Array, Array]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for index, tensor in enumerate(tensors, start=1):
        norm_sq = np.maximum(
            np.einsum("nij,nij->n", tensor, tensor),
            0.0,
        )
        result[f"X{index}"] = {
            "L2": float(math.sqrt(4.0 * math.pi * np.mean(norm_sq))),
            "Linf": float(math.sqrt(np.max(norm_sq))),
            "maximum_round_trace": float(
                np.max(np.abs(np.einsum("nij,nij->n", grid.projector, tensor)))
            ),
        }
    return result


def _transferred_shear(
    grid: PointSphereGrid,
    reference: Array,
    metric: Array,
    amplitude: float,
) -> Array:
    raw = np.matmul(np.matmul(reference, grid.projector), metric)
    raw = 0.5 * (raw + np.swapaxes(raw, -1, -2))
    inverse = tangent_inverse(grid, metric)
    return amplitude * tensor_tracefree(raw, metric, inverse)


def _rk4_nodes(
    nodes: Array,
    initial: tuple[Array, ...],
    rhs: Callable[[float, tuple[Array, ...]], tuple[Array, ...]],
    substeps: int,
) -> tuple[Array, ...]:
    outputs = [
        np.empty((len(nodes), *value.shape), dtype=float) for value in initial
    ]
    for output, value in zip(outputs, initial, strict=True):
        output[0] = value
    current = tuple(np.asarray(value, dtype=float).copy() for value in initial)
    for index in range(len(nodes) - 1):
        left = float(nodes[index])
        step = float(nodes[index + 1] - nodes[index]) / substeps
        for substep in range(substeps):
            x0 = left + substep * step
            k1 = rhs(x0, current)
            middle = tuple(
                value + 0.5 * step * derivative
                for value, derivative in zip(current, k1, strict=True)
            )
            k2 = rhs(x0 + 0.5 * step, middle)
            middle = tuple(
                value + 0.5 * step * derivative
                for value, derivative in zip(current, k2, strict=True)
            )
            k3 = rhs(x0 + 0.5 * step, middle)
            right = tuple(
                value + step * derivative
                for value, derivative in zip(current, k3, strict=True)
            )
            k4 = rhs(x0 + step, right)
            current = tuple(
                value
                + (step / 6.0)
                * (first + 2.0 * second + 2.0 * third + fourth)
                for value, first, second, third, fourth in zip(
                    current, k1, k2, k3, k4, strict=True
                )
            )
        for output, value in zip(outputs, current, strict=True):
            output[index + 1] = value
    return tuple(np.moveaxis(output, 0, 1) for output in outputs)


def solve_outgoing_face(
    grid: PointSphereGrid,
    v: Array,
    x1: Array,
    *,
    substeps: int,
) -> dict[str, Array]:
    round_metric = grid.projector
    initial_metric = round_metric.copy()
    initial_expansion = np.full(grid.count, 2.0)

    def metric_rhs(
        value_v: float, state: tuple[Array, ...]
    ) -> tuple[Array, ...]:
        metric, expansion = state
        shear = _transferred_shear(
            grid, x1, metric, math.sqrt(max(value_v, 0.0))
        )
        inverse = tangent_inverse(grid, metric)
        shear_norm_sq = tensor_norm_sq(shear, inverse)
        return (
            expansion[:, None, None] * metric + 2.0 * shear,
            -0.5 * expansion**2 - shear_norm_sq,
        )

    metric, expansion = _rk4_nodes(
        v,
        (initial_metric, initial_expansion),
        metric_rhs,
        substeps,
    )
    inverse = tangent_inverse(grid, metric)
    shear = np.stack(
        [
            _transferred_shear(grid, x1, metric[:, index], math.sqrt(value))
            for index, value in enumerate(v)
        ],
        axis=1,
    )
    initial_zeta = np.zeros((grid.count, 3))
    initial_shift = rotation_shift(grid.points)

    def interpolate(field: Array, value_v: float) -> Array:
        if value_v <= v[0]:
            return field[:, 0]
        if value_v >= v[-1]:
            return field[:, -1]
        right = int(np.searchsorted(v, value_v))
        left = right - 1
        fraction = (value_v - v[left]) / (v[right] - v[left])
        return (1.0 - fraction) * field[:, left] + fraction * field[:, right]

    def torsion_rhs(
        value_v: float, state: tuple[Array, ...]
    ) -> tuple[Array, ...]:
        zeta_up, _shift = state
        value_metric = interpolate(metric, value_v)
        value_inverse = tangent_inverse(grid, value_metric)
        value_expansion = interpolate(expansion, value_v)
        value_shear = interpolate(shear, value_v)
        difference, _ = connection_difference(
            grid, value_metric, value_inverse
        )
        divergence = tensor_divergence(
            grid, value_shear, difference, value_inverse
        )
        gradient = scalar_gradient(grid, value_expansion)
        mixed = np.matmul(value_shear, value_metric)
        # Sigma^A_B zeta^B; the first factor is raised with g^{-1}.
        mixed = np.matmul(value_inverse, value_shear)
        derivative_zeta = (
            -2.0 * value_expansion[:, None] * zeta_up
            - 2.0 * np.einsum("nij,nj->ni", mixed, zeta_up)
            + np.einsum("nij,nj->ni", value_inverse, divergence)
            - 0.5 * np.einsum("nij,nj->ni", value_inverse, gradient)
        )
        derivative_shift = -4.0 * zeta_up
        return derivative_zeta, derivative_shift

    zeta_up, shift = _rk4_nodes(
        v,
        (initial_zeta, initial_shift),
        torsion_rhs,
        substeps,
    )
    return {
        "metric": metric,
        "inverse": inverse,
        "expansion": expansion,
        "shear": shear,
        "omega": np.ones_like(expansion),
        "weighted_omega": np.zeros_like(expansion),
        "zeta_up": zeta_up,
        "shift": shift,
    }


def solve_incoming_face(
    grid: PointSphereGrid,
    u: Array,
    x2: Array,
    *,
    substeps: int,
) -> dict[str, Array]:
    round_metric = grid.projector
    shift = rotation_shift(grid.points)
    initial_metric = round_metric.copy()
    initial_in_expansion = np.full(grid.count, -2.0)
    initial_zeta = np.zeros((grid.count, 3))
    initial_out_expansion = np.full(grid.count, 2.0)

    def rhs(
        value_u: float, state: tuple[Array, ...]
    ) -> tuple[Array, ...]:
        metric, in_expansion, zeta, out_expansion = state
        inverse = tangent_inverse(grid, metric)
        shear = _transferred_shear(
            grid, x2, metric, math.sqrt(max(value_u + 1.0, 0.0))
        )
        weighted_chib = (
            0.5 * in_expansion[:, None, None] * metric + shear
        )
        difference, _ = connection_difference(grid, metric, inverse)
        lie_metric = lie_covariant_tensor(grid, shift, metric)
        gradient_in = scalar_gradient(grid, in_expansion)
        advected_in = np.einsum(
            "ni,ni->n", shift, gradient_in
        )
        metric_rhs = (
            in_expansion[:, None, None] * metric
            + 2.0 * shear
            - lie_metric
        )
        in_rhs = (
            -0.5 * in_expansion**2
            - tensor_norm_sq(shear, inverse)
            - advected_in
        )

        divergence_shear = tensor_divergence(
            grid, shear, difference, inverse
        )
        shear_mixed = np.matmul(shear, inverse)
        target_d3_zeta = (
            -1.5 * in_expansion[:, None] * zeta
            - np.einsum("nij,nj->ni", shear_mixed, zeta)
            - divergence_shear
            + 0.5 * gradient_in
        )
        coordinate_zeta_rhs = (
            target_d3_zeta
            - one_form_lie_derivative(grid, shift, zeta)
            + np.einsum(
                "nij,nj->ni",
                np.matmul(weighted_chib, inverse),
                zeta,
            )
        )

        eta_up = np.einsum("nij,nj->ni", inverse, zeta)
        div_eta = vector_divergence(grid, eta_up, difference)
        eta_norm_sq = np.einsum(
            "ni,nij,nj->n", zeta, inverse, zeta
        )
        curvature, _, _ = gaussian_curvature(grid, metric)
        gradient_out = scalar_gradient(grid, out_expansion)
        out_rhs = (
            -out_expansion * in_expansion
            + 2.0 * div_eta
            + 2.0 * eta_norm_sq
            - 2.0 * curvature
            - np.einsum("ni,ni->n", shift, gradient_out)
        )
        return metric_rhs, in_rhs, coordinate_zeta_rhs, out_rhs

    metric, in_expansion, zeta, out_expansion = _rk4_nodes(
        u,
        (
            initial_metric,
            initial_in_expansion,
            initial_zeta,
            initial_out_expansion,
        ),
        rhs,
        substeps,
    )
    inverse = tangent_inverse(grid, metric)
    shear = np.stack(
        [
            _transferred_shear(
                grid, x2, metric[:, index], math.sqrt(value + 1.0)
            )
            for index, value in enumerate(u)
        ],
        axis=1,
    )
    weighted_chib = (
        0.5 * in_expansion[..., None, None] * metric + shear
    )
    zeta_up = np.einsum("nuij,nuj->nui", inverse, zeta)
    return {
        "metric": metric,
        "weighted_tr_chib": in_expansion,
        "weighted_hatchib": shear,
        "weighted_chib": weighted_chib,
        "weighted_tr_chi": out_expansion,
        "omega": np.ones_like(in_expansion),
        "weighted_omegab": np.zeros_like(in_expansion),
        "zeta_up": zeta_up,
        "shift": np.broadcast_to(
            shift[:, None], (grid.count, len(u), 3)
        ).copy(),
    }


def construct_boundary_data(
    grid: PointSphereGrid,
    mesh: DoubleSqrtLGLMesh,
    *,
    substeps: int,
) -> tuple[BoundaryData, dict[str, Any]]:
    tensors = quadratic_hessian_tensors(grid)
    certificates = tensor_norm_certificates(grid, tensors)
    for name, certificate in certificates.items():
        if certificate["L2"] < 0.5 - 1.0e-12:
            raise ValueError(f"{name} violates the L2 lower bound")
        if certificate["Linf"] > 1.0 + 1.0e-12:
            raise ValueError(f"{name} violates the Linf upper bound")
    outgoing = solve_outgoing_face(
        grid, mesh.v, tensors[0], substeps=substeps
    )
    incoming = solve_incoming_face(
        grid, mesh.u, tensors[1], substeps=substeps
    )
    for name in ("metric", "zeta_up", "shift"):
        mismatch = float(
            np.max(np.abs(outgoing[name][:, 0] - incoming[name][:, 0]))
        )
        if mismatch > 2.0e-10:
            raise ValueError(f"corner mismatch in {name}: {mismatch:.3e}")
    return BoundaryData.create(outgoing, incoming), certificates
