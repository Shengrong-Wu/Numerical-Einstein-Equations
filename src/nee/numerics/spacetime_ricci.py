"""Independent coordinate-free four-dimensional Ricci audit on R2 x S2.

The two sphere directions are represented redundantly by three ambient
Cartesian components with the tangent projector.  The reference connection is
the product of the flat ``(u,v)`` connection and the unit round-sphere
connection.  Ricci is reconstructed from the full metric's connection
difference, independently of the null construction equations.
"""

from __future__ import annotations

import math

import numpy as np

from .coordinate_differentiation import high_order_differentiate
from .vacuum_state import GlobalState
from .sphere import PointSphereGrid, tangent_inverse


Array = np.ndarray


def hybrid_projector(grid: PointSphereGrid) -> Array:
    projector = np.zeros((grid.count, 5, 5), dtype=float)
    projector[:, 0, 0] = 1.0
    projector[:, 1, 1] = 1.0
    projector[:, 2:5, 2:5] = grid.projector
    return projector


def angular_reference_derivative(
    grid: PointSphereGrid, values: Array, tensor_rank: int
) -> Array:
    """Round-sphere derivative of hybrid R2 x TS2 tensor components."""

    first, second = grid.directional_derivatives(values)
    batch_rank = values.ndim - tensor_rank - 1
    frame_shape = (grid.count,) + (1,) * batch_rank + (3,) + (1,) * tensor_rank
    raw = (
        grid.first.reshape(frame_shape)
        * np.expand_dims(first, axis=values.ndim - tensor_rank)
        + grid.second.reshape(frame_shape)
        * np.expand_dims(second, axis=values.ndim - tensor_rank)
    )
    projector = hybrid_projector(grid)
    if tensor_rank == 2:
        projected = np.einsum(
            "npa,nqb,n...rab->n...rpq", projector, projector, raw
        )
    elif tensor_rank == 3:
        projected = np.einsum(
            "npa,nqb,nsc,n...rabc->n...rpqs",
            projector,
            projector,
            projector,
            raw,
        )
    else:
        raise ValueError("hybrid derivative is implemented for ranks two and three")
    result = np.zeros(values.shape[:-tensor_rank] + (5,) + (5,) * tensor_rank)
    result[..., 2:5, *(slice(None),) * tensor_rank] = projected
    return result


def spacetime_reference_derivative(
    grid: PointSphereGrid,
    values: Array,
    tensor_rank: int,
    u: Array,
    v: Array,
) -> Array:
    result = angular_reference_derivative(grid, values, tensor_rank)
    slots = (slice(None),) * tensor_rank
    result[..., 0, *slots] = high_order_differentiate(
        values, u, axis=1, stencil=min(17, len(u))
    )
    result[..., 1, *slots] = high_order_differentiate(
        values, v, axis=2, stencil=min(17, len(v))
    )
    return result


def build_metric_and_inverse(
    grid: PointSphereGrid, state: GlobalState
) -> tuple[Array, Array, Array]:
    shape = state.omega.shape
    metric = np.zeros(shape + (5, 5), dtype=float)
    inverse = np.zeros_like(metric)
    sphere_inverse = tangent_inverse(grid, state.metric)
    shift_cov = np.einsum("n...ij,n...j->n...i", state.metric, state.shift)
    shift_norm_sq = np.einsum("n...i,n...i->n...", shift_cov, state.shift)
    metric[..., 0, 0] = shift_norm_sq
    metric[..., 0, 1] = -2.0 * state.omega**2
    metric[..., 1, 0] = metric[..., 0, 1]
    metric[..., 0, 2:5] = -shift_cov
    metric[..., 2:5, 0] = -shift_cov
    metric[..., 2:5, 2:5] = state.metric

    null_inverse = -0.5 / state.omega**2
    inverse[..., 0, 1] = null_inverse
    inverse[..., 1, 0] = null_inverse
    inverse[..., 1, 2:5] = null_inverse[..., None] * state.shift
    inverse[..., 2:5, 1] = inverse[..., 1, 2:5]
    inverse[..., 2:5, 2:5] = sphere_inverse
    return metric, inverse, sphere_inverse


def direct_ricci_block(
    grid: PointSphereGrid, state: GlobalState, u: Array, v: Array
) -> tuple[Array, Array, Array]:
    metric, inverse, sphere_inverse = build_metric_and_inverse(grid, state)
    derivative = spacetime_reference_derivative(grid, metric, 2, u, v)
    lower = 0.5 * (
        derivative
        + np.swapaxes(derivative, -3, -2)
        - np.einsum("n...kij->n...ijk", derivative)
    )
    difference = np.einsum("n...lk,n...ijk->n...lij", inverse, lower)
    ricci = np.zeros_like(metric)
    ricci[..., 2:5, 2:5] = grid.projector[:, None, None]
    # Contract nabla^0 C direction by direction.  The complete rank-four
    # derivative has 625 components per point; Ricci needs only
    # D_i C^i_jk and D_k C^i_ji.
    for direction, coordinate in [(0, u), (1, v)]:
        derivative_direction = high_order_differentiate(
            difference,
            coordinate,
            axis=direction + 1,
            stencil=min(17, len(coordinate)),
        )
        ricci += derivative_direction[..., direction, :, :]
        trace = np.einsum("n...iji->n...j", derivative_direction)
        ricci[..., :, direction] -= trace
    projector = hybrid_projector(grid)
    for angular in range(3):
        direction = angular + 2
        matrix = (
            grid.first[:, angular, None] * grid.derivative_first
            + grid.second[:, angular, None] * grid.derivative_second
        )
        raw = np.tensordot(matrix, difference, axes=(1, 0))
        ricci += np.einsum(
            "na,njb,nkc,n...abc->n...jk",
            projector[:, direction, :],
            projector,
            projector,
            raw,
        )
        trace = np.einsum(
            "nia,njb,nic,n...abc->n...j",
            projector,
            projector,
            projector,
            raw,
        )
        ricci[..., :, direction] -= trace
    for j in range(5):
        for k in range(5):
            for i in range(5):
                for m in range(5):
                    ricci[..., j, k] += (
                        difference[..., i, i, m] * difference[..., m, j, k]
                    )
                    ricci[..., j, k] -= (
                        difference[..., i, k, m] * difference[..., m, j, i]
                    )
    return ricci, sphere_inverse, metric


def adapted_ricci_norm(
    ricci: Array, state: GlobalState, sphere_inverse: Array
) -> Array:
    omega_sq = state.omega**2
    shift = state.shift
    angular = ricci[..., 2:5, 2:5]
    r44 = ricci[..., 1, 1] / omega_sq
    r33 = (
        ricci[..., 0, 0]
        + 2.0 * np.einsum("n...i,n...i->n...", shift, ricci[..., 0, 2:5])
        + np.einsum("n...i,n...ij,n...j->n...", shift, angular, shift)
    ) / omega_sq
    r34 = (
        ricci[..., 0, 1]
        + np.einsum("n...i,n...i->n...", shift, ricci[..., 2:5, 1])
    ) / omega_sq
    r4 = ricci[..., 1, 2:5] / state.omega[..., None]
    r3 = (
        ricci[..., 0, 2:5]
        + np.einsum("n...i,n...ij->n...j", shift, angular)
    ) / state.omega[..., None]
    value = r33**2 + r44**2 + 2.0 * r34**2
    value += np.einsum("n...i,n...ij,n...j->n...", r3, sphere_inverse, r3)
    value += np.einsum("n...i,n...ij,n...j->n...", r4, sphere_inverse, r4)
    value += np.einsum(
        "n...ik,n...jl,n...ij,n...kl->n...",
        sphere_inverse,
        sphere_inverse,
        angular,
        angular,
    )
    return np.sqrt(np.maximum(value, 0.0))


def subset_state(state: GlobalState, v_indices: Array) -> GlobalState:
    return GlobalState(
        **{
            name: getattr(state, name)[:, :, v_indices].copy()
            for name in GlobalState.__dataclass_fields__
        }
    )


def audit_state(
    grid: PointSphereGrid,
    state: GlobalState,
    u: Array,
    v: Array,
    block_size: int = 7,
) -> tuple[Array, Array]:
    """Return (-u)||Ric||_L2(S) and the pointwise positive Ricci norm."""

    residual = np.empty((len(u), len(v)), dtype=float)
    rho = np.empty((grid.count, len(u), len(v)), dtype=float)
    for target_start in range(0, len(v), block_size):
        target_stop = min(target_start + block_size, len(v))
        local_start = max(0, target_start - 16)
        local_stop = min(len(v), target_stop + 16)
        if local_stop - local_start < min(33, len(v)):
            if local_start == 0:
                local_stop = min(len(v), 33)
            else:
                local_start = max(0, len(v) - 33)
        indices = np.arange(local_start, local_stop)
        local_state = subset_state(state, indices)
        ricci, sphere_inverse, _ = direct_ricci_block(
            grid, local_state, u, v[indices]
        )
        local_rho = adapted_ricci_norm(ricci, local_state, sphere_inverse)
        for target in range(target_start, target_stop):
            local = target - local_start
            rho[:, :, target] = local_rho[:, :, local]
            local_metric = np.einsum(
                "nia,n...ij,njb->n...ab",
                grid.frames,
                local_state.metric[:, :, local],
                grid.frames,
            )
            area_ratio = np.sqrt(np.maximum(np.linalg.det(local_metric), 0.0))
            l2 = np.sqrt(
                np.maximum(
                    4.0
                    * math.pi
                    * np.mean(local_rho[:, :, local] ** 2 * area_ratio, axis=0),
                    0.0,
                )
            )
            residual[:, target] = (-u) * l2
    return residual, rho


def minkowski_state(grid: PointSphereGrid, u: Array, v: Array) -> GlobalState:
    radius = v[None, None, :] - u[None, :, None]
    projector = grid.projector[:, None, None]
    metric = radius[..., None, None] ** 2 * projector
    shape = (grid.count, len(u), len(v))
    return GlobalState(
        metric=metric,
        omega=np.ones(shape),
        zeta_up=np.zeros(shape + (3,)),
        shift=np.zeros(shape + (3,)),
        q=np.broadcast_to(2.0 / radius, shape).copy(),
        shear=np.zeros_like(metric),
    )


def minkowski_audit(point_count: int, n_u: int, n_v: int) -> dict:
    grid = PointSphereGrid.create(point_count, neighbor_count=24)
    u = np.linspace(-1.0, -0.8, n_u)
    v = np.linspace(0.0005, 0.005, n_v)
    residual, _ = audit_state(grid, minkowski_state(grid, u, v), u, v)
    core = residual[5:-5, 5:-5]
    return {
        "point_count": point_count,
        "n_u": n_u,
        "n_v": n_v,
        "core_rms": float(np.sqrt(np.mean(core**2))),
        "core_max": float(np.max(core)),
    }
