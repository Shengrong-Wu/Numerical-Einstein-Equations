"""Independent graph-expansion geometry for Experiment 8.

The outgoing expansion of a graph ``u=h(theta)`` on an incoming null cone is
reconstructed from the saved primitive four-metric and its connection.  The
stored null expansions are deliberately not used in the MOTS equation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from nee.diagnostics.independent_audit import (
    angular_reference_derivative,
    build_metric_and_inverse,
)
from nee.numerics.sphere import PointSphereGrid
from nee.state.fields import PrimitiveFields


Array = np.ndarray


def composite_value_and_derivative(
    mesh: object, values: Array, target: float, axis: int
) -> tuple[Array, Array]:
    """Evaluate a composite-LGL polynomial and its derivative at ``target``."""

    moved = np.moveaxis(values, axis, 0)
    values_at_target = []
    derivatives_at_target = []
    tolerance = 2.0e-14
    for segment, indices in zip(mesh.segments, mesh.indices, strict=True):
        if segment.left - tolerance <= target <= segment.right + tolerance:
            local = moved[indices]
            values_at_target.append(
                np.tensordot(
                    segment.interpolation_matrix(np.array([target]))[0],
                    local,
                    axes=(0, 0),
                )
            )
            derivatives_at_target.append(
                np.tensordot(
                    segment.derivative_interpolation_matrix(
                        np.array([target])
                    )[0],
                    local,
                    axes=(0, 0),
                )
            )
    if not values_at_target:
        raise ValueError("target is outside the composite mesh")
    return (
        np.mean(np.stack(values_at_target), axis=0),
        np.mean(np.stack(derivatives_at_target), axis=0),
    )


def _local_change_of_basis(grid: PointSphereGrid) -> tuple[Array, Array]:
    """Return the embedding and angular left inverse for ``(u,v,e1,e2)``."""

    count = grid.count
    embedding = np.zeros((count, 5, 4), dtype=float)
    inverse = np.zeros((count, 4, 5), dtype=float)
    embedding[:, 0, 0] = 1.0
    embedding[:, 1, 1] = 1.0
    embedding[:, 2:5, 2:4] = grid.frames
    inverse[:, 0, 0] = 1.0
    inverse[:, 1, 1] = 1.0
    inverse[:, 2:4, 2:5] = np.swapaxes(grid.frames, 1, 2)
    return embedding, inverse


def metric_and_connection_at_v(
    grid: PointSphereGrid,
    fields: PrimitiveFields,
    coordinates: object,
    v_value: float,
) -> tuple[Array, Array]:
    """Reconstruct the four-metric connection at an arbitrary physical ``v``."""

    if not 0.0 <= v_value <= float(coordinates.v1):
        raise ValueError("v_value leaves the available null interval")
    metric5, _, _ = build_metric_and_inverse(
        grid, fields.g, fields.log_Omega, fields.b
    )
    s_value = (float(v_value) / float(coordinates.v1)) ** float(
        coordinates.delta
    )
    metric_slice, derivative_s = composite_value_and_derivative(
        coordinates.s, metric5, s_value, axis=2
    )
    exponent = 1.0 / float(coordinates.delta)
    dv_ds = float(coordinates.v1) * exponent * s_value ** (exponent - 1.0)
    if dv_ds <= 0.0:
        raise ValueError("arbitrary-v reconstruction is not used at v=0")
    inverse_slice = np.linalg.pinv(
        metric_slice, rcond=1.0e-13, hermitian=True
    )
    derivative = angular_reference_derivative(
        grid, metric_slice, tensor_rank=2
    )
    derivative[..., 0, :, :] = coordinates.differentiate_u(
        metric_slice, axis=1
    )
    derivative[..., 1, :, :] = derivative_s / dv_ds
    lower = 0.5 * (
        np.einsum("n...ijl->n...lij", derivative)
        + np.einsum("n...jil->n...lij", derivative)
        - derivative
    )
    connection5 = np.einsum("nrml,nrlab->nrmab", inverse_slice, lower)
    embedding, inverse = _local_change_of_basis(grid)
    metric4 = np.einsum(
        "naj,nrab,nbk->nrjk", embedding, metric_slice, embedding
    )
    connection4 = np.einsum(
        "nim,nrmab,naj,nbk->nrijk",
        inverse,
        connection5,
        embedding,
        embedding,
        optimize=True,
    )
    return metric4, connection4


def _interpolate_per_generator(
    splines: list[CubicSpline], values: Array
) -> Array:
    return np.stack(
        [spline(value) for spline, value in zip(splines, values, strict=True)]
    )


@dataclass(frozen=True)
class GraphExpansion:
    theta_out: Array
    theta_in: Array
    area_density: Array
    outgoing_normal: Array
    incoming_normal: Array
    induced_metric: Array
    spacetime_metric: Array


@dataclass
class NullConeGeometry:
    grid: PointSphereGrid
    u: Array
    v_value: float
    metric_splines: list[CubicSpline]
    connection_splines: list[CubicSpline]
    b_splines: list[CubicSpline]

    @classmethod
    def create_at_v(
        cls,
        grid: PointSphereGrid,
        fields: PrimitiveFields,
        coordinates: object,
        v_value: float,
    ) -> "NullConeGeometry":
        metric, connection = metric_and_connection_at_v(
            grid, fields, coordinates, v_value
        )
        s_value = (float(v_value) / float(coordinates.v1)) ** float(
            coordinates.delta
        )
        b_ambient, _ = composite_value_and_derivative(
            coordinates.s, fields.b, s_value, axis=2
        )
        b_local = np.einsum("nia,nui->nua", grid.frames, b_ambient)
        u = np.asarray(coordinates.u)
        return cls(
            grid=grid,
            u=u,
            v_value=float(v_value),
            metric_splines=[CubicSpline(u, value, axis=0) for value in metric],
            connection_splines=[
                CubicSpline(u, value, axis=0) for value in connection
            ],
            b_splines=[CubicSpline(u, value, axis=0) for value in b_local],
        )

    def expansion(self, h: Array) -> GraphExpansion:
        """Evaluate both future null expansions of the graph ``u=h``."""

        h = np.asarray(h, dtype=float)
        if h.shape != (self.grid.count,):
            raise ValueError(f"h must have shape ({self.grid.count},)")
        if np.min(h) < self.u[0] or np.max(h) > self.u[-1]:
            raise ValueError("graph leaves the available u interval")

        metric = _interpolate_per_generator(self.metric_splines, h)
        connection = _interpolate_per_generator(self.connection_splines, h)
        b = _interpolate_per_generator(self.b_splines, h)

        gradient_ambient = self.grid.reference_derivative(h, tensor_rank=0)
        gradient = np.einsum("nia,ni->na", self.grid.frames, gradient_ambient)
        hessian_ambient = self.grid.reference_derivative(
            gradient_ambient, tensor_rank=1
        )
        hessian = np.einsum(
            "nia,nij,njb->nab",
            self.grid.frames,
            hessian_ambient,
            self.grid.frames,
        )
        hessian = 0.5 * (hessian + np.swapaxes(hessian, -1, -2))

        tangent = np.zeros((self.grid.count, 2, 4), dtype=float)
        tangent[:, :, 0] = gradient
        tangent[:, 0, 2] = 1.0
        tangent[:, 1, 3] = 1.0
        induced = np.einsum("nai,nij,nbj->nab", tangent, metric, tangent)
        induced_inverse = np.linalg.inv(induced)

        incoming = np.zeros((self.grid.count, 4), dtype=float)
        incoming[:, 0] = 1.0
        incoming[:, 2:4] = b
        outgoing = np.empty_like(incoming)
        for index in range(self.grid.count):
            cov_tangent = metric[index] @ tangent[index].T
            cov_incoming = metric[index] @ incoming[index]
            constraints = np.vstack([cov_tangent.T, cov_incoming[None, :]])
            seed = np.linalg.lstsq(
                constraints, np.array([0.0, 0.0, -2.0]), rcond=None
            )[0]
            seed_norm = float(seed @ metric[index] @ seed)
            outgoing[index] = seed + 0.25 * seed_norm * incoming[index]

        acceleration = np.einsum(
            "nmjk,naj,nbk->nabm", connection, tangent, tangent, optimize=True
        )
        acceleration[:, :, :, 0] += hessian
        outgoing_cov = np.einsum("ni,nij->nj", outgoing, metric)
        incoming_cov = np.einsum("ni,nij->nj", incoming, metric)
        second_out = -np.einsum("ni,nabi->nab", outgoing_cov, acceleration)
        second_in = -np.einsum("ni,nabi->nab", incoming_cov, acceleration)
        theta_out = np.einsum("nab,nab->n", induced_inverse, second_out)
        theta_in = np.einsum("nab,nab->n", induced_inverse, second_in)
        area_density = np.sqrt(np.maximum(np.linalg.det(induced), 0.0))
        return GraphExpansion(
            theta_out=theta_out,
            theta_in=theta_in,
            area_density=area_density,
            outgoing_normal=outgoing,
            incoming_normal=incoming,
            induced_metric=induced,
            spacetime_metric=metric,
        )
