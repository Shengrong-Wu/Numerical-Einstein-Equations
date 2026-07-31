"""One-dimensional Legendre--Gauss--Lobatto spectral elements.

The numerical solver can use unrelated local polynomials for RK
stage interpolation, cumulative integration, and fresh differentiation.  This
module provides one polynomial per coordinate element, with its nodes,
derivative, integral, interpolation, and quadrature operators derived from the
same Lagrange basis.

The implementation is NumPy-only and intentionally independent of the
Einstein variables.  It is the coordinate-space foundation for the
hp/collocation solver; the existing angular Galerkin projection remains a
separate semidiscretization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.polynomial.legendre import Legendre, legvander


Array = np.ndarray
InterfaceRule = Literal["average", "left", "right"]


def _barycentric_weights(nodes: Array) -> Array:
    differences = nodes[:, None] - nodes[None, :]
    np.fill_diagonal(differences, 1.0)
    return 1.0 / np.prod(differences, axis=1)


def _reference_lgl(degree: int) -> tuple[Array, Array, Array, Array, Array]:
    """Return nodes, quadrature, D, Q, and barycentric weights on [-1,1]."""

    if degree < 2:
        raise ValueError("an LGL element requires polynomial degree at least 2")
    polynomial = Legendre.basis(degree)
    interior = np.sort(polynomial.deriv().roots())
    nodes = np.concatenate(([-1.0], interior, [1.0]))
    quadrature = 2.0 / (
        degree * (degree + 1.0) * polynomial(nodes) ** 2
    )

    barycentric = _barycentric_weights(nodes)
    differences = nodes[:, None] - nodes[None, :]
    derivative = np.zeros((degree + 1, degree + 1), dtype=float)
    mask = ~np.eye(degree + 1, dtype=bool)
    safe_differences = differences.copy()
    np.fill_diagonal(safe_differences, 1.0)
    derivative[mask] = (
        barycentric[None, :] / barycentric[:, None] / safe_differences
    )[mask]
    derivative[np.diag_indices_from(derivative)] = -np.sum(
        derivative, axis=1
    )

    # Q[i,j] is the integral from -1 to node i of the j-th Lagrange
    # polynomial.  Both Q and D therefore come from the same degree-p nodal
    # polynomial; no local stencil or independent fit enters.
    vandermonde = legvander(nodes, degree)
    inverse_vandermonde = np.linalg.inv(vandermonde)
    integrated_basis = np.empty_like(vandermonde)
    for mode in range(degree + 1):
        antiderivative = Legendre.basis(mode).integ()
        integrated_basis[:, mode] = (
            antiderivative(nodes) - antiderivative(-1.0)
        )
    integral = integrated_basis @ inverse_vandermonde
    integral[0] = 0.0
    return nodes, quadrature, derivative, integral, barycentric


@dataclass(frozen=True)
class LGLSegment:
    """A physical interval carrying one degree-p LGL polynomial."""

    left: float
    right: float
    degree: int
    nodes: Array
    weights: Array
    derivative: Array
    integral: Array
    barycentric: Array
    reference_nodes: Array

    @classmethod
    def create(cls, left: float, right: float, degree: int) -> "LGLSegment":
        if not np.isfinite(left) or not np.isfinite(right) or not left < right:
            raise ValueError("require finite element endpoints left < right")
        reference, weights, derivative, integral, barycentric = _reference_lgl(
            degree
        )
        half = 0.5 * (right - left)
        center = 0.5 * (right + left)
        return cls(
            left=float(left),
            right=float(right),
            degree=int(degree),
            nodes=center + half * reference,
            weights=half * weights,
            derivative=derivative / half,
            integral=half * integral,
            barycentric=barycentric,
            reference_nodes=reference,
        )

    def interpolation_matrix(self, targets: Array) -> Array:
        """Barycentric interpolation from element nodes to physical targets."""

        target = np.atleast_1d(np.asarray(targets, dtype=float))
        if np.any(target < self.left - 1.0e-14) or np.any(
            target > self.right + 1.0e-14
        ):
            raise ValueError("interpolation target lies outside the element")
        result = np.empty((len(target), len(self.nodes)), dtype=float)
        for row, value in enumerate(target):
            distance = value - self.nodes
            exact = np.flatnonzero(np.abs(distance) <= 8.0 * np.finfo(float).eps)
            if exact.size:
                result[row] = 0.0
                result[row, exact[0]] = 1.0
                continue
            terms = self.barycentric / distance
            result[row] = terms / np.sum(terms)
        return result

    def derivative_interpolation_matrix(self, targets: Array) -> Array:
        """Differentiate the element polynomial at physical targets."""

        target = np.atleast_1d(np.asarray(targets, dtype=float))
        if np.any(target < self.left - 1.0e-14) or np.any(
            target > self.right + 1.0e-14
        ):
            raise ValueError("derivative target lies outside the element")
        reference_target = (
            2.0 * target - (self.left + self.right)
        ) / (self.right - self.left)
        derivative_basis = np.empty(
            (len(target), self.degree + 1), dtype=float
        )
        for mode in range(self.degree + 1):
            derivative_basis[:, mode] = Legendre.basis(mode).deriv()(
                reference_target
            )
        inverse = np.linalg.inv(
            legvander(self.reference_nodes, self.degree)
        )
        return (2.0 / (self.right - self.left)) * (
            derivative_basis @ inverse
        )

    def interpolate(self, values: Array, targets: Array, axis: int = 0) -> Array:
        moved = np.moveaxis(values, axis, 0)
        if moved.shape[0] != len(self.nodes):
            raise ValueError("the interpolation axis does not match this element")
        output = np.tensordot(
            self.interpolation_matrix(targets), moved, axes=(1, 0)
        )
        return np.moveaxis(output, 0, axis)

    def modal_coefficients(self, values: Array, axis: int = 0) -> Array:
        """Legendre coefficients of the element interpolation polynomial."""

        moved = np.moveaxis(values, axis, 0)
        if moved.shape[0] != len(self.nodes):
            raise ValueError("the modal axis does not match this element")
        inverse = np.linalg.inv(legvander(self.reference_nodes, self.degree))
        coefficients = np.tensordot(inverse, moved, axes=(1, 0))
        return np.moveaxis(coefficients, 0, axis)


@dataclass(frozen=True)
class CompositeLGLMesh:
    """Continuous multi-element LGL grid with shared interface nodes."""

    segments: tuple[LGLSegment, ...]
    indices: tuple[Array, ...]
    nodes: Array

    @classmethod
    def create(
        cls, breakpoints: Array, degrees: int | Array
    ) -> "CompositeLGLMesh":
        points = np.asarray(breakpoints, dtype=float)
        if points.ndim != 1 or len(points) < 2 or np.any(np.diff(points) <= 0.0):
            raise ValueError("breakpoints must be a strictly increasing vector")
        if np.isscalar(degrees):
            degree_values = np.full(len(points) - 1, int(degrees), dtype=int)
        else:
            degree_values = np.asarray(degrees, dtype=int)
        if degree_values.shape != (len(points) - 1,):
            raise ValueError("one polynomial degree is required per element")

        segments: list[LGLSegment] = []
        indices: list[Array] = []
        global_nodes: list[float] = []
        start = 0
        for element, degree in enumerate(degree_values):
            segment = LGLSegment.create(
                float(points[element]), float(points[element + 1]), int(degree)
            )
            segments.append(segment)
            if element == 0:
                global_nodes.extend(float(value) for value in segment.nodes)
                local = np.arange(degree + 1, dtype=int)
                start = degree
            else:
                global_nodes.extend(float(value) for value in segment.nodes[1:])
                local = np.arange(start, start + degree + 1, dtype=int)
                start += degree
            indices.append(local)
        return cls(
            segments=tuple(segments),
            indices=tuple(indices),
            nodes=np.asarray(global_nodes),
        )

    def _check_axis(self, values: Array, axis: int) -> Array:
        moved = np.moveaxis(values, axis, 0)
        if moved.shape[0] != len(self.nodes):
            raise ValueError("the coordinate axis does not match the mesh")
        return moved

    def differentiate(
        self,
        values: Array,
        axis: int = 0,
        interface_rule: InterfaceRule = "average",
    ) -> Array:
        """Differentiate each element and select an interface trace."""

        moved = self._check_axis(values, axis)
        result = np.zeros_like(moved)
        multiplicity = np.zeros(len(self.nodes), dtype=float)
        for element, (segment, index) in enumerate(
            zip(self.segments, self.indices, strict=True)
        ):
            local = np.tensordot(segment.derivative, moved[index], axes=(1, 0))
            if interface_rule == "left":
                selected = index if element == 0 else index[1:]
                local_selected = local if element == 0 else local[1:]
            elif interface_rule == "right":
                selected = index if element == len(self.segments) - 1 else index[:-1]
                local_selected = (
                    local if element == len(self.segments) - 1 else local[:-1]
                )
            elif interface_rule == "average":
                selected = index
                local_selected = local
            else:
                raise ValueError(f"unknown interface rule: {interface_rule}")
            result[selected] += local_selected
            multiplicity[selected] += 1.0
        reshape = (len(multiplicity),) + (1,) * (result.ndim - 1)
        result /= multiplicity.reshape(reshape)
        return np.moveaxis(result, 0, axis)

    def integrate(self, values: Array, axis: int = 0) -> Array:
        """Cumulatively integrate the shared element polynomial."""

        moved = self._check_axis(values, axis)
        result = np.zeros_like(moved)
        carry = np.zeros_like(moved[0])
        for segment, index in zip(self.segments, self.indices, strict=True):
            local = np.tensordot(segment.integral, moved[index], axes=(1, 0))
            local = local + carry
            result[index] = local
            carry = local[-1]
        return np.moveaxis(result, 0, axis)

    def interface_derivative_jumps(self, values: Array, axis: int = 0) -> Array:
        """Return right-minus-left derivative traces at internal interfaces."""

        moved = self._check_axis(values, axis)
        jumps = []
        for element in range(len(self.segments) - 1):
            left_segment = self.segments[element]
            right_segment = self.segments[element + 1]
            left_values = moved[self.indices[element]]
            right_values = moved[self.indices[element + 1]]
            left_derivative = np.tensordot(
                left_segment.derivative[-1], left_values, axes=(0, 0)
            )
            right_derivative = np.tensordot(
                right_segment.derivative[0], right_values, axes=(0, 0)
            )
            jumps.append(right_derivative - left_derivative)
        if not jumps:
            return np.empty((0, *moved.shape[1:]), dtype=moved.dtype)
        return np.stack(jumps, axis=0)

    def maximum_modal_tail(
        self, values: Array, axis: int = 0, tail_modes: int = 2
    ) -> float:
        """Largest last-mode energy ratio over all coordinate elements."""

        if tail_modes < 1:
            raise ValueError("tail_modes must be positive")
        moved = self._check_axis(values, axis)
        maximum = 0.0
        for segment, index in zip(self.segments, self.indices, strict=True):
            coefficients = segment.modal_coefficients(moved[index], axis=0)
            scale = float(np.max(np.abs(coefficients)))
            if scale == 0.0:
                continue
            normalized = np.abs(coefficients) / scale
            total = float(np.sum(normalized**2))
            tail = float(np.sum(normalized[-tail_modes:] ** 2))
            maximum = max(maximum, np.sqrt(tail / max(total, 1.0e-300)))
        return maximum


@dataclass(frozen=True)
class CharacteristicLGLMesh:
    """Tensor-product mesh in tau=-log(-u) and s=sqrt(v/v1)."""

    tau: CompositeLGLMesh
    s: CompositeLGLMesh
    v1: float
    u: Array
    v: Array

    @classmethod
    def create(
        cls,
        tau_breakpoints: Array,
        tau_degrees: int | Array,
        s_breakpoints: Array,
        s_degrees: int | Array,
        v1: float,
    ) -> "CharacteristicLGLMesh":
        if not np.isfinite(v1) or v1 <= 0.0:
            raise ValueError("v1 must be positive")
        tau = CompositeLGLMesh.create(tau_breakpoints, tau_degrees)
        s = CompositeLGLMesh.create(s_breakpoints, s_degrees)
        return cls(
            tau=tau,
            s=s,
            v1=float(v1),
            u=-np.exp(-tau.nodes),
            v=float(v1) * s.nodes**2,
        )

    def differentiate_u(self, values: Array, axis: int = 1) -> Array:
        derivative_tau = self.tau.differentiate(
            values, axis=axis, interface_rule="average"
        )
        shape = [1] * values.ndim
        shape[axis] = len(self.u)
        return derivative_tau / (-self.u).reshape(shape)

    def integrate_v(self, source: Array, axis: int = 2) -> Array:
        shape = [1] * source.ndim
        shape[axis] = len(self.s.nodes)
        jacobian = (2.0 * self.v1 * self.s.nodes).reshape(shape)
        return self.s.integrate(jacobian * source, axis=axis)

    def differentiate_v(self, values: Array, axis: int = 2) -> Array:
        """Differentiate in physical v through the common s polynomial.

        The prescribed corner is allowed to be only smooth in ``s=sqrt(v/v1)``.
        Consequently a physical-v derivative need not have a finite value at
        ``s=0``.  For array-shaped diagnostics we copy the first positive-node
        derivative into that single boundary slot; production construction
        residuals use their stored ODE sources there instead.
        """

        derivative_s = self.s.differentiate(
            values, axis=axis, interface_rule="average"
        )
        moved = np.moveaxis(derivative_s, axis, 0)
        result = np.empty_like(moved)
        jacobian = 2.0 * self.v1 * self.s.nodes
        result[1:] = moved[1:] / jacobian[1:].reshape(
            (len(jacobian) - 1,) + (1,) * (moved.ndim - 1)
        )
        result[0] = result[1]
        return np.moveaxis(result, 0, axis)

    def diagnostics(self) -> dict[str, object]:
        return {
            "coordinate_system": "tau=-log(-u), s=sqrt(v/v1)",
            "tau_breakpoints": [
                self.tau.segments[0].left,
                *[segment.right for segment in self.tau.segments],
            ],
            "tau_degrees": [segment.degree for segment in self.tau.segments],
            "s_breakpoints": [
                self.s.segments[0].left,
                *[segment.right for segment in self.s.segments],
            ],
            "s_degrees": [segment.degree for segment in self.s.segments],
            "u_count": len(self.u),
            "v_count": len(self.v),
        }
