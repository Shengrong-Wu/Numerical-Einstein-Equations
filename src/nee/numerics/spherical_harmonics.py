"""Typed harmonic Galerkin projections for numerical angular fields.

The existing point-sphere derivative is a least-squares scalar-harmonic
operator.  Applying it to raw nodal values while performing nonlinear algebra
outside its represented space creates a hidden nodal subspace.  This module
builds explicit scalar, tangent-vector, and tangent-symmetric-tensor bases so
that every projected field is an analysis/synthesis image of one declared
angular space.

The retained degree is the evolution space.  A larger work degree represents
nonlinear products before their high modes are discarded.  No equation is
altered: a semidiscrete right-hand side is defined as ``Pi_L F(U_L)``.
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
    spherical_harmonic_collocation,
    tangent_inverse,
)


Array = np.ndarray


def _reference_derivative(
    grid: PointSphereGrid,
    ambient_derivative: Array,
    values: Array,
    tensor_rank: int,
) -> Array:
    """Round covariant derivative using one explicit spectral space.

    The ambient Cartesian components of the gradient of a degree-``ell``
    scalar harmonic contain modes through degree ``ell + 1``.  Building the
    vector and tensor families with the derivative operator already stored on
    ``grid`` is therefore correct only when that undocumented operator happens
    to have sufficient degree.  The Galerkin family instead owns the degree
    ``work_degree + 1`` operator passed here.
    """

    if tensor_rank < 0 or tensor_rank > 2:
        raise ValueError("typed Galerkin bases use tensor ranks zero through two")
    if tensor_rank and values.shape[-tensor_rank:] != (3,) * tensor_rank:
        raise ValueError("ambient tensor slots must all have length three")
    if values.shape[0] != grid.count:
        raise ValueError("the first array axis must be the sphere point")

    # tensordot initially puts the derivative slot directly after the point
    # slot.  Move it behind all batch slots and before the tensor slots, which
    # is the convention of PointSphereGrid.reference_derivative.
    raw = np.tensordot(ambient_derivative, values, axes=(2, 0))
    insert = values.ndim - tensor_rank
    raw = np.moveaxis(raw, 1, insert)
    if tensor_rank == 0:
        return raw
    if tensor_rank == 1:
        return np.einsum("nja,n...ia->n...ij", grid.projector, raw)
    return np.einsum(
        "nja,nkb,n...iab->n...ijk", grid.projector, grid.projector, raw
    )


def scalar_ell_labels(degree: int) -> Array:
    return np.concatenate(
        [np.full(2 * ell + 1, ell, dtype=int) for ell in range(degree + 1)]
    )


def _basis_matrix(basis: Array, slot_rank: int) -> Array:
    """Flatten sphere and tensor slots while retaining the mode axis."""

    slot_axes = list(range(2, 2 + slot_rank))
    ordered = np.transpose(basis, [0, *slot_axes, 1])
    return ordered.reshape(-1, basis.shape[1])


def _flatten_values(values: Array, slot_rank: int) -> tuple[Array, tuple[int, ...]]:
    batch_shape = values.shape[1 : values.ndim - slot_rank]
    batch_axes = list(range(1, 1 + len(batch_shape)))
    slot_axes = list(range(1 + len(batch_shape), values.ndim))
    ordered = np.transpose(values, [0, *slot_axes, *batch_axes])
    slot_size = 3**slot_rank
    return ordered.reshape(values.shape[0] * slot_size, -1), batch_shape


def _unflatten_values(
    flat: Array, count: int, batch_shape: tuple[int, ...], slot_rank: int
) -> Array:
    slot_shape = (3,) * slot_rank
    ordered = flat.reshape((count, *slot_shape, *batch_shape))
    slot_axes = list(range(1, 1 + slot_rank))
    batch_axes = list(range(1 + slot_rank, ordered.ndim))
    return np.transpose(ordered, [0, *batch_axes, *slot_axes])


def _minimum_norm_constraint_correction(
    constraint: Array,
    rhs: Array,
    *,
    rcond: float = 1.0e-12,
) -> Array:
    """Solve an underdetermined full-row-rank constraint by thin QR.

    For ``C delta = rhs`` with at least as many columns as rows, factor
    ``C.T = Q R``.  The minimum-Euclidean-norm solution is then
    ``delta = Q solve(R.T, rhs)``.  For a safely full-row-rank block this is
    algebraically equivalent to the minimum-norm least-squares solve used
    previously, but it avoids the SVD driver failure observed for a finite,
    condition-one 144-by-424 moving trace block.  A reciprocal infinity-norm
    condition gate is deliberately conservative relative to an SVD rank
    cutoff; the backward-residual check then fails closed if the solve is
    inaccurate.
    """

    matrix = np.asarray(constraint, dtype=float)
    target = np.asarray(rhs, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("constraint must be a two-dimensional matrix")
    row_count, column_count = matrix.shape
    if row_count > column_count:
        raise ValueError("minimum-norm constraint solve requires rows <= columns")
    if target.shape != (row_count,):
        raise ValueError(
            f"constraint rhs has shape {target.shape}, expected {(row_count,)}"
        )
    if not math.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("constraint rcond must be positive and finite")
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise FloatingPointError("constraint solve received nonfinite values")
    if row_count == 0:
        return np.zeros(column_count, dtype=float)

    orthogonal, triangular = np.linalg.qr(matrix.T, mode="reduced")
    diagonal = np.abs(np.diag(triangular))
    diagonal_scale = float(np.max(diagonal))
    if (
        not math.isfinite(diagonal_scale)
        or diagonal_scale <= 0.0
        or float(np.min(diagonal)) <= rcond * diagonal_scale
    ):
        raise np.linalg.LinAlgError(
            "moving trace constraint is rank deficient or ill-conditioned"
        )
    condition = float(np.linalg.cond(triangular, p=np.inf))
    if not math.isfinite(condition) or condition * rcond >= 1.0:
        raise np.linalg.LinAlgError(
            "moving trace constraint is rank deficient or ill-conditioned"
        )
    reduced = np.linalg.solve(triangular.T, target)
    correction = orthogonal @ reduced

    residual = matrix @ correction - target
    residual_norm = float(np.linalg.norm(residual))
    backward_scale = float(
        np.linalg.norm(target)
        + np.linalg.norm(matrix) * np.linalg.norm(correction)
    )
    backward_tolerance = (
        256.0
        * np.finfo(float).eps
        * max(backward_scale, np.finfo(float).tiny)
    )
    if not math.isfinite(residual_norm) or residual_norm > backward_tolerance:
        raise np.linalg.LinAlgError(
            "moving trace constraint QR solve failed its backward-error gate"
        )
    return correction


@dataclass(frozen=True)
class HarmonicFamily:
    basis: Array
    matrix: Array
    pseudoinverse: Array
    ell: Array
    retained: Array
    retained_indices: Array
    retained_matrix: Array
    retained_pseudoinverse: Array
    retained_basis: Array
    slot_rank: int
    condition: float

    @classmethod
    def build(
        cls,
        basis: Array,
        ell: Array,
        retained_degree: int,
        slot_rank: int,
    ) -> "HarmonicFamily":
        matrix = _basis_matrix(basis, slot_rank)
        pseudoinverse = np.linalg.pinv(matrix, rcond=1.0e-12)
        retained = ell <= retained_degree
        retained_indices = np.flatnonzero(retained)

        # Boolean/advanced indexing of ``basis`` in an ODE right-hand side
        # silently copies a large four-dimensional array.  Materialize the
        # retained flattened family once and recover a tensor-shaped view of
        # it.  At L=9 on 450 sphere points this avoids a roughly 9 MiB copy at
        # every half-shear stage.
        retained_matrix = np.ascontiguousarray(matrix[:, retained_indices])
        slot_shape = (3,) * slot_rank
        retained_ordered = retained_matrix.reshape(
            (basis.shape[0], *slot_shape, len(retained_indices))
        )
        retained_basis = np.moveaxis(retained_ordered, -1, 1)
        retained_pseudoinverse = np.ascontiguousarray(
            pseudoinverse[retained_indices]
        )
        return cls(
            basis=basis,
            matrix=matrix,
            pseudoinverse=pseudoinverse,
            ell=ell,
            retained=retained,
            retained_indices=retained_indices,
            retained_matrix=retained_matrix,
            retained_pseudoinverse=retained_pseudoinverse,
            retained_basis=retained_basis,
            slot_rank=slot_rank,
            condition=float(np.linalg.cond(matrix)),
        )

    def analyze(self, values: Array) -> Array:
        flat, batch_shape = _flatten_values(values, self.slot_rank)
        coefficients = self.pseudoinverse @ flat
        return coefficients.reshape((len(self.ell), *batch_shape))

    def synthesize(self, coefficients: Array) -> Array:
        batch_shape = coefficients.shape[1:]
        flat = self.matrix @ coefficients.reshape((len(self.ell), -1))
        return _unflatten_values(
            flat, self.basis.shape[0], batch_shape, self.slot_rank
        )

    def analyze_retained(self, values: Array) -> Array:
        """Analyze directly into the retained coefficient family."""

        flat, batch_shape = _flatten_values(values, self.slot_rank)
        coefficients = self.retained_pseudoinverse @ flat
        return coefficients.reshape((len(self.retained_indices), *batch_shape))

    def synthesize_retained(self, coefficients: Array) -> Array:
        """Synthesize retained coefficients without zero work-mode columns."""

        values = np.asarray(coefficients)
        if values.ndim < 1 or values.shape[0] != len(self.retained_indices):
            raise ValueError(
                "the leading coefficient axis must match the retained family"
            )
        batch_shape = values.shape[1:]
        flat = self.retained_matrix @ values.reshape(
            (len(self.retained_indices), -1)
        )
        return _unflatten_values(
            flat, self.basis.shape[0], batch_shape, self.slot_rank
        )

    def truncate(self, coefficients: Array) -> Array:
        result = coefficients.copy()
        result[~self.retained] = 0.0
        return result

    def project(self, values: Array) -> Array:
        return self.synthesize(self.truncate(self.analyze(values)))

    def tail_ratio(self, values: Array) -> Array:
        coefficients = self.analyze(values)
        total = np.sum(np.abs(coefficients) ** 2, axis=0)
        tail = np.sum(np.abs(coefficients[~self.retained]) ** 2, axis=0)
        return np.sqrt(tail / np.maximum(total, 1.0e-300))


class AngularGalerkin:
    """Typed retained/work harmonic spaces on one Fibonacci grid."""

    def __init__(
        self,
        grid: PointSphereGrid,
        retained_degree: int,
        work_degree: int,
    ) -> None:
        if retained_degree < 0 or work_degree < retained_degree:
            raise ValueError("require 0 <= retained_degree <= work_degree")
        differentiation_degree = work_degree + 1
        if (differentiation_degree + 1) ** 2 >= grid.count:
            raise ValueError(
                "degree-(work_degree + 1) differentiation space must be "
                "overdetermined"
            )
        self.grid = grid
        self.retained_degree = retained_degree
        self.work_degree = work_degree
        self.differentiation_degree = differentiation_degree

        ambient_derivative, differentiation_basis, differentiation_condition = (
            spherical_harmonic_collocation(grid.points, differentiation_degree)
        )
        scalar_mode_count = (work_degree + 1) ** 2
        scalar_basis = differentiation_basis[:, :scalar_mode_count]
        self.differentiation_condition = differentiation_condition
        scalar_ell = scalar_ell_labels(work_degree)
        self.scalar = HarmonicFamily.build(
            scalar_basis,
            scalar_ell,
            retained_degree,
            slot_rank=0,
        )
        gradient = _reference_derivative(
            grid, ambient_derivative, scalar_basis, tensor_rank=0
        )
        vector_modes = []
        vector_ell = []
        for mode, ell in enumerate(scalar_ell):
            if ell == 0:
                continue
            normalization = math.sqrt(float(ell * (ell + 1)))
            electric = gradient[:, mode] / normalization
            magnetic = np.cross(grid.points, gradient[:, mode]) / normalization
            vector_modes.extend([electric, magnetic])
            vector_ell.extend([ell, ell])
        vector_basis = np.stack(vector_modes, axis=1)
        self.vector = HarmonicFamily.build(
            vector_basis,
            np.asarray(vector_ell, dtype=int),
            retained_degree,
            slot_rank=1,
        )

        hessian = _reference_derivative(
            grid, ambient_derivative, gradient, tensor_rank=1
        )
        hessian = 0.5 * (hessian + np.swapaxes(hessian, -1, -2))
        star_gradient = np.cross(grid.points[:, None, :], gradient)
        star_derivative = _reference_derivative(
            grid, ambient_derivative, star_gradient, tensor_rank=1
        )
        tensor_modes = []
        tensor_ell = []
        tensor_kind = []
        for mode, ell in enumerate(scalar_ell):
            trace_mode = (
                scalar_basis[:, mode, None, None]
                * grid.projector
                / math.sqrt(2.0)
            )
            tensor_modes.append(trace_mode)
            tensor_ell.append(ell)
            tensor_kind.append("trace")
            if ell < 2:
                continue
            lam = float(ell * (ell + 1))
            normalization = math.sqrt(0.5 * lam * (lam - 2.0))
            electric = hessian[:, mode] + 0.5 * lam * scalar_basis[
                :, mode, None, None
            ] * grid.projector
            magnetic = star_derivative[:, mode] + np.swapaxes(
                star_derivative[:, mode], -1, -2
            )
            # Remove only round-trace roundoff from the analytic STF modes.
            for value, kind in (
                (electric / normalization, "electric"),
                (magnetic / (2.0 * normalization), "magnetic"),
            ):
                trace = np.einsum("nij,nij->n", grid.projector, value)
                value = value - 0.5 * trace[:, None, None] * grid.projector
                tensor_modes.append(value)
                tensor_ell.append(ell)
                tensor_kind.append(kind)
        tensor_basis = np.stack(tensor_modes, axis=1)
        self.sym2 = HarmonicFamily.build(
            tensor_basis,
            np.asarray(tensor_ell, dtype=int),
            retained_degree,
            slot_rank=2,
        )
        self.tensor_kind = np.asarray(tensor_kind)

        # Use the same work-space analysis followed by retained truncation as
        # project_scalar.  A fresh least-squares fit using only retained modes
        # would let discarded work modes alias back into this constraint.
        self.scalar_retained_analysis = self.scalar.pseudoinverse[
            self.scalar.retained
        ]

    def project_scalar(self, values: Array) -> Array:
        return self.scalar.project(values)

    def project_vector(self, values: Array) -> Array:
        return self.vector.project(values)

    def project_sym2(self, values: Array) -> Array:
        symmetric = 0.5 * (values + np.swapaxes(values, -1, -2))
        return self.sym2.project(symmetric)

    def project_g_tracefree(self, tensor: Array, inverse: Array) -> Array:
        """Least-change retained coefficients with zero retained g-trace."""

        if tensor.shape != inverse.shape:
            raise ValueError("tensor and inverse must have identical shapes")

        coefficients = self.sym2.analyze_retained(
            0.5 * (tensor + np.swapaxes(tensor, -1, -2))
        )
        retained_basis = self.sym2.retained_basis
        batch_shape = inverse.shape[1:-2]
        coefficient_flat = coefficients.reshape((coefficients.shape[0], -1))
        inverse_flat = inverse.reshape((inverse.shape[0], -1, 3, 3))
        corrected = np.empty_like(coefficient_flat)
        for batch in range(coefficient_flat.shape[1]):
            traces = np.einsum(
                "nij,nkij->nk", inverse_flat[:, batch], retained_basis
            )
            constraint = self.scalar_retained_analysis @ traces
            rhs = constraint @ coefficient_flat[:, batch]
            correction = _minimum_norm_constraint_correction(
                constraint, rhs, rcond=1.0e-12
            )
            corrected[:, batch] = coefficient_flat[:, batch] - correction
        return self.sym2.synthesize_retained(
            corrected.reshape((len(self.sym2.retained_indices), *batch_shape))
        )

    def g_tracefree_derivative_retained_coefficients(
        self,
        coefficients: Array,
        derivative_coefficients: Array,
        inverse: Array,
        inverse_derivative: Array,
    ) -> Array:
        """Apply the moving-constraint tangent projection in coefficient space.

        This is the coefficient-space core of
        :meth:`project_g_tracefree_derivative`.  Keeping it separate lets a
        reduced-coordinate DAE solver carry the already known retained
        coefficients through the projection and return only its independent
        electric/magnetic rows.  The same ``C_dot`` term and the same
        minimum-norm least-squares correction are retained exactly.
        """

        coefficient_values = np.asarray(coefficients, dtype=float)
        derivative_values = np.asarray(derivative_coefficients, dtype=float)
        expected_shape = (
            len(self.sym2.retained_indices),
            *inverse.shape[1:-2],
        )
        if coefficient_values.shape != expected_shape:
            raise ValueError(
                "retained tensor coefficients have shape "
                f"{coefficient_values.shape}, expected {expected_shape}"
            )
        if derivative_values.shape != expected_shape:
            raise ValueError(
                "retained derivative coefficients have shape "
                f"{derivative_values.shape}, expected {expected_shape}"
            )
        if inverse.shape != inverse_derivative.shape:
            raise ValueError(
                "inverse and inverse derivative must have identical shapes"
            )

        retained_basis = self.sym2.retained_basis
        coefficient_flat = coefficient_values.reshape(
            (coefficient_values.shape[0], -1)
        )
        derivative_flat = derivative_values.reshape(
            (derivative_values.shape[0], -1)
        )
        inverse_flat = inverse.reshape((inverse.shape[0], -1, 3, 3))
        inverse_derivative_flat = inverse_derivative.reshape(
            (inverse.shape[0], -1, 3, 3)
        )
        corrected = np.empty_like(derivative_flat)
        for batch in range(coefficient_flat.shape[1]):
            traces = np.einsum(
                "nij,nkij->nk", inverse_flat[:, batch], retained_basis
            )
            traces_derivative = np.einsum(
                "nij,nkij->nk",
                inverse_derivative_flat[:, batch],
                retained_basis,
            )
            constraint = self.scalar_retained_analysis @ traces
            constraint_derivative = (
                self.scalar_retained_analysis @ traces_derivative
            )
            defect = (
                constraint @ derivative_flat[:, batch]
                + constraint_derivative @ coefficient_flat[:, batch]
            )
            correction = _minimum_norm_constraint_correction(
                constraint, -defect, rcond=1.0e-12
            )
            corrected[:, batch] = derivative_flat[:, batch] + correction
        return corrected.reshape(expected_shape)

    def project_g_tracefree_derivative(
        self,
        tensor: Array,
        raw_derivative: Array,
        inverse: Array,
        inverse_derivative: Array,
    ) -> Array:
        """Project a tensor derivative tangent to the moving trace constraint.

        If retained coefficients ``c`` obey ``C(g)c=0``, their derivative
        must satisfy ``C c_dot + C_dot c=0``.  Merely making ``c_dot``
        trace-free drops the ``C_dot c`` term and is incorrect when the
        section metric varies.  This routine first forms the retained raw
        Galerkin derivative and then applies the minimum-norm coefficient
        correction that satisfies the differentiated constraint.
        """

        if not (
            tensor.shape
            == raw_derivative.shape
            == inverse.shape
            == inverse_derivative.shape
        ):
            raise ValueError(
                "tensor, derivative, inverse, and inverse derivative must "
                "have identical shapes"
            )
        coefficients = self.sym2.analyze_retained(
            0.5 * (tensor + np.swapaxes(tensor, -1, -2))
        )
        derivative_coefficients = self.sym2.analyze_retained(
            0.5
            * (raw_derivative + np.swapaxes(raw_derivative, -1, -2))
        )
        batch_shape = inverse.shape[1:-2]
        corrected = self.g_tracefree_derivative_retained_coefficients(
            coefficients,
            derivative_coefficients,
            inverse,
            inverse_derivative,
        )
        return self.sym2.synthesize_retained(
            corrected.reshape((len(self.sym2.retained_indices), *batch_shape))
        )

    def project_state(self, state: object) -> object:
        """Project primary numerical fields in place and return the state.

        ``Omega`` is represented through retained ``log(Omega)`` so that the
        projection preserves positivity.  ``Omega*chib`` is a general
        symmetric tensor (its trace is an evolved coefficient), whereas only
        the outgoing shear is subject to the g-trace-free constraint.
        """

        state.metric = self.project_sym2(state.metric)
        inverse = tangent_inverse(self.grid, state.metric)
        state.omega = np.exp(self.project_scalar(np.log(state.omega)))
        state.zeta_up = self.project_vector(state.zeta_up)
        state.shift = self.project_vector(state.shift)
        state.q = self.project_scalar(state.q)
        shear_trace = np.einsum(
            "n...ij,n...ij->n...", inverse, state.shear
        )
        state.shear = (
            0.5 * (state.shear + np.swapaxes(state.shear, -1, -2))
            - 0.5 * shear_trace[..., None, None] * state.metric
        )
        state.weighted_chib = self.project_sym2(state.weighted_chib)
        state.weighted_omega = self.project_scalar(state.weighted_omega)
        state.weighted_omegab = self.project_scalar(state.weighted_omegab)
        return state

    def diagnostics(self) -> dict[str, float | int]:
        return {
            "retained_degree": self.retained_degree,
            "work_degree": self.work_degree,
            "differentiation_degree": self.differentiation_degree,
            "scalar_modes": len(self.scalar.ell),
            "vector_modes": len(self.vector.ell),
            "sym2_modes": len(self.sym2.ell),
            "scalar_condition": self.scalar.condition,
            "vector_condition": self.vector.condition,
            "sym2_condition": self.sym2.condition,
            "differentiation_condition": self.differentiation_condition,
        }
