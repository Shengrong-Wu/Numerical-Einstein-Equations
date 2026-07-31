"""Equation-specific tau-LGL/SDC solvers for the numerical construction.

The four incoming-direction solves are wired into ``picard_step`` through a
local import (to avoid a module cycle).  They use primitive same-element
interpolation, independent overgrid gates, explicit shared-interface traces,
and off-grid geometric-cone audits.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable

import numpy as np

from .vacuum_iteration import (  # Imported one-way until production wiring.
    FirstOrderState,
    ProjectionTails,
    _minimum_tangent_eigenvalue,
    _project_scalar,
    _project_sym2,
    _project_tracefree_sym2,
    _record_tail,
)
from .spherical_harmonics import AngularGalerkin
from .lgl import CharacteristicLGLMesh, LGLSegment
from .sphere import (
    PointSphereGrid,
    connection_difference,
    lie_covariant_tensor,
    scalar_gradient,
    tangent_inverse,
    tensor_norm_sq,
    tensor_trace,
    tensor_tracefree,
    tracefree_square,
    tracefree_symmetric_gradient,
    vector_divergence,
)
from .tau_mesh import (
    TauElementEvaluator,
    assign_composite_element,
    composite_element_initial,
    expand_element_profiles,
    half_shear_trace_certificate,
    json_safe_diagnostic,
    tau_sdc_result_diagnostic,
)
from .tau_sdc import solve_tau_sdc


Array = np.ndarray


def _record_tail_coefficients(
    family: Any,
    coefficients: Array,
    tails: ProjectionTails | None,
    label: str,
) -> None:
    """Record a projection tail from coefficients already analyzed.

    This is algebraically identical to ``vacuum_iteration._record_tail``
    but avoids analyzing the half-shear right-hand side a second time merely
    to obtain its retained rows.
    """

    if tails is None:
        return
    values = np.asarray(coefficients)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError(
            f"angular projection '{label}' produced nonfinite coefficients"
        )
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        ratio = 0.0
    else:
        scaled = np.abs(values) / scale
        total = float(np.sum(scaled**2))
        discarded = float(np.sum(scaled[~family.retained] ** 2))
        ratio = math.sqrt(discarded / max(total, 1.0e-300))
    tails[label] = max(tails.get(label, 0.0), ratio)


class _BoundedCoordinateCache:
    """Byte-bounded exact-coordinate cache with deterministic counters."""

    def __init__(self, *, enabled: bool, maximum_bytes: int) -> None:
        if maximum_bytes < 0:
            raise ValueError("known-stage cache byte limit must be nonnegative")
        self.enabled = bool(enabled and maximum_bytes > 0)
        self.maximum_bytes = int(maximum_bytes)
        self._values: OrderedDict[float, tuple[Any, int]] = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.oversize_skips = 0

    @staticmethod
    def _nbytes(value: Any, seen: set[int] | None = None) -> int:
        visited = set() if seen is None else seen
        identity = id(value)
        if identity in visited:
            return 0
        visited.add(identity)
        if isinstance(value, np.ndarray):
            return int(value.nbytes)
        if is_dataclass(value) and not isinstance(value, type):
            return sum(
                _BoundedCoordinateCache._nbytes(
                    getattr(value, field.name), visited
                )
                for field in fields(value)
            )
        if isinstance(value, dict):
            return sum(
                _BoundedCoordinateCache._nbytes(item, visited)
                for item in value.values()
            )
        if isinstance(value, (tuple, list)):
            return sum(
                _BoundedCoordinateCache._nbytes(item, visited) for item in value
            )
        return 0

    def get(self, coordinate: float, builder: Callable[[], Any]) -> Any:
        key = float(coordinate)
        if self.enabled and key in self._values:
            self.hits += 1
            value, size = self._values.pop(key)
            self._values[key] = (value, size)
            return value
        self.misses += 1
        value = builder()
        if not self.enabled:
            return value
        size = self._nbytes(value)
        if size > self.maximum_bytes:
            self.oversize_skips += 1
            return value
        while self._values and self.bytes + size > self.maximum_bytes:
            _, (_, removed_size) = self._values.popitem(last=False)
            self.bytes -= removed_size
            self.evictions += 1
        self._values[key] = (value, size)
        self.bytes += size
        return value

    def diagnostics(self) -> dict[str, int | bool]:
        return {
            "known_stage_cache_enabled": self.enabled,
            "known_stage_cache_hits": self.hits,
            "known_stage_cache_misses": self.misses,
            "known_stage_cache_entries": len(self._values),
            "known_stage_cache_bytes": self.bytes,
            "known_stage_cache_maximum_bytes": self.maximum_bytes,
            "known_stage_cache_evictions": self.evictions,
            "known_stage_cache_oversize_skips": self.oversize_skips,
        }


@dataclass(frozen=True)
class TauConstructionResult:
    """A constructed global field with certified element diagnostics."""

    values: Array
    diagnostics: list[dict[str, Any]]
    defect_maps: dict[str, Array]
    auxiliary: dict[str, Array] | None = None


class TauConstructionFailure(FloatingPointError):
    """Rejected tau construction carrying JSON-safe localization evidence."""

    def __init__(
        self,
        field: str,
        diagnostics: list[dict[str, Any]],
        defect_maps: dict[str, Array],
    ) -> None:
        rejected = [item for item in diagnostics if not item.get("accepted", False)]

        def failed_gates(item: dict[str, Any]) -> list[str]:
            result: list[str] = []
            for name, value in item.items():
                if name == "accepted" or not isinstance(value, bool):
                    continue
                if (
                    name in {"converged", "resolved"}
                    or name.endswith("_accepted")
                    or name.endswith("_resolved")
                    or name.endswith("_consistent")
                ) and not value:
                    result.append(name)
            return result

        defect_names = (
            "collocation_defect",
            "overgrid_defect",
            "physical_overgrid_defect",
            "trace_overgrid_defect",
            "trace_retained_overgrid_defect",
            "trace_strong_angular_defect",
            "trace_block_condition",
            "source_cross_equation_overgrid_defect",
            "fresh_source_interface_jump",
            "rhs_interface_jump",
            "primitive_cone_violation",
        )

        def rejection_rank(item: dict[str, Any]) -> tuple[int, float]:
            defects = []
            for name in defect_names:
                value = item.get(name)
                if isinstance(value, (float, int)):
                    defects.append(float(value))
            return len(failed_gates(item)), max(defects, default=0.0)

        worst = max(
            rejected,
            key=rejection_rank,
            default=None,
        )
        reason = "no accepted tau elements"
        if worst is not None:
            gates = failed_gates(worst)
            gate_text = ", ".join(gates) if gates else "unspecified acceptance gate"
            reason = (
                f"element {worst['element']} on tau=[{worst['tau_left']}, "
                f"{worst['tau_right']}] was rejected; failed gates: {gate_text}"
            )
        super().__init__(f"{field} tau-SDC construction failed: {reason}")
        self.field = field
        self.diagnostics = diagnostics
        self.defect_maps = defect_maps


def _check_coordinates(
    state: FirstOrderState, scalar_coordinates: CharacteristicLGLMesh
) -> None:
    if state.q.shape[1:3] != (len(scalar_coordinates.u), len(scalar_coordinates.v)):
        raise ValueError("the characteristic tau/s grid does not match the state")


def _element_metadata(
    field: str, element: int, segment: LGLSegment
) -> dict[str, Any]:
    return {
        "field": field,
        "element": element + 1,
        "degree": segment.degree,
        "tau_left": float(segment.left),
        "tau_right": float(segment.right),
        "u_left": float(-math.exp(-segment.left)),
        "u_right": float(-math.exp(-segment.right)),
    }


def _du_dtau(tau: float) -> float:
    """Return ``du/dtau=-u`` for ``u=-exp(-tau)``."""

    return math.exp(-float(tau))


def _stage_geometry(
    grid: PointSphereGrid,
    metric: Array,
    omega: Array,
    zeta_up: Array,
    weighted_chib: Array,
    inverse: Array | None = None,
) -> dict[str, Array]:
    difference, inverse = connection_difference(grid, metric, inverse)
    weighted_tr_chib = tensor_trace(weighted_chib, inverse)
    weighted_hatchib = tensor_tracefree(weighted_chib, metric, inverse)
    grad_log_omega = scalar_gradient(grid, np.log(omega))
    zeta = np.einsum("n...ij,n...j->n...i", metric, zeta_up)
    eta = zeta + grad_log_omega
    etab = -zeta + grad_log_omega
    return {
        "difference": difference,
        "inverse": inverse,
        "weighted_tr_chib": weighted_tr_chib,
        "weighted_hatchib": weighted_hatchib,
        "hatchib": weighted_hatchib / omega[..., None, None],
        "eta": eta,
        "etab": etab,
        "eta_grad_hat": tracefree_symmetric_gradient(
            grid, eta, metric, difference, inverse
        ),
        "eta_square_hat": tracefree_square(eta, metric, inverse),
    }


def _lie_covariant_tensor_known_vector_derivative(
    grid: PointSphereGrid,
    vector: Array,
    vector_derivative: Array,
    tensor: Array,
) -> Array:
    """Evaluate a Lie derivative with a coordinate-known vector derivative.

    The half-shear equation revisits the same ``(tau, v)`` shift at every SDC
    correction.  Only the tensor derivative depends on the current unknown;
    caching the shift derivative removes one unchanged global sphere
    differentiation without changing the formula.
    """

    derivative_tensor = grid.reference_derivative(tensor, tensor_rank=2)
    result = np.einsum(
        "n...k,n...kij->n...ij", vector, derivative_tensor
    )
    result += np.einsum(
        "n...kj,n...ik->n...ij", tensor, vector_derivative
    )
    result += np.einsum(
        "n...ik,n...jk->n...ij", tensor, vector_derivative
    )
    return result


def _moving_tracefree_derivative(
    tensor: Array,
    raw_derivative: Array,
    metric: Array,
    inverse: Array,
    inverse_derivative: Array,
    angular: AngularGalerkin | None,
) -> Array:
    if angular is not None:
        return angular.project_g_tracefree_derivative(
            tensor,
            raw_derivative,
            inverse,
            inverse_derivative,
        )
    symmetric = 0.5 * (raw_derivative + np.swapaxes(raw_derivative, -1, -2))
    desired_trace = -np.einsum(
        "n...ij,n...ij->n...", inverse_derivative, tensor
    )
    raw_trace = tensor_trace(symmetric, inverse)
    # The tangent metric has dimension two, so tr_g(alpha*g)=2 alpha.
    correction = 0.5 * (desired_trace - raw_trace)
    return symmetric + correction[..., None, None] * metric


@dataclass(frozen=True)
class _WeakTraceModalLayout:
    """Retained tensor-mode partition for the moving weak trace DAE.

    The trace modes are algebraic variables.  Electric and magnetic modes are
    the independent evolution variables.  Positions are relative to the
    retained coefficient vector, while ``retained_indices`` address the full
    work-space coefficient vector owned by ``AngularGalerkin.sym2``.
    """

    retained_indices: Array
    trace_positions: Array
    free_positions: Array


@dataclass(frozen=True)
class _WeakTraceReduction:
    """Coordinate-local graph of trace coefficients over the free modes."""

    trace_from_free: Array
    condition_by_batch: Array


def _weak_trace_modal_layout(angular: AngularGalerkin) -> _WeakTraceModalLayout:
    retained_indices = angular.sym2.retained_indices
    kinds = np.asarray(angular.tensor_kind)[retained_indices]
    trace_positions = np.flatnonzero(kinds == "trace")
    free_positions = np.flatnonzero(
        np.logical_or(kinds == "electric", kinds == "magnetic")
    )
    unexpected = np.logical_not(
        np.logical_or(
            kinds == "trace",
            np.logical_or(kinds == "electric", kinds == "magnetic"),
        )
    )
    scalar_count = int(np.count_nonzero(angular.scalar.retained))
    if np.any(unexpected):
        raise ValueError("the retained tensor family has an unknown mode kind")
    if len(trace_positions) != scalar_count:
        raise ValueError(
            "the retained trace block must have one mode per retained scalar"
        )
    if not len(free_positions):
        raise ValueError(
            "the retained tensor family has no electric/magnetic free modes"
        )
    if len(trace_positions) + len(free_positions) != len(retained_indices):
        raise ValueError("the retained tensor-mode partition is incomplete")
    return _WeakTraceModalLayout(
        retained_indices=retained_indices,
        trace_positions=trace_positions,
        free_positions=free_positions,
    )


def _weak_trace_reduction(
    angular: AngularGalerkin,
    inverse: Array,
    layout: _WeakTraceModalLayout,
) -> _WeakTraceReduction:
    """Build ``c_trace=A(tau)c_free`` for every batch value.

    ``inverse`` has shape ``(sphere,batch,3,3)``.  The constraint is exactly
    the retained scalar analysis of ``tr_g S`` used by
    :meth:`AngularGalerkin.project_g_tracefree`.  Solving its square trace
    block eliminates the algebraic variables without a coordinate-dependent
    retraction in the SDC iteration.
    """

    inverse_values = np.asarray(inverse, dtype=float)
    if inverse_values.ndim != 4 or inverse_values.shape[0] != angular.grid.count:
        raise ValueError(
            "a weak-trace reduction requires (sphere,batch,3,3) inverse data"
        )
    retained_basis = angular.sym2.retained_basis
    pointwise_traces = np.einsum(
        "nvij,nkij->nvk", inverse_values, retained_basis
    )
    constraint = np.einsum(
        "an,nvk->vak", angular.scalar_retained_analysis, pointwise_traces
    )
    trace_block = constraint[..., layout.trace_positions]
    free_block = constraint[..., layout.free_positions]
    condition = np.asarray(np.linalg.cond(trace_block), dtype=float)
    if np.any(~np.isfinite(condition)):
        raise FloatingPointError("the moving weak-trace block is singular")
    try:
        trace_from_free = np.linalg.solve(trace_block, -free_block)
    except np.linalg.LinAlgError as error:
        raise FloatingPointError(
            "the moving weak-trace block cannot be solved"
        ) from error
    return _WeakTraceReduction(
        trace_from_free=np.asarray(trace_from_free),
        condition_by_batch=condition,
    )


def _weak_trace_reconstruct(
    angular: AngularGalerkin,
    layout: _WeakTraceModalLayout,
    reduction: _WeakTraceReduction,
    free_coefficients: Array,
) -> tuple[Array, Array]:
    """Reconstruct one exactly weak-trace-free retained tensor value."""

    free = np.asarray(free_coefficients, dtype=float)
    expected = (len(layout.free_positions), reduction.trace_from_free.shape[0])
    if free.shape != expected:
        raise ValueError(
            f"free half-shear coefficients have shape {free.shape}, "
            f"expected {expected}"
        )
    retained = np.zeros(
        (len(layout.retained_indices), free.shape[1]), dtype=free.dtype
    )
    retained[layout.free_positions] = free
    retained[layout.trace_positions] = np.einsum(
        "vtf,fv->tv", reduction.trace_from_free, free
    )
    return angular.sym2.synthesize_retained(retained), retained


def _weak_trace_constraint_profile(
    reduction: _WeakTraceReduction,
    layout: _WeakTraceModalLayout,
    retained_coefficients: Array,
    *,
    absolute_floor: float,
) -> Array:
    """Return one normalized retained-constraint defect per batch value."""

    retained = np.asarray(retained_coefficients, dtype=float)
    expected = (
        len(layout.retained_indices),
        reduction.trace_from_free.shape[0],
    )
    if retained.shape != expected:
        raise ValueError(
            f"retained coefficients have shape {retained.shape}, expected {expected}"
        )
    expected_trace = np.einsum(
        "vtf,fv->tv",
        reduction.trace_from_free,
        retained[layout.free_positions],
    )
    residual = retained[layout.trace_positions] - expected_trace
    numerator = np.linalg.norm(residual, axis=0)
    denominator = np.maximum(
        float(absolute_floor), np.linalg.norm(retained, axis=0)
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        profile = numerator / denominator
    return np.where(np.isfinite(profile), profile, np.inf)


def _per_v_scaled_profile(residual: Array, reference: Array) -> tuple[float, Array]:
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = np.abs(residual) / np.maximum(1.0, np.abs(reference))
    scaled = np.where(np.isfinite(scaled), scaled, np.inf)
    if scaled.ndim < 3:
        raise ValueError("a physical defect requires leading node, sphere, and v axes")
    reduction = (0, 1, *range(3, scaled.ndim))
    profile = np.max(scaled, axis=reduction)
    return float(np.max(profile)), np.asarray(profile)


def _per_v_scaled_difference(
    first: Array,
    second: Array,
    *,
    absolute_floor: float = 1.0,
) -> tuple[float, Array]:
    """Symmetric mixed absolute/relative difference retaining the v axis."""

    if not np.isfinite(absolute_floor) or absolute_floor <= 0.0:
        raise ValueError("the scaled-difference absolute floor must be positive")
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    if first_values.shape != second_values.shape or first_values.ndim < 2:
        raise ValueError("source traces must have matching sphere/v shapes")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        denominator = np.maximum(
            absolute_floor,
            np.maximum(np.abs(first_values), np.abs(second_values)),
        )
        scaled = np.abs(first_values - second_values) / denominator
    scaled = np.where(np.isfinite(scaled), scaled, np.inf)
    # Arrays are either (sphere,v,...) interface traces or
    # (overgrid,sphere,v,...) independent source samples.
    v_axis = 1 if scaled.ndim == 2 else 2
    reduction = tuple(axis for axis in range(scaled.ndim) if axis != v_axis)
    profile = np.max(scaled, axis=reduction)
    return float(np.max(profile)), np.asarray(profile)


def _relative_trace_profile(
    trace: Array,
    norm: Array,
    *,
    absolute_floor: float,
) -> tuple[float, Array, float]:
    """Return a scale-aware relative trace defect and its effective floor.

    ``absolute_floor`` supplies the small-field absolute part of the mixed
    norm.  A roundoff floor proportional to the element's tensor scale avoids
    amplifying trace noise at isolated zeros of an otherwise large tensor.
    Unlike the former ``max(1,|C|)`` denominator, this is invariant under
    rescaling whenever the tensor is resolved above the stated floor.
    """

    if not np.isfinite(absolute_floor) or absolute_floor <= 0.0:
        raise ValueError("the half-shear trace absolute floor must be positive")
    trace_values = np.asarray(trace, dtype=float)
    norm_values = np.asarray(norm, dtype=float)
    if trace_values.shape != norm_values.shape or trace_values.ndim != 3:
        raise ValueError("trace and norm samples must have (node,sphere,v) shape")
    tensor_scale = float(np.max(np.abs(norm_values)))
    effective_floor = max(
        float(absolute_floor),
        64.0 * np.finfo(float).eps * tensor_scale,
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        scaled = np.abs(trace_values) / np.maximum(
            effective_floor, np.abs(norm_values)
        )
    scaled = np.where(np.isfinite(scaled), scaled, np.inf)
    profile = np.max(scaled, axis=(0, 1))
    return float(np.max(profile)), np.asarray(profile), effective_floor


def _interface_profile_map(
    scalar_coordinates: CharacteristicLGLMesh,
    profiles: list[Array],
) -> Array:
    """Place right-minus-left profiles only at shared tau interfaces."""

    expected = max(len(scalar_coordinates.tau.segments) - 1, 0)
    if len(profiles) != expected:
        raise ValueError("one profile is required for every internal tau interface")
    result = np.zeros((len(scalar_coordinates.u), len(scalar_coordinates.v)), dtype=float)
    for element, profile in enumerate(profiles, start=1):
        values = np.asarray(profile, dtype=float)
        if values.shape != (len(scalar_coordinates.v),):
            raise ValueError("an interface profile has the wrong v shape")
        result[int(scalar_coordinates.tau.indices[element][0])] = values
    return result


def _primitive_cone_audit(
    grid: PointSphereGrid,
    metric: Array,
    omega: Array,
    scalar_coordinates: CharacteristicLGLMesh,
    *,
    overgrid_extra_degree: int,
    minimum_metric_eigenvalue: float,
    minimum_lapse: float,
) -> tuple[list[dict[str, Any]], dict[str, Array]]:
    """Audit known metric/lapse primitives on independent overgrid nodes."""

    if (
        overgrid_extra_degree < 1
        or not np.isfinite(minimum_metric_eigenvalue)
        or not np.isfinite(minimum_lapse)
        or minimum_metric_eigenvalue <= 0.0
        or minimum_lapse <= 0.0
    ):
        raise ValueError("invalid primitive cone audit parameters")
    metric_profiles: list[Array] = []
    lapse_profiles: list[Array] = []
    violation_profiles: list[Array] = []
    diagnostics: list[dict[str, Any]] = []
    for element, (segment, indices) in enumerate(
        zip(scalar_coordinates.tau.segments, scalar_coordinates.tau.indices, strict=True)
    ):
        over = LGLSegment.create(
            segment.left, segment.right, segment.degree + overgrid_extra_degree
        )
        metric_over = TauElementEvaluator(
            segment, indices, metric, axis=1
        ).values_at(over.nodes)
        lapse_over = TauElementEvaluator(
            segment, indices, omega, axis=1
        ).values_at(over.nodes)

        metric_finite = np.all(np.isfinite(metric_over), axis=(-2, -1))
        safe_metric = np.where(np.isfinite(metric_over), metric_over, 0.0)
        local_metric = np.einsum(
            "nia,knvij,njb->knvab",
            grid.frames,
            safe_metric,
            grid.frames,
        )
        local_metric = 0.5 * (
            local_metric + np.swapaxes(local_metric, -1, -2)
        )
        minimum_at_sample = np.linalg.eigvalsh(local_metric)[..., 0]
        minimum_at_sample = np.where(
            metric_finite, minimum_at_sample, -np.inf
        )
        lapse_at_sample = np.where(np.isfinite(lapse_over), lapse_over, -np.inf)
        metric_profile = np.min(minimum_at_sample, axis=(0, 1))
        lapse_profile = np.min(lapse_at_sample, axis=(0, 1))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            metric_violation = np.maximum(
                0.0,
                (minimum_metric_eigenvalue - metric_profile)
                / minimum_metric_eigenvalue,
            )
            lapse_violation = np.maximum(
                0.0, (minimum_lapse - lapse_profile) / minimum_lapse
            )
        violation = np.maximum(metric_violation, lapse_violation)
        violation = np.where(np.isfinite(violation), violation, np.inf)
        resolved = bool(
            np.all(metric_profile > minimum_metric_eigenvalue)
            and np.all(lapse_profile > minimum_lapse)
        )
        metric_profiles.append(np.asarray(metric_profile))
        lapse_profiles.append(np.asarray(lapse_profile))
        violation_profiles.append(np.asarray(violation))
        diagnostics.append(
            {
                **_element_metadata("primitive_cone", element, segment),
                "primitive_metric_minimum_eigenvalue": float(
                    np.min(metric_profile)
                ),
                "primitive_lapse_minimum": float(np.min(lapse_profile)),
                "primitive_cone_violation": float(np.max(violation)),
                "primitive_cone_resolved": resolved,
            }
        )
    maps = {
        "primitive_metric_minimum_eigenvalue": -expand_element_profiles(
            scalar_coordinates.tau, [-profile for profile in metric_profiles]
        ),
        "primitive_lapse_minimum": -expand_element_profiles(
            scalar_coordinates.tau, [-profile for profile in lapse_profiles]
        ),
        "primitive_cone_violation": expand_element_profiles(
            scalar_coordinates.tau, violation_profiles
        ),
    }
    return diagnostics, maps


def _reject_invalid_primitive_cone(
    field: str,
    cone_diagnostics: list[dict[str, Any]],
    cone_maps: dict[str, Array],
) -> None:
    """Reject an undefined off-grid geometry with localized evidence."""

    if all(item["primitive_cone_resolved"] for item in cone_diagnostics):
        return
    diagnostics: list[dict[str, Any]] = []
    for item in cone_diagnostics:
        diagnostic = dict(item)
        diagnostic["accepted"] = bool(item["primitive_cone_resolved"])
        converted = json_safe_diagnostic(diagnostic)
        if not isinstance(converted, dict):
            raise AssertionError("primitive cone diagnostic is not a mapping")
        diagnostics.append(converted)
    raise TauConstructionFailure(field, diagnostics, cone_maps)


def _raise_rejected(
    field: str,
    diagnostics: list[dict[str, Any]],
    defect_maps: dict[str, Array],
    require_acceptance: bool,
) -> None:
    if require_acceptance and any(
        not item.get("accepted", False) for item in diagnostics
    ):
        raise TauConstructionFailure(field, diagnostics, defect_maps)


def solve_half_shear_tau_sdc(
    grid: PointSphereGrid,
    state: FirstOrderState,
    boundary: dict[str, Array | float],
    scalar_coordinates: CharacteristicLGLMesh,
    *,
    angular: AngularGalerkin | None = None,
    projection_tails: ProjectionTails | None = None,
    tolerance: float = 1.0e-10,
    overgrid_tolerance: float = 1.0e-7,
    trace_tolerance: float = 1.0e-9,
    trace_absolute_floor: float = 1.0e-14,
    trace_overgrid_tolerance: float | None = None,
    strong_trace_tolerance: float = 1.0e-5,
    trace_block_condition_limit: float = 1.0e10,
    maximum_corrections: int = 12,
    overgrid_extra_degree: int = 3,
    minimum_primitive_metric_eigenvalue: float = 1.0e-14,
    minimum_primitive_lapse: float = 1.0e-14,
    cache_known_stages: bool = True,
    known_cache_maximum_bytes: int = 256 * 1024 * 1024,
    require_acceptance: bool = True,
) -> TauConstructionResult:
    """Construct the half outgoing shear with the moving trace-free DAE.

    In angular-Galerkin mode only the retained electric/magnetic coefficients
    are evolved.  At every RHS coordinate the retained trace coefficients are
    recovered from the square moving weak-trace block.  The SDC unknowns
    therefore live in fixed free scalar_coordinates and need no retraction; their raw
    collocation and independent overgrid defects have their usual meanings.
    """

    _check_coordinates(state, scalar_coordinates)
    if trace_overgrid_tolerance is None:
        trace_overgrid_tolerance = overgrid_tolerance
    if (
        trace_tolerance <= 0.0
        or trace_absolute_floor <= 0.0
        or trace_overgrid_tolerance <= 0.0
        or strong_trace_tolerance <= 0.0
        or not np.isfinite(trace_block_condition_limit)
        or trace_block_condition_limit <= 1.0
    ):
        raise ValueError("the half-shear trace tolerance must be positive")
    cone_diagnostics, cone_maps = _primitive_cone_audit(
        grid,
        state.metric,
        state.omega,
        scalar_coordinates,
        overgrid_extra_degree=overgrid_extra_degree,
        minimum_metric_eigenvalue=minimum_primitive_metric_eigenvalue,
        minimum_lapse=minimum_primitive_lapse,
    )
    _reject_invalid_primitive_cone(
        "half_shear", cone_diagnostics, cone_maps
    )
    result = np.zeros_like(state.metric)
    boundary_shear = np.asarray(boundary["shear"])
    if "inverse" in boundary:
        boundary_inverse = np.asarray(boundary["inverse"])
    else:
        boundary_inverse = tangent_inverse(grid, np.asarray(boundary["metric"]))
    transfer = np.matmul(
        np.matmul(boundary_shear, boundary_inverse), state.metric[:, 0]
    )
    transfer = 0.5 * (transfer + np.swapaxes(transfer, -1, -2))
    initial_inverse = tangent_inverse(grid, state.metric[:, 0])
    result[:, 0] = _project_tracefree_sym2(
        angular,
        transfer,
        state.metric[:, 0],
        initial_inverse,
        projection_tails,
        "half_shear_tau_boundary",
    )

    diagnostics: list[dict[str, Any]] = []
    collocation_profiles: list[Array] = []
    evolution_profiles: list[Array] = []
    established_trace_profiles: list[Array] = []
    condition_profiles: list[Array] = []
    retained_overgrid_traces: list[Array] = []
    overgrid_tensor_norms: list[Array] = []
    strong_angular_traces: list[Array] = []
    strong_angular_tensor_norms: list[Array] = []
    free_result: Array | None = None
    layout = None if angular is None else _weak_trace_modal_layout(angular)
    if layout is not None:
        free_result = np.zeros(
            (
                len(layout.free_positions),
                len(scalar_coordinates.u),
                len(scalar_coordinates.v),
            ),
            dtype=float,
        )
    primitives = {
        "metric": state.metric,
        "omega": state.omega,
        "zeta_up": state.zeta_up,
        "q": state.q,
        "weighted_chib": state.weighted_chib,
        "shift": state.shift,
    }
    for element, (segment, indices) in enumerate(
        zip(scalar_coordinates.tau.segments, scalar_coordinates.tau.indices, strict=True)
    ):
        evaluators = {
            name: TauElementEvaluator(segment, indices, value, axis=1)
            for name, value in primitives.items()
        }
        known_cache = _BoundedCoordinateCache(
            enabled=cache_known_stages,
            maximum_bytes=known_cache_maximum_bytes,
        )

        def known_stage(tau: float) -> dict[str, Any]:
            def build() -> dict[str, Any]:
                metric = evaluators["metric"].value(tau)
                metric_tau = evaluators["metric"].derivative(tau)
                inverse = tangent_inverse(grid, metric)
                inverse_tau = -np.matmul(
                    np.matmul(inverse, metric_tau), inverse
                )
                omega = evaluators["omega"].value(tau)
                zeta_up = evaluators["zeta_up"].value(tau)
                weighted_chib = evaluators["weighted_chib"].value(tau)
                q = evaluators["q"].value(tau)
                shift = evaluators["shift"].value(tau)
                geometry = _stage_geometry(
                    grid,
                    metric,
                    omega,
                    zeta_up,
                    weighted_chib,
                    inverse,
                )
                tr_chi = omega * q
                source = omega[..., None, None] ** 2 * (
                    geometry["eta_grad_hat"]
                    + geometry["eta_square_hat"]
                    - 0.5
                    * tr_chi[..., None, None]
                    * geometry["hatchib"]
                )
                stage: dict[str, Any] = {
                    "metric": metric,
                    "inverse": inverse,
                    "inverse_tau": inverse_tau,
                    "half_source": source,
                    "half_mixed": np.matmul(
                        geometry["weighted_hatchib"], inverse
                    ),
                    "half_weighted_tr_chib": geometry[
                        "weighted_tr_chib"
                    ],
                    "shift": shift,
                    "shift_derivative": grid.reference_derivative(
                        shift, tensor_rank=1
                    ),
                }
                if angular is not None:
                    if layout is None:
                        raise AssertionError("the weak-trace layout is missing")
                    stage["trace_reduction"] = _weak_trace_reduction(
                        angular, inverse, layout
                    )
                return stage

            return known_cache.get(tau, build)

        def retract(tau: float, tensor: Array) -> Array:
            stage = known_stage(tau)
            metric = stage["metric"]
            inverse = stage["inverse"]
            if angular is None:
                return tensor_tracefree(tensor, metric, inverse)
            return angular.project_g_tracefree(tensor, inverse)

        def physical_raw_rhs(
            tau: float,
            stage_tensor: Array,
            stage: dict[str, Any] | None = None,
            *,
            record_tail: bool = True,
        ) -> Array:
            current = known_stage(tau) if stage is None else stage
            shift = current["shift"]
            mixed = current["half_mixed"]
            physical_rhs = (
                0.5
                * current["half_weighted_tr_chib"][..., None, None]
                * stage_tensor
                + np.matmul(mixed, stage_tensor)
                + np.matmul(stage_tensor, np.swapaxes(mixed, -1, -2))
                + current["half_source"]
                - _lie_covariant_tensor_known_vector_derivative(
                    grid,
                    shift,
                    current["shift_derivative"],
                    stage_tensor,
                )
            )
            tau_rhs = _du_dtau(tau) * physical_rhs
            if angular is not None and record_tail:
                _record_tail(
                    angular.sym2,
                    tau_rhs,
                    projection_tails,
                    "half_shear_tau_rhs",
                )
            return tau_rhs

        def physical_tangent_rhs(tau: float, stage_tensor: Array) -> Array:
            stage = known_stage(tau)
            tau_rhs = physical_raw_rhs(tau, stage_tensor, stage)
            return _moving_tracefree_derivative(
                stage_tensor,
                tau_rhs,
                stage["metric"],
                stage["inverse"],
                stage["inverse_tau"],
                angular,
            )

        if angular is None:
            initial = composite_element_initial(result, indices, axis=1)

            def rhs(tau: float, tensor: Array) -> Array:
                return physical_tangent_rhs(tau, retract(tau, tensor))

            solved = solve_tau_sdc(
                segment,
                initial,
                rhs,
                tolerance=tolerance,
                overgrid_tolerance=overgrid_tolerance,
                maximum_corrections=maximum_corrections,
                retract=retract,
                overgrid_extra_degree=overgrid_extra_degree,
                batch_axes=(1,),
            )
            local_tensors = solved.values
            collocation_profile = np.full(
                len(scalar_coordinates.v), solved.collocation_defect
            )
        else:
            if layout is None or free_result is None:
                raise AssertionError("the weak-trace free-coordinate state is missing")

            def reconstruct(
                tau: float, free_coefficients: Array
            ) -> tuple[Array, Array]:
                reduction = known_stage(tau)["trace_reduction"]
                return _weak_trace_reconstruct(
                    angular, layout, reduction, free_coefficients
                )

            if element == 0:
                boundary_coefficients = angular.sym2.analyze_retained(
                    result[:, 0]
                )
                free_result[:, indices[0]] = boundary_coefficients[
                    layout.free_positions
                ]
            initial_free = composite_element_initial(
                free_result, indices, axis=1
            )

            def rhs_free(tau: float, free_coefficients: Array) -> Array:
                stage = known_stage(tau)
                stage_tensor, retained = _weak_trace_reconstruct(
                    angular,
                    layout,
                    stage["trace_reduction"],
                    free_coefficients,
                )
                tau_rhs = physical_raw_rhs(
                    tau,
                    stage_tensor,
                    stage,
                    record_tail=False,
                )
                derivative_coefficients = angular.sym2.analyze(tau_rhs)
                _record_tail_coefficients(
                    angular.sym2,
                    derivative_coefficients,
                    projection_tails,
                    "half_shear_tau_rhs",
                )
                retained_derivative = derivative_coefficients[
                    layout.retained_indices
                ]
                constrained = (
                    angular.g_tracefree_derivative_retained_coefficients(
                        retained,
                        retained_derivative,
                        stage["inverse"],
                        stage["inverse_tau"],
                    )
                )
                return constrained[layout.free_positions]

            solved = solve_tau_sdc(
                segment,
                initial_free,
                rhs_free,
                tolerance=tolerance,
                overgrid_tolerance=overgrid_tolerance,
                maximum_corrections=maximum_corrections,
                overgrid_extra_degree=overgrid_extra_degree,
                batch_axes=(1,),
            )
            assign_composite_element(free_result, indices, solved.values)
            reconstructed_nodes = [
                reconstruct(float(tau), free)[0]
                for tau, free in zip(
                    segment.nodes, solved.values, strict=True
                )
            ]
            local_tensors = np.stack(reconstructed_nodes)
            collocation_profile = np.asarray(
                solved.collocation_defect_by_batch
            ).copy()
        assign_composite_element(result, indices, local_tensors)

        over = LGLSegment.create(
            segment.left, segment.right, segment.degree + overgrid_extra_degree
        )
        interpolation = segment.interpolation_matrix(over.nodes)
        over_tensor = np.tensordot(interpolation, local_tensors, axes=(1, 0))
        over_trace = np.empty(over_tensor.shape[:3], dtype=float)
        over_norm = np.empty(over_tensor.shape[:3], dtype=float)
        strong_trace = np.empty_like(over_trace)
        strong_norm = np.empty_like(over_norm)
        condition_samples: list[Array] = []
        for node, (tau, tensor) in enumerate(
            zip(over.nodes, over_tensor, strict=True)
        ):
            stage = known_stage(float(tau))
            inverse = stage["inverse"]
            over_trace[node] = tensor_trace(tensor, inverse)
            over_norm[node] = np.sqrt(
                np.maximum(tensor_norm_sq(tensor, inverse), 0.0)
            )
            if angular is None:
                strong_tensor = retract(float(tau), tensor)
            else:
                strong_tensor = angular.project_g_tracefree(tensor, inverse)
                condition_samples.append(
                    stage["trace_reduction"].condition_by_batch
                )
            strong_trace[node] = tensor_trace(strong_tensor, inverse)
            strong_norm[node] = np.sqrt(
                np.maximum(tensor_norm_sq(strong_tensor, inverse), 0.0)
            )
        trace_defect, trace_profile, effective_trace_floor = (
            _relative_trace_profile(
                over_trace,
                over_norm,
                absolute_floor=trace_absolute_floor,
            )
        )
        if condition_samples:
            condition_profile = np.max(np.stack(condition_samples), axis=0)
        else:
            condition_profile = np.ones(len(scalar_coordinates.v), dtype=float)
        condition_resolved = bool(
            np.max(condition_profile) <= trace_block_condition_limit
        )
        cone = cone_diagnostics[element]
        combined = bool(
            solved.accepted
            and (angular is not None or trace_defect <= trace_tolerance)
            and condition_resolved
            and cone["primitive_cone_resolved"]
        )
        diagnostic = tau_sdc_result_diagnostic(
            solved, **_element_metadata("half_shear", element, segment)
        )
        trace_metadata = json_safe_diagnostic(
            {
                "evolution_accepted": bool(solved.accepted),
                "trace_overgrid_defect": trace_defect,
                "trace_overgrid_defect_by_v": trace_profile,
                "trace_absolute_floor": float(trace_absolute_floor),
                "trace_effective_absolute_floor": effective_trace_floor,
                "trace_resolved": bool(trace_defect <= trace_tolerance),
                "trace_block_condition": float(np.max(condition_profile)),
                "trace_block_condition_by_v": condition_profile,
                "trace_block_condition_limit": float(
                    trace_block_condition_limit
                ),
                "trace_block_resolved": condition_resolved,
                "free_coordinate_count": (
                    None if layout is None else len(layout.free_positions)
                ),
                "trace_coordinate_count": (
                    None if layout is None else len(layout.trace_positions)
                ),
                "coordinate_semantics": (
                    "ambient projected tensor"
                    if layout is None
                    else (
                        "retained electric/magnetic free coefficients with "
                        "algebraically eliminated moving trace coefficients"
                    )
                ),
                "primitive_metric_minimum_eigenvalue": cone[
                    "primitive_metric_minimum_eigenvalue"
                ],
                "primitive_lapse_minimum": cone["primitive_lapse_minimum"],
                "primitive_cone_violation": cone[
                    "primitive_cone_violation"
                ],
                "primitive_cone_resolved": cone[
                    "primitive_cone_resolved"
                ],
                **known_cache.diagnostics(),
                "accepted": combined,
            }
        )
        if not isinstance(trace_metadata, dict):
            raise AssertionError("trace diagnostic metadata is not a mapping")
        diagnostic.update(trace_metadata)
        diagnostics.append(diagnostic)
        collocation_profiles.append(collocation_profile)
        evolution_profiles.append(solved.overgrid_defect_by_batch.copy())
        established_trace_profiles.append(trace_profile)
        condition_profiles.append(condition_profile)
        retained_overgrid_traces.append(
            np.stack(
                [angular.project_scalar(trace) for trace in over_trace]
            )
            if angular is not None
            else over_trace.copy()
        )
        overgrid_tensor_norms.append(over_norm)
        strong_angular_traces.append(strong_trace)
        strong_angular_tensor_norms.append(strong_norm)

    maps = {
        "collocation": expand_element_profiles(
            scalar_coordinates.tau, collocation_profiles
        ),
        "overgrid": expand_element_profiles(
            scalar_coordinates.tau, evolution_profiles
        ),
        "trace_block_condition": expand_element_profiles(
            scalar_coordinates.tau, condition_profiles
        ),
        **cone_maps,
    }
    auxiliary: dict[str, Array] | None = None
    if angular is None:
        maps["trace_overgrid"] = expand_element_profiles(
            scalar_coordinates.tau, established_trace_profiles
        )
    else:
        inverse_nodes = tangent_inverse(grid, state.metric)
        node_trace = tensor_trace(result, inverse_nodes)
        retained_node_trace = angular.project_scalar(node_trace)
        node_norm = np.sqrt(
            np.maximum(tensor_norm_sq(result, inverse_nodes), 0.0)
        )
        certificate = half_shear_trace_certificate(
            scalar_coordinates.tau,
            np.moveaxis(retained_node_trace, 1, 0),
            np.moveaxis(node_norm, 1, 0),
            retained_overgrid_traces,
            overgrid_tensor_norms,
            strong_angular_traces,
            strong_angular_tensor_norms,
            absolute_floor=trace_absolute_floor,
            retained_node_tolerance=trace_tolerance,
            retained_overgrid_tolerance=trace_overgrid_tolerance,
            strong_angular_tolerance=strong_trace_tolerance,
        )
        maps.update(certificate["maps"])
        trace_metadata = certificate["diagnostics"]
        # The retained node/overgrid conditions are algebraic constraints of
        # the semidiscrete DAE and therefore belong to solver acceptance.
        # ``trace_strong_angular`` instead measures angular truncation after
        # that DAE has been solved.  Keep it as an independent reliability
        # certificate: an under-resolved angular band must be mapped and
        # refined, but must not abort the whole Q1 march before such a map can
        # be produced.
        trace_constraint_ok = bool(certificate["constraint_resolved"])
        for diagnostic in diagnostics:
            diagnostic.update(trace_metadata)
            # Compatibility aliases now carry the retained weak constraint,
            # never the topology-unstable pointwise ratio.
            diagnostic["trace_overgrid_defect"] = trace_metadata[
                "trace_retained_overgrid_defect"
            ]
            diagnostic["trace_overgrid_defect_by_v"] = trace_metadata[
                "trace_retained_overgrid_defect_by_v"
            ]
            diagnostic["trace_resolved"] = trace_constraint_ok
            diagnostic["angular_trace_certified"] = bool(
                certificate["strong_angular_resolved"]
            )
            diagnostic["accepted"] = bool(
                diagnostic["accepted"] and trace_constraint_ok
            )
        if free_result is None:
            raise AssertionError("the free-coordinate result is missing")
        auxiliary = {"free_coefficients": free_result}
    _raise_rejected("half_shear", diagnostics, maps, require_acceptance)
    return TauConstructionResult(result, diagnostics, maps, auxiliary)


def solve_log_omega_tau_sdc(
    grid: PointSphereGrid,
    state: FirstOrderState,
    weighted_omegab_half: Array,
    scalar_coordinates: CharacteristicLGLMesh,
    initial_log_omega: Array,
    *,
    angular: AngularGalerkin | None = None,
    projection_tails: ProjectionTails | None = None,
    tolerance: float = 1.0e-10,
    overgrid_tolerance: float = 1.0e-7,
    maximum_corrections: int = 12,
    overgrid_extra_degree: int = 3,
    require_acceptance: bool = True,
) -> TauConstructionResult:
    """Solve ``(partial_u+b^i) log(Omega)=-2 B_half`` in tau."""

    _check_coordinates(state, scalar_coordinates)
    source_values = np.asarray(weighted_omegab_half)
    if source_values.shape != state.omega.shape:
        raise ValueError("weighted_omegab_half has the wrong state shape")
    result = np.zeros_like(state.omega)
    initial = np.broadcast_to(np.asarray(initial_log_omega), result[:, 0].shape)
    result[:, 0] = _project_scalar(
        angular,
        initial,
        projection_tails,
        "log_omega_tau_boundary",
    )
    diagnostics: list[dict[str, Any]] = []
    profiles: list[Array] = []
    for element, (segment, indices) in enumerate(
        zip(scalar_coordinates.tau.segments, scalar_coordinates.tau.indices, strict=True)
    ):
        shift = TauElementEvaluator(segment, indices, state.shift, axis=1)
        source = TauElementEvaluator(segment, indices, source_values, axis=1)

        def rhs(tau: float, value: Array) -> Array:
            physical_rhs = -np.einsum(
                "n...i,n...i->n...",
                shift.value(tau),
                scalar_gradient(grid, value),
            ) - 2.0 * source.value(tau)
            complete = _du_dtau(tau) * physical_rhs
            return _project_scalar(
                angular,
                complete,
                projection_tails,
                "log_omega_tau_rhs",
            )

        solved = solve_tau_sdc(
            segment,
            composite_element_initial(result, indices, axis=1),
            rhs,
            tolerance=tolerance,
            overgrid_tolerance=overgrid_tolerance,
            maximum_corrections=maximum_corrections,
            overgrid_extra_degree=overgrid_extra_degree,
            batch_axes=(1,),
        )
        assign_composite_element(result, indices, solved.values)
        diagnostics.append(
            tau_sdc_result_diagnostic(
                solved, **_element_metadata("log_omega", element, segment)
            )
        )
        profiles.append(solved.overgrid_defect_by_batch.copy())

    with np.errstate(over="ignore", invalid="ignore"):
        lapse = np.exp(result)
    if not np.all(np.isfinite(lapse)) or float(np.min(lapse)) <= 0.0:
        raise FloatingPointError("tau log(Omega) solve left the positive cone")
    maps = {"overgrid": expand_element_profiles(scalar_coordinates.tau, profiles)}
    _raise_rejected("log_omega", diagnostics, maps, require_acceptance)
    return TauConstructionResult(result, diagnostics, maps, {"omega": lapse})


def _fresh_omegab_v_source(
    grid: PointSphereGrid,
    tau: float,
    evaluators: dict[str, TauElementEvaluator],
    angular: AngularGalerkin | None,
    projection_tails: ProjectionTails | None,
) -> Array:
    metric = evaluators["metric"].value(tau)
    omega = evaluators["omega"].value(tau)
    omega_tau = evaluators["omega"].derivative(tau)
    q = evaluators["q"].value(tau)
    q_tau = evaluators["q"].derivative(tau)
    zeta_up = evaluators["zeta_up"].value(tau)
    shift = evaluators["shift"].value(tau)
    weighted_chib = evaluators["weighted_chib"].value(tau)
    geometry = _stage_geometry(grid, metric, omega, zeta_up, weighted_chib)
    inverse = geometry["inverse"]
    half_raw = evaluators["half"].value(tau)
    if angular is None:
        half = tensor_tracefree(half_raw, metric, inverse)
    else:
        half = angular.project_g_tracefree(half_raw, inverse)
    weighted_tr_chi = omega**2 * q
    weighted_tr_chi_tau = 2.0 * omega * omega_tau * q + omega**2 * q_tau
    jacobian = _du_dtau(tau)
    d3_weighted_tr_chi = weighted_tr_chi_tau / jacobian
    d3_weighted_tr_chi += np.einsum(
        "n...i,n...i->n...",
        shift,
        scalar_gradient(grid, weighted_tr_chi),
    )
    eta_etab = np.einsum(
        "n...i,n...ij,n...j->n...",
        geometry["eta"],
        inverse,
        geometry["etab"],
    )
    eta_up = np.einsum(
        "n...ij,n...j->n...i", inverse, geometry["eta"]
    )
    div_eta = vector_divergence(grid, eta_up, geometry["difference"])
    shear_dot = np.einsum(
        "n...ik,n...jl,n...ij,n...kl->n...",
        inverse,
        inverse,
        half,
        geometry["weighted_hatchib"],
    )
    source = 0.25 * (
        shear_dot
        + 0.5 * weighted_tr_chi * geometry["weighted_tr_chib"]
        - 4.0 * omega**2 * eta_etab
        + d3_weighted_tr_chi
        - 2.0 * omega**2 * div_eta
    )
    return _project_scalar(
        angular,
        source,
        projection_tails,
        "weighted_omegab_tau_fresh_v_source",
    )


def solve_weighted_omega_tau_sdc(
    grid: PointSphereGrid,
    state: FirstOrderState,
    half_shear: Array,
    log_omega: Array,
    scalar_coordinates: CharacteristicLGLMesh,
    initial_value: Array,
    *,
    angular: AngularGalerkin | None = None,
    projection_tails: ProjectionTails | None = None,
    tolerance: float = 1.0e-10,
    overgrid_tolerance: float = 1.0e-7,
    reference_v_source: Array | None = None,
    source_cross_equation_tolerance: float | None = None,
    interface_tolerance: float | None = None,
    source_scale_floor: float = 1.0,
    maximum_corrections: int = 12,
    overgrid_extra_degree: int = 3,
    minimum_primitive_metric_eigenvalue: float = 1.0e-14,
    minimum_primitive_lapse: float = 1.0e-14,
    cache_known_stages: bool = True,
    known_cache_maximum_bytes: int = 256 * 1024 * 1024,
    require_acceptance: bool = True,
) -> TauConstructionResult:
    """Construct ``Omega*omega`` with a fresh primitive ``partial_v B`` RHS.

    A composite tau mesh has two legitimate derivative traces at every
    internal interface.  The fresh ``B_v`` reconstruction therefore has two
    traces as well.  They are retained separately in ``auxiliary``; no average
    is silently substituted into construction diagnostics.  When the caller
    supplies ``reference_v_source`` (the source used to construct ``B`` in
    the v march), independent overgrid closure against that second equation is
    also an acceptance gate.
    """

    _check_coordinates(state, scalar_coordinates)
    if np.asarray(half_shear).shape != state.metric.shape:
        raise ValueError("half_shear has the wrong state shape")
    if np.asarray(log_omega).shape != state.omega.shape:
        raise ValueError("log_omega has the wrong state shape")
    reference_values: Array | None = None
    if reference_v_source is not None:
        reference_values = np.asarray(reference_v_source)
        if reference_values.shape != state.omega.shape:
            raise ValueError("reference_v_source has the wrong state shape")
    if source_cross_equation_tolerance is None:
        source_cross_equation_tolerance = overgrid_tolerance
    if interface_tolerance is None:
        interface_tolerance = overgrid_tolerance
    if (
        np.isnan(source_cross_equation_tolerance)
        or source_cross_equation_tolerance <= 0.0
        or np.isnan(interface_tolerance)
        or interface_tolerance <= 0.0
        or not np.isfinite(source_scale_floor)
        or source_scale_floor <= 0.0
    ):
        raise ValueError("invalid weighted-omega source audit tolerance")

    cone_diagnostics, cone_maps = _primitive_cone_audit(
        grid,
        state.metric,
        state.omega,
        scalar_coordinates,
        overgrid_extra_degree=overgrid_extra_degree,
        minimum_metric_eigenvalue=minimum_primitive_metric_eigenvalue,
        minimum_lapse=minimum_primitive_lapse,
    )
    _reject_invalid_primitive_cone(
        "weighted_omega", cone_diagnostics, cone_maps
    )

    result = np.zeros_like(state.weighted_omega)
    boundary = np.broadcast_to(np.asarray(initial_value), result[:, 0].shape)
    result[:, 0] = _project_scalar(
        angular,
        boundary,
        projection_tails,
        "weighted_omega_tau_boundary",
    )
    diagnostics: list[dict[str, Any]] = []
    profiles: list[Array] = []
    source_closure_profiles: list[Array] = []
    source_interface_profiles: list[Array] = []
    rhs_interface_profiles: list[Array] = []
    fresh_source_left_trace = np.full_like(state.weighted_omega, np.nan)
    fresh_source_right_trace = np.full_like(state.weighted_omega, np.nan)
    previous_interface_source: Array | None = None
    previous_interface_rhs: Array | None = None
    primitive_values = {
        "metric": state.metric,
        "omega": state.omega,
        "q": state.q,
        "zeta_up": state.zeta_up,
        "shift": state.shift,
        "weighted_chib": state.weighted_chib,
        "half": np.asarray(half_shear),
        "log_omega": np.asarray(log_omega),
    }
    for element, (segment, indices) in enumerate(
        zip(scalar_coordinates.tau.segments, scalar_coordinates.tau.indices, strict=True)
    ):
        evaluators = {
            name: TauElementEvaluator(segment, indices, value, axis=1)
            for name, value in primitive_values.items()
        }
        reference = (
            None
            if reference_values is None
            else TauElementEvaluator(
                segment, indices, reference_values, axis=1
            )
        )
        known_cache = _BoundedCoordinateCache(
            enabled=cache_known_stages,
            maximum_bytes=known_cache_maximum_bytes,
        )

        def known_forcing(tau: float) -> dict[str, Array]:
            def build() -> dict[str, Array]:
                omega = evaluators["omega"].value(tau)
                zeta_up = evaluators["zeta_up"].value(tau)
                shift = evaluators["shift"].value(tau)
                source = _fresh_omegab_v_source(
                    grid,
                    tau,
                    evaluators,
                    angular,
                    projection_tails,
                )
                fixed_forcing = source - 2.0 * omega**2 * np.einsum(
                    "n...i,n...i->n...",
                    zeta_up,
                    scalar_gradient(
                        grid, evaluators["log_omega"].value(tau)
                    ),
                )
                return {
                    "source": source,
                    "fixed_forcing": fixed_forcing,
                    "shift": shift,
                }

            return known_cache.get(tau, build)

        def rhs(tau: float, value: Array) -> Array:
            known = known_forcing(tau)
            physical_rhs = known["fixed_forcing"] - np.einsum(
                "n...i,n...i->n...",
                known["shift"],
                scalar_gradient(grid, value),
            )
            return _project_scalar(
                angular,
                _du_dtau(tau) * physical_rhs,
                projection_tails,
                "weighted_omega_tau_rhs",
            )

        solved = solve_tau_sdc(
            segment,
            composite_element_initial(result, indices, axis=1),
            rhs,
            tolerance=tolerance,
            overgrid_tolerance=overgrid_tolerance,
            maximum_corrections=maximum_corrections,
            overgrid_extra_degree=overgrid_extra_degree,
            batch_axes=(1,),
        )
        assign_composite_element(result, indices, solved.values)

        over = LGLSegment.create(
            segment.left, segment.right, segment.degree + overgrid_extra_degree
        )
        fresh_overgrid_source = np.stack(
            [
                known_forcing(float(tau))["source"]
                for tau in over.nodes
            ]
        )
        if reference is None:
            source_closure_defect = 0.0
            source_closure_profile = np.zeros(len(scalar_coordinates.v), dtype=float)
            source_cross_equation_available = False
        else:
            reference_overgrid_source = reference.values_at(over.nodes)
            (
                source_closure_defect,
                source_closure_profile,
            ) = _per_v_scaled_difference(
                fresh_overgrid_source,
                reference_overgrid_source,
                absolute_floor=source_scale_floor,
            )
            source_cross_equation_available = True
        source_cross_equation_resolved = bool(
            source_closure_defect <= source_cross_equation_tolerance
        )

        local_fresh_sources = np.stack(
            [
                known_forcing(float(tau))["source"]
                for tau in segment.nodes
            ]
        )
        local_sources_u_first = np.moveaxis(local_fresh_sources, 0, 1)
        if element == 0:
            fresh_source_left_trace[:, indices] = local_sources_u_first
        else:
            fresh_source_left_trace[:, indices[1:]] = local_sources_u_first[:, 1:]
        fresh_source_right_trace[:, indices] = local_sources_u_first

        current_left_rhs = rhs(float(segment.left), solved.values[0])
        if previous_interface_source is None or previous_interface_rhs is None:
            source_interface_jump = 0.0
            rhs_interface_jump = 0.0
            source_interface_profile = np.zeros(len(scalar_coordinates.v), dtype=float)
            rhs_interface_profile = np.zeros(len(scalar_coordinates.v), dtype=float)
            has_left_interface = False
        else:
            (
                source_interface_jump,
                source_interface_profile,
            ) = _per_v_scaled_difference(
                previous_interface_source,
                local_fresh_sources[0],
                absolute_floor=source_scale_floor,
            )
            rhs_interface_jump, rhs_interface_profile = _per_v_scaled_difference(
                previous_interface_rhs,
                current_left_rhs,
                absolute_floor=source_scale_floor,
            )
            source_interface_profiles.append(source_interface_profile)
            rhs_interface_profiles.append(rhs_interface_profile)
            has_left_interface = True
        interface_resolved = bool(
            max(source_interface_jump, rhs_interface_jump)
            <= interface_tolerance
        )
        cone = cone_diagnostics[element]
        combined = bool(
            solved.accepted
            and source_cross_equation_resolved
            and interface_resolved
            and cone["primitive_cone_resolved"]
        )
        next_interface_rhs = rhs(
            float(segment.right), solved.values[-1]
        ).copy()
        diagnostic = tau_sdc_result_diagnostic(
            solved,
            **_element_metadata("weighted_omega", element, segment),
            source_semantics=(
                "fresh primitive B_v reconstruction with distinct left/right "
                "composite-interface traces"
            ),
        )
        source_metadata = json_safe_diagnostic(
            {
                "evolution_accepted": bool(solved.accepted),
                "source_cross_equation_available": source_cross_equation_available,
                "source_cross_equation_overgrid_defect": source_closure_defect,
                "source_cross_equation_overgrid_defect_by_v": (
                    source_closure_profile
                ),
                "source_cross_equation_tolerance": float(
                    source_cross_equation_tolerance
                ),
                "source_cross_equation_resolved": (
                    source_cross_equation_resolved
                ),
                "has_left_interface": has_left_interface,
                "fresh_source_interface_jump": source_interface_jump,
                "fresh_source_interface_jump_by_v": source_interface_profile,
                "rhs_interface_jump": rhs_interface_jump,
                "rhs_interface_jump_by_v": rhs_interface_profile,
                "interface_tolerance": float(interface_tolerance),
                "interface_resolved": interface_resolved,
                "source_scale_floor": float(source_scale_floor),
                "primitive_metric_minimum_eigenvalue": cone[
                    "primitive_metric_minimum_eigenvalue"
                ],
                "primitive_lapse_minimum": cone["primitive_lapse_minimum"],
                "primitive_cone_violation": cone[
                    "primitive_cone_violation"
                ],
                "primitive_cone_resolved": cone[
                    "primitive_cone_resolved"
                ],
                **known_cache.diagnostics(),
                "accepted": combined,
            }
        )
        if not isinstance(source_metadata, dict):
            raise AssertionError("weighted-omega source diagnostic is not a mapping")
        diagnostic.update(source_metadata)
        diagnostics.append(diagnostic)
        profiles.append(solved.overgrid_defect_by_batch.copy())
        source_closure_profiles.append(source_closure_profile)
        previous_interface_source = local_fresh_sources[-1].copy()
        previous_interface_rhs = next_interface_rhs

    if not np.all(np.isfinite(fresh_source_left_trace)) or not np.all(
        np.isfinite(fresh_source_right_trace)
    ):
        raise FloatingPointError("weighted-omega source traces are incomplete")
    maps = {
        "overgrid": expand_element_profiles(scalar_coordinates.tau, profiles),
        "source_cross_equation_overgrid": expand_element_profiles(
            scalar_coordinates.tau, source_closure_profiles
        ),
        "fresh_source_interface_jump": _interface_profile_map(
            scalar_coordinates, source_interface_profiles
        ),
        "rhs_interface_jump": _interface_profile_map(
            scalar_coordinates, rhs_interface_profiles
        ),
        **cone_maps,
    }
    _raise_rejected("weighted_omega", diagnostics, maps, require_acceptance)
    return TauConstructionResult(
        result,
        diagnostics,
        maps,
        {
            "fresh_v_source_left_trace": fresh_source_left_trace,
            "fresh_v_source_right_trace": fresh_source_right_trace,
        },
    )


def solve_incoming_metric_tau_sdc(
    grid: PointSphereGrid,
    metric_on_hminus1: Array,
    weighted_chib: Array,
    shift: Array,
    scalar_coordinates: CharacteristicLGLMesh,
    *,
    kinematic_factor: float = 2.0,
    angular: AngularGalerkin | None = None,
    projection_tails: ProjectionTails | None = None,
    tolerance: float = 1.0e-10,
    overgrid_tolerance: float = 1.0e-7,
    physical_overgrid_tolerance: float | None = None,
    maximum_corrections: int = 12,
    overgrid_extra_degree: int = 3,
    minimum_metric_eigenvalue: float = 1.0e-14,
    require_acceptance: bool = True,
) -> TauConstructionResult:
    """Reconstruct the incoming metric through an SPD tangent factor."""

    metric_values = np.asarray(metric_on_hminus1)
    chib_values = np.asarray(weighted_chib)
    shift_values = np.asarray(shift)
    expected_tensor = (
        grid.count,
        len(scalar_coordinates.u),
        len(scalar_coordinates.v),
        3,
        3,
    )
    expected_vector = (
        grid.count,
        len(scalar_coordinates.u),
        len(scalar_coordinates.v),
        3,
    )
    if chib_values.shape != expected_tensor or shift_values.shape != expected_vector:
        raise ValueError("incoming metric primitive fields do not match the grid")
    expected_boundary = (grid.count, len(scalar_coordinates.v), 3, 3)
    if metric_values.shape != expected_boundary:
        raise ValueError("incoming metric boundary has the wrong shape")
    if physical_overgrid_tolerance is None:
        physical_overgrid_tolerance = overgrid_tolerance
    if physical_overgrid_tolerance <= 0.0 or minimum_metric_eigenvalue <= 0.0:
        raise ValueError("invalid incoming metric acceptance tolerance")

    projected_initial = _project_sym2(
        angular,
        metric_values,
        projection_tails,
        "incoming_metric_boundary",
    )
    local_initial = np.einsum(
        "nia,n...ij,njb->n...ab", grid.frames, projected_initial, grid.frames
    )
    factor = np.zeros(
        (grid.count, len(scalar_coordinates.u), len(scalar_coordinates.v), 2, 2),
        dtype=float,
    )
    factor[:, 0] = np.linalg.cholesky(local_initial)
    diagnostics: list[dict[str, Any]] = []
    factor_profiles: list[Array] = []
    physical_profiles: list[Array] = []

    def ambient_metric(value_factor: Array) -> Array:
        local_metric = np.matmul(value_factor, np.swapaxes(value_factor, -1, -2))
        return np.einsum(
            "nia,n...ab,njb->n...ij", grid.frames, local_metric, grid.frames
        )

    for element, (segment, indices) in enumerate(
        zip(scalar_coordinates.tau.segments, scalar_coordinates.tau.indices, strict=True)
    ):
        chib = TauElementEvaluator(segment, indices, chib_values, axis=1)
        shift_evaluator = TauElementEvaluator(
            segment, indices, shift_values, axis=1
        )

        def physical_tau_rhs(tau: float, value_metric: Array) -> Array:
            physical = kinematic_factor * chib.value(tau) - lie_covariant_tensor(
                grid, shift_evaluator.value(tau), value_metric
            )
            physical = _project_sym2(
                angular,
                physical,
                projection_tails,
                "incoming_metric_tau_rhs",
            )
            return _du_dtau(tau) * physical

        def rhs(tau: float, value_factor: Array) -> Array:
            value_metric = ambient_metric(value_factor)
            if _minimum_tangent_eigenvalue(grid, value_metric) <= minimum_metric_eigenvalue:
                raise FloatingPointError("incoming metric factor approached singularity")
            metric_rhs = physical_tau_rhs(tau, value_metric)
            local_rhs = np.einsum(
                "nia,n...ij,njb->n...ab", grid.frames, metric_rhs, grid.frames
            )
            try:
                inverse_transpose = np.swapaxes(
                    np.linalg.inv(value_factor), -1, -2
                )
            except np.linalg.LinAlgError as error:
                raise FloatingPointError("incoming metric factor became singular") from error
            return 0.5 * np.matmul(local_rhs, inverse_transpose)

        solved = solve_tau_sdc(
            segment,
            composite_element_initial(factor, indices, axis=1),
            rhs,
            tolerance=tolerance,
            overgrid_tolerance=overgrid_tolerance,
            maximum_corrections=maximum_corrections,
            overgrid_extra_degree=overgrid_extra_degree,
            batch_axes=(1,),
        )
        assign_composite_element(factor, indices, solved.values)

        over = LGLSegment.create(
            segment.left, segment.right, segment.degree + overgrid_extra_degree
        )
        interpolation = segment.interpolation_matrix(over.nodes)
        derivative = segment.derivative_interpolation_matrix(over.nodes)
        over_factor = np.tensordot(interpolation, solved.values, axes=(1, 0))
        over_factor_tau = np.tensordot(derivative, solved.values, axes=(1, 0))
        physical_residuals = []
        physical_references = []
        minimum_eigenvalue_seen = math.inf
        for tau, value_factor, value_factor_tau in zip(
            over.nodes, over_factor, over_factor_tau, strict=True
        ):
            value_metric = ambient_metric(value_factor)
            local_metric_tau = np.matmul(
                value_factor_tau, np.swapaxes(value_factor, -1, -2)
            ) + np.matmul(
                value_factor, np.swapaxes(value_factor_tau, -1, -2)
            )
            value_metric_tau = np.einsum(
                "nia,n...ab,njb->n...ij",
                grid.frames,
                local_metric_tau,
                grid.frames,
            )
            reference = physical_tau_rhs(float(tau), value_metric)
            physical_residuals.append(value_metric_tau - reference)
            physical_references.append(reference)
            minimum_eigenvalue_seen = min(
                minimum_eigenvalue_seen,
                _minimum_tangent_eigenvalue(grid, value_metric),
            )
        physical_defect, physical_profile = _per_v_scaled_profile(
            np.stack(physical_residuals), np.stack(physical_references)
        )
        physical_resolved = bool(
            physical_defect <= physical_overgrid_tolerance
            and minimum_eigenvalue_seen > minimum_metric_eigenvalue
        )
        combined = bool(solved.accepted and physical_resolved)
        diagnostic = tau_sdc_result_diagnostic(
            solved, **_element_metadata("incoming_metric", element, segment)
        )
        physical_metadata = json_safe_diagnostic(
            {
                "factor_accepted": bool(solved.accepted),
                "physical_overgrid_defect": physical_defect,
                "physical_overgrid_defect_by_v": physical_profile,
                "physical_resolved": physical_resolved,
                "minimum_metric_eigenvalue": float(minimum_eigenvalue_seen),
                "accepted": combined,
            }
        )
        if not isinstance(physical_metadata, dict):
            raise AssertionError("physical diagnostic metadata is not a mapping")
        diagnostic.update(physical_metadata)
        diagnostics.append(diagnostic)
        factor_profiles.append(solved.overgrid_defect_by_batch.copy())
        physical_profiles.append(physical_profile)

    metric = ambient_metric(factor)
    maps = {
        "factor_overgrid": expand_element_profiles(
            scalar_coordinates.tau, factor_profiles
        ),
        "physical_overgrid": expand_element_profiles(
            scalar_coordinates.tau, physical_profiles
        ),
    }
    _raise_rejected("incoming_metric", diagnostics, maps, require_acceptance)
    return TauConstructionResult(metric, diagnostics, maps, {"factor": factor})
