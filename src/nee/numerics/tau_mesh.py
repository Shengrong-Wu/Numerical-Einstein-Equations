"""Reusable element plumbing for future tau-collocation construction solves.

This module is intentionally independent of the Einstein iteration driver.  It
contains only coordinate interpolation, checked composite assembly, diagnostic
map expansion, and JSON conversion.  Keeping these operations separate makes
the later production patch small and gives shared-interface behavior a direct
test rather than burying it in an equation-specific solver.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .lgl import CompositeLGLMesh, LGLSegment


Array = np.ndarray


HALF_SHEAR_TRACE_SCHEMA_VERSION = 2
HALF_SHEAR_TRACE_MAP_SEMANTICS = {
    "trace_retained_nodes": (
        "hard Galerkin-DAE constraint at stored tau nodes: "
        "||Pi_L tr_g C_L||_L2(S)/||C_L||_L2(S,g)"
    ),
    "trace_retained_overgrid": (
        "hard dense-output Galerkin-DAE constraint on the unretracted "
        "independent tau overgrid"
    ),
    "trace_strong_angular": (
        "strong pointwise nonlinear trace after weak Galerkin retraction, "
        "measured in surface L2 as angular-consistency evidence"
    ),
}
@dataclass(frozen=True)
class SurfaceL2TraceDefect:
    """A topology-safe trace defect with the tau and v localization retained."""

    maximum: float
    by_v: Array
    by_node_v: Array
    effective_absolute_floor: float


def _surface_rms(values: Array) -> Array:
    """Stable equal-area sphere RMS for ``(tau,sphere,v)`` samples."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 3:
        raise ValueError("surface samples must have (tau,sphere,v) shape")
    scale = np.max(np.abs(array), axis=1)
    safe_scale = np.where(scale > 0.0, scale, 1.0)
    normalized = array / safe_scale[:, None, :]
    result = safe_scale * np.sqrt(np.mean(normalized**2, axis=1))
    return np.where(scale > 0.0, result, 0.0)


def surface_l2_relative_trace_defect(
    trace: Array,
    tensor_norm: Array,
    *,
    absolute_floor: float = 1.0e-14,
    reference_tensor_scale: float | None = None,
) -> SurfaceL2TraceDefect:
    """Measure a relative trace without dividing at angular zeros.

    ``trace`` and ``tensor_norm`` have shape ``(tau,sphere,v)``.  The
    numerator and denominator are reduced over the *whole sphere* before
    division.  This is essential for a trace-free two-tensor on ``S^2``:
    its norm can vanish at individual angular points, so the pointwise ratio
    ``|tr C|/|C|`` is not a stable acceptance norm.

    The equal-weight RMS is the discrete ``L2`` norm associated with the
    Fibonacci/Galerkin reference sphere; the common ``sqrt(4*pi)`` factor
    cancels in the ratio.  ``trace`` may be either ``Pi_L tr_g C_L`` for the
    retained DAE constraint or the unprojected trace for strong angular
    consistency.
    """

    if not math.isfinite(absolute_floor) or absolute_floor <= 0.0:
        raise ValueError("the trace absolute floor must be positive and finite")
    trace_values = np.asarray(trace, dtype=float)
    norm_values = np.asarray(tensor_norm, dtype=float)
    if trace_values.shape != norm_values.shape or trace_values.ndim != 3:
        raise ValueError(
            "trace and tensor norm must have identical (tau,sphere,v) shape"
        )
    if not np.all(np.isfinite(trace_values)) or not np.all(np.isfinite(norm_values)):
        raise ValueError("trace diagnostic samples must be finite")
    if np.any(norm_values < 0.0):
        raise ValueError("tensor norm samples must be nonnegative")

    trace_rms = _surface_rms(trace_values)
    tensor_rms = _surface_rms(norm_values)
    local_scale = float(np.max(tensor_rms))
    if reference_tensor_scale is None:
        reference_scale = local_scale
    else:
        reference_scale = float(reference_tensor_scale)
        if not math.isfinite(reference_scale) or reference_scale < 0.0:
            raise ValueError("the reference tensor scale must be finite and nonnegative")
        if reference_scale + 32.0 * np.finfo(float).eps * max(
            1.0, local_scale
        ) < local_scale:
            raise ValueError("the reference tensor scale is smaller than the samples")
    effective_floor = max(
        float(absolute_floor),
        64.0 * np.finfo(float).eps * reference_scale,
    )
    ratio = trace_rms / np.maximum(tensor_rms, effective_floor)
    return SurfaceL2TraceDefect(
        maximum=float(np.max(ratio)),
        by_v=np.max(ratio, axis=0),
        by_node_v=ratio,
        effective_absolute_floor=effective_floor,
    )


def _normalized_axis(axis: int, ndim: int) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += ndim
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"axis {axis} is invalid for a {ndim}-D array")
    return normalized


class TauElementEvaluator:
    """Interpolate one primitive field on one tau element.

    ``global_indices`` select the element nodes from the declared coordinate
    axis of ``field``.  Multi-target methods return a leading target axis and
    otherwise preserve the ordering of the non-coordinate field axes.  The
    scalar ``value`` and ``derivative`` methods remove that leading target
    axis, which is the shape expected by an ODE right-hand side.
    """

    def __init__(
        self,
        segment: LGLSegment,
        global_indices: Array,
        field: Array,
        *,
        axis: int = 1,
    ) -> None:
        values = np.asarray(field)
        coordinate_axis = _normalized_axis(axis, values.ndim)
        indices = np.asarray(global_indices, dtype=int)
        if indices.ndim != 1 or len(indices) != len(segment.nodes):
            raise ValueError(
                "one global index is required for every element node"
            )
        if len(np.unique(indices)) != len(indices) or np.any(np.diff(indices) <= 0):
            raise ValueError("element indices must be unique and increasing")
        if np.any(indices < 0) or np.any(indices >= values.shape[coordinate_axis]):
            raise ValueError("an element index lies outside the field coordinate axis")

        local = np.take(values, indices, axis=coordinate_axis)
        self.segment = segment
        self.global_indices = indices.copy()
        self.axis = coordinate_axis
        self._local = np.moveaxis(local, coordinate_axis, 0)
        # An SDC sweep repeatedly revisits the same collocation, RK-midpoint,
        # and independent-overgrid scalar_coordinates.  Cache the small barycentric
        # rows per primitive evaluator; the (usually much larger) field
        # contractions remain fresh and therefore cannot become stale.
        self._interpolation_rows: dict[float, Array] = {}
        self._derivative_rows: dict[float, Array] = {}

    @property
    def value_shape(self) -> tuple[int, ...]:
        """Shape of one field value after removing the tau coordinate axis."""

        return self._local.shape[1:]

    def cache_info(self) -> dict[str, int]:
        """Return cache cardinalities for diagnostics and deterministic tests."""

        return {
            "interpolation_rows": len(self._interpolation_rows),
            "derivative_rows": len(self._derivative_rows),
        }

    def _interpolation_row(self, tau: float) -> Array:
        coordinate = float(tau)
        row = self._interpolation_rows.get(coordinate)
        if row is None:
            row = self.segment.interpolation_matrix(
                np.array([coordinate], dtype=float)
            )[0]
            row.setflags(write=False)
            self._interpolation_rows[coordinate] = row
        return row

    def _derivative_row(self, tau: float) -> Array:
        coordinate = float(tau)
        row = self._derivative_rows.get(coordinate)
        if row is None:
            row = self.segment.derivative_interpolation_matrix(
                np.array([coordinate], dtype=float)
            )[0]
            row.setflags(write=False)
            self._derivative_rows[coordinate] = row
        return row

    def _target_vector(self, tau: Array) -> Array:
        targets = np.atleast_1d(np.asarray(tau, dtype=float))
        if targets.ndim != 1:
            raise ValueError("tau targets must be a scalar or one-dimensional vector")
        return targets

    def values_at(self, tau: Array) -> Array:
        """Return interpolated values with a leading target-coordinate axis."""

        targets = self._target_vector(tau)
        if not len(targets):
            return np.empty((0, *self.value_shape), dtype=self._local.dtype)
        matrix = np.stack(
            [self._interpolation_row(float(target)) for target in targets]
        )
        return np.tensordot(matrix, self._local, axes=(1, 0))

    def derivatives_at(self, tau: Array) -> Array:
        """Return tau derivatives with a leading target-coordinate axis."""

        targets = self._target_vector(tau)
        if not len(targets):
            return np.empty((0, *self.value_shape), dtype=self._local.dtype)
        matrix = np.stack(
            [self._derivative_row(float(target)) for target in targets]
        )
        return np.tensordot(matrix, self._local, axes=(1, 0))

    def value(self, tau: float) -> Array:
        return self.values_at(np.array([tau], dtype=float))[0]

    def derivative(self, tau: float) -> Array:
        return self.derivatives_at(np.array([tau], dtype=float))[0]


def composite_element_initial(
    destination: Array,
    global_indices: Array,
    *,
    axis: int = 1,
) -> Array:
    """Return a copy of an element's carried left endpoint."""

    values = np.asarray(destination)
    coordinate_axis = _normalized_axis(axis, values.ndim)
    indices = np.asarray(global_indices, dtype=int)
    if indices.ndim != 1 or not len(indices):
        raise ValueError("element indices must be a nonempty vector")
    if indices[0] < 0 or indices[0] >= values.shape[coordinate_axis]:
        raise ValueError("the carried endpoint lies outside the destination")
    return np.take(values, int(indices[0]), axis=coordinate_axis).copy()


def assign_composite_element(
    destination: Array,
    global_indices: Array,
    local_values: Array,
    *,
    destination_axis: int = 1,
    local_axis: int = 0,
    check_left_carry: bool = True,
    relative_tolerance: float = 1.0e-12,
    absolute_tolerance: float = 1.0e-13,
) -> Array:
    """Assign an element after checking its carried shared endpoint.

    The destination is updated in place and returned for convenience.  The
    check is performed before any assignment, so a mismatched interface cannot
    partially corrupt the global array.
    """

    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("interface tolerances must be nonnegative")
    global_values = np.asarray(destination)
    local = np.asarray(local_values)
    global_axis = _normalized_axis(destination_axis, global_values.ndim)
    element_axis = _normalized_axis(local_axis, local.ndim)
    indices = np.asarray(global_indices, dtype=int)
    if indices.ndim != 1 or not len(indices) or np.any(np.diff(indices) <= 0):
        raise ValueError("element indices must be a nonempty increasing vector")
    if np.any(indices < 0) or np.any(indices >= global_values.shape[global_axis]):
        raise ValueError("an element index lies outside the destination")

    moved_global = np.moveaxis(global_values, global_axis, 0)
    moved_local = np.moveaxis(local, element_axis, 0)
    expected = (len(indices), *moved_global.shape[1:])
    if moved_local.shape != expected:
        raise ValueError(
            f"local element has shape {moved_local.shape}, expected {expected}"
        )
    if check_left_carry and not np.allclose(
        moved_global[indices[0]],
        moved_local[0],
        rtol=relative_tolerance,
        atol=absolute_tolerance,
        equal_nan=False,
    ):
        scale = np.maximum(
            1.0,
            np.maximum(
                np.abs(moved_global[indices[0]]), np.abs(moved_local[0])
            ),
        )
        mismatch = float(
            np.max(np.abs(moved_global[indices[0]] - moved_local[0]) / scale)
        )
        raise ValueError(
            "tau element left endpoint does not match the carried interface: "
            f"scaled maximum={mismatch:.12g}"
        )
    moved_global[indices] = moved_local
    return destination


def expand_element_profiles(
    mesh: CompositeLGLMesh,
    profiles: Sequence[Array],
    *,
    interface_rule: str = "maximum",
) -> Array:
    """Expand one batch profile per element onto the shared global tau grid.

    Defect profiles are normally nonnegative and use ``maximum`` at a shared
    interface so one well-resolved neighbor cannot hide an underresolved one.
    ``left`` and ``right`` are provided for diagnostics whose trace convention
    is directional.
    """

    if len(profiles) != len(mesh.segments):
        raise ValueError("one profile is required for every tau element")
    if interface_rule not in {"maximum", "left", "right"}:
        raise ValueError("interface_rule must be 'maximum', 'left', or 'right'")
    converted = [np.asarray(profile, dtype=float) for profile in profiles]
    if not converted:
        raise ValueError("the composite mesh has no elements")
    profile_shape = converted[0].shape
    if any(profile.shape != profile_shape for profile in converted):
        raise ValueError("all element profiles must have the same shape")

    result = np.full((len(mesh.nodes), *profile_shape), np.nan, dtype=float)
    assigned = np.zeros(len(mesh.nodes), dtype=bool)
    for profile, indices in zip(converted, mesh.indices, strict=True):
        for index in indices:
            global_index = int(index)
            if not assigned[global_index]:
                result[global_index] = profile
                assigned[global_index] = True
            elif interface_rule == "maximum":
                result[global_index] = np.maximum(result[global_index], profile)
            elif interface_rule == "right":
                result[global_index] = profile
            # The left rule deliberately retains the already assigned trace.
    if not np.all(assigned):
        raise ValueError("the element indices do not cover the composite grid")
    return result


def expand_diagnostic_profiles(
    mesh: CompositeLGLMesh,
    diagnostics: Sequence[Mapping[str, Any]],
    key: str = "overgrid_defect_by_batch",
    *,
    interface_rule: str = "maximum",
) -> Array:
    """Extract and expand one named array profile from each diagnostic."""

    missing = [index for index, item in enumerate(diagnostics) if key not in item]
    if missing:
        raise ValueError(
            f"diagnostics {missing} do not contain the profile key {key!r}"
        )
    return expand_element_profiles(
        mesh,
        [np.asarray(item[key], dtype=float) for item in diagnostics],
        interface_rule=interface_rule,
    )


def json_safe_diagnostic(value: Any) -> Any:
    """Recursively convert a diagnostic to strict-JSON-compatible objects.

    Nonfinite numbers are encoded as the strings ``"nan"``, ``"+inf"``, and
    ``"-inf"``.  This retains the kind of numerical failure while allowing
    ``json.dumps(..., allow_nan=False)`` to enforce standards-compliant JSON.
    Unknown object types raise instead of being silently stringified.
    """

    if is_dataclass(value) and not isinstance(value, type):
        return json_safe_diagnostic(asdict(value))
    if isinstance(value, np.ndarray):
        return json_safe_diagnostic(value.tolist())
    if isinstance(value, np.generic):
        return json_safe_diagnostic(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(json_safe_diagnostic(key)): json_safe_diagnostic(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [json_safe_diagnostic(item) for item in value]
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "+inf" if value > 0.0 else "-inf"
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(
        f"diagnostic value of type {type(value).__name__} is not JSON-safe"
    )


def half_shear_trace_certificate(
    mesh: CompositeLGLMesh,
    retained_node_trace: Array,
    node_tensor_norm: Array,
    retained_overgrid_traces: Sequence[Array],
    overgrid_tensor_norms: Sequence[Array],
    strong_angular_traces: Sequence[Array],
    strong_angular_tensor_norms: Sequence[Array],
    *,
    absolute_floor: float = 1.0e-14,
    retained_node_tolerance: float = 1.0e-9,
    retained_overgrid_tolerance: float = 1.0e-7,
    strong_angular_tolerance: float = 1.0e-5,
) -> dict[str, Any]:
    """Build the three non-conflated half-shear trace certificates.

    Callers must supply already projected retained traces:

    * ``retained_node_trace = Pi_L(tr_g C_L)`` at stored tau nodes;
    * one unretracted ``Pi_L(tr_g C_p)`` array per independent overgrid;
    * one unprojected ``tr_g(retract(C_p))`` array per overgrid for the
      strong angular-consistency diagnostic.

    In particular, the retained overgrid input must be formed *before* tau
    retraction; otherwise that independent DAE constraint test is circular.
    Conversely, the strong trace is formed after weak retraction so it does
    not mix tau interpolation drift with angular truncation.

    Returned map suffixes are the stable schema documented in
    ``HALF_SHEAR_TRACE_MAP_SEMANTICS``.  The former ``trace_overgrid`` map is
    deliberately not emitted because it used a topology-unstable pointwise
    local division and cannot be converted into any of these quantities.
    """

    tolerances = {
        "retained_node_tolerance": retained_node_tolerance,
        "retained_overgrid_tolerance": retained_overgrid_tolerance,
        "strong_angular_tolerance": strong_angular_tolerance,
    }
    for name, value in tolerances.items():
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(absolute_floor) or absolute_floor <= 0.0:
        raise ValueError("the trace absolute floor must be positive and finite")
    groups = {
        "retained_overgrid": (
            retained_overgrid_traces,
            overgrid_tensor_norms,
        ),
        "strong_angular": (
            strong_angular_traces,
            strong_angular_tensor_norms,
        ),
    }
    for name, (traces, norms) in groups.items():
        if len(traces) != len(mesh.segments) or len(norms) != len(mesh.segments):
            raise ValueError(
                f"{name} requires one trace and norm array per tau element"
            )

    node_trace = np.asarray(retained_node_trace, dtype=float)
    node_norm = np.asarray(node_tensor_norm, dtype=float)
    if node_trace.shape != node_norm.shape or node_trace.ndim != 3:
        raise ValueError(
            "retained node trace and norm must have identical "
            "(tau,sphere,v) shape"
        )
    if node_trace.shape[0] != len(mesh.nodes):
        raise ValueError("retained node samples do not match the composite mesh")

    all_norms = [node_norm]
    for _, norms in groups.values():
        all_norms.extend(np.asarray(value, dtype=float) for value in norms)
    reference_scale = max(
        (float(np.max(_surface_rms(value))) for value in all_norms),
        default=0.0,
    )
    node = surface_l2_relative_trace_defect(
        node_trace,
        node_norm,
        absolute_floor=absolute_floor,
        reference_tensor_scale=reference_scale,
    )

    evaluated: dict[str, list[SurfaceL2TraceDefect]] = {}
    for name, (traces, norms) in groups.items():
        evaluated[name] = [
            surface_l2_relative_trace_defect(
                np.asarray(trace, dtype=float),
                np.asarray(norm, dtype=float),
                absolute_floor=absolute_floor,
                reference_tensor_scale=reference_scale,
            )
            for trace, norm in zip(traces, norms, strict=True)
        ]
    retained_overgrid = evaluated["retained_overgrid"]
    strong_angular = evaluated["strong_angular"]
    retained_overgrid_maximum = max(
        (item.maximum for item in retained_overgrid), default=0.0
    )
    strong_angular_maximum = max(
        (item.maximum for item in strong_angular), default=0.0
    )
    constraint_resolved = bool(
        node.maximum <= retained_node_tolerance
        and retained_overgrid_maximum <= retained_overgrid_tolerance
    )
    strong_angular_resolved = bool(
        strong_angular_maximum <= strong_angular_tolerance
    )
    diagnostics = json_safe_diagnostic(
        {
            "trace_schema_version": HALF_SHEAR_TRACE_SCHEMA_VERSION,
            "trace_norm": "surface-L2-before-relative-division",
            "trace_retained_nodes_defect": node.maximum,
            "trace_retained_nodes_defect_by_v": node.by_v,
            "trace_retained_nodes_tolerance": retained_node_tolerance,
            "trace_retained_nodes_resolved": bool(
                node.maximum <= retained_node_tolerance
            ),
            "trace_retained_overgrid_defect": retained_overgrid_maximum,
            "trace_retained_overgrid_defect_by_v": np.maximum.reduce(
                [item.by_v for item in retained_overgrid]
            ),
            "trace_retained_overgrid_tolerance": retained_overgrid_tolerance,
            "trace_retained_overgrid_resolved": bool(
                retained_overgrid_maximum <= retained_overgrid_tolerance
            ),
            "trace_constraint_resolved": constraint_resolved,
            "trace_strong_angular_defect": strong_angular_maximum,
            "trace_strong_angular_defect_by_v": np.maximum.reduce(
                [item.by_v for item in strong_angular]
            ),
            "trace_strong_angular_tolerance": strong_angular_tolerance,
            "trace_strong_angular_resolved": strong_angular_resolved,
            "trace_effective_absolute_floor": node.effective_absolute_floor,
            "trace_established_pointwise_ratio_used_for_acceptance": False,
        }
    )
    if not isinstance(diagnostics, dict):
        raise AssertionError("half-shear trace diagnostics are not a mapping")
    maps = {
        "trace_retained_nodes": node.by_node_v.copy(),
        "trace_retained_overgrid": expand_element_profiles(
            mesh, [item.by_v for item in retained_overgrid]
        ),
        "trace_strong_angular": expand_element_profiles(
            mesh, [item.by_v for item in strong_angular]
        ),
    }
    return {
        "diagnostics": diagnostics,
        "maps": maps,
        "constraint_resolved": constraint_resolved,
        "strong_angular_resolved": strong_angular_resolved,
    }


def tau_sdc_result_diagnostic(result: Any, **metadata: Any) -> dict[str, Any]:
    """Return the small, JSON-safe portion of a tau-SDC result.

    The large nodal ``values`` array is intentionally excluded.  Equation
    wrappers can add element endpoints, field names, constraint defects, and
    other metadata through keyword arguments.
    """

    required = (
        "corrections",
        "converged",
        "resolved",
        "accepted",
        "collocation_defect",
        "overgrid_defect",
        "overgrid_defect_by_batch",
        "batch_axes",
        "correction_history",
    )
    missing = [name for name in required if not hasattr(result, name)]
    if missing:
        raise ValueError(f"tau-SDC result is missing attributes {missing}")
    raw = {
        **metadata,
        **{name: getattr(result, name) for name in required},
    }
    converted = json_safe_diagnostic(raw)
    if not isinstance(converted, dict):
        raise AssertionError("the converted tau-SDC diagnostic is not a mapping")
    return converted
