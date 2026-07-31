"""Progressive full-Q1 continuation on characteristic LGL panels.

A full-slab Picard sweep starts the strong-pulse part of Q1 from a spherical
guess and ceases to be admissible near v=.36.  This driver converges causal
prefixes first, prolongs the state into the next panel, and then resumes the
*same* continuum Picard map.  The established constant-tail predictor remains
available, while an opt-in boundary-lift predictor transports the known
outgoing-data increment smoothly through the slab.  The driver records where
this globalization succeeds, loses contraction, loses coordinate/angular
resolution, or leaves the geometric positive cone.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import uuid
from dataclasses import fields
from pathlib import Path

import numpy as np

# Import the package-local campaign helpers before optional checkpoint support.
from .pulse_campaign import (  # noqa: E402
    CHECKPOINT_MANIFEST_KIND,
    CHECKPOINT_MANIFEST_SCHEMA_VERSION,
    FINGERPRINT_SOURCES,
    SOLVER_SEMANTICS,
    checkpoint_manifest_path,
    file_fingerprint,
    low_band_s_breakpoints,
    runtime_provenance,
    safe_scalar_summary,
    scalar_boundary_summary,
    solver_source_fingerprint,
    validate_checkpoint_manifest,
)
from .adaptive_mesh import load_s_breakpoints
from .spherical_harmonics import AngularGalerkin
from .lgl import CharacteristicLGLMesh, LGLSegment
from .vacuum_state_io import full_ricci_map, load_state, save_state
from .vacuum_iteration import (
    FirstOrderState,
    boundary_compatible_initial_state,
    initial_state,
    impose_outgoing_boundary,
    metric_closure,
    picard_step,
    update_map,
    update_norm,
)
from .vacuum_residual import components, physical_component_l2_maps
from .sphere import PointSphereGrid
from .short_pulse import (
    calibrate_low_band_profiles,
    solve_low_band_boundary,
)
from .sphere import (
    tangent_inverse,
    tensor_norm_sq,
    tensor_trace,
    tensor_tracefree,
)
from .tau_mesh import surface_l2_relative_trace_defect
from .checkpoint import (
    atomic_write_json,
    prepare_output_directory,
    resolve_operational_restart_checkpoint,
    write_operational_checkpoint,
)


Array = np.ndarray
ROOT = Path(__file__).resolve().parent.parent
CONTINUATION_CAP_RECORD_SCHEMA_VERSION = 1
CONTINUATION_CAP_CANONICALIZATION = (
    "json-sort-keys-indent-2-allow-nan-false-newline"
)


def continuation_source_fingerprint() -> str:
    """Fingerprint the discrete map and this globalization driver."""

    digest = hashlib.sha256()
    digest.update(solver_source_fingerprint().encode("utf-8"))
    digest.update(b"\0")
    digest.update(Path(__file__).read_bytes())
    return f"sha256:{digest.hexdigest()}"


def continuation_schedule(
    s_breakpoints: Array,
    *,
    v1: float,
    specification: str | None,
) -> tuple[Array, list[int]]:
    """Insert requested physical caps and return their prefix indices.

    Coordinate h-refinement and nonlinear endpoint continuation serve
    different purposes.  The former may require dozens of small elements;
    using every such element as a nonlinear continuation stage creates an
    unnecessary O(number-of-elements^2) solve.  Explicit caps let one retain
    the fine common coordinate representation while globalizing Picard on a
    much smaller causal schedule.
    """

    base = np.asarray(s_breakpoints, dtype=float)
    if specification is None:
        return base, list(range(1, len(base)))
    try:
        caps = np.asarray(
            [float(value) for value in specification.split(",") if value.strip()],
            dtype=float,
        )
    except ValueError as error:
        raise ValueError("continuation caps must be comma-separated numbers") from error
    if (
        caps.ndim != 1
        or not len(caps)
        or np.any(~np.isfinite(caps))
        or np.any(caps <= 0.0)
        or np.any(caps > v1 + 2.0e-14)
        or np.any(np.diff(caps) <= 0.0)
    ):
        raise ValueError("continuation caps must increase strictly in (0,v1]")
    if not np.isclose(caps[-1], v1, rtol=0.0, atol=2.0e-14):
        caps = np.concatenate((caps, [v1]))
    requested_s = np.sqrt(np.maximum(caps / v1, 0.0))
    merged = np.unique(np.concatenate((base, requested_s)))
    indices: list[int] = []
    for value in requested_s:
        match = np.flatnonzero(np.isclose(merged, value, rtol=0.0, atol=2.0e-14))
        if len(match) != 1 or match[0] == 0:
            raise ValueError(f"could not place continuation cap s={value:.17g}")
        indices.append(int(match[0]))
    return merged, indices


def continuation_execution_indices(
    s_breakpoints: Array,
    continuation_indices: list[int],
    *,
    v1: float,
    stop_after_v: float | None,
) -> list[int]:
    """Return the full-schedule prefix executed by this invocation.

    ``stop_after_v`` is deliberately controller metadata, not part of the
    scientific discrete-map contract.  A completed cap can therefore seed a
    later invocation with a larger stop while an operational crash restart,
    whose run contract includes the stop, remains exact.  Requiring the stop
    to be an existing nonlinear cap also prevents a nominal endpoint from
    landing inside an element or being silently rounded to a nearby cap.
    """

    indices = [int(value) for value in continuation_indices]
    if not indices:
        raise ValueError("the continuation schedule has no nonlinear caps")
    if stop_after_v is None:
        return indices
    stop = float(stop_after_v)
    if not math.isfinite(stop) or not 0.0 < stop <= v1:
        raise ValueError("stop-after-v must lie in (0,v1]")
    endpoints = np.asarray(
        [v1 * float(s_breakpoints[index]) ** 2 for index in indices],
        dtype=float,
    )
    matches = np.flatnonzero(
        np.isclose(endpoints, stop, rtol=0.0, atol=2.0e-14)
    )
    if len(matches) != 1:
        raise ValueError(
            "stop-after-v must exactly match one configured nonlinear "
            "continuation cap"
        )
    return indices[: int(matches[0]) + 1]


def q1_import_solver_parameters(
    continuation_parameters: dict[str, object],
    prefix_coordinates: CharacteristicLGLMesh,
    *,
    v_endpoint: float,
) -> dict[str, object]:
    """Translate a continuation contract into the exact Q1-run contract.

    The continuum/discrete Picard map sections must agree exactly.  Only the
    producer-specific coordinate wrapper and the continuation controller are
    changed: a standalone ``pulse_campaign`` records its prefix mesh and
    physical endpoint, whereas the continuation records the full mesh and cap
    schedule.  No output label or fixed-sweep count enters either scientific
    contract.
    """

    required = (
        "boundary",
        "sphere",
        "galerkin",
        "metric_construction",
        "incoming_construction",
    )
    missing = [name for name in required if name not in continuation_parameters]
    if missing:
        raise ValueError(
            "continuation solver contract lacks sections: "
            + ", ".join(missing)
        )
    boundary = dict(continuation_parameters["boundary"])
    boundary["v_endpoint"] = float(v_endpoint)
    return {
        "boundary": boundary,
        "sphere": dict(continuation_parameters["sphere"]),
        "galerkin": dict(continuation_parameters["galerkin"]),
        "scalar_coordinates": {
            "method": "lgl",
            "mesh": prefix_coordinates.diagnostics(),
        },
        "metric_construction": dict(
            continuation_parameters["metric_construction"]
        ),
        "incoming_construction": dict(
            continuation_parameters["incoming_construction"]
        ),
    }


def _read_resume_manifest(state_path: Path) -> tuple[Path, dict[str, object]]:
    state = Path(state_path).resolve()
    manifest = checkpoint_manifest_path(state)
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"resume provenance manifest {manifest} is unreadable: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"resume provenance manifest {manifest} is not an object")
    return manifest, payload


def _validate_completed_q1_summary(
    state_path: Path,
    manifest_path: Path,
    *,
    solver_fingerprint: str,
) -> dict[str, object]:
    """Require the q1 producer to have durably published a completed run."""

    summary_path = Path(state_path).resolve().parent / "summary.json"
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(
            f"pulse_campaign scientific import lacks completed summary {summary_path}"
        ) from error
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"pulse_campaign summary {summary_path} is unreadable: {error}"
        ) from error
    if not isinstance(summary, dict):
        raise ValueError(f"pulse_campaign summary {summary_path} is not an object")
    if summary.get("status") != "completed":
        raise ValueError(
            "pulse_campaign scientific import requires status='completed'; "
            f"received {summary.get('status')!r}"
        )
    if summary.get("solver_semantics") != SOLVER_SEMANTICS:
        raise ValueError("pulse_campaign summary has different solver semantics")
    if summary.get("solver_source_fingerprint") != solver_fingerprint:
        raise ValueError("pulse_campaign summary has a different source fingerprint")
    declared_manifest = summary.get("checkpoint_manifest")
    if not isinstance(declared_manifest, str):
        raise ValueError("completed pulse_campaign summary lacks checkpoint_manifest")
    declared_path = Path(declared_manifest)
    if not declared_path.is_absolute():
        declared_path = summary_path.parent / declared_path
    if declared_path.resolve() != manifest_path.resolve():
        raise ValueError(
            "pulse_campaign summary does not declare the imported state manifest"
        )
    return {
        "path": str(summary_path.resolve()),
        "sha256": file_fingerprint(summary_path),
        "status": "completed",
    }


def canonical_continuation_cap_stage(stage: dict[str, object]) -> bytes:
    """Return the one declared canonical byte representation of a cap stage."""

    return (
        json.dumps(stage, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def continuation_cap_record_binding(
    stage: dict[str, object],
) -> dict[str, object]:
    """Content-address the immutable stage record embedded in the summary."""

    canonical = canonical_continuation_cap_stage(stage)
    return {
        "schema_version": CONTINUATION_CAP_RECORD_SCHEMA_VERSION,
        "canonicalization": CONTINUATION_CAP_CANONICALIZATION,
        "stage_sha256": f"sha256:{hashlib.sha256(canonical).hexdigest()}",
        "stage_bytes": len(canonical),
    }


def _continuation_checkpoint_lineage(
    resume_validation: dict[str, object] | None,
    operational_restart_validation: dict[str, object] | None,
) -> dict[str, object]:
    """Match the shared manifested-state lineage semantics without mutation."""

    if operational_restart_validation is not None:
        operational_lineage = operational_restart_validation.get("lineage")
        if not isinstance(operational_lineage, dict):
            operational_lineage = {}
        operational_status = operational_restart_validation.get("status")
        return {
            "status": (
                "verified"
                if operational_status == "verified"
                else "unsafe-unverified"
            ),
            "origin": "operational-restart",
            "root_origin": operational_lineage.get("root_origin"),
            "operational_restart_count": int(
                operational_lineage.get("operational_restart_count", 0)
            )
            + 1,
            "parent_manifest_sha256": operational_restart_validation.get(
                "manifest_sha256"
            ),
        }
    if resume_validation is None:
        return {"status": "verified", "origin": "fresh-run"}
    if resume_validation.get("status") == "verified":
        return {
            "status": "verified",
            "origin": "verified-resume",
            "parent_manifest_sha256": resume_validation.get(
                "manifest_sha256"
            ),
        }
    return {
        "status": "unsafe-unverified",
        "origin": "unsafe-resume",
        "reason": resume_validation.get("reason"),
    }


def write_continuation_cap_manifest(
    state_path: Path,
    *,
    solver_fingerprint: str,
    continuation_fingerprint: str,
    solver_parameters: dict[str, object],
    cap_record: dict[str, object],
    resume_validation: dict[str, object] | None = None,
    operational_restart_validation: dict[str, object] | None = None,
) -> Path:
    """Atomically publish a state manifest that binds its convergence record."""

    state = Path(state_path).resolve()
    if not state.is_file():
        raise FileNotFoundError(state)
    payload = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "manifest_kind": CHECKPOINT_MANIFEST_KIND,
        "producer": "pulse_continuation",
        "solver_semantics": SOLVER_SEMANTICS,
        "solver_source_fingerprint": solver_fingerprint,
        "producer_source_fingerprint": continuation_fingerprint,
        "solver_parameters": solver_parameters,
        "lineage": _continuation_checkpoint_lineage(
            resume_validation, operational_restart_validation
        ),
        "state": {
            "filename": state.name,
            "bytes": state.stat().st_size,
            "sha256": file_fingerprint(state),
        },
        "continuation_cap_record": cap_record,
    }
    manifest = checkpoint_manifest_path(state)
    atomic_write_json(manifest, payload)
    return manifest


def _strict_json_object(path: Path, *, description: str) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {token}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{description} {path} is unreadable: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} {path} is not an object")
    return value


def _validate_bound_continuation_stage(
    state_path: Path,
    manifest_path: Path,
    manifest_payload: dict[str, object],
    *,
    solver_fingerprint: str,
    continuation_fingerprint: str,
) -> dict[str, object]:
    """Verify that the cap manifest binds its published convergence stage."""

    binding = manifest_payload.get("continuation_cap_record")
    expected_keys = {
        "schema_version",
        "canonicalization",
        "stage_sha256",
        "stage_bytes",
    }
    if not isinstance(binding, dict) or set(binding) != expected_keys:
        raise ValueError(
            "continuation cap manifest lacks the canonical stage-record binding"
        )
    if (
        binding.get("schema_version")
        != CONTINUATION_CAP_RECORD_SCHEMA_VERSION
        or binding.get("canonicalization")
        != CONTINUATION_CAP_CANONICALIZATION
        or not isinstance(binding.get("stage_sha256"), str)
        or isinstance(binding.get("stage_bytes"), bool)
        or not isinstance(binding.get("stage_bytes"), int)
        or int(binding["stage_bytes"]) <= 0
    ):
        raise ValueError("continuation cap stage-record binding is malformed")

    summary_path = Path(state_path).resolve().parent.parent / (
        "continuation-summary.json"
    )
    summary = _strict_json_object(
        summary_path, description="continuation summary"
    )
    if summary.get("solver_semantics") != SOLVER_SEMANTICS:
        raise ValueError("continuation summary has different solver semantics")
    if summary.get("solver_source_fingerprint") != solver_fingerprint:
        raise ValueError("continuation summary has a different source fingerprint")
    if (
        summary.get("continuation_source_fingerprint")
        != continuation_fingerprint
    ):
        raise ValueError(
            "continuation summary has a different producer source fingerprint"
        )
    if summary.get("status") not in {"completed", "stopped"}:
        raise ValueError("continuation summary has no terminal publication status")
    stages = summary.get("stages")
    if not isinstance(stages, list):
        raise ValueError("continuation summary lacks stage records")
    matches: list[dict[str, object]] = []
    for item in stages:
        if not isinstance(item, dict):
            continue
        declared = item.get("checkpoint_manifest")
        if not isinstance(declared, str):
            continue
        declared_path = Path(declared)
        if not declared_path.is_absolute():
            declared_path = summary_path.parent / declared_path
        if declared_path.resolve() == manifest_path.resolve():
            matches.append(item)
    if len(matches) != 1:
        raise ValueError(
            "continuation summary does not uniquely declare the cap manifest"
        )
    stage = dict(matches[0])
    stage.pop("checkpoint_manifest")
    canonical = canonical_continuation_cap_stage(stage)
    actual_sha256 = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    if (
        len(canonical) != int(binding["stage_bytes"])
        or actual_sha256 != binding["stage_sha256"]
    ):
        raise ValueError(
            "continuation summary stage does not match its cap-manifest binding"
        )
    status = stage.get("status")
    sweeps = stage.get("sweeps")
    if status not in {"converged", "maximum-sweeps"} or not isinstance(
        sweeps, list
    ) or not sweeps:
        raise ValueError("bound continuation cap is not a usable terminal stage")
    last_sweep = sweeps[-1]
    if not isinstance(last_sweep, dict):
        raise ValueError("bound continuation cap has a malformed final sweep")
    solver_parameters = manifest_payload.get("solver_parameters")
    nonlinear = (
        solver_parameters.get("nonlinear_continuation")
        if isinstance(solver_parameters, dict)
        else None
    )
    if not isinstance(nonlinear, dict):
        raise ValueError("continuation manifest lacks convergence parameters")
    try:
        completed_sweep = int(last_sweep["sweep"])
        final_update = float(last_sweep["global_update"])
        minimum_sweeps = int(nonlinear["minimum_sweeps"])
        maximum_sweeps = int(nonlinear["maximum_sweeps"])
        tolerance = float(nonlinear["tolerance"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("bound continuation convergence record is malformed") from error
    if not math.isfinite(final_update) or final_update < 0.0:
        raise ValueError("bound continuation final update is invalid")
    if status == "converged" and not (
        completed_sweep >= minimum_sweeps and final_update <= tolerance
    ):
        raise ValueError(
            "cap is labeled converged but does not meet its bound tolerance"
        )
    if status == "maximum-sweeps" and completed_sweep != maximum_sweeps:
        raise ValueError(
            "maximum-sweeps cap does not end at the configured sweep limit"
        )
    require_convergence = nonlinear.get("require_convergence")
    if not isinstance(require_convergence, bool):
        raise ValueError("continuation convergence requirement is malformed")
    return {
        "path": str(summary_path.resolve()),
        "sha256": file_fingerprint(summary_path),
        "status": summary["status"],
        "cap_stage": int(stage["stage"]),
        "cap_status": status,
        "cap_record": dict(binding),
        "requires_cap_resolve": bool(
            status == "maximum-sweeps" and require_convergence
        ),
    }


def validate_scientific_resume(
    state_path: Path,
    *,
    continuation_parameters: dict[str, object],
    first_prefix_coordinates: CharacteristicLGLMesh,
    expected_first_v_endpoint: float,
    solver_fingerprint: str,
    continuation_fingerprint: str,
    unsafe_allow_unverified: bool = False,
) -> dict[str, object]:
    """Validate either an exact continuation cap or a Q1 warm-start state.

    Continuation-produced caps retain the exact-contract check.
    A completed ``pulse_campaign`` state is admitted only at the first cap and
    only after its verified manifest, state content hash, completed summary,
    source fingerprint, physical data, prefix mesh, angular discretization,
    and both construction-integrator contracts all match.  It is a warm start
    and is never treated as proof that the continuation convergence tolerance
    has already been met.
    """

    state = Path(state_path).resolve()
    manifest = checkpoint_manifest_path(state)
    if not manifest.is_file():
        # Preserve the existing diagnostic-only established escape for continuation
        # caps.  A producer cannot be safely inferred without a manifest, so
        # this path is never eligible for the pulse_campaign import exception.
        validation = validate_checkpoint_manifest(
            state,
            producer="pulse_continuation",
            producer_source_fingerprint=continuation_fingerprint,
            solver_parameters=continuation_parameters,
            unsafe_allow_unverified=unsafe_allow_unverified,
            solver_source_fingerprint_value=solver_fingerprint,
        )
        return {**validation, "source_producer": "established-unknown"}

    manifest_path, payload = _read_resume_manifest(state)
    producer = payload.get("producer")
    if producer == "pulse_continuation":
        validation = validate_checkpoint_manifest(
            state,
            producer="pulse_continuation",
            producer_source_fingerprint=continuation_fingerprint,
            solver_parameters=continuation_parameters,
            unsafe_allow_unverified=unsafe_allow_unverified,
            solver_source_fingerprint_value=solver_fingerprint,
        )
        cap_summary = _validate_bound_continuation_stage(
            state,
            manifest_path,
            payload,
            solver_fingerprint=solver_fingerprint,
            continuation_fingerprint=continuation_fingerprint,
        )
        return {
            **validation,
            "source_producer": producer,
            "source_summary": cap_summary,
            "warm_start_requires_continuation_convergence": bool(
                cap_summary["requires_cap_resolve"]
            ),
        }
    if producer != "pulse_campaign":
        raise ValueError(
            "scientific resume producer must be pulse_continuation or "
            f"pulse_campaign; received {producer!r}"
        )

    source_parameters = payload.get("solver_parameters")
    source_boundary = (
        source_parameters.get("boundary")
        if isinstance(source_parameters, dict)
        else None
    )
    source_endpoint = (
        source_boundary.get("v_endpoint")
        if isinstance(source_boundary, dict)
        else None
    )
    if isinstance(source_endpoint, bool) or not isinstance(
        source_endpoint, (int, float)
    ):
        raise ValueError("pulse_campaign manifest lacks a numeric v_endpoint")
    source_endpoint = float(source_endpoint)
    if not math.isfinite(source_endpoint) or not math.isclose(
        source_endpoint,
        float(expected_first_v_endpoint),
        rel_tol=0.0,
        abs_tol=2.0e-14,
    ):
        raise ValueError(
            "pulse_campaign state is not at the first continuation cap: "
            f"expected {expected_first_v_endpoint:.17g}, "
            f"received {source_endpoint:.17g}"
        )
    q1_parameters = q1_import_solver_parameters(
        continuation_parameters,
        first_prefix_coordinates,
        v_endpoint=source_endpoint,
    )
    # Deliberately do not pass the unsafe switch: cross-producer import is
    # available only for a fully verified pulse_campaign lineage.
    validation = validate_checkpoint_manifest(
        state,
        producer="pulse_campaign",
        producer_source_fingerprint=solver_fingerprint,
        solver_parameters=q1_parameters,
        unsafe_allow_unverified=False,
        solver_source_fingerprint_value=solver_fingerprint,
    )
    completed_summary = _validate_completed_q1_summary(
        state,
        manifest_path,
        solver_fingerprint=solver_fingerprint,
    )
    return {
        **validation,
        "source_producer": producer,
        "source_summary": completed_summary,
        "import_mode": "verified-q1-warm-start",
        "warm_start_requires_continuation_convergence": True,
        "compatibility": {
            "first_cap_v_endpoint": source_endpoint,
            "prefix_mesh": first_prefix_coordinates.diagnostics(),
            "checked_contract_sections": [
                "boundary",
                "sphere",
                "galerkin",
                "scalar_coordinates",
                "metric_construction",
                "incoming_construction",
            ],
        },
    }


def continuation_resume_solver_parameters(
    args: argparse.Namespace,
    full_coordinates: CharacteristicLGLMesh,
    continuation_indices: list[int],
) -> dict[str, object]:
    """Return the complete discrete map/globalization contract for resume."""

    return {
        "boundary": {
            "c": args.c,
            "shear_divisor": args.shear_divisor,
            "mode": "low-band-hemisphere",
            "v1": args.v1,
        },
        "sphere": {
            "points": args.points,
            "neighbors": args.neighbors,
            "angular_degree": args.angular_degree,
            "spectral_degree": args.spectral_degree,
        },
        "galerkin": {
            "retained_degree": args.galerkin_retained_degree,
            "work_degree": args.galerkin_work_degree,
        },
        "scalar_coordinates": {
            "mesh": full_coordinates.diagnostics(),
            "continuation_breakpoint_indices": continuation_indices,
        },
        "metric_construction": {
            "substeps": args.metric_substeps,
            "parameterization": args.metric_parameterization,
            "integrator": args.metric_integrator,
            "sdc_tolerance": args.sdc_tolerance,
            "sdc_overgrid_tolerance": args.sdc_overgrid_tolerance,
            "sdc_maximum_corrections": args.sdc_maximum_corrections,
        },
        "incoming_construction": {
            "integrator": args.u_integrator,
            "sdc_tolerance": args.u_sdc_tolerance,
            "sdc_overgrid_tolerance": args.u_sdc_overgrid_tolerance,
            "sdc_maximum_corrections": args.u_sdc_maximum_corrections,
            "sdc_half_trace_tolerance": args.u_sdc_half_trace_tolerance,
            "sdc_half_trace_absolute_floor": (
                args.u_sdc_half_trace_absolute_floor
            ),
        },
        "nonlinear_continuation": {
            "prolongation_predictor": args.continuation_predictor,
            "minimum_sweeps": args.minimum_sweeps,
            "maximum_sweeps": args.maximum_sweeps,
            "tolerance": args.tolerance,
            "active_update_floor": args.active_update_floor,
            "require_convergence": args.require_convergence,
        },
    }


def continuation_operational_run_contract(
    args: argparse.Namespace,
    solver_parameters: dict[str, object],
) -> dict[str, object]:
    return {
        "driver": "pulse_continuation",
        "output_label": args.output_label,
        "solver_parameters": solver_parameters,
        "controller": {
            "stop_after_v": args.stop_after_v,
            "u_halo": args.u_halo,
            "continuation_predictor": args.continuation_predictor,
            "checkpoint_every_sweep": args.checkpoint_every_sweep,
            "checkpoint_retention_generations": (
                args.checkpoint_retain_generations
            ),
        },
    }


def sliced_boundary(
    boundary: dict[str, Array | float], full_count: int, count: int
) -> dict[str, Array | float]:
    result: dict[str, Array | float] = {}
    for name, value in boundary.items():
        if (
            isinstance(value, np.ndarray)
            and value.ndim >= 2
            and value.shape[1] == full_count
        ):
            result[name] = value[:, :count].copy()
        else:
            result[name] = value
    return result


OUTGOING_STATE_BOUNDARY_SPECIFICATIONS = (
    ("metric", "metric", None, True),
    ("q", "expansion", None, True),
    ("shear", "shear", None, True),
    ("omega", "omega", 1.0, True),
    ("weighted_omega", "weighted_omega", 0.0, True),
    ("weighted_omegab", "weighted_omegab", None, False),
    ("weighted_chib", "weighted_chib", None, False),
    ("zeta_up", "zeta_up", None, False),
    ("shift", "shift", None, False),
)


def validate_exact_outgoing_boundary(
    state: FirstOrderState,
    boundary: dict[str, Array | float],
    *,
    description: str,
) -> dict[str, bool]:
    """Validate an operational checkpoint without mutating its state.

    ``impose_outgoing_boundary`` deliberately mutates its argument.  Applying
    it directly to a loaded checkpoint and then comparing the returned object
    with that checkpoint is therefore both vacuous (the two names alias) and
    destructive.  Operational restart requires exact reproducibility, so
    compare only the constrained H_-1 traces with the regenerated targets.
    This also avoids copying the potentially multi-gigabyte interior state.
    """

    exact_by_field: dict[str, bool] = {}
    for state_name, boundary_name, default, always_imposed in (
        OUTGOING_STATE_BOUNDARY_SPECIFICATIONS
    ):
        actual = getattr(state, state_name)[:, 0]
        if boundary_name in boundary:
            supplied = np.asarray(boundary[boundary_name])
        elif always_imposed and default is not None:
            supplied = np.asarray(default)
        elif always_imposed:
            raise ValueError(
                f"{description} boundary is missing required field "
                f"{boundary_name!r}"
            )
        else:
            continue
        try:
            target = np.broadcast_to(supplied, actual.shape)
        except ValueError as error:
            raise ValueError(
                f"{description} boundary field {boundary_name!r} has an "
                "invalid shape"
            ) from error
        exact_by_field[state_name] = bool(np.array_equal(actual, target))
    changed = [name for name, exact in exact_by_field.items() if not exact]
    if changed:
        raise ValueError(
            f"{description} boundary mismatch in fields " + ", ".join(changed)
        )
    return exact_by_field


def canonicalize_continuation_boundary(
    old: FirstOrderState,
    grid: PointSphereGrid,
    boundary: dict[str, Array | float],
    v_count: int,
) -> tuple[dict[str, Array | float], dict[str, object]]:
    """Make the accepted H_-1 prefix authoritative for one stage.

    Solving the same outgoing ODE on a longer composite s mesh can change old
    prefix values by roundoff.  A Picard step calls ``impose_outgoing_boundary``
    internally, so merely preserving the prolonged state is insufficient: the
    stage must use one canonical boundary whose old columns are copied from
    the accepted state and whose new columns are the freshly solved data.
    Material recomputation changes fail closed; roundoff-sized changes are
    recorded and canonicalized.  The inverse prefix is recomputed from the
    accepted metric rather than copied from a possibly stale boundary array.
    """

    old_count = old.q.shape[2]
    if not 0 < old_count <= v_count:
        raise ValueError("canonical boundary has an invalid accepted prefix")
    result: dict[str, Array | float] = {
        name: value.copy() if isinstance(value, np.ndarray) else value
        for name, value in boundary.items()
    }
    mismatch_by_field: dict[str, float] = {}
    tolerance_by_field: dict[str, float] = {}
    new_tail_exact_by_field: dict[str, bool] = {}
    prefix_bitwise_by_field: dict[str, bool] = {}
    canonical_values: dict[str, Array] = {}
    for state_name, boundary_name, default, always_imposed in (
        OUTGOING_STATE_BOUNDARY_SPECIFICATIONS
    ):
        accepted = getattr(old, state_name)[:, 0]
        target_shape = (grid.count, v_count, *accepted.shape[2:])
        if boundary_name in boundary:
            supplied = np.asarray(boundary[boundary_name], dtype=float)
        elif always_imposed and default is not None:
            supplied = np.asarray(default, dtype=float)
        elif always_imposed:
            raise ValueError(
                f"outgoing boundary is missing required field {boundary_name!r}"
            )
        else:
            continue
        try:
            recomputed = np.broadcast_to(supplied, target_shape).copy()
        except ValueError as error:
            raise ValueError(
                f"outgoing boundary field {boundary_name!r} has an invalid shape"
            ) from error
        mismatch = float(
            np.max(np.abs(recomputed[:, :old_count] - accepted))
        )
        scale = max(
            1.0,
            float(np.max(np.abs(recomputed[:, :old_count]))),
            float(np.max(np.abs(accepted))),
        )
        tolerance = 256.0 * np.finfo(float).eps * scale
        if mismatch > tolerance:
            raise ValueError(
                f"recomputed outgoing field {boundary_name!r} changed the "
                "accepted prefix: maximum mismatch="
                f"{mismatch:.6g}, roundoff tolerance={tolerance:.6g}"
            )
        new_tail = recomputed[:, old_count:].copy()
        recomputed[:, :old_count] = accepted
        result[boundary_name] = recomputed
        canonical_values[boundary_name] = recomputed
        mismatch_by_field[state_name] = mismatch
        tolerance_by_field[state_name] = tolerance
        new_tail_exact_by_field[state_name] = bool(
            np.array_equal(recomputed[:, old_count:], new_tail)
        )
        prefix_bitwise_by_field[state_name] = bool(
            np.array_equal(recomputed[:, :old_count], accepted)
        )

    accepted_inverse = tangent_inverse(grid, old.metric[:, 0])
    if "inverse" in boundary:
        try:
            recomputed_inverse = np.broadcast_to(
                np.asarray(boundary["inverse"], dtype=float),
                (grid.count, v_count, 3, 3),
            ).copy()
        except ValueError as error:
            raise ValueError("outgoing inverse has an invalid shape") from error
    else:
        recomputed_inverse = tangent_inverse(
            grid, np.asarray(canonical_values["metric"])
        )
    inverse_mismatch = float(
        np.max(
            np.abs(
                recomputed_inverse[:, :old_count] - accepted_inverse
            )
        )
    )
    inverse_scale = max(
        1.0,
        float(np.max(np.abs(recomputed_inverse[:, :old_count]))),
        float(np.max(np.abs(accepted_inverse))),
    )
    inverse_tolerance = 256.0 * np.finfo(float).eps * inverse_scale
    if inverse_mismatch > inverse_tolerance:
        raise ValueError(
            "recomputed outgoing inverse changed the accepted metric inverse: "
            f"maximum mismatch={inverse_mismatch:.6g}, roundoff tolerance="
            f"{inverse_tolerance:.6g}"
        )
    inverse_new_tail = recomputed_inverse[:, old_count:].copy()
    recomputed_inverse[:, :old_count] = accepted_inverse
    result["inverse"] = recomputed_inverse
    mismatch_by_field["inverse"] = inverse_mismatch
    tolerance_by_field["inverse"] = inverse_tolerance
    new_tail_exact_by_field["inverse"] = bool(
        np.array_equal(
            recomputed_inverse[:, old_count:], inverse_new_tail
        )
    )
    prefix_bitwise_by_field["inverse"] = bool(
        np.array_equal(
            recomputed_inverse[:, :old_count], accepted_inverse
        )
    )
    diagnostics: dict[str, object] = {
        "semantics": (
            "accepted H_-1 prefix is bitwise authoritative; recomputed "
            "new-v tail is exact; inverse prefix is recomputed from the "
            "accepted metric"
        ),
        "old_v_count": old_count,
        "stage_v_count": v_count,
        "recomputed_prefix_maximum_mismatch": max(
            mismatch_by_field.values(), default=0.0
        ),
        "recomputed_prefix_mismatch_by_field": mismatch_by_field,
        "recomputed_prefix_tolerance_by_field": tolerance_by_field,
        "prefix_bitwise_by_field": prefix_bitwise_by_field,
        "prefix_bitwise_preserved": all(prefix_bitwise_by_field.values()),
        "new_tail_exact_by_field": new_tail_exact_by_field,
        "new_tail_exact": all(new_tail_exact_by_field.values()),
    }
    return result, diagnostics


def _prolongation_trace_diagnostics(
    grid: PointSphereGrid,
    angular: AngularGalerkin,
    scalar_coordinates: CharacteristicLGLMesh,
    raw_tail_shear: Array,
    state: FirstOrderState,
    old_count: int,
    *,
    absolute_floor: float,
    retained_node_tolerance: float,
    retained_overgrid_tolerance: float,
    overgrid_extra_degree: int = 3,
) -> dict[str, object]:
    """Audit the new-panel shear before the first Picard construction.

    The raw nodal statistic shows how far the additive predictor leaves the
    moving trace-free manifold.  The hard checks are the unprojected
    pointwise trace at nodes and on the unretracted independent tau overgrid.
    Retained harmonic traces and the trace after an explicit pointwise
    retraction are recorded only as secondary diagnostics; neither is allowed
    to hide interpolation drift in the stored exact trace-free field.
    """

    if scalar_coordinates.u.shape != state.q.shape[1:2] or not np.array_equal(
        scalar_coordinates.u, -np.exp(-scalar_coordinates.tau.nodes)
    ):
        raise ValueError("predictor trace audit has incompatible tau scalar_coordinates")
    if not 0 < old_count < len(scalar_coordinates.v):
        raise ValueError("predictor trace audit requires a nonempty new v panel")
    metric = state.metric[:, :, old_count:]
    shear = state.shear[:, :, old_count:]
    if raw_tail_shear.shape != shear.shape:
        raise ValueError("raw predictor shear has the wrong new-panel shape")

    node_inverse = tangent_inverse(grid, metric)
    node_norm = np.sqrt(
        np.maximum(tensor_norm_sq(shear, node_inverse), 0.0)
    )
    node_trace = tensor_trace(shear, node_inverse)
    retained_node_trace = angular.project_scalar(node_trace)
    raw_norm = np.sqrt(
        np.maximum(tensor_norm_sq(raw_tail_shear, node_inverse), 0.0)
    )
    raw_trace = tensor_trace(raw_tail_shear, node_inverse)
    raw_node = surface_l2_relative_trace_defect(
        np.moveaxis(raw_trace, 1, 0),
        np.moveaxis(raw_norm, 1, 0),
        absolute_floor=absolute_floor,
    )
    pointwise_node = surface_l2_relative_trace_defect(
        np.moveaxis(node_trace, 1, 0),
        np.moveaxis(node_norm, 1, 0),
        absolute_floor=absolute_floor,
    )
    retained_node = surface_l2_relative_trace_defect(
        np.moveaxis(retained_node_trace, 1, 0),
        np.moveaxis(node_norm, 1, 0),
        absolute_floor=absolute_floor,
    )

    pointwise_overgrid_traces: list[Array] = []
    retained_overgrid_traces: list[Array] = []
    overgrid_tensor_norms: list[Array] = []
    retracted_overgrid_traces: list[Array] = []
    retracted_overgrid_tensor_norms: list[Array] = []
    overgrid_metric_minimum = math.inf
    overgrid_lapse_minimum = math.inf
    lapse = state.omega[:, :, old_count:]
    for segment, indices in zip(
        scalar_coordinates.tau.segments,
        scalar_coordinates.tau.indices,
        strict=True,
    ):
        over = LGLSegment.create(
            segment.left,
            segment.right,
            segment.degree + overgrid_extra_degree,
        )
        interpolation = segment.interpolation_matrix(over.nodes)
        local_metric = np.moveaxis(metric[:, indices], 1, 0)
        local_shear = np.moveaxis(shear[:, indices], 1, 0)
        local_lapse = np.moveaxis(lapse[:, indices], 1, 0)
        over_metric = np.tensordot(interpolation, local_metric, axes=(1, 0))
        over_shear = np.tensordot(interpolation, local_shear, axes=(1, 0))
        over_lapse = np.tensordot(interpolation, local_lapse, axes=(1, 0))
        sample_lapse_minimum = float(np.min(over_lapse))
        overgrid_lapse_minimum = min(
            overgrid_lapse_minimum, sample_lapse_minimum
        )
        if sample_lapse_minimum <= 0.0:
            raise FloatingPointError(
                "boundary-lift predictor has a nonpositive tau-overgrid "
                f"lapse {sample_lapse_minimum:.6g}"
            )
        pointwise_samples: list[Array] = []
        retained_samples: list[Array] = []
        norm_samples: list[Array] = []
        retracted_trace_samples: list[Array] = []
        retracted_norm_samples: list[Array] = []
        for sample_metric, sample_shear in zip(
            over_metric, over_shear, strict=True
        ):
            local_metric = np.einsum(
                "nia,n...ij,njb->n...ab",
                grid.frames,
                sample_metric,
                grid.frames,
            )
            sample_metric_minimum = float(
                np.min(np.linalg.eigvalsh(local_metric))
            )
            overgrid_metric_minimum = min(
                overgrid_metric_minimum, sample_metric_minimum
            )
            if sample_metric_minimum <= 0.0:
                raise FloatingPointError(
                    "boundary-lift predictor has a nonpositive tau-overgrid "
                    f"metric eigenvalue {sample_metric_minimum:.6g}"
                )
            inverse = tangent_inverse(grid, sample_metric)
            trace = tensor_trace(sample_shear, inverse)
            norm = np.sqrt(
                np.maximum(tensor_norm_sq(sample_shear, inverse), 0.0)
            )
            pointwise_samples.append(trace)
            retained_samples.append(angular.project_scalar(trace))
            norm_samples.append(norm)
            retracted_shear = tensor_tracefree(
                sample_shear, sample_metric, inverse
            )
            retracted_trace_samples.append(
                tensor_trace(retracted_shear, inverse)
            )
            retracted_norm_samples.append(
                np.sqrt(
                    np.maximum(
                        tensor_norm_sq(retracted_shear, inverse), 0.0
                    )
                )
            )
        pointwise_overgrid_traces.append(np.stack(pointwise_samples))
        retained_overgrid_traces.append(np.stack(retained_samples))
        overgrid_tensor_norms.append(np.stack(norm_samples))
        retracted_overgrid_traces.append(np.stack(retracted_trace_samples))
        retracted_overgrid_tensor_norms.append(
            np.stack(retracted_norm_samples)
        )

    def combined_overgrid_defect(
        traces: list[Array], norms: list[Array]
    ) -> tuple[float, Array]:
        evaluated = [
            surface_l2_relative_trace_defect(
                trace,
                norm,
                absolute_floor=absolute_floor,
            )
            for trace, norm in zip(traces, norms, strict=True)
        ]
        return (
            max((item.maximum for item in evaluated), default=0.0),
            np.maximum.reduce([item.by_v for item in evaluated]),
        )

    pointwise_overgrid, pointwise_overgrid_by_v = combined_overgrid_defect(
        pointwise_overgrid_traces, overgrid_tensor_norms
    )
    retained_overgrid, retained_overgrid_by_v = combined_overgrid_defect(
        retained_overgrid_traces, overgrid_tensor_norms
    )
    retracted_overgrid, retracted_overgrid_by_v = combined_overgrid_defect(
        retracted_overgrid_traces, retracted_overgrid_tensor_norms
    )
    pointwise_resolved = bool(
        pointwise_node.maximum <= retained_node_tolerance
        and pointwise_overgrid <= retained_overgrid_tolerance
    )
    return {
        "trace_schema": "continuation-predictor-pointwise-v1",
        "trace_norm": "surface-L2-before-relative-division",
        "raw_trace_pointwise_nodes_defect": raw_node.maximum,
        "raw_trace_pointwise_nodes_defect_by_v": raw_node.by_v.tolist(),
        "trace_pointwise_nodes_defect": pointwise_node.maximum,
        "trace_pointwise_nodes_defect_by_v": pointwise_node.by_v.tolist(),
        "trace_pointwise_nodes_tolerance": retained_node_tolerance,
        "trace_pointwise_tau_overgrid_defect": pointwise_overgrid,
        "trace_pointwise_tau_overgrid_defect_by_v": (
            pointwise_overgrid_by_v.tolist()
        ),
        "trace_pointwise_tau_overgrid_tolerance": (
            retained_overgrid_tolerance
        ),
        "trace_pointwise_resolved": pointwise_resolved,
        # These are supporting diagnostics only.  In particular, the
        # post-retraction trace is circular and is not an angular certificate.
        "trace_retained_nodes_defect": retained_node.maximum,
        "trace_retained_nodes_defect_by_v": retained_node.by_v.tolist(),
        "trace_retained_tau_overgrid_defect": retained_overgrid,
        "trace_retained_tau_overgrid_defect_by_v": (
            retained_overgrid_by_v.tolist()
        ),
        "trace_post_retraction_tau_overgrid_defect": retracted_overgrid,
        "trace_post_retraction_tau_overgrid_defect_by_v": (
            retracted_overgrid_by_v.tolist()
        ),
        "post_retraction_is_not_angular_certificate": True,
        "new_v_count": len(scalar_coordinates.v) - old_count,
        "projection": "new-tail pointwise tensor_tracefree",
        "tau_overgrid_minimum_metric_eigenvalue": overgrid_metric_minimum,
        "tau_overgrid_minimum_lapse": overgrid_lapse_minimum,
    }


def prolonged_state(
    old: FirstOrderState,
    grid: PointSphereGrid,
    u: Array,
    v: Array,
    boundary: dict[str, Array | float],
    *,
    predictor: str = "constant",
    angular: AngularGalerkin | None = None,
    scalar_coordinates: CharacteristicLGLMesh | None = None,
    trace_absolute_floor: float = 1.0e-14,
    trace_retained_node_tolerance: float = 1.0e-9,
    trace_retained_overgrid_tolerance: float = 1.0e-7,
) -> tuple[FirstOrderState, dict[str, object]]:
    """Copy a converged prefix and predict the newly appended causal panel.

    ``constant`` uses a constant-tail predictor.  ``boundary-lift``
    reuses :func:`boundary_compatible_initial_state` as a smooth transfinite
    lift and applies only its increment from the accepted endpoint ``v_a``:

    ``F_new(u,v) = F_old(u,v_a) + F_lift(u,v) - F_lift(u,v_a)``.

    Thus the accepted prefix is never extrapolated or altered, while the new
    outgoing trace is introduced across all u nodes instead of as an
    endpoint-cardinal mode on the first u element.
    """

    template = initial_state(grid, u, v)
    old_count = old.q.shape[2]
    if old_count >= len(v):
        raise ValueError("continuation must add at least one v node")
    if predictor not in {"constant", "boundary-lift"}:
        raise ValueError(f"unknown continuation predictor {predictor!r}")
    lift = (
        None
        if predictor == "constant"
        else boundary_compatible_initial_state(grid, u, v, boundary)
    )
    values: dict[str, Array] = {}
    tail_increment_by_field: dict[str, float] = {}
    for description in fields(FirstOrderState):
        name = description.name
        previous = getattr(old, name)
        current = getattr(template, name)
        if previous.shape[:2] != current.shape[:2] or (
            previous.shape[3:] != current.shape[3:]
        ):
            raise ValueError(
                f"continuation field {name!r} is incompatible with the target mesh"
            )
        if previous.shape[2] != old_count:
            raise ValueError(
                f"continuation field {name!r} has an inconsistent v prefix"
            )
        current[:, :, :old_count] = previous
        tail = previous[:, :, -1:]
        if lift is None:
            predicted = np.broadcast_to(
                tail, current[:, :, old_count:].shape
            )
        else:
            lifted = getattr(lift, name)
            predicted = (
                tail
                + lifted[:, :, old_count:]
                - lifted[:, :, old_count - 1 : old_count]
            )
        current[:, :, old_count:] = predicted
        tail_increment_by_field[name] = float(
            np.max(np.abs(current[:, :, old_count:] - tail))
        )
        values[name] = current
    raw_tail_shear = values["shear"][:, :, old_count:].copy()
    prolonged = FirstOrderState(**values)
    if predictor == "boundary-lift":
        if angular is None or scalar_coordinates is None:
            raise ValueError(
                "the boundary-lift predictor requires angular and coordinate "
                "objects for its trace-free projection certificate"
            )
        if not np.array_equal(scalar_coordinates.u, u) or not np.array_equal(
            scalar_coordinates.v, v
        ):
            raise ValueError(
                "the continuation predictor scalar_coordinates do not match u and v"
            )
        tail_inverse = tangent_inverse(
            grid, prolonged.metric[:, :, old_count:]
        )
        prolonged.shear[:, :, old_count:] = tensor_tracefree(
            prolonged.shear[:, :, old_count:],
            prolonged.metric[:, :, old_count:],
            tail_inverse,
        )
    boundary_mismatch_by_field: dict[str, float] = {}
    boundary_exact_by_field: dict[str, bool] = {}
    new_boundary_exact_by_field: dict[str, bool] = {}
    recomputed_prefix_mismatch_by_field: dict[str, float] = {}
    boundary_restoration_by_field: dict[str, float] = {}
    boundary_restoration_tolerance_by_field: dict[str, float] = {}
    for state_name, boundary_name, default, always_imposed in (
        OUTGOING_STATE_BOUNDARY_SPECIFICATIONS
    ):
        actual = getattr(prolonged, state_name)[:, 0]
        if boundary_name in boundary:
            target = np.asarray(boundary[boundary_name])
        elif always_imposed and default is not None:
            target = np.asarray(default)
        elif always_imposed:
            raise ValueError(
                f"outgoing boundary is missing required field {boundary_name!r}"
            )
        else:
            continue
        try:
            recomputed = np.broadcast_to(target, actual.shape)
        except ValueError as error:
            raise ValueError(
                f"outgoing boundary field {boundary_name!r} has an invalid shape"
            ) from error
        accepted_prefix = getattr(old, state_name)[:, 0]
        prefix_mismatch = float(
            np.max(
                np.abs(
                    recomputed[:, :old_count] - accepted_prefix
                )
            )
        )
        prefix_scale = max(
            1.0,
            float(np.max(np.abs(recomputed[:, :old_count]))),
            float(np.max(np.abs(accepted_prefix))),
        )
        prefix_tolerance = 256.0 * np.finfo(float).eps * prefix_scale
        if prefix_mismatch > prefix_tolerance:
            raise ValueError(
                f"recomputed outgoing field {boundary_name!r} changed the "
                "accepted boundary prefix: maximum mismatch="
                f"{prefix_mismatch:.6g}, roundoff tolerance="
                f"{prefix_tolerance:.6g}"
            )
        recomputed_prefix_mismatch_by_field[state_name] = prefix_mismatch
        # The converged prefix is authoritative.  Impose the recomputed exact
        # outgoing data only on the appended v panel, so harmless differences
        # from evaluating the boundary solve on a larger composite mesh cannot
        # mutate accepted bits.
        pre_boundary_tail = actual[:, old_count:].copy()
        actual[:, old_count:] = recomputed[:, old_count:]
        restoration = float(
            np.max(np.abs(actual[:, old_count:] - pre_boundary_tail))
        )
        restoration_scale = max(
            1.0,
            float(np.max(np.abs(actual[:, old_count:]))),
            float(np.max(np.abs(pre_boundary_tail))),
        )
        restoration_tolerance = (
            512.0 * np.finfo(float).eps * restoration_scale
        )
        boundary_restoration_by_field[state_name] = restoration
        boundary_restoration_tolerance_by_field[state_name] = (
            restoration_tolerance
        )
        if predictor == "boundary-lift" and restoration > (
            restoration_tolerance
        ):
            raise ValueError(
                f"boundary-lift field {state_name!r} requires a material "
                "post-predictor boundary restoration: maximum correction="
                f"{restoration:.6g}, roundoff tolerance="
                f"{restoration_tolerance:.6g}"
            )
        canonical = np.array(recomputed, copy=True)
        canonical[:, :old_count] = accepted_prefix
        boundary_mismatch_by_field[state_name] = float(
            np.max(np.abs(actual - canonical))
        )
        boundary_exact_by_field[state_name] = bool(
            np.array_equal(actual, canonical)
        )
        new_boundary_exact_by_field[state_name] = bool(
            np.array_equal(
                actual[:, old_count:], recomputed[:, old_count:]
            )
        )
    boundary_exact = all(boundary_exact_by_field.values())
    if not boundary_exact:
        changed = [
            name for name, exact in boundary_exact_by_field.items() if not exact
        ]
        raise ValueError(
            "continuation predictor failed the exact outgoing boundary in fields "
            + ", ".join(changed)
        )

    prefix_by_field = {
        description.name: bool(
            np.array_equal(
                getattr(prolonged, description.name)[:, :, :old_count],
                getattr(old, description.name),
            )
        )
        for description in fields(FirstOrderState)
    }
    prefix_preserved = all(prefix_by_field.values())
    if not prefix_preserved:
        changed = [name for name, same in prefix_by_field.items() if not same]
        raise ValueError(
            "continuation predictor changed the accepted prefix in fields "
            + ", ".join(changed)
        )

    nonfinite = [
        description.name
        for description in fields(FirstOrderState)
        if not np.all(np.isfinite(getattr(prolonged, description.name)))
    ]
    if nonfinite:
        raise FloatingPointError(
            "continuation predictor produced nonfinite fields "
            + ", ".join(nonfinite)
        )
    metric_minimum = minimum_metric_eigenvalue(grid, prolonged)
    lapse_minimum = float(np.min(prolonged.omega))
    q_minimum = float(np.min(prolonged.q))
    # The geometric primitive cone requires an SPD section metric and positive
    # lapse.  q=Omega^{-1} tr(chi) is deliberately *not* sign constrained:
    # trapped-surface formation requires it to cross through zero.
    if metric_minimum <= 0.0 or lapse_minimum <= 0.0:
        raise FloatingPointError(
            "continuation predictor left the positive cone: "
            f"metric={metric_minimum:.6g}, lapse={lapse_minimum:.6g}"
        )
    trace_diagnostics: dict[str, object] | None = None
    if angular is not None and scalar_coordinates is not None:
        trace_diagnostics = _prolongation_trace_diagnostics(
            grid,
            angular,
            scalar_coordinates,
            raw_tail_shear,
            prolonged,
            old_count,
            absolute_floor=trace_absolute_floor,
            retained_node_tolerance=trace_retained_node_tolerance,
            retained_overgrid_tolerance=(
                trace_retained_overgrid_tolerance
            ),
        )
        if predictor == "boundary-lift" and not (
            trace_diagnostics["trace_pointwise_resolved"]
            and trace_diagnostics[
                "tau_overgrid_minimum_metric_eigenvalue"
            ]
            > 0.0
            and trace_diagnostics["tau_overgrid_minimum_lapse"] > 0.0
        ):
            raise ValueError(
                "boundary-lift predictor failed its pre-Picard trace "
                "certificate: pointwise_nodes="
                f"{trace_diagnostics['trace_pointwise_nodes_defect']:.6g}, "
                "pointwise_tau_overgrid="
                f"{trace_diagnostics['trace_pointwise_tau_overgrid_defect']:.6g}, "
                "tau_overgrid_metric="
                f"{trace_diagnostics['tau_overgrid_minimum_metric_eigenvalue']:.6g}, "
                "tau_overgrid_lapse="
                f"{trace_diagnostics['tau_overgrid_minimum_lapse']:.6g}"
            )
    tail_increment_by_field["shear"] = float(
        np.max(
            np.abs(
                prolonged.shear[:, :, old_count:]
                - old.shear[:, :, -1:]
            )
        )
    )
    diagnostics: dict[str, object] = {
        "applied": True,
        "predictor": predictor,
        "old_v_count": old_count,
        "new_v_count": len(v),
        "added_v_count": len(v) - old_count,
        "old_v_endpoint": float(v[old_count - 1]),
        "new_v_endpoint": float(v[-1]),
        "prefix_bitwise_preserved": prefix_preserved,
        "prefix_bitwise_by_field": prefix_by_field,
        "outgoing_boundary_exact": boundary_exact,
        "outgoing_boundary_exact_by_field": boundary_exact_by_field,
        "outgoing_boundary_maximum_mismatch": max(
            boundary_mismatch_by_field.values(), default=0.0
        ),
        "outgoing_boundary_mismatch_by_field": boundary_mismatch_by_field,
        "new_outgoing_boundary_exact_by_field": (
            new_boundary_exact_by_field
        ),
        "recomputed_outgoing_prefix_maximum_mismatch": max(
            recomputed_prefix_mismatch_by_field.values(), default=0.0
        ),
        "recomputed_outgoing_prefix_mismatch_by_field": (
            recomputed_prefix_mismatch_by_field
        ),
        "post_predictor_boundary_restoration_maximum": max(
            boundary_restoration_by_field.values(), default=0.0
        ),
        "post_predictor_boundary_restoration_by_field": (
            boundary_restoration_by_field
        ),
        "post_predictor_boundary_restoration_tolerance_by_field": (
            boundary_restoration_tolerance_by_field
        ),
        "finite": True,
        "minimum_metric_eigenvalue": metric_minimum,
        "tau_overgrid_minimum_metric_eigenvalue": (
            None
            if trace_diagnostics is None
            else trace_diagnostics[
                "tau_overgrid_minimum_metric_eigenvalue"
            ]
        ),
        "tau_overgrid_minimum_lapse": (
            None
            if trace_diagnostics is None
            else trace_diagnostics["tau_overgrid_minimum_lapse"]
        ),
        "minimum_lapse": lapse_minimum,
        "minimum_q": q_minimum,
        "tail_increment_maximum_by_field": tail_increment_by_field,
        "trace": trace_diagnostics,
    }
    return prolonged, diagnostics


def minimum_metric_eigenvalue(
    grid: PointSphereGrid, state: FirstOrderState
) -> float:
    local = np.einsum(
        "nia,n...ij,njb->n...ab",
        grid.frames,
        state.metric,
        grid.frames,
    )
    return float(np.min(np.linalg.eigvalsh(local)))


def active_contraction_summary(
    current_update: Array,
    previous_update: Array,
    *,
    u_halo: int,
    active_update_floor: float,
) -> dict[str, float | int]:
    """Summarize contraction only where at least one update is resolved.

    Ratios of two roundoff-sized numbers have no numerical meaning and used
    to dominate the reported median and maximum.  A cell is active when the
    current or previous update exceeds one declared absolute floor.  Growth
    from below the floor remains visible because the denominator is clipped
    at the same floor; cells below it on both sweeps are excluded entirely.
    """

    current = np.abs(np.asarray(current_update, dtype=float))
    previous = np.abs(np.asarray(previous_update, dtype=float))
    if current.shape != previous.shape or current.ndim != 2:
        raise ValueError("local update maps must have one identical (u,v) shape")
    if (
        not math.isfinite(active_update_floor)
        or active_update_floor < 0.0
    ):
        raise ValueError("active update floor must be finite and nonnegative")
    if u_halo < 0 or 2 * u_halo >= current.shape[0]:
        raise ValueError("u halo must leave at least one contraction row")
    safe_u = slice(u_halo, -u_halo if u_halo else None)
    current = current[safe_u]
    previous = previous[safe_u]
    finite = np.isfinite(current) & np.isfinite(previous)
    active = finite & (
        (current > active_update_floor)
        | (previous > active_update_floor)
    )
    active_count = int(np.count_nonzero(active))
    finite_count = int(np.count_nonzero(finite))
    if active_count:
        ratio = current[active] / np.maximum(
            previous[active], active_update_floor
        )
        minimum = float(np.min(ratio))
        median = float(np.median(ratio))
        maximum = float(np.max(ratio))
        contracting_fraction = float(np.mean(ratio < 1.0))
    else:
        minimum = median = maximum = 0.0
        contracting_fraction = 1.0
    return {
        "minimum": minimum,
        "median": median,
        "maximum": maximum,
        "contracting_fraction": contracting_fraction,
        "active_cell_count": active_count,
        "inactive_finite_cell_count": finite_count - active_count,
        "nonfinite_cell_count": int(finite.size - finite_count),
        "newly_active_cell_count": int(
            np.count_nonzero(
                active
                & (previous <= active_update_floor)
                & (current > active_update_floor)
            )
        ),
        "active_update_floor": float(active_update_floor),
    }


def sdc_defect_arrays(
    diagnostics: list[dict[str, object]],
    u: Array,
    v: Array,
) -> dict[str, Array]:
    """Expand final-sweep element SDC defects onto saved v nodes.

    Shared element endpoints conservatively receive the larger adjacent
    defect.  Missing or failed-element diagnostics remain NaN rather than
    being silently interpreted as zero error.
    """

    coordinate = np.asarray(v, dtype=float)
    collocation = np.full(len(coordinate), np.nan)
    overgrid = np.full(len(coordinate), np.nan)
    for item in diagnostics:
        if item.get("v_left") is None or item.get("v_right") is None:
            continue
        left = float(item["v_left"])
        right = float(item["v_right"])
        tolerance = 1.0e-12 * max(1.0, abs(left), abs(right))
        selected = (coordinate >= left - tolerance) & (
            coordinate <= right + tolerance
        )
        for name, target in (
            ("collocation_defect", collocation),
            ("overgrid_defect", overgrid),
        ):
            if item.get(name) is None:
                continue
            value = float(item[name])
            if not math.isfinite(value):
                continue
            prior = target[selected]
            target[selected] = np.where(
                np.isnan(prior), value, np.maximum(prior, value)
            )
    u_count = len(np.asarray(u))
    return {
        "metric_sdc_collocation_defect_by_v": collocation,
        "metric_sdc_overgrid_defect_by_v": overgrid,
        "metric_sdc_collocation_defect": np.broadcast_to(
            collocation[None, :], (u_count, len(coordinate))
        ).copy(),
        "metric_sdc_overgrid_defect": np.broadcast_to(
            overgrid[None, :], (u_count, len(coordinate))
        ).copy(),
    }


def checkpoint_diagnostic_arrays(
    closure: dict[str, Array | float],
    incoming_metric: Array,
    construction_residual_history: list[Array],
    metric_sdc_diagnostics: list[dict[str, object]],
    u: Array,
    v: Array,
    *,
    u_sdc_maps: dict[str, Array] | None = None,
) -> dict[str, Array]:
    """Return the spatial diagnostics persisted beside a continuation state."""

    required = ("pointwise", "differential_pointwise")
    missing = [name for name in required if name not in closure]
    if missing:
        raise ValueError("metric closure lacks arrays: " + ", ".join(missing))
    result = {
        "metric_closure": np.asarray(closure["pointwise"]),
        "differential_metric_closure": np.asarray(
            closure["differential_pointwise"]
        ),
        "incoming_metric": np.asarray(incoming_metric),
    }
    if construction_residual_history:
        history = np.stack(construction_residual_history)
        expected = (len(u), len(v))
        if history.shape[1:] != expected:
            raise ValueError(
                "construction residual history has incompatible shape "
                f"{history.shape}; expected (sweeps,{expected[0]},{expected[1]})"
            )
        result["construction_residual_history"] = history
    if metric_sdc_diagnostics:
        result.update(sdc_defect_arrays(metric_sdc_diagnostics, u, v))
    for name, value in (u_sdc_maps or {}).items():
        if not name.startswith("u_sdc_"):
            raise ValueError(f"u-SDC checkpoint map lacks prefix: {name}")
        array = np.asarray(value, dtype=float)
        if array.shape != (len(u), len(v)):
            raise ValueError(
                f"u-SDC checkpoint map {name} has shape {array.shape}; "
                f"expected ({len(u)},{len(v)})"
            )
        result[name] = array
    return result


def continuation_sweep_evidence(
    grid: PointSphereGrid,
    state: FirstOrderState,
    context: dict[str, object],
    values: dict[str, Array],
    construction_residual_history: list[Array],
    update_history: list[Array],
    scalar_coordinates: CharacteristicLGLMesh,
    *,
    u_halo: int,
) -> tuple[dict[str, object], dict[str, Array]]:
    """Build the finalizable diagnostics saved after every continuation sweep."""

    metric_sdc_diagnostics = list(
        context.get("metric_sdc_diagnostics", [])
    )
    closure = metric_closure(
        grid,
        state,
        context["incoming_metric"],
        scalar_coordinates.u,
        scalar_coordinates=scalar_coordinates,
    )
    maps = physical_component_l2_maps(
        grid, state, scalar_coordinates.u, values
    )
    total = construction_residual_history[-1]
    sdc_arrays = (
        sdc_defect_arrays(
            metric_sdc_diagnostics,
            scalar_coordinates.u,
            scalar_coordinates.v,
        )
        if metric_sdc_diagnostics
        else {}
    )
    summary: dict[str, object] = {
        "metric_closure_maximum": float(closure["maximum"]),
        "differential_closure_maximum": float(
            closure["differential_maximum"]
        ),
        "residual": safe_scalar_summary(total, u_halo, 0),
        "construction_residual_history": [
            safe_scalar_summary(value, u_halo, 0)
            for value in construction_residual_history
        ],
        "projection_tails": context["projection_tails"],
        "metric_sdc_maximum_collocation_defect": (
            finite_maximum(
                sdc_arrays["metric_sdc_collocation_defect_by_v"]
            )
            if sdc_arrays
            else None
        ),
        "metric_sdc_maximum_overgrid_defect": (
            finite_maximum(
                sdc_arrays["metric_sdc_overgrid_defect_by_v"]
            )
            if sdc_arrays
            else None
        ),
        "u_sdc_maximum_defects": {
            name: finite_maximum(value)
            for name, value in context.get("u_sdc_maps", {}).items()
        },
    }
    diagnostic_arrays = checkpoint_diagnostic_arrays(
        closure,
        np.asarray(context["incoming_metric"]),
        construction_residual_history,
        metric_sdc_diagnostics,
        scalar_coordinates.u,
        scalar_coordinates.v,
        u_sdc_maps=context.get("u_sdc_maps", {}),
    )
    extra = {
        "construction_total_residual": total,
        "picard_update_history": np.stack(update_history),
        # Persist the actual half-step shear so an independent angular grid
        # can audit the nonlinear strong trace after the expensive sweep.
        # This is diagnostic state, not an additional evolved unknown.
        "half_shear": np.asarray(context["half_shear"]),
        **diagnostic_arrays,
        **{f"map_{name}": value for name, value in maps.items()},
    }
    return summary, extra


def load_continuation_checkpoint_arrays(
    state_path: Path,
) -> tuple[list[Array], list[Array], dict[str, Array]]:
    with np.load(state_path, allow_pickle=False) as data:
        residual = [
            np.asarray(value)
            for value in data["construction_residual_history"]
        ]
        updates = [
            np.asarray(value) for value in data["picard_update_history"]
        ]
        state_keys = {
            "u",
            "v",
            "state_schema_version",
            "weighted_omegab_semantics",
            "metric",
            "omega",
            "zeta_up",
            "shift",
            "q",
            "shear",
            "weighted_chib",
            "weighted_omega",
            "weighted_omegab",
        }
        extra = {
            name: np.asarray(data[name])
            for name in data.files
            if name not in state_keys
        }
    return residual, updates, extra


def finite_maximum(value: Array) -> float | None:
    finite = np.asarray(value, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.max(finite)) if finite.size else None


def run(args: argparse.Namespace) -> dict[str, object]:
    run_solver_source_fingerprint = solver_source_fingerprint()
    run_continuation_source_fingerprint = continuation_source_fingerprint()
    run_runtime_provenance = runtime_provenance()
    output = ROOT / "results" / "Q1" / args.output_label
    output = prepare_output_directory(
        output,
        operational_restart=args.restart_checkpoint is not None,
    )

    if args.checkpoint_retain_generations < 2:
        raise ValueError("checkpoint retention must keep at least two generations")
    if args.restart_checkpoint is not None and args.resume_cap is not None:
        raise ValueError(
            "--restart-checkpoint and --resume-cap have different meanings "
            "and are mutually exclusive"
        )

    unsafe_allow_unverified_resume = bool(
        getattr(args, "unsafe_allow_unverified_resume", False)
    )
    if unsafe_allow_unverified_resume and args.resume_cap is None:
        raise ValueError(
            "--unsafe-allow-unverified-resume requires --resume-cap"
        )
    if args.unsafe_allow_unverified_restart and args.restart_checkpoint is None:
        raise ValueError(
            "--unsafe-allow-unverified-restart requires --restart-checkpoint"
        )

    active_update_floor = float(
        getattr(args, "active_update_floor", 1.0e-10)
    )
    if not math.isfinite(active_update_floor) or active_update_floor < 0.0:
        raise ValueError("active update floor must be finite and nonnegative")

    grid = PointSphereGrid.create(
        args.points,
        neighbor_count=args.neighbors,
        degree=args.angular_degree,
        spectral_degree=args.spectral_degree,
    )
    angular = AngularGalerkin(
        grid,
        retained_degree=args.galerkin_retained_degree,
        work_degree=args.galerkin_work_degree,
    )
    u_right = float(getattr(args, "u_right", -0.5))
    if not -1.0 < u_right < 0.0:
        raise ValueError("u-right must lie strictly between -1 and 0")
    tau_breakpoints = np.linspace(
        0.0, -math.log(-u_right), args.lgl_u_elements + 1
    )
    if args.lgl_s_breakpoints_json is None:
        s_breakpoints = low_band_s_breakpoints(args.v1, args.v1)
    else:
        s_breakpoints = load_s_breakpoints(
            args.lgl_s_breakpoints_json,
            v1=args.v1,
            v_endpoint=args.v1,
        )
    s_breakpoints, continuation_indices = continuation_schedule(
        s_breakpoints,
        v1=args.v1,
        specification=args.continuation_v_caps,
    )
    execution_indices = continuation_execution_indices(
        s_breakpoints,
        continuation_indices,
        v1=args.v1,
        stop_after_v=args.stop_after_v,
    )
    full_coordinates = CharacteristicLGLMesh.create(
        tau_breakpoints,
        args.lgl_u_degree,
        s_breakpoints,
        args.lgl_s_degree,
        args.v1,
    )
    execution_coordinates = CharacteristicLGLMesh.create(
        tau_breakpoints,
        args.lgl_u_degree,
        s_breakpoints[: execution_indices[-1] + 1],
        args.lgl_s_degree,
        args.v1,
    )
    resume_solver_parameters = continuation_resume_solver_parameters(
        args, full_coordinates, continuation_indices
    )
    operational_run_contract = continuation_operational_run_contract(
        args, resume_solver_parameters
    )
    operational_restart_validation: dict[str, object] | None = None
    if args.restart_checkpoint is not None:
        operational_restart_validation = resolve_operational_restart_checkpoint(
            args.restart_checkpoint,
            producer="pulse_continuation",
            solver_semantics=SOLVER_SEMANTICS,
            solver_source_fingerprint=run_solver_source_fingerprint,
            producer_source_fingerprint=(
                run_continuation_source_fingerprint
            ),
            runtime_provenance=run_runtime_provenance,
            run_contract=operational_run_contract,
            unsafe_allow_unverified=args.unsafe_allow_unverified_restart,
        )
        operational_manifest = Path(
            str(operational_restart_validation["manifest"])
        ).resolve()
        if not operational_manifest.is_relative_to(
            (output / "checkpoints").resolve()
        ):
            raise ValueError(
                "operational checkpoint does not belong to this output directory"
            )
    resume_validation: dict[str, object] | None = None
    if args.resume_cap is not None:
        first_panel = continuation_indices[0]
        first_prefix_coordinates = CharacteristicLGLMesh.create(
            tau_breakpoints,
            args.lgl_u_degree,
            s_breakpoints[: first_panel + 1],
            args.lgl_s_degree,
            args.v1,
        )
        resume_validation = validate_scientific_resume(
            args.resume_cap,
            continuation_parameters=resume_solver_parameters,
            first_prefix_coordinates=first_prefix_coordinates,
            expected_first_v_endpoint=(
                args.v1 * float(s_breakpoints[first_panel]) ** 2
            ),
            solver_fingerprint=run_solver_source_fingerprint,
            continuation_fingerprint=run_continuation_source_fingerprint,
            unsafe_allow_unverified=unsafe_allow_unverified_resume,
        )
    calibration = calibrate_low_band_profiles(
        args.v1, args.c, divisor=args.shear_divisor
    )
    execution_boundary = solve_low_band_boundary(
        grid,
        execution_coordinates.v,
        calibration,
        angular=angular,
        scalar_coordinates=execution_coordinates,
        coordinate_integrator=args.metric_integrator,
        sdc_overgrid_tolerance=args.sdc_overgrid_tolerance,
    )

    state: FirstOrderState | None = None
    resume_metadata: dict[str, object] | None = None
    resume_panel = 0
    operational_active_panel: int | None = None
    operational_completed_sweep = 0
    operational_stage_number = 1
    operational_stage_status = "running"
    operational_progress: dict[str, object] | None = None
    run_id = uuid.uuid4().hex
    parent_operational_manifest_sha256: str | None = None
    operational_restart_count = 0
    if resume_validation is None:
        root_origin = "fresh-boundary-seed"
        operational_lineage_status = "verified"
    elif resume_validation.get("status") == "verified":
        root_origin = "verified-scientific-resume"
        operational_lineage_status = "verified"
    else:
        root_origin = "unsafe-scientific-resume"
        operational_lineage_status = "unsafe-unverified"
    if resume_validation is not None and isinstance(
        resume_validation.get("manifest_sha256"), str
    ):
        # The first operational generation after a scientific resume must
        # retain the scientific parent's content-addressed lineage.  Later
        # generations replace this with the immediately preceding operational
        # manifest hash, yielding one unbroken crash-restart chain.
        parent_operational_manifest_sha256 = str(
            resume_validation["manifest_sha256"]
        )

    if operational_restart_validation is not None:
        position = operational_restart_validation["position"]
        progress = operational_restart_validation["progress"]
        if (
            not isinstance(position, dict)
            or position.get("driver") != "pulse_continuation"
            or not isinstance(progress, dict)
        ):
            raise ValueError(
                "operational checkpoint is not a continuation sweep checkpoint"
            )
        operational_active_panel = int(position.get("panel", -1))
        operational_completed_sweep = int(
            position.get("completed_sweeps", -1)
        )
        operational_stage_number = int(position.get("stage", -1))
        operational_stage_status = str(position.get("stage_status", ""))
        if (
            operational_active_panel not in execution_indices
            or not 1 <= operational_completed_sweep <= args.maximum_sweeps
            or operational_stage_number < 1
            or operational_stage_status
            not in {"running", "converged", "maximum-sweeps"}
        ):
            raise ValueError("operational continuation position is invalid")
        operational_progress = progress
        state, saved_u, saved_v = load_state(
            Path(str(operational_restart_validation["state"]))
        )
        expected_count = operational_active_panel * args.lgl_s_degree + 1
        if (
            not np.array_equal(saved_u, full_coordinates.u)
            or len(saved_v) != expected_count
            or not np.array_equal(
                saved_v, full_coordinates.v[:expected_count]
            )
        ):
            raise ValueError(
                "operational continuation checkpoint grid is not its declared prefix"
            )
        active_boundary = sliced_boundary(
            execution_boundary, len(execution_coordinates.v), len(saved_v)
        )
        validate_exact_outgoing_boundary(
            state,
            active_boundary,
            description="operational continuation",
        )
        run_id = str(operational_restart_validation["run_id"])
        parent_operational_manifest_sha256 = str(
            operational_restart_validation["manifest_sha256"]
        )
        prior_lineage = operational_restart_validation.get("lineage")
        if not isinstance(prior_lineage, dict):
            raise ValueError("operational continuation lineage is malformed")
        root_origin = str(prior_lineage.get("root_origin", "unknown"))
        operational_lineage_status = str(
            prior_lineage.get("status", "unsafe-unverified")
        )
        operational_restart_count = int(
            prior_lineage.get("operational_restart_count", 0)
        ) + 1
    elif args.resume_cap is not None:
        state, saved_u, saved_v = load_state(args.resume_cap)
        if not np.array_equal(saved_u, full_coordinates.u):
            raise ValueError("resume cap u grid does not match the continuation mesh")
        if len(saved_v) > len(full_coordinates.v) or not np.array_equal(
            saved_v, full_coordinates.v[: len(saved_v)]
        ):
            raise ValueError(
                "resume cap v grid is not a prefix of the refined mesh"
            )
        resume_matches = [
            index
            for index in continuation_indices
            if index * args.lgl_s_degree + 1 == len(saved_v)
        ]
        if len(resume_matches) != 1:
            raise ValueError(
                "resume endpoint is not one of the nonlinear continuation caps"
            )
        resume_panel = resume_matches[0]
        source_producer = resume_validation.get("source_producer")
        if source_producer == "pulse_campaign" and (
            resume_panel != continuation_indices[0]
        ):
            raise ValueError(
                "pulse_campaign state may warm-start only the first "
                "nonlinear continuation cap"
            )
        if resume_panel not in execution_indices:
            raise ValueError(
                "resume cap lies beyond this invocation's stop-after-v"
            )
        resume_boundary = sliced_boundary(
            execution_boundary, len(execution_coordinates.v), len(saved_v)
        )
        # A scientific cap is accepted up to the declared 256-epsilon
        # regeneration tolerance.  Validate it read-only here; the stage loop
        # will build one canonical boundary with the accepted prefix and reuse
        # that same object in every Picard call.
        canonicalize_continuation_boundary(
            state, grid, resume_boundary, len(saved_v)
        )
        resume_metadata = {
            "path": str(args.resume_cap.resolve()),
            "sha256": file_fingerprint(args.resume_cap),
            "v_endpoint": float(saved_v[-1]),
            "panel": resume_panel,
            "source_producer": source_producer,
            "warm_start_requires_continuation_convergence": (
                bool(
                    resume_validation.get(
                        "warm_start_requires_continuation_convergence",
                        False,
                    )
                )
            ),
            "provenance": resume_validation,
        }
    if operational_active_panel is None:
        if (
            resume_validation is not None
            and resume_validation.get(
                "warm_start_requires_continuation_convergence", False
            )
        ):
            # A fixed-sweep Q1 state, or a continuation cap that exhausted its
            # sweep limit under a require-convergence contract, has not earned
            # the destination tolerance.  Re-solve its cap from the imported
            # state before any prolongation.
            scheduled_indices = [
                index for index in execution_indices if index >= resume_panel
            ]
        else:
            scheduled_indices = [
                index for index in execution_indices if index > resume_panel
            ]
        stages: list[dict[str, object]] = []
        stage_number_start = 1
    else:
        scheduled_indices = [operational_active_panel] + [
            index
            for index in execution_indices
            if index > operational_active_panel
        ]
        prior_stages = operational_progress.get("completed_stages")
        if not isinstance(prior_stages, list):
            raise ValueError(
                "operational continuation lacks completed-stage journal"
            )
        stages = [dict(item) for item in prior_stages]
        stage_number_start = operational_stage_number
    expected_total_stage_records = len(stages) + len(scheduled_indices)
    stop_reason = ""
    for stage_number, panel_count in enumerate(
        scheduled_indices, start=stage_number_start
    ):
        scalar_coordinates = CharacteristicLGLMesh.create(
            tau_breakpoints,
            args.lgl_u_degree,
            s_breakpoints[: panel_count + 1],
            args.lgl_s_degree,
            args.v1,
        )
        count = len(scalar_coordinates.v)
        boundary = sliced_boundary(
            execution_boundary, len(execution_coordinates.v), count
        )
        restarting_active_stage = (
            operational_active_panel is not None
            and panel_count == operational_active_panel
            and stage_number == operational_stage_number
        )
        warming_scientific_cap = (
            resume_validation is not None
            and resume_validation.get(
                "warm_start_requires_continuation_convergence", False
            )
            and panel_count == resume_panel
            and stage_number == stage_number_start
        )
        prestage_error: Exception | None = None
        boundary_canonicalization: dict[str, object] | None = None
        if state is not None:
            try:
                boundary, boundary_canonicalization = (
                    canonicalize_continuation_boundary(
                        state, grid, boundary, count
                    )
                )
            except (
                FloatingPointError,
                np.linalg.LinAlgError,
                ValueError,
            ) as error:
                prestage_error = error
        prolongation_diagnostics: dict[str, object]
        if restarting_active_stage:
            if operational_progress is None:
                raise ValueError("missing operational continuation journal")
            restored_prolongation = operational_progress.get(
                "prolongation_diagnostics"
            )
            if not isinstance(restored_prolongation, dict):
                raise ValueError(
                    "operational continuation lacks prolongation diagnostics"
                )
            if restored_prolongation.get("predictor") != (
                args.continuation_predictor
            ):
                raise ValueError(
                    "operational continuation predictor diagnostics do not "
                    "match the run contract"
                )
            prolongation_diagnostics = dict(restored_prolongation)
            prolongation_diagnostics["stage_boundary_canonicalization"] = (
                boundary_canonicalization
            )
        elif warming_scientific_cap:
            prolongation_diagnostics = {
                "applied": False,
                "predictor": args.continuation_predictor,
                "reason": "scientific-cap-warm-start",
                "stage_boundary_canonicalization": (
                    boundary_canonicalization
                ),
            }
        elif state is None:
            prolongation_diagnostics = {
                "applied": False,
                "predictor": args.continuation_predictor,
                "reason": "fresh-boundary-seed",
            }
        else:
            prolongation_diagnostics = {}
        if prestage_error is None:
            try:
                if state is None:
                    state = boundary_compatible_initial_state(
                        grid, scalar_coordinates.u, scalar_coordinates.v, boundary
                    )
                elif not restarting_active_stage and not warming_scientific_cap:
                    state, prolongation_diagnostics = prolonged_state(
                        state,
                        grid,
                        scalar_coordinates.u,
                        scalar_coordinates.v,
                        boundary,
                        predictor=args.continuation_predictor,
                        angular=angular,
                        scalar_coordinates=scalar_coordinates,
                        trace_absolute_floor=(
                            args.u_sdc_half_trace_absolute_floor
                        ),
                        trace_retained_node_tolerance=(
                            args.u_sdc_half_trace_tolerance
                        ),
                        trace_retained_overgrid_tolerance=(
                            args.u_sdc_overgrid_tolerance
                        ),
                    )
                    prolongation_diagnostics[
                        "stage_boundary_canonicalization"
                    ] = boundary_canonicalization
            except (
                FloatingPointError,
                np.linalg.LinAlgError,
                ValueError,
            ) as error:
                prestage_error = error
        if prestage_error is not None:
            prolongation_diagnostics = {
                **prolongation_diagnostics,
                "applied": bool(
                    state is not None
                    and not restarting_active_stage
                    and not warming_scientific_cap
                ),
                "predictor": args.continuation_predictor,
                "accepted": False,
                "failure": str(prestage_error),
                "stage_boundary_canonicalization": (
                    boundary_canonicalization
                ),
            }

        stage_started = time.perf_counter()
        sweep_records: list[dict[str, object]] = []
        update_history: list[Array] = []
        construction_residual_history: list[Array] = []
        stage_elapsed_before_restart = 0.0
        last_stage_evidence: dict[str, object] | None = None
        last_checkpoint_extra: dict[str, Array] | None = None
        last_metric_sdc_diagnostics: list[dict[str, object]] = []
        last_u_sdc_diagnostics: dict[str, list[dict[str, object]]] = {}
        status = "failed" if prestage_error is not None else "maximum-sweeps"
        reason = "" if prestage_error is None else str(prestage_error)
        failure_diagnostics: list[dict[str, object]] = []
        failure_field: str | None = (
            None if prestage_error is None else "continuation_predictor"
        )
        sweep_start = 1 if prestage_error is None else args.maximum_sweeps + 1
        if restarting_active_stage and prestage_error is None:
            if operational_progress is None:
                raise ValueError("missing operational continuation journal")
            restored_records = operational_progress.get("sweep_records")
            restored_evidence = operational_progress.get("stage_evidence")
            restored_metric = operational_progress.get(
                "metric_sdc_diagnostics"
            )
            restored_u = operational_progress.get("u_sdc_diagnostics")
            if (
                not isinstance(restored_records, list)
                or len(restored_records) != operational_completed_sweep
                or not isinstance(restored_evidence, dict)
                or not isinstance(restored_metric, list)
                or not isinstance(restored_u, dict)
            ):
                raise ValueError(
                    "operational continuation journal is incomplete"
                )
            sweep_records = [dict(item) for item in restored_records]
            construction_residual_history, update_history, last_checkpoint_extra = (
                load_continuation_checkpoint_arrays(
                    Path(str(operational_restart_validation["state"]))
                )
            )
            if not (
                len(construction_residual_history)
                == operational_completed_sweep
                == len(update_history)
            ):
                raise ValueError(
                    "operational continuation array history has wrong length"
                )
            last_stage_evidence = dict(restored_evidence)
            last_metric_sdc_diagnostics = [
                dict(item) for item in restored_metric
            ]
            last_u_sdc_diagnostics = {
                str(name): [dict(item) for item in items]
                for name, items in restored_u.items()
            }
            stage_elapsed_before_restart = float(
                operational_progress.get("stage_elapsed_seconds", 0.0)
            )
            sweep_start = operational_completed_sweep + 1
            if operational_stage_status != "running":
                status = operational_stage_status
                # A terminal generation already contains the finalized state
                # and controller evidence for this cap.  Replaying additional
                # Picard sweeps would change the restart trajectory before the
                # state is prolonged to the next panel.  Leave the range empty
                # and finalize this restored cap directly.
                sweep_start = args.maximum_sweeps + 1
        for sweep in range(sweep_start, args.maximum_sweeps + 1):
            previous = state
            try:
                candidate, context = picard_step(
                    grid,
                    previous,
                    boundary,
                    scalar_coordinates.u,
                    scalar_coordinates.v,
                    metric_substeps=args.metric_substeps,
                    angular=angular,
                    scalar_coordinates=scalar_coordinates,
                    metric_parameterization=args.metric_parameterization,
                    metric_integrator=args.metric_integrator,
                    sdc_tolerance=args.sdc_tolerance,
                    sdc_overgrid_tolerance=args.sdc_overgrid_tolerance,
                    sdc_maximum_corrections=args.sdc_maximum_corrections,
                    u_integrator=args.u_integrator,
                    u_sdc_tolerance=args.u_sdc_tolerance,
                    u_sdc_overgrid_tolerance=args.u_sdc_overgrid_tolerance,
                    u_sdc_maximum_corrections=args.u_sdc_maximum_corrections,
                    u_sdc_half_trace_tolerance=(
                        args.u_sdc_half_trace_tolerance
                    ),
                    u_sdc_half_trace_absolute_floor=(
                        args.u_sdc_half_trace_absolute_floor
                    ),
                )
                sweep_values = components(
                    grid,
                    candidate,
                    scalar_coordinates.u,
                    scalar_coordinates.v,
                    mode="construction",
                    previous_state=previous,
                    construction_context=context,
                    scalar_coordinates=scalar_coordinates,
                )
                sweep_residual = full_ricci_map(
                    grid, candidate, scalar_coordinates.u, sweep_values
                )
            except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
                status = "failed"
                reason = str(error)
                failure_diagnostics = list(
                    getattr(error, "diagnostics", [])
                )
                raw_failure_field = getattr(error, "field", None)
                failure_field = (
                    None
                    if raw_failure_field is None
                    else str(raw_failure_field)
                )
                break

            local_update = update_map(candidate, previous)
            global_update = update_norm(candidate, previous)
            contraction = None
            if update_history:
                contraction = active_contraction_summary(
                    local_update,
                    update_history[-1],
                    u_halo=args.u_halo,
                    active_update_floor=active_update_floor,
                )
            sweep_records.append(
                {
                    "sweep": sweep,
                    "global_update": global_update,
                    "local_update": safe_scalar_summary(
                        local_update, args.u_halo, 0
                    ),
                    "local_contraction": contraction,
                    "construction_residual": safe_scalar_summary(
                        sweep_residual, args.u_halo, 0
                    ),
                    "minimum_metric_eigenvalue": minimum_metric_eigenvalue(
                        grid, candidate
                    ),
                    "minimum_lapse": float(np.min(candidate.omega)),
                    "minimum_q": float(np.min(candidate.q)),
                    "metric_sdc_diagnostics": context.get(
                        "metric_sdc_diagnostics", []
                    ),
                    "u_sdc_diagnostics": context.get(
                        "u_sdc_diagnostics", {}
                    ),
                    "u_sdc_maximum_defects": {
                        name: finite_maximum(value)
                        for name, value in context.get(
                            "u_sdc_maps", {}
                        ).items()
                    },
                }
            )
            update_history.append(local_update)
            construction_residual_history.append(sweep_residual)
            state = candidate
            last_metric_sdc_diagnostics = list(
                context.get("metric_sdc_diagnostics", [])
            )
            last_u_sdc_diagnostics = {
                str(name): list(items)
                for name, items in context.get(
                    "u_sdc_diagnostics", {}
                ).items()
            }
            last_stage_evidence, last_checkpoint_extra = (
                continuation_sweep_evidence(
                    grid,
                    state,
                    context,
                    sweep_values,
                    construction_residual_history,
                    update_history,
                    scalar_coordinates,
                    u_halo=args.u_halo,
                )
            )
            sweep_converged = (
                sweep >= args.minimum_sweeps
                and global_update <= args.tolerance
            )
            checkpoint_stage_status = (
                "converged"
                if sweep_converged
                else "maximum-sweeps"
                if sweep == args.maximum_sweeps
                else "running"
            )
            if args.checkpoint_every_sweep:
                position = {
                    "driver": "pulse_continuation",
                    "stage": stage_number,
                    "panel": panel_count,
                    "completed_sweeps": sweep,
                    "stage_status": checkpoint_stage_status,
                }
                lineage = {
                    "status": operational_lineage_status,
                    "root_origin": root_origin,
                    "operational_restart_count": operational_restart_count,
                    "parent_manifest_sha256": (
                        parent_operational_manifest_sha256
                    ),
                }

                def write_state(checkpoint_path: Path) -> None:
                    save_state(
                        checkpoint_path,
                        state,
                        scalar_coordinates.u,
                        scalar_coordinates.v,
                        extra=last_checkpoint_extra,
                    )

                checkpoint = write_operational_checkpoint(
                    output / "checkpoints",
                    f"panel-{panel_count:06d}-sweep-{sweep:06d}",
                    state_writer=write_state,
                    producer="pulse_continuation",
                    solver_semantics=SOLVER_SEMANTICS,
                    solver_source_fingerprint=(
                        run_solver_source_fingerprint
                    ),
                    producer_source_fingerprint=(
                        run_continuation_source_fingerprint
                    ),
                    runtime_provenance=run_runtime_provenance,
                    run_id=run_id,
                    run_contract=operational_run_contract,
                    position=position,
                    progress={
                        "driver": "pulse_continuation",
                        "completed_stages": stages,
                        "sweep_records": sweep_records,
                        "stage_elapsed_seconds": (
                            stage_elapsed_before_restart
                            + time.perf_counter()
                            - stage_started
                        ),
                        "stage_evidence": last_stage_evidence,
                        "prolongation_diagnostics": (
                            prolongation_diagnostics
                        ),
                        "metric_sdc_diagnostics": (
                            last_metric_sdc_diagnostics
                        ),
                        "u_sdc_diagnostics": last_u_sdc_diagnostics,
                    },
                    lineage=lineage,
                    retain_valid_generations=(
                        args.checkpoint_retain_generations
                    ),
                    source_guard=lambda: (
                        solver_source_fingerprint()
                        == run_solver_source_fingerprint
                        and continuation_source_fingerprint()
                        == run_continuation_source_fingerprint
                    ),
                )
                parent_operational_manifest_sha256 = str(
                    checkpoint["manifest_sha256"]
                )
            if sweep_converged:
                status = "converged"
                break

        if status == "failed" and failure_field is None:
            metric_sdc_diagnostics = failure_diagnostics
            u_sdc_diagnostics: dict[str, list[dict[str, object]]] = {}
        elif status == "failed":
            metric_sdc_diagnostics = []
            u_sdc_diagnostics = {failure_field: failure_diagnostics}
        else:
            metric_sdc_diagnostics = last_metric_sdc_diagnostics
            u_sdc_diagnostics = last_u_sdc_diagnostics
        # Complete the stage payload before it is content-addressed.  In
        # particular, a fresh first stage has no earlier predictor from which
        # to inherit this key.  Mutating the nested diagnostics after hashing
        # would make the published summary differ from its cap-manifest
        # binding even though the numerical state itself is unchanged.
        prolongation_diagnostics = {
            **prolongation_diagnostics,
            "stage_boundary_canonicalization": boundary_canonicalization,
        }
        stage: dict[str, object] = {
            "stage": stage_number,
            "panel": panel_count,
            "v_endpoint": float(scalar_coordinates.v[-1]),
            "u_count": len(scalar_coordinates.u),
            "v_count": len(scalar_coordinates.v),
            "seconds": (
                stage_elapsed_before_restart
                + time.perf_counter()
                - stage_started
            ),
            "status": status,
            "reason": reason,
            "failure_diagnostics": failure_diagnostics,
            "failure_field": failure_field,
            "metric_sdc_diagnostics": metric_sdc_diagnostics,
            "u_sdc_diagnostics": u_sdc_diagnostics,
            "sweeps": sweep_records,
            "prolongation": prolongation_diagnostics,
            "initial_state": (
                (
                    "verified-q1-warm-start"
                    if resume_validation is not None
                    and resume_validation.get("source_producer")
                    == "pulse_campaign"
                    else "continuation-cap-warm-start"
                )
                if warming_scientific_cap
                else "operational-restart"
                if restarting_active_stage
                else "causal-prolongation"
            ),
        }
        if (
            status != "failed"
            and state is not None
            and last_stage_evidence is not None
            and last_checkpoint_extra is not None
            and construction_residual_history
        ):
            stage.update(last_stage_evidence)
            stage_dir = output / f"cap-{float(scalar_coordinates.v[-1]):.8f}"
            stage_dir.mkdir(exist_ok=True)
            state_path = stage_dir / "state.npz"
            if (
                solver_source_fingerprint()
                != run_solver_source_fingerprint
                or continuation_source_fingerprint()
                != run_continuation_source_fingerprint
            ):
                raise RuntimeError(
                    "a fingerprinted solver source changed while the Q1 "
                    "continuation was executing; refusing to write a "
                    "mismatched checkpoint"
                )
            save_state(
                state_path,
                state,
                scalar_coordinates.u,
                scalar_coordinates.v,
                extra=last_checkpoint_extra,
            )
            cap_record = continuation_cap_record_binding(stage)
            final_operational_restart_validation = (
                operational_restart_validation
            )
            if (
                final_operational_restart_validation is not None
                and parent_operational_manifest_sha256 is not None
            ):
                final_operational_restart_validation = {
                    **final_operational_restart_validation,
                    "manifest_sha256": (
                        parent_operational_manifest_sha256
                    ),
                }
            manifest_path = write_continuation_cap_manifest(
                state_path,
                solver_fingerprint=run_solver_source_fingerprint,
                continuation_fingerprint=(
                    run_continuation_source_fingerprint
                ),
                solver_parameters=resume_solver_parameters,
                cap_record=cap_record,
                resume_validation=resume_validation,
                operational_restart_validation=(
                    final_operational_restart_validation
                ),
            )
            stage["checkpoint_manifest"] = str(manifest_path)
        stages.append(stage)
        print(json.dumps(stage, indent=2), flush=True)

        if status == "failed":
            stop_reason = reason
            break
        if status != "converged" and args.require_convergence:
            stop_reason = (
                f"cap {float(scalar_coordinates.v[-1]):.8g} did not converge in "
                f"{args.maximum_sweeps} sweeps"
            )
            break

    parameters = vars(args).copy()
    if args.lgl_s_breakpoints_json is not None:
        parameters["lgl_s_breakpoints_json"] = str(
            args.lgl_s_breakpoints_json.resolve()
        )
        parameters["lgl_s_breakpoints_sha256"] = file_fingerprint(
            args.lgl_s_breakpoints_json
        )
    if args.resume_cap is not None:
        parameters["resume_cap"] = str(args.resume_cap.resolve())
    if args.restart_checkpoint is not None:
        parameters["restart_checkpoint"] = str(
            args.restart_checkpoint.resolve()
        )
    if (
        solver_source_fingerprint() != run_solver_source_fingerprint
        or continuation_source_fingerprint()
        != run_continuation_source_fingerprint
    ):
        raise RuntimeError(
            "a fingerprinted solver source changed while the Q1 continuation "
            "was executing; refusing to write a mismatched summary"
        )
    requested_terminal_v = (
        args.v1 if args.stop_after_v is None else float(args.stop_after_v)
    )
    terminal_v_endpoint = (
        float(stages[-1]["v_endpoint"])
        if stages
        else (
            None
            if resume_metadata is None
            else float(resume_metadata["v_endpoint"])
        )
    )
    terminal_cap_reached = bool(
        terminal_v_endpoint is not None
        and math.isclose(
            terminal_v_endpoint,
            requested_terminal_v,
            rel_tol=0.0,
            abs_tol=2.0e-14,
        )
    )
    completed = bool(
        len(stages) == expected_total_stage_records
        and not stop_reason
        and all(stage.get("status") != "failed" for stage in stages)
        and terminal_cap_reached
    )
    result: dict[str, object] = {
        "experiment": "full-Q1 characteristic-LGL endpoint continuation",
        "solver_semantics": SOLVER_SEMANTICS,
        "solver_source_fingerprint": run_solver_source_fingerprint,
        "fingerprinted_sources": list(FINGERPRINT_SOURCES),
        "continuation_source_fingerprint": (
            run_continuation_source_fingerprint
        ),
        "runtime_provenance": run_runtime_provenance,
        "status": "completed" if completed else "stopped",
        "stop_reason": stop_reason,
        "requested_stop_after_v": args.stop_after_v,
        "terminal_v_endpoint": terminal_v_endpoint,
        "terminal_cap_reached": terminal_cap_reached,
        "parameters": parameters,
        "resume": resume_metadata,
        "operational_restart_provenance": (
            None
            if operational_restart_validation is None
            else {
                "status": operational_restart_validation["status"],
                "manifest": operational_restart_validation["manifest"],
                "manifest_sha256": operational_restart_validation[
                    "manifest_sha256"
                ],
                "position": operational_restart_validation["position"],
                "lineage": operational_restart_validation["lineage"],
            }
        ),
        "execution_provenance": {
            "run_id": run_id,
            "root_origin": root_origin,
            "operational_restart_count": operational_restart_count,
            "checkpoint_retention_generations": (
                args.checkpoint_retain_generations
            ),
        },
        "angular": angular.diagnostics(),
        "full_coordinate_mesh": full_coordinates.diagnostics(),
        "execution_coordinate_mesh": execution_coordinates.diagnostics(),
        "boundary": scalar_boundary_summary(execution_boundary),
        "stages": stages,
    }
    atomic_write_json(output / "continuation-summary.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--output-label", required=True)
    result.add_argument("--c", type=float, default=1.0)
    result.add_argument("--shear-divisor", type=float, default=3.2)
    result.add_argument("--v1", type=float, default=0.5)
    result.add_argument("--points", type=int, default=200)
    result.add_argument("--neighbors", type=int, default=30)
    result.add_argument("--angular-degree", type=int, default=4)
    result.add_argument("--spectral-degree", type=int, default=11)
    result.add_argument("--galerkin-retained-degree", type=int, default=5)
    result.add_argument("--galerkin-work-degree", type=int, default=10)
    result.add_argument(
        "--lgl-u-elements",
        type=int,
        default=2,
        help=(
            "u spectral elements; 2xp8 is the minimum seed mesh certified "
            "below the 1e-7 independent overgrid gate"
        ),
    )
    result.add_argument("--lgl-u-degree", type=int, default=8)
    result.add_argument(
        "--u-right",
        type=float,
        default=-0.5,
        help=(
            "terminal incoming coordinate (the left endpoint remains u=-1); "
            "values closer to zero search for focusing beyond Q1"
        ),
    )
    result.add_argument("--lgl-s-degree", type=int, default=8)
    result.add_argument(
        "--lgl-s-breakpoints-json",
        type=Path,
        help="audited custom s=sqrt(v/v1) mesh specification",
    )
    result.add_argument(
        "--continuation-v-caps",
        help=(
            "comma-separated physical v endpoints for nonlinear continuation; "
            "they are inserted into the fine LGL mesh, and v1 is appended to "
            "the full scientific schedule"
        ),
    )
    result.add_argument(
        "--continuation-predictor",
        choices=("constant", "boundary-lift"),
        default="constant",
        help=(
            "warm-start for newly appended causal panels: 'constant' keeps "
            "the terminal trace, while 'boundary-lift' adds the "
            "smooth boundary-compatible seed increment (opt-in)"
        ),
    )
    result.add_argument(
        "--stop-after-v",
        type=float,
        help=(
            "cleanly finish this invocation after this configured nonlinear "
            "cap; later caps remain in the scientific resume contract but "
            "are not executed"
        ),
    )
    result.add_argument(
        "--resume-cap",
        type=Path,
        help=(
            "manifested schema-2 state.npz at a scheduled cap on an "
            "unchanged prefix mesh and discrete solver map; a completed, "
            "verified pulse_campaign state may warm-start the first cap and "
            "will be re-swept to the continuation tolerance"
        ),
    )
    result.add_argument(
        "--restart-checkpoint",
        type=Path,
        help=(
            "operationally restart the same interrupted continuation stage "
            "from an immutable per-sweep checkpoint"
        ),
    )
    result.add_argument(
        "--unsafe-allow-unverified-resume",
        action="store_true",
        help=(
            "diagnostic-only: allow a established cap with no manifest or an "
            "already unverified lineage; semantic, parameter, and content "
            "mismatches are never ignored"
        ),
    )
    result.add_argument(
        "--unsafe-allow-unverified-restart",
        action="store_true",
        help=(
            "diagnostic-only: restart an operational checkpoint whose root "
            "lineage was already unsafe"
        ),
    )
    result.add_argument(
        "--checkpoint-every-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    result.add_argument(
        "--checkpoint-retain-generations",
        type=int,
        default=2,
        help="number of valid same-run generations retained (minimum two)",
    )
    result.add_argument("--metric-substeps", type=int, default=2)
    result.add_argument(
        "--metric-parameterization",
        choices=("direct", "cholesky"),
        default="cholesky",
    )
    result.add_argument(
        "--metric-integrator", choices=("rk4", "sdc"), default="rk4"
    )
    result.add_argument("--sdc-tolerance", type=float, default=1.0e-10)
    result.add_argument("--sdc-overgrid-tolerance", type=float, default=1.0e-7)
    result.add_argument("--sdc-maximum-corrections", type=int, default=12)
    result.add_argument(
        "--u-integrator",
        choices=("rk4", "sdc"),
        default="rk4",
        help="integrator for the four incoming-direction construction marches",
    )
    result.add_argument("--u-sdc-tolerance", type=float, default=1.0e-10)
    result.add_argument(
        "--u-sdc-overgrid-tolerance", type=float, default=1.0e-7
    )
    result.add_argument(
        "--u-sdc-maximum-corrections", type=int, default=12
    )
    result.add_argument(
        "--u-sdc-half-trace-tolerance", type=float, default=1.0e-9
    )
    result.add_argument(
        "--u-sdc-half-trace-absolute-floor", type=float, default=1.0e-14
    )
    result.add_argument("--minimum-sweeps", type=int, default=2)
    result.add_argument("--maximum-sweeps", type=int, default=8)
    result.add_argument("--tolerance", type=float, default=1.0e-7)
    result.add_argument(
        "--active-update-floor",
        type=float,
        default=1.0e-10,
        help=(
            "exclude cells whose consecutive local Picard updates are both "
            "below this absolute floor from contraction ratios"
        ),
    )
    result.add_argument("--u-halo", type=int, default=2)
    result.add_argument(
        "--require-convergence", action=argparse.BooleanOptionalAction, default=True
    )
    return result


if __name__ == "__main__":
    run(parser().parse_args())
