"""Run the numerical first-order formulation on Q1."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from nee.io.boundary_artifact import load_boundary_data, save_boundary_data
from nee.state.boundary import BoundaryData


HERE = Path(__file__).resolve().parent
ROOT = HERE

from .vacuum_iteration import (  # noqa: E402
    boundary_compatible_initial_state,
    initial_state,
    impose_outgoing_boundary,
    metric_closure,
    picard_step,
    solve_incoming_metric,
    update_map,
    update_norm,
)
from .spherical_harmonics import AngularGalerkin  # noqa: E402
from .adaptive_mesh import load_s_breakpoints  # noqa: E402
from .lgl import CharacteristicLGLMesh  # noqa: E402
from .vacuum_residual import (  # noqa: E402
    components,
    physical_component_l2_maps,
    safe_maxima,
)
from .vacuum_state_io import full_ricci_map, load_state, save_state  # noqa: E402
from .sphere import PointSphereGrid  # noqa: E402
from .boundary_shear import solve_linear_fixed_boundary  # noqa: E402
from .short_pulse import (  # noqa: E402
    CORNER_SCALE_FRACTION,
    PROFILE_TRANSITION_FRACTION,
    SUPPORT_MARGIN_FRACTION,
    calibrate_low_band_profiles,
    solve_low_band_boundary,
)
from .checkpoint import (  # noqa: E402
    OperationalCheckpointError,
    atomic_write_json,
    prepare_output_directory,
    resolve_operational_restart_checkpoint,
    write_operational_checkpoint,
)
from .smooth_pulse import (  # noqa: E402
    calibrate_profiles,
    scaled_calibration,
    solve_outgoing_boundary,
)


Array = np.ndarray

SOLVER_SEMANTICS = (
    "nee-galerkin-flat-bandlimited-pulse-common-coordinate-"
    "operators-exact-derived-Omega_chih-full-omegab-boundary-compatible-seed-"
    "reduced-coordinate-half-Omega_chih-optional-tau-sdc-u-marches"
)
FINGERPRINT_SOURCE_PATHS = (
    (
        "numerical/src/adaptive_mesh.py",
        HERE / "adaptive_mesh.py",
    ),
    ("numerical/src/spherical_harmonics.py", HERE / "spherical_harmonics.py"),
    (
        "numerical/src/lgl.py",
        HERE / "lgl.py",
    ),
    (
        "numerical/src/sdc.py",
        HERE / "sdc.py",
    ),
    (
        "numerical/src/tau_sdc.py",
        HERE / "tau_sdc.py",
    ),
    (
        "numerical/src/tau_mesh.py",
        HERE / "tau_mesh.py",
    ),
    (
        "numerical/src/tau_solvers.py",
        HERE / "tau_solvers.py",
    ),
    ("numerical/src/vacuum_state_io.py", HERE / "vacuum_state_io.py"),
    ("numerical/src/vacuum_iteration.py", HERE / "vacuum_iteration.py"),
    ("numerical/src/vacuum_residual.py", HERE / "vacuum_residual.py"),
    ("numerical/src/boundary_shear.py", HERE / "boundary_shear.py"),
    (
        "numerical/src/short_pulse.py",
        HERE / "short_pulse.py",
    ),
    (
        "numerical/src/checkpoint.py",
        HERE / "checkpoint.py",
    ),
    ("numerical/src/pulse_campaign.py", HERE / "pulse_campaign.py"),
    (
        "numerical/src/coordinate_differentiation.py",
        HERE / "coordinate_differentiation.py",
    ),
    (
        "numerical/src/coordinate_quadrature.py",
        HERE / "coordinate_quadrature.py",
    ),
    (
        "numerical/src/vacuum_state.py",
        HERE / "vacuum_state.py",
    ),
    (
        "numerical/src/pulse_design.py",
        HERE / "pulse_design.py",
    ),
    (
        "numerical/src/ricci_residual.py",
        HERE / "ricci_residual.py",
    ),
    (
        "numerical/src/sphere.py",
        HERE / "sphere.py",
    ),
    (
        "numerical/src/smooth_pulse.py",
        HERE / "smooth_pulse.py",
    ),
)
FINGERPRINT_SOURCES = tuple(label for label, _ in FINGERPRINT_SOURCE_PATHS)
CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1
CHECKPOINT_MANIFEST_KIND = "numerical-resumable-state"


def solver_source_fingerprint() -> str:
    """Hash the source files that define this experiment's discrete map."""

    digest = hashlib.sha256()
    for label, path in FINGERPRINT_SOURCE_PATHS:
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


def file_fingerprint(path: Path) -> str:
    """Hash a non-source input artifact such as an adaptive mesh proposal."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def checkpoint_manifest_path(state_path: Path) -> Path:
    """Return the fail-closed provenance sidecar for one state archive."""

    path = Path(state_path)
    return path.with_name(path.name + ".manifest.json")


def _manifest_mismatches(
    expected: Any, actual: Any, prefix: str = ""
) -> list[str]:
    """Describe exact, recursively localized manifest mismatches."""

    label = prefix or "value"
    if type(expected) is not type(actual):
        return [
            f"{label}: expected {type(expected).__name__}, "
            f"received {type(actual).__name__}"
        ]
    if isinstance(expected, dict):
        differences: list[str] = []
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            differences.append(f"{label}.{key}: missing")
        for key in sorted(actual_keys - expected_keys):
            differences.append(f"{label}.{key}: unexpected")
        for key in sorted(expected_keys & actual_keys):
            differences.extend(
                _manifest_mismatches(
                    expected[key], actual[key], f"{label}.{key}"
                )
            )
        return differences
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return [
                f"{label}: expected length {len(expected)}, "
                f"received {len(actual)}"
            ]
        differences = []
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            differences.extend(
                _manifest_mismatches(
                    expected_item, actual_item, f"{label}[{index}]"
                )
            )
        return differences
    return [] if expected == actual else [
        f"{label}: expected {expected!r}, received {actual!r}"
    ]


def write_checkpoint_manifest(
    state_path: Path,
    *,
    producer: str,
    producer_source_fingerprint: str,
    solver_parameters: dict[str, object],
    resume_validation: dict[str, object] | None = None,
    operational_restart_validation: dict[str, object] | None = None,
    solver_source_fingerprint_value: str | None = None,
) -> Path:
    """Bind a saved state to its exact solver map and verified lineage."""

    state = Path(state_path).resolve()
    if not state.is_file():
        raise FileNotFoundError(state)
    if operational_restart_validation is not None:
        operational_lineage = operational_restart_validation.get("lineage")
        if not isinstance(operational_lineage, dict):
            operational_lineage = {}
        operational_status = operational_restart_validation.get("status")
        lineage = {
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
    elif resume_validation is None:
        lineage: dict[str, object] = {
            "status": "verified",
            "origin": "fresh-run",
        }
    elif resume_validation.get("status") == "verified":
        lineage = {
            "status": "verified",
            "origin": "verified-resume",
            "parent_manifest_sha256": resume_validation.get(
                "manifest_sha256"
            ),
        }
    else:
        # An unsafe diagnostic resume must never be laundered into a newly
        # certified checkpoint merely because this process can fingerprint it.
        lineage = {
            "status": "unsafe-unverified",
            "origin": "unsafe-resume",
            "reason": resume_validation.get("reason"),
        }
    source_fingerprint = (
        solver_source_fingerprint()
        if solver_source_fingerprint_value is None
        else str(solver_source_fingerprint_value)
    )
    payload = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "manifest_kind": CHECKPOINT_MANIFEST_KIND,
        "producer": producer,
        "solver_semantics": SOLVER_SEMANTICS,
        "solver_source_fingerprint": source_fingerprint,
        "producer_source_fingerprint": producer_source_fingerprint,
        "solver_parameters": solver_parameters,
        "lineage": lineage,
        "state": {
            "filename": state.name,
            "bytes": state.stat().st_size,
            "sha256": file_fingerprint(state),
        },
    }
    manifest = checkpoint_manifest_path(state)
    atomic_write_json(manifest, payload)
    return manifest


def validate_checkpoint_manifest(
    state_path: Path,
    *,
    producer: str,
    producer_source_fingerprint: str,
    solver_parameters: dict[str, object],
    unsafe_allow_unverified: bool = False,
    solver_source_fingerprint_value: str | None = None,
) -> dict[str, object]:
    """Validate provenance before parsing a checkpoint's numerical state.

    The unsafe switch permits only a missing established sidecar or an explicitly
    unverified lineage.  It never overrides a present semantic, parameter,
    producer, source-fingerprint, size, or content-hash mismatch.
    """

    state = Path(state_path).resolve()
    manifest = checkpoint_manifest_path(state)
    if not manifest.is_file():
        reason = (
            f"resume checkpoint {state} has no provenance manifest {manifest}; "
            "established checkpoints are rejected because their solver map cannot "
            "be verified"
        )
        if unsafe_allow_unverified:
            return {
                "status": "unsafe-unverified",
                "reason": reason,
                "state": str(state),
                "manifest": str(manifest),
            }
        raise ValueError(
            reason
            + "; use --unsafe-allow-unverified-resume only for a diagnostic "
            "run whose outputs will remain uncertified"
        )
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"resume provenance manifest {manifest} is unreadable: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"resume provenance manifest {manifest} is not an object")
    source_fingerprint = (
        solver_source_fingerprint()
        if solver_source_fingerprint_value is None
        else str(solver_source_fingerprint_value)
    )
    expected = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "manifest_kind": CHECKPOINT_MANIFEST_KIND,
        "producer": producer,
        "solver_semantics": SOLVER_SEMANTICS,
        "solver_source_fingerprint": source_fingerprint,
        "producer_source_fingerprint": producer_source_fingerprint,
        "solver_parameters": solver_parameters,
    }
    actual = {name: payload.get(name) for name in expected}
    differences = _manifest_mismatches(expected, actual, "manifest")
    if differences:
        preview = "; ".join(differences[:8])
        if len(differences) > 8:
            preview += f"; and {len(differences) - 8} more"
        raise ValueError(
            "resume checkpoint provenance does not match the requested "
            f"solver: {preview}"
        )
    state_record = payload.get("state")
    if not isinstance(state_record, dict):
        raise ValueError(f"resume provenance manifest {manifest} lacks state data")
    expected_state = {
        "filename": state.name,
        "bytes": state.stat().st_size,
        "sha256": file_fingerprint(state),
    }
    state_differences = _manifest_mismatches(
        expected_state, state_record, "manifest.state"
    )
    if state_differences:
        raise ValueError(
            "resume checkpoint content does not match its provenance manifest: "
            + "; ".join(state_differences[:8])
        )
    lineage = payload.get("lineage")
    lineage_status = (
        lineage.get("status") if isinstance(lineage, dict) else None
    )
    if lineage_status != "verified":
        reason = (
            f"resume checkpoint {state} has unverified provenance lineage "
            f"{lineage_status!r}"
        )
        if not unsafe_allow_unverified:
            raise ValueError(
                reason
                + "; use --unsafe-allow-unverified-resume only for a "
                "diagnostic run whose outputs will remain uncertified"
            )
        status = "unsafe-unverified"
    else:
        reason = ""
        status = "verified"
    return {
        "status": status,
        "reason": reason,
        "state": str(state),
        "state_sha256": expected_state["sha256"],
        "manifest": str(manifest),
        "manifest_sha256": file_fingerprint(manifest),
        "lineage": lineage,
    }


def runtime_provenance() -> dict[str, str]:
    """Return runtime versions that can affect floating-point artifacts."""

    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def q1_resume_solver_parameters(
    args: argparse.Namespace,
    *,
    v_endpoint: float,
    u: Array,
    v: Array,
    scalar_coordinates: CharacteristicLGLMesh | None,
) -> dict[str, object]:
    """Return only parameters that define the resumed discrete Picard map."""

    retained = args.galerkin_retained_degree
    effective_work = (
        None
        if retained is None
        else (
            2 * retained
            if args.galerkin_work_degree is None
            else args.galerkin_work_degree
        )
    )
    if scalar_coordinates is None:
        coordinate_specification: dict[str, object] = {
            "method": "local-polynomial",
            "v_grid_power": args.v_grid_power,
            "u": [float(value) for value in u],
            "v": [float(value) for value in v],
        }
    else:
        coordinate_specification = {
            "method": "lgl",
            "mesh": scalar_coordinates.diagnostics(),
        }
    return {
        "boundary": {
            "c": args.c,
            "Omega_chih_divisor": args.Omega_chih_divisor,
            "mode": args.boundary_mode,
            "v1": args.v1,
            "v_endpoint": v_endpoint,
        },
        "sphere": {
            "points": args.points,
            "neighbors": args.neighbors,
            "angular_degree": args.angular_degree,
            "spectral_degree": args.spectral_degree,
        },
        "galerkin": {
            "retained_degree": retained,
            "work_degree": effective_work,
        },
        "scalar_coordinates": coordinate_specification,
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
    }


def q1_operational_run_contract(
    args: argparse.Namespace,
    *,
    solver_parameters: dict[str, object],
    output_label: str,
) -> dict[str, object]:
    """Bind the controller state that a crash restart must not change."""

    return {
        "driver": "pulse_campaign",
        "output_label": output_label,
        "solver_parameters": solver_parameters,
        "controller": {
            "iterations_requested": args.iterations,
            "iteration_offset": args.iteration_offset,
            "u_halo": args.u_halo,
            "v_halo": args.v_halo,
            "construction_v_halo": args.construction_v_halo,
            "rho_epsilon": args.rho_epsilon,
            "rho_min": args.rho_min,
            "rho_max": args.rho_max,
            "checkpoint_every_sweep": args.checkpoint_every_sweep,
            "checkpoint_retention_generations": (
                args.checkpoint_retain_generations
            ),
        },
    }


def resolve_q1_operational_restart(
    checkpoint: Path,
    *,
    output: Path,
    solver_fingerprint: str,
    run_runtime_provenance: dict[str, str],
    run_contract: dict[str, object],
    unsafe_allow_unverified: bool = False,
) -> dict[str, object]:
    """Driver-level resolver with fallback and output-ownership checks."""

    validation = resolve_operational_restart_checkpoint(
        checkpoint,
        producer="pulse_campaign",
        solver_semantics=SOLVER_SEMANTICS,
        solver_source_fingerprint=solver_fingerprint,
        producer_source_fingerprint=solver_fingerprint,
        runtime_provenance=run_runtime_provenance,
        run_contract=run_contract,
        unsafe_allow_unverified=unsafe_allow_unverified,
    )
    operational_manifest = Path(str(validation["manifest"])).resolve()
    expected_checkpoint_root = (Path(output).resolve() / "checkpoints").resolve()
    if not operational_manifest.is_relative_to(expected_checkpoint_root):
        raise ValueError(
            "operational checkpoint does not belong to this output directory"
        )
    return validation


def q1_checkpoint_extra(
    *,
    last_maps: dict[str, Array] | None,
    last_residual: Array | None,
    last_closure: dict[str, Array | float] | None,
    last_context: dict[str, object] | None,
    residual_history: list[Array],
    update_history: list[Array],
) -> dict[str, Array]:
    """Return all arrays needed to continue and finalize without another sweep."""

    extra: dict[str, Array] = {}
    if last_closure is not None:
        extra["metric_closure"] = np.asarray(last_closure["pointwise"])
        extra["differential_metric_closure"] = np.asarray(
            last_closure["differential_pointwise"]
        )
    if last_maps is not None:
        extra.update(
            {f"map_{name}": np.asarray(value) for name, value in last_maps.items()}
        )
    if last_residual is not None:
        extra["construction_total_residual"] = np.asarray(last_residual)
    if residual_history:
        extra["construction_residual_history"] = np.stack(residual_history)
    if update_history:
        extra["picard_update_history"] = np.stack(update_history)
    if last_context is not None:
        for name in (
            "half_shear",
            "incoming_metric",
            "weighted_omegab_half",
        ):
            if name in last_context:
                extra[name] = np.asarray(last_context[name])
        extra.update(
            {
                str(name): np.asarray(value)
                for name, value in last_context.get("u_sdc_maps", {}).items()
            }
        )
    return extra


def restore_q1_operational_arrays(
    state_path: Path,
) -> dict[str, object]:
    """Restore accumulated array diagnostics from one operational generation."""

    with np.load(state_path, allow_pickle=False) as data:
        residual_history = (
            [np.asarray(value) for value in data["construction_residual_history"]]
            if "construction_residual_history" in data.files
            else []
        )
        update_history = (
            [np.asarray(value) for value in data["picard_update_history"]]
            if "picard_update_history" in data.files
            else []
        )
        maps = {
            name[4:]: np.asarray(data[name])
            for name in data.files
            if name.startswith("map_")
        }
        closure = (
            {
                "pointwise": np.asarray(data["metric_closure"]),
                "differential_pointwise": np.asarray(
                    data["differential_metric_closure"]
                ),
            }
            if {
                "metric_closure",
                "differential_metric_closure",
            }.issubset(data.files)
            else None
        )
        context: dict[str, object] = {}
        for name in (
            "half_shear",
            "incoming_metric",
            "weighted_omegab_half",
        ):
            if name in data.files:
                context[name] = np.asarray(data[name])
        context["u_sdc_maps"] = {
            name: np.asarray(data[name])
            for name in data.files
            if name.startswith("u_sdc_")
        }
        residual = (
            np.asarray(data["construction_total_residual"])
            if "construction_total_residual" in data.files
            else None
        )
    return {
        "residual_history": residual_history,
        "update_history": update_history,
        "last_maps": maps or None,
        "last_closure": closure,
        "last_context": context or None,
        "last_residual": residual,
    }


def safe_maximum(value: Array, u_halo: int, v_halo: int) -> float:
    u_slice = slice(u_halo, -u_halo if u_halo else None)
    v_slice = slice(v_halo, -v_halo if v_halo else None)
    return float(np.max(value[:, u_slice, v_slice]))


def scalar_boundary_summary(
    boundary: dict[str, Array | float | object],
) -> dict[str, object]:
    result: dict[str, object] = {
        name: float(value)
        for name, value in boundary.items()
        if np.isscalar(value)
    }
    if "boundary_sdc_diagnostics" in boundary:
        result["boundary_sdc_diagnostics"] = boundary[
            "boundary_sdc_diagnostics"
        ]
    return result


def asymptotic_power_fit(
    residual: Array,
    u: Array,
    v: Array,
    *,
    u_halo: int,
    epsilon: float,
    rho_min: float,
    rho_max: float,
    minimum_points: int = 6,
) -> dict[str, float | int | None]:
    """Fit log(f)=a+p log(rho) separately on each safe u section."""

    slopes: list[float] = []
    sample_count = 0
    first = u_halo
    last = len(u) - u_halo
    for index in range(first, last):
        rho = v / (-u[index]) ** (1.0 + epsilon)
        values = residual[index]
        mask = (
            (rho >= rho_min)
            & (rho <= rho_max)
            & np.isfinite(values)
            & (values > 1.0e-15)
        )
        if int(np.count_nonzero(mask)) < minimum_points:
            continue
        slope, _ = np.polyfit(np.log(rho[mask]), np.log(values[mask]), 1)
        slopes.append(float(slope))
        sample_count += int(np.count_nonzero(mask))
    if not slopes:
        return {
            "section_count": 0,
            "sample_count": 0,
            "median_power": None,
            "minimum_power": None,
            "maximum_power": None,
        }
    return {
        "section_count": len(slopes),
        "sample_count": sample_count,
        "median_power": float(np.median(slopes)),
        "minimum_power": float(np.min(slopes)),
        "maximum_power": float(np.max(slopes)),
    }


def safe_scalar_summary(
    value: Array, u_halo: int, v_halo: int = 0
) -> dict[str, float]:
    u_slice = slice(u_halo, -u_halo if u_halo else None)
    v_slice = slice(v_halo, -v_halo if v_halo else None)
    safe = value[u_slice, v_slice]
    finite = safe[np.isfinite(safe)]
    if finite.size == 0:
        raise ValueError("the requested scalar-map halo leaves no finite cells")
    return {
        "minimum": float(np.min(finite)),
        "median": float(np.median(finite)),
        "maximum": float(np.max(finite)),
    }


def mutation_test(
    grid: PointSphereGrid,
    u: Array,
    v: Array,
    angular: AngularGalerkin | None = None,
    scalar_coordinates: CharacteristicLGLMesh | None = None,
) -> dict[str, float]:
    state = initial_state(grid, u, v)
    correct = solve_incoming_metric(
        grid,
        state.g[:, 0],
        state.Omega_chib,
        state.b,
        u,
        kinematic_factor=2.0,
        angular=angular,
    )
    mutated = solve_incoming_metric(
        grid,
        state.g[:, 0],
        state.Omega_chib,
        state.b,
        u,
        kinematic_factor=0.5,
        angular=angular,
    )
    correct_error = metric_closure(
        grid, state, correct, u, scalar_coordinates=scalar_coordinates
    )["maximum"]
    mutated_error = metric_closure(
        grid, state, mutated, u, scalar_coordinates=scalar_coordinates
    )["maximum"]
    if not correct_error < 1.0e-11 or not mutated_error > 1.0e-2:
        raise AssertionError(
            "kinematic coefficient mutation test failed: "
            f"correct={correct_error}, mutated={mutated_error}"
        )
    return {
        "factor_2_maximum": float(correct_error),
        "factor_one_half_maximum": float(mutated_error),
        "separation": float(mutated_error / max(correct_error, 1.0e-30)),
    }


def low_band_s_breakpoints(v1: float, v_endpoint: float) -> Array:
    """Place s-elements at every pulse corner/support transition."""

    split = 0.5 * v1
    left_end = (1.0 - SUPPORT_MARGIN_FRACTION) * split
    left_transition = PROFILE_TRANSITION_FRACTION * left_end
    corner_scale = CORNER_SCALE_FRACTION * left_end
    half_length = v1 - split
    right_start = split + SUPPORT_MARGIN_FRACTION * half_length
    right_end = v1 - SUPPORT_MARGIN_FRACTION * half_length
    right_transition = PROFILE_TRANSITION_FRACTION * (right_end - right_start)
    continuation_caps = v1 * np.array(
        [0.01, 0.02, 0.04, 0.08, 0.16, 0.32]
    )
    physical = np.concatenate(
        (
            np.array(
                [
                    0.0,
                    corner_scale,
                    left_end - left_transition,
                    left_end,
                    split,
                    right_start,
                    right_start + right_transition,
                    right_end - right_transition,
                    right_end,
                    v1,
                ]
            ),
            continuation_caps,
        )
    )
    s_endpoint = math.sqrt(v_endpoint / v1)
    transformed = np.sort(np.sqrt(np.maximum(physical / v1, 0.0)))
    interior = transformed[(transformed > 0.0) & (transformed < s_endpoint)]
    return np.unique(np.concatenate(([0.0], interior, [s_endpoint])))


def run(args: argparse.Namespace) -> dict:
    run_solver_source_fingerprint = solver_source_fingerprint()
    run_runtime_provenance = runtime_provenance()
    if args.u_integrator == "sdc" and args.coordinate_method != "lgl":
        raise ValueError("--u-integrator=sdc requires --coordinate-method=lgl")
    if args.checkpoint_retain_generations < 2:
        raise ValueError("checkpoint retention must keep at least two generations")
    if args.restart_checkpoint is not None and args.resume_state is not None:
        raise ValueError(
            "--restart-checkpoint and --resume-state have different meanings "
            "and are mutually exclusive"
        )
    if (
        args.unsafe_allow_unverified_resume
        and args.resume_state is None
    ):
        raise ValueError(
            "--unsafe-allow-unverified-resume requires --resume-state"
        )
    if (
        args.unsafe_allow_unverified_restart
        and args.restart_checkpoint is None
    ):
        raise ValueError(
            "--unsafe-allow-unverified-restart requires --restart-checkpoint"
        )
    grid = PointSphereGrid.create(
        args.points,
        neighbor_count=args.neighbors,
        degree=args.angular_degree,
        spectral_degree=args.spectral_degree,
    )
    angular = None
    if args.galerkin_retained_degree is not None:
        work_degree = (
            2 * args.galerkin_retained_degree
            if args.galerkin_work_degree is None
            else args.galerkin_work_degree
        )
        if args.boundary_mode != "low-band-hemisphere":
            raise ValueError(
                "Galerkin Q1 currently requires the compatible "
                "low-band-hemisphere boundary solver"
            )
        if args.spectral_degree is None or args.spectral_degree < work_degree + 1:
            raise ValueError(
                "spectral-degree must be at least Galerkin work-degree + 1"
            )
        angular = AngularGalerkin(
            grid,
            retained_degree=args.galerkin_retained_degree,
            work_degree=work_degree,
        )
    elif args.galerkin_work_degree is not None:
        raise ValueError(
            "--galerkin-work-degree requires --galerkin-retained-degree"
        )
    v_endpoint = args.v1 if args.v_endpoint is None else args.v_endpoint
    if not 0.0 < v_endpoint <= args.v1:
        raise ValueError("v_endpoint must lie in (0,v1]")
    scalar_coordinates = None
    if args.coordinate_method == "lgl":
        if args.boundary_mode != "low-band-hemisphere":
            raise ValueError(
                "the characteristic LGL pilot currently requires the "
                "low-band-hemisphere datum"
            )
        if args.lgl_u_elements < 1 or args.lgl_u_degree < 2:
            raise ValueError("invalid LGL u element configuration")
        if args.lgl_s_degree < 2:
            raise ValueError("the LGL s degree must be at least two")
        tau_breakpoints = np.linspace(
            0.0, math.log(2.0), args.lgl_u_elements + 1
        )
        if args.lgl_s_breakpoints_json is None:
            s_breakpoints = low_band_s_breakpoints(args.v1, v_endpoint)
        else:
            s_breakpoints = load_s_breakpoints(
                args.lgl_s_breakpoints_json,
                v1=args.v1,
                v_endpoint=v_endpoint,
            )
        scalar_coordinates = CharacteristicLGLMesh.create(
            tau_breakpoints,
            args.lgl_u_degree,
            s_breakpoints,
            args.lgl_s_degree,
            args.v1,
        )
        u = scalar_coordinates.u
        v = scalar_coordinates.v
    else:
        if args.v_grid_power <= 0.0:
            raise ValueError("v_grid_power must be positive")
        u = -np.exp(np.linspace(0.0, math.log(0.5), args.u_count))
        v_parameter = np.linspace(0.0, 1.0, args.v_count)
        v = v_endpoint * v_parameter**args.v_grid_power
    resume_solver_parameters = q1_resume_solver_parameters(
        args,
        v_endpoint=v_endpoint,
        u=u,
        v=v,
        scalar_coordinates=scalar_coordinates,
    )
    resume_validation: dict[str, object] | None = None
    if args.resume_state is not None:
        resume_validation = validate_checkpoint_manifest(
            args.resume_state,
            producer="pulse_campaign",
            producer_source_fingerprint=run_solver_source_fingerprint,
            solver_parameters=resume_solver_parameters,
            unsafe_allow_unverified=args.unsafe_allow_unverified_resume,
            solver_source_fingerprint_value=(
                run_solver_source_fingerprint
            ),
        )
    label = args.output_label or (
        f"sphere-{args.points}_u-{len(u)}_v-{len(v)}"
    )
    output = ROOT / "results" / "Q1" / label
    output = prepare_output_directory(
        output,
        operational_restart=args.restart_checkpoint is not None,
    )
    operational_run_contract = q1_operational_run_contract(
        args,
        solver_parameters=resume_solver_parameters,
        output_label=label,
    )
    operational_restart_validation: dict[str, object] | None = None
    if args.restart_checkpoint is not None:
        operational_restart_validation = resolve_q1_operational_restart(
            args.restart_checkpoint,
            output=output,
            solver_fingerprint=run_solver_source_fingerprint,
            run_runtime_provenance=run_runtime_provenance,
            run_contract=operational_run_contract,
            unsafe_allow_unverified=args.unsafe_allow_unverified_restart,
        )
    if args.boundary_mode == "linear-fixed":
        boundary = solve_linear_fixed_boundary(grid, v)
    elif args.boundary_mode == "low-band-hemisphere":
        calibration = calibrate_low_band_profiles(
            args.v1, args.c, divisor=args.Omega_chih_divisor
        )
        boundary = solve_low_band_boundary(
            grid,
            v,
            calibration,
            angular=angular,
            scalar_coordinates=scalar_coordinates,
            coordinate_integrator=args.metric_integrator,
            sdc_overgrid_tolerance=args.sdc_overgrid_tolerance,
        )
    else:
        base_calibration = calibrate_profiles(args.v1, args.c)
        calibration = scaled_calibration(base_calibration, args.Omega_chih_divisor)
        boundary = solve_outgoing_boundary(grid, v, calibration)

    # Persist and reload the physical characteristic traces before building
    # the zeroth iterate. This separates data generation from the sweep and
    # ensures the numerical kernel consumes content-verified immutable arrays.
    preview = boundary_compatible_initial_state(grid, u, v, boundary)
    outgoing_fields = {
        name: np.asarray(boundary[name])
        for name in (
            "g", "inverse_g", "Omega_trchi", "Omega_chih", "zeta", "b"
        )
        if name in boundary
    }
    incoming_fields = {
        "g": preview.g[:, :, 0],
        "Omega_trchi": preview.Omega_trchi[:, :, 0],
        "Omega": preview.Omega[:, :, 0],
        "Omega_chib": preview.Omega_chib[:, :, 0],
        "Omega_omega": preview.Omega_omega[:, :, 0],
        "Omega_omegab": preview.Omega_omegab[:, :, 0],
        "zeta": preview.zeta[:, :, 0],
        "b": preview.b[:, :, 0],
    }
    immutable_boundary = BoundaryData.create(
        outgoing_fields,
        incoming_fields,
        metadata={"experiment": "vacuum-strong-short-pulse"},
    )
    boundary_artifact_path = output / "boundary-data.npz"
    save_boundary_data(
        boundary_artifact_path,
        immutable_boundary,
        u=u,
        v=v,
    )
    verified_boundary, artifact_u, artifact_v = load_boundary_data(
        boundary_artifact_path
    )
    np.testing.assert_array_equal(artifact_u, u)
    np.testing.assert_array_equal(artifact_v, v)
    boundary = {**boundary, **dict(verified_boundary.outgoing)}
    if operational_restart_validation is not None:
        restart_state_path = Path(
            str(operational_restart_validation["state"])
        )
        state, saved_u, saved_v = load_state(restart_state_path)
        if not (
            np.array_equal(saved_u, u)
            and np.array_equal(saved_v, v)
            and state.g.shape[0] == grid.count
        ):
            raise ValueError(
                "operational restart grid does not match the requested Q1 grid"
            )
        bounded = impose_outgoing_boundary(state, boundary)
        for description in state.__dataclass_fields__:
            if not np.array_equal(
                getattr(state, description)[:, 0],
                getattr(bounded, description)[:, 0],
            ):
                raise ValueError(
                    "operational checkpoint outgoing boundary differs from "
                    f"the regenerated boundary in field {description}"
                )
    elif args.resume_state is None:
        state = boundary_compatible_initial_state(grid, u, v, boundary)
    else:
        state, saved_u, saved_v = load_state(args.resume_state)
        if not (
            np.array_equal(saved_u, u)
            and np.array_equal(saved_v, v)
            and state.g.shape[0] == grid.count
        ):
            raise ValueError(
                "resume checkpoint grid does not match the requested Q1 grid"
            )
        state = impose_outgoing_boundary(state, boundary)
    mutation = mutation_test(
        grid, u, v, angular=angular, scalar_coordinates=scalar_coordinates
    )
    completed_sweeps = 0
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

    records: list[dict[str, object]] = []
    residual_history: list[Array] = []
    update_history: list[Array] = []
    last_maps = None
    last_residual = None
    last_closure = None
    last_context = None
    status = "completed"
    reason = ""
    failure_diagnostics: list[dict[str, object]] = []
    failure_field: str | None = None
    last_operational_checkpoint: dict[str, object] | None = None

    if operational_restart_validation is not None:
        run_id = str(operational_restart_validation["run_id"])
        position = operational_restart_validation["position"]
        progress = operational_restart_validation["progress"]
        if not isinstance(position, dict) or position.get("driver") != "pulse_campaign":
            raise ValueError("operational checkpoint is not a Q1 sweep checkpoint")
        completed_sweeps = int(position.get("completed_sweeps", -1))
        if not 0 <= completed_sweeps <= args.iterations:
            raise ValueError("operational checkpoint has an invalid sweep ordinal")
        restored_records = progress.get("records") if isinstance(progress, dict) else None
        if not isinstance(restored_records, list) or len(restored_records) != completed_sweeps:
            raise ValueError(
                "operational checkpoint record count does not match its sweep ordinal"
            )
        records = [dict(item) for item in restored_records]
        restored = restore_q1_operational_arrays(
            Path(str(operational_restart_validation["state"]))
        )
        residual_history = list(restored["residual_history"])
        update_history = list(restored["update_history"])
        if not (
            len(residual_history) == completed_sweeps
            and len(update_history) == completed_sweeps
        ):
            raise ValueError(
                "operational checkpoint array history does not match its sweep ordinal"
            )
        last_maps = restored["last_maps"]
        last_residual = restored["last_residual"]
        last_closure = restored["last_closure"]
        last_context = restored["last_context"]
        parent_operational_manifest_sha256 = str(
            operational_restart_validation["manifest_sha256"]
        )
        prior_lineage = operational_restart_validation.get("lineage")
        if not isinstance(prior_lineage, dict):
            raise ValueError("operational checkpoint lineage is malformed")
        root_origin = str(prior_lineage.get("root_origin", "unknown"))
        operational_lineage_status = str(
            prior_lineage.get("status", "unsafe-unverified")
        )
        operational_restart_count = int(
            prior_lineage.get("operational_restart_count", 0)
        ) + 1
        last_operational_checkpoint = {
            "manifest": operational_restart_validation["manifest"],
            "manifest_sha256": operational_restart_validation[
                "manifest_sha256"
            ],
            "position": position,
        }

    for local_iteration in range(completed_sweeps + 1, args.iterations + 1):
        iteration = args.iteration_offset + local_iteration
        started = time.perf_counter()
        try:
            new_state, context = picard_step(
                grid,
                state,
                boundary,
                u,
                v,
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
                u_sdc_half_trace_tolerance=args.u_sdc_half_trace_tolerance,
                u_sdc_half_trace_absolute_floor=(
                    args.u_sdc_half_trace_absolute_floor
                ),
            )
            closure = metric_closure(
                grid,
                new_state,
                context["incoming_metric"],
                u,
                scalar_coordinates=scalar_coordinates,
            )
            values = components(
                grid,
                new_state,
                u,
                v,
                mode="construction",
                previous_state=state,
                construction_context=context,
                scalar_coordinates=scalar_coordinates,
            )
            maps = physical_component_l2_maps(
                grid, new_state, u, values
            )
            total_residual = full_ricci_map(
                grid, new_state, u, values
            )
            local_update = update_map(new_state, state)
            local_update_summary = safe_scalar_summary(
                local_update, args.u_halo, 0
            )
            contraction_summary = None
            if update_history:
                previous_update = update_history[-1]
                contraction = local_update / np.maximum(
                    previous_update, 1.0e-30
                )
                safe_u = slice(
                    args.u_halo,
                    -args.u_halo if args.u_halo else None,
                )
                safe_contraction = contraction[safe_u]
                contraction_summary = {
                    "minimum": float(np.min(safe_contraction)),
                    "median": float(np.median(safe_contraction)),
                    "maximum": float(np.max(safe_contraction)),
                    "contracting_fraction": float(
                        np.mean(safe_contraction < 1.0)
                    ),
                }
            maxima = safe_maxima(maps, args.u_halo, args.v_halo)
            residual_summary = safe_scalar_summary(
                total_residual, args.u_halo, args.construction_v_halo
            )
            power_fit = asymptotic_power_fit(
                total_residual,
                u,
                v,
                u_halo=args.u_halo,
                epsilon=args.rho_epsilon,
                rho_min=args.rho_min,
                rho_max=args.rho_max,
            )
            record = {
                "iteration": iteration,
                "seconds": time.perf_counter() - started,
                "picard_update": update_norm(new_state, state),
                "picard_update_map": local_update_summary,
                "local_contraction": contraction_summary,
                "metric_closure_maximum": closure["maximum"],
                "metric_closure_safe_maximum": safe_maximum(
                    closure["pointwise"], args.u_halo, args.v_halo
                ),
                "differential_closure_maximum": closure[
                    "differential_maximum"
                ],
                "differential_closure_safe_maximum": safe_maximum(
                    closure["differential_pointwise"],
                    args.u_halo,
                    args.v_halo,
                ),
                "closure_below_1e-5": bool(closure["maximum"] < 1.0e-5),
                "construction_component_maxima": maxima,
                "construction_total_residual": residual_summary,
                "construction_asymptotic_fit": power_fit,
                "construction_discrete_closure_maxima": {
                    "raychaudhuri_projection_defect": safe_maximum(
                        np.abs(values["Ric44_construction_closure"]),
                        args.u_halo,
                        args.construction_v_halo,
                    ),
                    "metric_projection_defect": safe_maximum(
                        values["metric_projection_defect"],
                        args.u_halo,
                        args.construction_v_halo,
                    ),
                    "outgoing_lapse_fresh_closure": safe_maximum(
                        np.abs(values["outgoing_lapse_closure"]),
                        args.u_halo,
                        args.v_halo,
                    ),
                    "incoming_lapse_fresh_closure": safe_maximum(
                        np.abs(values["incoming_lapse_closure"]),
                        args.u_halo,
                        args.construction_v_halo,
                    ),
                    "omegab_source_fresh_closure": safe_maximum(
                        np.abs(values["omegab_source_closure"]),
                        args.u_halo,
                        args.v_halo,
                    ),
                },
                "galerkin_projection_tails": context.get(
                    "projection_tails", {}
                ),
                "half_shear_trace_drift": context.get(
                    "half_shear_trace_drift", {}
                ),
                "preoverwrite_boundary_mismatch": context.get(
                    "preoverwrite_boundary_mismatch", {}
                ),
                "metric_sdc_diagnostics": context.get(
                    "metric_sdc_diagnostics", []
                ),
                "u_sdc_diagnostics": context.get(
                    "u_sdc_diagnostics", {}
                ),
                "u_sdc_maximum_defects": {
                    name: float(np.max(value))
                    for name, value in context.get("u_sdc_maps", {}).items()
                },
            }
            records.append(record)
            residual_history.append(total_residual.copy())
            update_history.append(local_update.copy())
            print(json.dumps(record, indent=2), flush=True)
            state = new_state
            last_maps = maps
            last_residual = total_residual
            last_closure = closure
            last_context = context
            if args.checkpoint_every_sweep:
                checkpoint_extra = q1_checkpoint_extra(
                    last_maps=last_maps,
                    last_residual=last_residual,
                    last_closure=last_closure,
                    last_context=last_context,
                    residual_history=residual_history,
                    update_history=update_history,
                )
                position = {
                    "driver": "pulse_campaign",
                    "completed_sweeps": local_iteration,
                    "iteration_label": iteration,
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
                        u,
                        v,
                        extra=checkpoint_extra,
                    )

                checkpoint = write_operational_checkpoint(
                    output / "checkpoints",
                    f"sweep-{local_iteration:06d}",
                    state_writer=write_state,
                    producer="pulse_campaign",
                    solver_semantics=SOLVER_SEMANTICS,
                    solver_source_fingerprint=(
                        run_solver_source_fingerprint
                    ),
                    producer_source_fingerprint=(
                        run_solver_source_fingerprint
                    ),
                    runtime_provenance=run_runtime_provenance,
                    run_id=run_id,
                    run_contract=operational_run_contract,
                    position=position,
                    progress={
                        "driver": "pulse_campaign",
                        "records": records,
                    },
                    lineage=lineage,
                    retain_valid_generations=(
                        args.checkpoint_retain_generations
                    ),
                    source_guard=lambda: (
                        solver_source_fingerprint()
                        == run_solver_source_fingerprint
                    ),
                )
                parent_operational_manifest_sha256 = str(
                    checkpoint["manifest_sha256"]
                )
                last_operational_checkpoint = {
                    "manifest": checkpoint["manifest"],
                    "manifest_sha256": checkpoint["manifest_sha256"],
                    "position": position,
                }
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
            if isinstance(error, OperationalCheckpointError):
                raise
            status = "failed"
            reason = str(error)
            failure_diagnostics = list(getattr(error, "diagnostics", []))
            raw_failure_field = getattr(error, "field", None)
            failure_field = (
                None if raw_failure_field is None else str(raw_failure_field)
            )
            print(json.dumps({"status": status, "reason": reason}), flush=True)
            break

    current_solver_source_fingerprint = solver_source_fingerprint()
    if current_solver_source_fingerprint != run_solver_source_fingerprint:
        raise RuntimeError(
            "a fingerprinted solver source changed while the Q1 run was "
            "executing; refusing to write a mismatched checkpoint"
        )
    result = {
        "experiment": "numerical first-order incoming Ricci-coefficient Q1 iteration",
        "solver_semantics": SOLVER_SEMANTICS,
        "solver_source_fingerprint": run_solver_source_fingerprint,
        "fingerprinted_sources": list(FINGERPRINT_SOURCES),
        "runtime_provenance": run_runtime_provenance,
        "status": status,
        "reason": reason,
        "failure_diagnostics": failure_diagnostics,
        "failure_field": failure_field,
        "resume_provenance": resume_validation,
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
            "last_operational_checkpoint": last_operational_checkpoint,
        },
        "parameters": {
            "c": args.c,
            "Omega_chih_divisor": args.Omega_chih_divisor,
            "boundary_mode": args.boundary_mode,
            "v1": args.v1,
            "v_endpoint": v_endpoint,
            "v_grid_power": args.v_grid_power,
            "points": args.points,
            "neighbors": args.neighbors,
            "angular_degree": args.angular_degree,
            "spectral_degree": args.spectral_degree,
            "u_count": len(u),
            "v_count": len(v),
            "coordinate_method": args.coordinate_method,
            "coordinate_mesh": (
                None if scalar_coordinates is None else scalar_coordinates.diagnostics()
            ),
            "lgl_s_breakpoints_spec": (
                None
                if args.lgl_s_breakpoints_json is None
                else {
                    "path": str(args.lgl_s_breakpoints_json.resolve()),
                    "sha256": file_fingerprint(args.lgl_s_breakpoints_json),
                }
            ),
            "iterations_requested": args.iterations,
            "iteration_offset": args.iteration_offset,
            "resume_state": (
                None if args.resume_state is None else str(args.resume_state)
            ),
            "restart_checkpoint": (
                None
                if args.restart_checkpoint is None
                else str(args.restart_checkpoint)
            ),
            "unsafe_allow_unverified_resume": (
                args.unsafe_allow_unverified_resume
            ),
            "unsafe_allow_unverified_restart": (
                args.unsafe_allow_unverified_restart
            ),
            "checkpoint_every_sweep": args.checkpoint_every_sweep,
            "checkpoint_retain_generations": (
                args.checkpoint_retain_generations
            ),
            "metric_substeps": args.metric_substeps,
            "metric_parameterization": args.metric_parameterization,
            "metric_integrator": args.metric_integrator,
            "sdc_tolerance": args.sdc_tolerance,
            "sdc_overgrid_tolerance": args.sdc_overgrid_tolerance,
            "sdc_maximum_corrections": args.sdc_maximum_corrections,
            "u_integrator": args.u_integrator,
            "u_sdc_tolerance": args.u_sdc_tolerance,
            "u_sdc_overgrid_tolerance": args.u_sdc_overgrid_tolerance,
            "u_sdc_maximum_corrections": args.u_sdc_maximum_corrections,
            "u_sdc_half_trace_tolerance": args.u_sdc_half_trace_tolerance,
            "u_sdc_half_trace_absolute_floor": (
                args.u_sdc_half_trace_absolute_floor
            ),
            "u_halo": args.u_halo,
            "v_halo": args.v_halo,
            "construction_v_halo": args.construction_v_halo,
            "rho_epsilon": args.rho_epsilon,
            "rho_fit_window": [args.rho_min, args.rho_max],
            "galerkin": (
                None if angular is None else angular.diagnostics()
            ),
        },
        "kinematic_convention": (
            "Lie_(partial_u+b) g = 2 (Omega chib); the attachment's later "
            "factor 1/2 is tested only as a deliberate mutation"
        ),
        "mutation_test": mutation,
        "boundary": scalar_boundary_summary(boundary),
        "records": records,
    }
    if records and last_maps is not None and last_closure is not None:
        extra = q1_checkpoint_extra(
            last_maps=last_maps,
            last_residual=last_residual,
            last_closure=last_closure,
            last_context=last_context,
            residual_history=residual_history,
            update_history=update_history,
        )
        state_path = output / "final-state.npz"
        save_state(state_path, state, u, v, extra=extra)
        final_operational_restart_validation = operational_restart_validation
        if (
            final_operational_restart_validation is not None
            and parent_operational_manifest_sha256 is not None
        ):
            final_operational_restart_validation = {
                **final_operational_restart_validation,
                "manifest_sha256": parent_operational_manifest_sha256,
            }
        manifest_path = write_checkpoint_manifest(
            state_path,
            producer="pulse_campaign",
            producer_source_fingerprint=run_solver_source_fingerprint,
            solver_parameters=resume_solver_parameters,
            resume_validation=resume_validation,
            operational_restart_validation=(
                final_operational_restart_validation
            ),
            solver_source_fingerprint_value=(
                run_solver_source_fingerprint
            ),
        )
        result["checkpoint_manifest"] = str(manifest_path)
    # Publish a completed summary only after its declared state and manifest
    # are durable.  A concurrent postprocessor must never observe a completed
    # run whose final scientific artifacts do not yet exist.
    atomic_write_json(output / "summary.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--c", type=float, default=1.0)
    result.add_argument("--Omega_chih-divisor", type=float, default=3.2)
    result.add_argument("--v1", type=float, default=0.5)
    result.add_argument(
        "--v-endpoint",
        type=float,
        help="computed v endpoint; defaults to the full pulse endpoint v1",
    )
    result.add_argument(
        "--v-grid-power",
        type=float,
        default=1.0,
        help="use v=v_endpoint*s^power on a uniform parameter grid",
    )
    result.add_argument(
        "--coordinate-method",
        choices=("local-polynomial", "lgl"),
        default="local-polynomial",
        help="coordinate representation used for first derivatives/integrals",
    )
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
    result.add_argument("--lgl-s-degree", type=int, default=8)
    result.add_argument(
        "--lgl-s-breakpoints-json",
        type=Path,
        help=(
            "audited custom s=sqrt(v/v1) mesh; accepts a JSON list or an "
            "adaptive-refinement object containing s_breakpoints"
        ),
    )
    result.add_argument(
        "--boundary-mode",
        choices=(
            "smooth-hemisphere",
            "low-band-hemisphere",
            "linear-fixed",
        ),
        default="smooth-hemisphere",
    )
    result.add_argument("--points", type=int, default=24)
    result.add_argument("--neighbors", type=int, default=20)
    result.add_argument("--angular-degree", type=int, default=4)
    result.add_argument("--spectral-degree", type=int, default=3)
    result.add_argument(
        "--galerkin-retained-degree",
        type=int,
        help="evolve typed angular harmonics through this degree",
    )
    result.add_argument(
        "--galerkin-work-degree",
        type=int,
        help="oversampled nonlinear work degree (default: twice retained)",
    )
    result.add_argument("--u-count", type=int, default=13)
    result.add_argument("--v-count", type=int, default=97)
    result.add_argument("--iterations", type=int, default=6)
    result.add_argument(
        "--resume-state",
        type=Path,
        help=(
            "continue from a manifested schema-2 checkpoint on the "
            "identical grid and discrete solver map"
        ),
    )
    result.add_argument(
        "--restart-checkpoint",
        type=Path,
        help=(
            "operationally restart this same interrupted invocation from an "
            "immutable per-sweep checkpoint; unlike --resume-state this "
            "restores accumulated records and controller position"
        ),
    )
    result.add_argument(
        "--unsafe-allow-unverified-resume",
        action="store_true",
        help=(
            "diagnostic-only: allow a established checkpoint with no manifest or "
            "an already unverified lineage; semantic, parameter, and content "
            "mismatches are never ignored"
        ),
    )
    result.add_argument(
        "--unsafe-allow-unverified-restart",
        action="store_true",
        help=(
            "diagnostic-only: restart an operational checkpoint whose root "
            "lineage was already marked unsafe; outputs remain uncertified"
        ),
    )
    result.add_argument(
        "--checkpoint-every-sweep",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="publish one immutable operational checkpoint after each sweep",
    )
    result.add_argument(
        "--checkpoint-retain-generations",
        type=int,
        default=2,
        help=(
            "retain this many valid same-run operational generations; must "
            "be at least two so a damaged newest generation has a fallback"
        ),
    )
    result.add_argument(
        "--iteration-offset",
        type=int,
        default=0,
        help="label the first new sweep as offset+1",
    )
    result.add_argument(
        "--g-substeps", dest="metric_substeps", type=int, default=2
    )
    result.add_argument(
        "--g-parameterization",
        dest="metric_parameterization",
        choices=("direct", "cholesky"),
        default="direct",
        help="direct RK g or an SPD-preserving g factor",
    )
    result.add_argument(
        "--g-integrator",
        dest="metric_integrator",
        choices=("rk4", "sdc"),
        default="rk4",
    )
    result.add_argument("--sdc-tolerance", type=float, default=1.0e-10)
    result.add_argument(
        "--sdc-overgrid-tolerance",
        type=float,
        default=1.0e-7,
        help="independent between-node SDC defect required for acceptance",
    )
    result.add_argument("--sdc-maximum-corrections", type=int, default=12)
    result.add_argument(
        "--u-integrator",
        choices=("rk4", "sdc"),
        default="rk4",
        help="integrator for the four incoming-direction construction marches",
    )
    result.add_argument("--u-sdc-tolerance", type=float, default=1.0e-10)
    result.add_argument(
        "--u-sdc-overgrid-tolerance",
        type=float,
        default=1.0e-7,
        help="independent between-node defect gate for each tau construction",
    )
    result.add_argument(
        "--u-sdc-maximum-corrections", type=int, default=12
    )
    result.add_argument(
        "--u-sdc-half-trace-tolerance",
        type=float,
        default=1.0e-9,
        help="independent moving trace-free gate for the half Omega_chih",
    )
    result.add_argument(
        "--u-sdc-half-trace-absolute-floor",
        type=float,
        default=1.0e-14,
        help=(
            "small-field absolute floor in the scale-aware relative "
            "half-Omega_chih trace gate"
        ),
    )
    result.add_argument("--u-halo", type=int, default=4)
    result.add_argument("--v-halo", type=int, default=5)
    result.add_argument(
        "--construction-v-halo",
        type=int,
        default=0,
        help="v halo for source-aware construction residual summaries",
    )
    result.add_argument("--rho-epsilon", type=float, default=0.2)
    result.add_argument("--rho-min", type=float, default=1.0e-7)
    result.add_argument("--rho-max", type=float, default=2.0e-2)
    result.add_argument("--output-label")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
