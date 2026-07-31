"""A posteriori h-refinement for the characteristic LGL mesh.

The collocation equations can be satisfied at the LGL nodes even when the
polynomial between those nodes is badly resolved.  This module turns the
independent overgrid defect (and any nonfinite element failure) into a new
mesh.  It deliberately operates on mesh metadata rather than Einstein state
arrays, so a refinement decision is reproducible and can be audited before a
large run is launched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np


Array = np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def validate_s_breakpoints(
    breakpoints: Iterable[float], *, v1: float, v_endpoint: float
) -> Array:
    """Validate an ``s=sqrt(v/v1)`` mesh for a requested slab endpoint."""

    points = np.asarray(tuple(breakpoints), dtype=float)
    endpoint = math.sqrt(v_endpoint / v1)
    if (
        points.ndim != 1
        or len(points) < 2
        or not np.all(np.isfinite(points))
        or np.any(np.diff(points) <= 0.0)
        or not np.isclose(points[0], 0.0, rtol=0.0, atol=2.0e-14)
        or not np.isclose(points[-1], endpoint, rtol=0.0, atol=2.0e-14)
    ):
        raise ValueError(
            "the custom s mesh must be strictly increasing from 0 to "
            f"sqrt(v_endpoint/v1)={endpoint:.17g}"
        )
    return points


def load_s_breakpoints(
    path: Path, *, v1: float, v_endpoint: float
) -> Array:
    """Load either a JSON list or an object containing ``s_breakpoints``."""

    payload = json.loads(path.read_text())
    values = payload.get("s_breakpoints") if isinstance(payload, dict) else payload
    if values is None:
        raise ValueError(f"{path} does not contain s_breakpoints")
    return validate_s_breakpoints(values, v1=v1, v_endpoint=v_endpoint)


def _diagnostic_interval(
    item: Mapping[str, object], *, v1: float
) -> tuple[float, float]:
    if "s_left" in item and "s_right" in item:
        return float(item["s_left"]), float(item["s_right"])
    if "v_left" in item and "v_right" in item:
        return (
            math.sqrt(max(float(item["v_left"]), 0.0) / v1),
            math.sqrt(max(float(item["v_right"]), 0.0) / v1),
        )
    raise ValueError("an element diagnostic lacks both s and v endpoints")


def refinement_factor(
    item: Mapping[str, object],
    *,
    polynomial_degree: int,
    overgrid_tolerance: float,
    failure_subdivisions: int = 2,
    maximum_subdivisions: int = 8,
) -> tuple[int, str]:
    """Return an h-refinement factor and a human-readable decision reason."""

    if polynomial_degree < 2:
        raise ValueError("polynomial_degree must be at least two")
    if overgrid_tolerance <= 0.0:
        raise ValueError("overgrid_tolerance must be positive")
    if failure_subdivisions < 2 or maximum_subdivisions < 2:
        raise ValueError("subdivision limits must be at least two")

    status = str(item.get("status", "completed"))
    if status != "completed" or "overgrid_defect" not in item:
        return min(failure_subdivisions, maximum_subdivisions), status

    defect = float(item["overgrid_defect"])
    if not np.isfinite(defect):
        return min(failure_subdivisions, maximum_subdivisions), "nonfinite defect"
    if defect <= overgrid_tolerance:
        return 1, "accepted"

    # For a smooth degree-p element, a derivative defect scales nominally as
    # h^p.  This estimate is only a first proposal: the next solve must pass
    # the independent overgrid test again.  Clamping prevents one poor pilot
    # panel from producing an impractically large mesh in a single step.
    estimate = int(
        math.ceil((defect / overgrid_tolerance) ** (1.0 / polynomial_degree))
    )
    factor = min(max(2, estimate), maximum_subdivisions)
    return factor, f"overgrid defect {defect:.6g} > {overgrid_tolerance:.6g}"


def refine_s_breakpoints(
    breakpoints: Iterable[float],
    diagnostics: Iterable[Mapping[str, object]],
    *,
    v1: float,
    polynomial_degree: int,
    overgrid_tolerance: float,
    failure_subdivisions: int = 2,
    maximum_subdivisions: int = 8,
) -> tuple[Array, list[dict[str, object]]]:
    """Refine diagnosed elements and return the mesh plus decision records."""

    base = np.asarray(tuple(breakpoints), dtype=float)
    if base.ndim != 1 or len(base) < 2 or np.any(np.diff(base) <= 0.0):
        raise ValueError("breakpoints must be strictly increasing")
    records = list(diagnostics)
    matched: set[int] = set()
    output: list[float] = [float(base[0])]
    decisions: list[dict[str, object]] = []

    for element, (left, right) in enumerate(zip(base[:-1], base[1:]), start=1):
        match_index = None
        match = None
        for index, item in enumerate(records):
            candidate_left, candidate_right = _diagnostic_interval(item, v1=v1)
            if np.isclose(candidate_left, left, rtol=0.0, atol=2.0e-12) and np.isclose(
                candidate_right, right, rtol=0.0, atol=2.0e-12
            ):
                match_index = index
                match = item
                break
        if match is None:
            factor, reason = 1, "no diagnostic"
        else:
            matched.add(int(match_index))
            factor, reason = refinement_factor(
                match,
                polynomial_degree=polynomial_degree,
                overgrid_tolerance=overgrid_tolerance,
                failure_subdivisions=failure_subdivisions,
                maximum_subdivisions=maximum_subdivisions,
            )

        local = np.linspace(left, right, factor + 1)[1:]
        output.extend(float(value) for value in local)
        decisions.append(
            {
                "element": element,
                "s_left": float(left),
                "s_right": float(right),
                "v_left": float(v1 * left**2),
                "v_right": float(v1 * right**2),
                "subdivisions": factor,
                "reason": reason,
            }
        )

    unmatched = set(range(len(records))) - matched
    if unmatched:
        raise ValueError(
            "diagnostic intervals do not match the base mesh: "
            + ", ".join(str(index + 1) for index in sorted(unmatched))
        )
    return np.asarray(output), decisions


def _coordinate_breakpoints(summary: Mapping[str, object]) -> Array:
    parameters = summary.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError("summary lacks parameters")
    mesh = parameters.get("coordinate_mesh")
    if not isinstance(mesh, dict):
        mesh = summary.get("full_coordinate_mesh")
    if not isinstance(mesh, dict) or "s_breakpoints" not in mesh:
        raise ValueError("summary lacks a characteristic coordinate mesh")
    return np.asarray(mesh["s_breakpoints"], dtype=float)


def build_refinement_spec(args: argparse.Namespace) -> dict[str, object]:
    summary_path = args.summary.resolve()
    diagnostics_path = args.diagnostics.resolve()
    summary = json.loads(summary_path.read_text())
    payload = json.loads(diagnostics_path.read_text())
    if isinstance(payload, dict):
        diagnostics = payload.get("element_diagnostics")
        if diagnostics is None:
            diagnostics = payload.get("failure_diagnostics")
        if diagnostics is None:
            stages = payload.get("stages")
            if isinstance(stages, list):
                for stage in reversed(stages):
                    if isinstance(stage, dict) and stage.get("failure_diagnostics"):
                        diagnostics = stage["failure_diagnostics"]
                        break
        if diagnostics is None:
            boundary = payload.get("boundary")
            if isinstance(boundary, dict):
                diagnostics = boundary.get("boundary_sdc_diagnostics")
    else:
        diagnostics = payload
    if not isinstance(diagnostics, list):
        raise ValueError("diagnostics must be a list or contain element_diagnostics")

    base = _coordinate_breakpoints(summary)
    refined, decisions = refine_s_breakpoints(
        base,
        diagnostics,
        v1=args.v1,
        polynomial_degree=args.polynomial_degree,
        overgrid_tolerance=args.overgrid_tolerance,
        failure_subdivisions=args.failure_subdivisions,
        maximum_subdivisions=args.maximum_subdivisions,
    )
    return {
        "method": "independent-overgrid-defect h-refinement",
        "v1": args.v1,
        "polynomial_degree": args.polynomial_degree,
        "overgrid_tolerance": args.overgrid_tolerance,
        "failure_subdivisions": args.failure_subdivisions,
        "maximum_subdivisions": args.maximum_subdivisions,
        "source_summary": str(summary_path),
        "source_summary_sha256": _sha256(summary_path),
        "source_diagnostics": str(diagnostics_path),
        "source_diagnostics_sha256": _sha256(diagnostics_path),
        "base_element_count": len(base) - 1,
        "refined_element_count": len(refined) - 1,
        "s_breakpoints": refined.tolist(),
        "decisions": decisions,
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--summary", type=Path, required=True)
    result.add_argument("--diagnostics", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--v1", type=float, default=0.5)
    result.add_argument("--polynomial-degree", type=int, default=8)
    result.add_argument("--overgrid-tolerance", type=float, default=1.0e-7)
    result.add_argument("--failure-subdivisions", type=int, default=2)
    result.add_argument("--maximum-subdivisions", type=int, default=8)
    return result


if __name__ == "__main__":
    arguments = parser().parse_args()
    specification = build_refinement_spec(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(specification, indent=2) + "\n")
    print(json.dumps(specification, indent=2))
