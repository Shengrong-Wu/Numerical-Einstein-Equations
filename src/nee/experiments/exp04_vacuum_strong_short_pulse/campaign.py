"""Experiment 4 two-hemisphere vacuum-pulse matrix.

The shared vacuum kernel regenerates the boundary constraint data for every
run and records failures without stopping later strengths or continuation
caps.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np


from nee.diagnostics.independent_audit import PrimitiveFields
from nee.diagnostics.mapped_audit import evaluate_mapped_overgrid
from nee.numerics import pulse_campaign as q1
from nee.numerics.lgl import CharacteristicLGLMesh
from nee.numerics.sphere import PointSphereGrid
from nee.numerics.vacuum_state_io import load_state


def _run_one(
    *,
    output_root: Path,
    label: str,
    strength: float,
    cap: float,
    u_count: int,
    v_count: int,
    retained: int,
    work: int,
    points: int,
    iterations: int,
) -> dict[str, Any]:
    target = output_root / "results" / "Q1" / label
    if (target / "summary.json").exists():
        return json.loads((target / "summary.json").read_text(encoding="utf-8"))
    arguments = [
        "--boundary-mode",
        "low-band-hemisphere",
        "--c",
        str(strength),
        "--Omega_chih-divisor",
        "3.2",
        "--v1",
        "0.5",
        "--v-endpoint",
        str(cap),
        "--v-grid-power",
        "2",
        "--coordinate-method",
        "local-polynomial",
        "--points",
        str(points),
        "--neighbors",
        str(min(32, points - 1)),
        "--spectral-degree",
        str(work + 1),
        "--galerkin-retained-degree",
        str(retained),
        "--galerkin-work-degree",
        str(work),
        "--u-count",
        str(u_count),
        "--v-count",
        str(v_count),
        "--iterations",
        str(iterations),
        "--g-substeps",
        "1",
        "--u-halo",
        "1",
        "--v-halo",
        "1",
        "--rho-max",
        "1.0",
        "--output-label",
        label,
        "--no-checkpoint-every-sweep",
    ]
    parsed = q1.parser().parse_args(arguments)
    old_root = q1.ROOT
    old_calibrator = q1.calibrate_low_band_profiles
    q1.ROOT = output_root
    if strength == 0.0:
        def zero_calibrator(
            v1: float, c: float, divisor: float = 3.2, **kwargs: Any
        ) -> Any:
            del c
            base = old_calibrator(v1, 1.0, divisor=divisor, **kwargs)
            return replace(
                base,
                c=0.0,
                first_amplitude=0.0,
                second_amplitude=0.0,
            )

        q1.calibrate_low_band_profiles = zero_calibrator
    try:
        return q1.run(parsed)
    finally:
        q1.ROOT = old_root
        q1.calibrate_low_band_profiles = old_calibrator


def _run_spectral_case(
    *,
    output_root: Path,
    label: str,
    strength: float,
    iterations: int,
    public,
) -> dict[str, Any]:
    """Run the retained mapped LGL/Galerkin configuration from fresh data."""

    mesh_path = output_root / f"{label}-s-breakpoints.json"
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text(
        json.dumps(np.sqrt(np.asarray(public.coordinates.v_breakpoints) / float(public.physics["v1"])).tolist()) + "\n"
    )
    arguments = [
        "--boundary-mode", "low-band-hemisphere",
        "--c", str(strength),
        "--Omega_chih-divisor", str(public.initial_data["Omega_chih_divisor"]),
        "--v1", str(public.physics["v1"]),
        "--v-endpoint", str(public.physics["cap"]),
        "--coordinate-method", "lgl",
        "--lgl-u-elements", str(len(public.coordinates.u_degrees)),
        "--lgl-u-degree", str(public.coordinates.u_degrees[0]),
        "--lgl-s-degree", str(public.coordinates.v_degrees[0]),
        "--lgl-s-breakpoints-json", str(mesh_path),
        "--points", str(public.angular.point_count),
        "--neighbors", str(public.angular.neighbor_count),
        "--angular-degree", "4",
        "--spectral-degree", str(public.angular.differentiation_degree),
        "--galerkin-retained-degree", str(public.angular.retained_degree),
        "--galerkin-work-degree", str(public.angular.work_degree),
        "--iterations", str(iterations),
        "--g-substeps", str(public.solver.metric_substeps),
        "--g-parameterization", "cholesky",
        "--g-integrator", "sdc",
        "--sdc-tolerance", str(public.solver.tolerance),
        "--sdc-overgrid-tolerance", "1e-7",
        "--sdc-maximum-corrections", str(public.solver.sdc_sweeps),
        "--u-integrator", "sdc",
        "--u-sdc-tolerance", str(public.solver.tolerance),
        "--u-sdc-overgrid-tolerance", "1e-7",
        "--u-sdc-maximum-corrections", str(public.solver.sdc_sweeps),
        "--u-sdc-half-trace-tolerance", "1e-9",
        "--u-sdc-half-trace-absolute-floor", "1e-14",
        "--u-halo", "2",
        "--v-halo", "1",
        "--output-label", label,
    ]
    parsed = q1.parser().parse_args(arguments)
    previous_root = q1.ROOT
    previous_calibrator = q1.calibrate_low_band_profiles
    q1.ROOT = output_root
    if strength == 0.0:
        def zero_calibrator(
            v1: float, c: float, divisor: float = 3.2, **kwargs: Any
        ) -> Any:
            del c
            base = previous_calibrator(v1, 1.0, divisor=divisor, **kwargs)
            return replace(base, c=0.0, first_amplitude=0.0, second_amplitude=0.0)

        q1.calibrate_low_band_profiles = zero_calibrator
    try:
        return q1.run(parsed)
    finally:
        q1.ROOT = previous_root
        q1.calibrate_low_band_profiles = previous_calibrator


def _audit_mesh(public):
    grid = PointSphereGrid.create(public.angular.point_count,
        neighbor_count=public.angular.neighbor_count, degree=4,
        spectral_degree=public.angular.differentiation_degree)
    coordinates = CharacteristicLGLMesh.create(
        -np.log(-np.asarray(public.coordinates.u_breakpoints)), public.coordinates.u_degrees[0],
        np.sqrt(np.asarray(public.coordinates.v_breakpoints)/float(public.physics['v1'])),
        public.coordinates.v_degrees[0], float(public.physics['v1']))
    return grid, coordinates


def _mapped_audit(output: Path, label: str, public) -> dict[str, Any]:
    target = output / "results" / "Q1" / label
    state, u, v = load_state(target / "final-state.npz")
    grid, coordinates = _audit_mesh(public)
    if not np.array_equal(u, coordinates.u) or not np.array_equal(
        v, coordinates.v
    ):
        raise ValueError("pulse state nodes do not match its declared LGL mesh")
    maximum_s = float(coordinates.s.nodes[-1])
    from nee.diagnostics.state_audit import audit
    result = audit(grid, state, u, v, coordinates=coordinates,
        retained_degree=public.angular.retained_degree,
        protected_s_values=tuple(fraction*maximum_s for fraction in (0.2, 0.4, 0.6, 0.8)))
    (target / "independent-first-order-audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return result


def _exact_zero_control_audit(public) -> dict[str, Any]:
    grid, coordinates = _audit_mesh(public)
    radius = coordinates.v[None, None, :] - coordinates.u[None, :, None]
    scalar_shape = (grid.count, len(coordinates.u), len(coordinates.v))
    maximum_s = float(coordinates.s.nodes[-1])
    from nee.diagnostics.state_audit import audit
    from nee.state.iterate import PicardState
    metric = radius[..., None, None]**2 * grid.projector[:, None, None]
    form = radius[..., None, None] * grid.projector[:, None, None]
    zero = np.zeros(scalar_shape)
    state = PicardState(g=metric, b=np.zeros(scalar_shape+(3,)), log_Omega=zero,
        Omega_chi=form, Omega_chib=-form, zeta=np.zeros(scalar_shape+(3,)),
        Omega_omega=zero.copy(), Omega_omegab=zero.copy())
    return audit(grid, state, coordinates.u, coordinates.v, coordinates=coordinates,
        retained_degree=public.angular.retained_degree,
        protected_s_values=tuple(fraction*maximum_s for fraction in (0.2, 0.4, 0.6, 0.8)))


def run_standard(
    output: Path, *, public, iterations: int = 6
) -> dict[str, Any]:
    """Run the strong pulse and same-discretization zero control."""

    if output.exists():
        raise FileExistsError(f"immutable run directory exists: {output}")
    output.mkdir(parents=True)
    strong = _run_spectral_case(
        output_root=output,
        label="strong-pulse",
        strength=float(public.physics["pulse_strength"]),
        iterations=iterations,
        public=public,
    )
    control = _run_spectral_case(
        output_root=output,
        label="zero-control",
        strength=float(public.physics["control_strength"]),
        iterations=iterations,
        public=public,
    )
    strong_audit = _mapped_audit(output, "strong-pulse", public)
    numerical_control_audit = _mapped_audit(output, "zero-control", public)
    exact_control_audit = _exact_zero_control_audit(public)
    aggregate = {
        "schema": "nee-vacuum-strong-pulse-1",
        "experiment": 4,
        "strong_pulse": strong,
        "zero_control": control,
        "strong_pulse_independent_first_order_residual": strong_audit,
        "numerical_zero_control_independent_first_order_residual": (
            numerical_control_audit
        ),
        "exact_zero_control_independent_first_order_residual": (
            exact_control_audit
        ),
    }
    (output / "aggregate-summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    return aggregate


def _failure(
    output_root: Path, label: str, error: Exception
) -> dict[str, Any]:
    target = output_root / "results" / "Q1" / label
    target.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "nee-official-vacuum-pulse-failure-v1",
        "case_id": label,
        "terminal_status": "failed",
        "failure_classification": (
            "Picard contraction, positivity, or discretization"
        ),
        "error": repr(error),
    }
    (target / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def run_experiment(output: Path, quick: bool) -> dict[str, Any]:
    if not quick:
        raise NotImplementedError(
            "the production pulse matrix is intentionally not implicit; "
            "run the audited continuation controller with a declared budget"
        )
    strengths = (0.0, 0.5, 1.0, 1.5)
    caps = (0.01, 0.02, 0.04, 0.10, 0.25, 0.50)
    summaries: list[dict[str, Any]] = []

    # Complete stress/cap matrix at a plan-compatible retained degree L=5.
    for strength in strengths:
        for cap in caps:
            label = f"strength-{strength:.1f}-cap-{cap:.2f}-base"
            try:
                summary = _run_one(
                    output_root=output,
                    label=label,
                    strength=strength,
                    cap=cap,
                    u_count=9,
                    v_count=17,
                    retained=5,
                    work=10,
                    points=180,
                    iterations=3,
                )
            except Exception as error:
                summary = _failure(output, label, error)
            summaries.append(summary)

    # Three coordinate levels for the source-data baseline at v=0.04.
    for level, (u_count, v_count) in enumerate(((9, 17), (13, 25), (17, 33))):
        if level == 0:
            continue
        label = f"strength-1.0-cap-0.04-coordinate-{level}"
        try:
            summary = _run_one(
                output_root=output,
                label=label,
                strength=1.0,
                cap=0.04,
                u_count=u_count,
                v_count=v_count,
                retained=5,
                work=10,
                points=180,
                iterations=4,
            )
        except Exception as error:
            summary = _failure(output, label, error)
        summaries.append(summary)

    # Three angular bands at the same coordinate grid.
    for level, (retained, work, points) in enumerate(
        ((5, 10, 180), (6, 12, 220), (7, 14, 300))
    ):
        if level == 0:
            continue
        label = f"strength-1.0-cap-0.04-angular-{level}"
        try:
            summary = _run_one(
                output_root=output,
                label=label,
                strength=1.0,
                cap=0.04,
                u_count=13,
                v_count=25,
                retained=retained,
                work=work,
                points=points,
                iterations=4,
            )
        except Exception as error:
            summary = _failure(output, label, error)
        summaries.append(summary)

    aggregate = {
        "schema": "nee-official-experiment-04-aggregate-v1",
        "experiment": 4,
        "runs": [
            {
                "case_id": item.get("case_id", item.get("configuration", {}).get("output_label")),
                "terminal_status": item.get("terminal_status", "completed"),
                "summary": item,
            }
            for item in summaries
        ],
        "matrix": {
            "strengths": strengths,
            "caps": caps,
            "base_resolution": {
                "u_count": 9,
                "v_count": 17,
                "retained_degree": 5,
                "work_degree": 10,
                "sphere_points": 180,
                "picard_sweeps": 3,
            },
            "note": (
                "The full strength/cap matrix is a stress survey.  Only the "
                "lambda=1, v_cap=0.04 baseline receives additional coordinate "
                "and angular levels in this fresh batch."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    (output / "aggregate-summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable run directory exists: {args.output}")
    args.output.mkdir(parents=True)
    run_experiment(args.output, args.quick)


if __name__ == "__main__":
    main()
