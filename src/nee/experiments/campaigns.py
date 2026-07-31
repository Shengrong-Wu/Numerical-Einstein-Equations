"""Campaign dispatch preserving the numerical kernels of each experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nee.config import ExperimentConfig
from nee.exact_solutions import fisher_jnw, vacuum_benchmarks

from . import _reference_campaign as reference


def _smoke(config: ExperimentConfig) -> bool:
    return config.source_path is not None and config.source_path.stem == "smoke"


def regular_vacuum(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    if not _smoke(config):
        return reference.run_experiment_1(output)
    summary = reference.vacuum_case(
        case_id="exp01-smoke",
        builder=lambda grid, u, v: vacuum_benchmarks.regular_schwarzschild_state(
            grid, u, v, 1.0
        ),
        u=np.linspace(-1.0, -0.5, 17),
        v=np.linspace(0.0, 0.5, 17),
        points=40,
        retained_degree=3,
        iterations=1,
        output=output / "regular-schwarzschild",
    )
    output.mkdir(parents=True, exist_ok=True)
    reference.write_json(output / "aggregate-summary.json", {"experiment": 1, "runs": [summary]})
    return {"experiment": 1, "runs": [summary]}


def schwarzschild_horizon(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    if not _smoke(config):
        return reference.run_experiment_2(output)
    summary = reference.vacuum_case(
        case_id="exp02-smoke-kruskal",
        builder=lambda grid, u, v: vacuum_benchmarks.kruskal_state(
            grid, u, v, 1.0, u_offset=0.75, v_offset=1.0
        ),
        u=np.linspace(-1.0, -0.5, 17),
        v=np.linspace(0.0, 0.1, 17),
        points=40,
        retained_degree=3,
        iterations=1,
        output=output / "kruskal-crossing",
    )
    output.mkdir(parents=True, exist_ok=True)
    reference.write_json(output / "aggregate-summary.json", {"experiment": 2, "runs": [summary]})
    return {"experiment": 2, "runs": [summary]}


def schwarzschild_interior(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    if not _smoke(config):
        return reference.run_experiment_3(output)
    from nee.geometry.curved_sphere import exact_fields, overgrid_audit, solve

    output.mkdir(parents=True)
    solution = solve(
        0.5,
        17,
        17,
        iterations=config.solver.maximum_sweeps,
        tolerance=config.solver.tolerance,
    )
    for record in solution.records:
        print(
            "exp03-smoke: sweep "
            f"{int(record['iteration'])}/{config.solver.maximum_sweeps} "
            f"update={record['update']:.6e}"
        )
    exact = exact_fields(
        solution.u, solution.xi, 0.5, 1.0, high_precision=False
    )
    np.savez_compressed(
        output / "boundary-data.npz",
        u=solution.u,
        v=solution.v,
        radius_u0=exact["radius"][:, 0],
        radius_v0=exact["radius"][0, :],
    )
    np.savez_compressed(
        output / "final-state.npz",
        u=solution.u,
        v=solution.v,
        radius=solution.radius,
        log_lapse=solution.log_omega,
    )
    residual = overgrid_audit(solution, 0.5)
    np.savez_compressed(output / "residual-maps.npz", radius_error=np.abs(solution.radius - exact["radius"]))
    summary = {
        "experiment": 3,
        "terminal_status": "completed",
        "epsilon": 0.5,
        "sweeps": len(solution.records),
        "radius_error_maximum": float(np.max(np.abs(solution.radius - exact["radius"]))),
        "independent_audit": residual,
    }
    reference.write_json(output / "summary.json", summary)
    return summary


def strong_vacuum_pulse(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    from .exp04_vacuum_strong_short_pulse import campaign

    if not _smoke(config):
        return campaign.run_standard(
            output, iterations=config.solver.maximum_sweeps
        )
    output.mkdir(parents=True)
    summary = campaign._run_one(
        output_root=output,
        label="smoke",
        strength=1.0,
        cap=0.005,
        u_count=5,
        v_count=9,
        retained=3,
        work=5,
        points=50,
        iterations=1,
    )
    aggregate = {"experiment": 4, "runs": [summary]}
    reference.write_json(output / "aggregate-summary.json", aggregate)
    return aggregate


def crossed_vacuum_pulses(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    from .exp05_vacuum_crossed_pulses import campaign

    if not _smoke(config):
        return campaign.run(output, quick=False, reported_only=True)
    output.mkdir(parents=True)
    resolution = campaign.Resolution(
        "smoke", 6, 4, 3, 6, 100, 24, 20, 4, 2, 1.0, 1.0e-8
    )
    summary = campaign.run_level(output / "smoke", resolution)
    aggregate = {"experiment": 5, "levels": [summary], "reported_summary": summary}
    reference.write_json(output / "aggregate-summary.json", aggregate)
    return aggregate


def regular_exact_scalar(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    if not _smoke(config):
        return reference.run_experiment_5(output)
    numerical = fisher_jnw._configuration(
        name="smoke",
        tau_elements=2,
        tau_degree=6,
        s_elements=2,
        s_degree=6,
        iterations=1,
        quick=True,
    )
    summary = reference.ese_case(nu=0.8, level=0, config=numerical, output=output / "nu-0.80")
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {"experiment": 6, "runs": [summary]}
    reference.write_json(output / "aggregate-summary.json", aggregate)
    return aggregate


def nonspherical_scalar(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    if not _smoke(config):
        return reference.run_experiment_6(output)
    output.mkdir(parents=True)
    summary = reference.nonspherical_ese_case(
        output=output,
        name="smoke",
        cap=0.02,
        multipliers=(1.0, 1.0, 1.0, 1.0),
        retained=3,
        work=6,
        points=100,
        tau_elements=2,
        s_elements=2,
        tau_degree=8,
        s_degree=11,
        metric_substeps=1,
        derivative_halo=2,
        iterations=1,
    )
    aggregate = {"experiment": 7, "runs": [summary]}
    reference.write_json(output / "aggregate-summary.json", aggregate)
    return aggregate


def trapped_scalar(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    from .exp08_scalar_trapped_section.campaign import run_configuration

    control = config.source_path is not None and config.source_path.stem == "angular-control"
    return run_configuration(output, control=control, quick=_smoke(config))
