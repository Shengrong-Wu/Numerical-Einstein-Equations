"""Campaign dispatch preserving the numerical kernels of each experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nee.config import ExperimentConfig
from nee.exact_solutions import fisher_jnw, vacuum_benchmarks

from . import _campaign_support as support


def _smoke(config: ExperimentConfig) -> bool:
    return config.experiment.mode == "smoke"


def regular_vacuum(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    if not _smoke(config):
        return support.run_experiment_1(output, config)
    summary = support.vacuum_case(
        case_id="exp01-smoke",
        builder=lambda grid, u, v: vacuum_benchmarks.regular_schwarzschild_state(
            grid, u, v, float(config.physics["mass"])
        ),
        u=np.linspace(config.coordinates.u_min, config.coordinates.u_max, config.coordinates.node_count(config.coordinates.u_degrees)),
        v=np.linspace(config.coordinates.v_min, config.coordinates.v_max, config.coordinates.node_count(config.coordinates.v_degrees)),
        points=config.angular.point_count,
        retained_degree=config.angular.retained_degree,
        iterations=config.solver.maximum_sweeps,
        output=output / "regular-schwarzschild",
    )
    output.mkdir(parents=True, exist_ok=True)
    support.write_json(output / "aggregate-summary.json", {"experiment": 1, "runs": [summary]})
    return {"experiment": 1, "runs": [summary]}


def schwarzschild_horizon(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    if not _smoke(config):
        return support.run_experiment_2(output, config)
    summary = support.vacuum_case(
        case_id="exp02-smoke-kruskal",
        builder=lambda grid, u, v: vacuum_benchmarks.kruskal_state(
            grid, u, v, float(config.physics["mass"]), u_offset=0.75, v_offset=1.0
        ),
        u=np.linspace(config.coordinates.u_min, config.coordinates.u_max, config.coordinates.node_count(config.coordinates.u_degrees)),
        v=np.linspace(config.coordinates.v_min, config.coordinates.v_max, config.coordinates.node_count(config.coordinates.v_degrees)),
        points=config.angular.point_count,
        retained_degree=config.angular.retained_degree,
        iterations=config.solver.maximum_sweeps,
        output=output / "kruskal-crossing",
    )
    output.mkdir(parents=True, exist_ok=True)
    support.write_json(output / "aggregate-summary.json", {"experiment": 2, "runs": [summary]})
    return {"experiment": 2, "runs": [summary]}


def schwarzschild_interior(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    if not _smoke(config):
        return support.run_experiment_3(output, config)
    from nee.geometry.curved_sphere import exact_fields, overgrid_audit, solve

    output.mkdir(parents=True)
    solution = solve(
        float(config.physics["epsilon"]),
        config.coordinates.node_count(config.coordinates.u_degrees),
        config.coordinates.node_count(config.coordinates.v_degrees),
        mass=float(config.physics["mass"]),
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
        solution.u, solution.xi, float(config.physics["epsilon"]), float(config.physics["mass"]), high_precision=False
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
        log_Omega=solution.log_Omega,
    )
    residual = overgrid_audit(solution, float(config.physics["epsilon"]), mass=float(config.physics["mass"]))
    np.savez_compressed(output / "residual-maps.npz", radius_error=np.abs(solution.radius - exact["radius"]))
    summary = {
        "experiment": 3,
        "terminal_status": "completed",
        "epsilon": float(config.physics["epsilon"]),
        "sweeps": len(solution.records),
        "radius_error_maximum": float(np.max(np.abs(solution.radius - exact["radius"]))),
        "independent_audit": residual,
    }
    support.write_json(output / "summary.json", summary)
    return summary


def strong_vacuum_pulse(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    from .exp04_vacuum_strong_short_pulse import campaign

    if not _smoke(config):
        return campaign.run_standard(
            output, public=config, iterations=config.solver.maximum_sweeps
        )
    output.mkdir(parents=True)
    summary = campaign._run_one(
        output_root=output,
        label="smoke",
        strength=float(config.physics["pulse_strength"]),
        cap=float(config.physics["cap"]),
        u_count=config.coordinates.node_count(config.coordinates.u_degrees),
        v_count=config.coordinates.node_count(config.coordinates.v_degrees),
        retained=config.angular.retained_degree,
        work=config.angular.work_degree,
        points=config.angular.point_count,
        iterations=config.solver.maximum_sweeps,
    )
    aggregate = {"experiment": 4, "runs": [summary]}
    support.write_json(output / "aggregate-summary.json", aggregate)
    return aggregate


def crossed_vacuum_pulses(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    from .exp05_vacuum_crossed_pulses import campaign

    output.mkdir(parents=True)
    degrees = (*config.coordinates.u_degrees, *config.coordinates.v_degrees)
    if len(set(degrees)) != 1 or len(config.coordinates.u_degrees) != len(config.coordinates.v_degrees):
        raise ValueError("crossed pulses require equal uniform u/v element degrees")
    resolution = campaign.Resolution(
        "smoke" if _smoke(config) else "level-3",
        len(config.coordinates.u_degrees), degrees[0],
        config.angular.retained_degree, config.angular.work_degree,
        config.angular.point_count, config.angular.neighbor_count,
        config.solver.maximum_sweeps, config.solver.boundary_substeps,
        config.solver.metric_substeps, config.solver.relaxation, config.solver.tolerance,
    )
    summary = campaign.run_level(
        output / resolution.name, resolution,
        outgoing_amplitude=float(config.physics["outgoing_amplitude"]),
        incoming_amplitude=float(config.physics["incoming_amplitude"]),
    )
    aggregate = {"experiment": 5, "levels": [summary], "reported_summary": summary}
    support.write_json(output / "aggregate-summary.json", aggregate)
    return aggregate


def regular_exact_scalar(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    if not _smoke(config):
        return support.run_experiment_6(output, config)
    numerical = fisher_jnw._configuration(
        name="smoke",
        tau_elements=len(config.coordinates.u_degrees),
        tau_degree=config.coordinates.u_degrees[0],
        s_elements=len(config.coordinates.v_degrees),
        s_degree=config.coordinates.v_degrees[0],
        iterations=config.solver.maximum_sweeps,
        quick=True,
    )
    summary = support.ese_case(nu=float(config.physics["nu_values"][0]), level=0, config=numerical, output=output / f"nu-{config.physics['nu_values'][0]:.2f}", sigma=float(config.physics["sigma"]))
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {"experiment": 6, "runs": [summary]}
    support.write_json(output / "aggregate-summary.json", aggregate)
    return aggregate


def nonspherical_scalar(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    if not _smoke(config):
        return support.run_experiment_7(output, config)
    output.mkdir(parents=True)
    summary = support.nonspherical_ese_case(
        output=output,
        name="smoke",
        cap=float(config.physics["central_cap"]),
        multipliers=(1.0, 1.0, 1.0, 1.0),
        public_config=config,
        retained=config.angular.retained_degree,
        work=config.angular.work_degree,
        points=config.angular.point_count,
        tau_elements=len(config.coordinates.u_degrees),
        s_elements=len(config.coordinates.v_degrees),
        tau_degree=config.coordinates.u_degrees[0],
        s_degree=config.coordinates.v_degrees[0],
        metric_substeps=config.solver.metric_substeps,
        derivative_halo=config.audit.derivative_halo_u,
        iterations=config.solver.maximum_sweeps,
    )
    aggregate = {"experiment": 7, "runs": [summary]}
    support.write_json(output / "aggregate-summary.json", aggregate)
    return aggregate


def trapped_scalar(config: ExperimentConfig, output: Path) -> dict[str, Any]:
    from .exp08_scalar_trapped_section.campaign import run_configuration

    return run_configuration(output, config)
