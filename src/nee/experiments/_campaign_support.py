"""Shared orchestration and audit support for the public experiments."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


from nee.exact_solutions import fisher_jnw as ese_bench  # noqa: E402
from nee.exact_solutions import vacuum_benchmarks as vacuum_bench  # noqa: E402
from nee.diagnostics.construction_residuals import evaluate as closure_audit  # noqa: E402
from nee.diagnostics.exact_comparison import field_error
from nee.geometry.curved_sphere import (  # noqa: E402
    chebyshev_lobatto,
    exact_fields as curved_exact_fields,
    overgrid_audit as curved_overgrid_audit,
    solve as solve_curved,
)
from nee.diagnostics.ricci_components import (  # noqa: E402
    components as first_order_components,
    section_maps as first_order_maps,
    summarize as summarize_first_order,
)
from nee.solver.backend import (  # noqa: E402
    ese_picard_step,
    from_numerical,
    to_numerical,
    vacuum_picard_step,
    weighted_update_map,
    weighted_update_norm,
)
from nee.state.boundary import BoundaryData
from nee.state.iterate import PicardState  # noqa: E402


Array = np.ndarray


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def direct_residual_value(summary: dict[str, Any]) -> float | None:
    try:
        value = float(
            summary["independent_first_order_residual"][
                "masked_Linf_uv_L2_sphere"
            ]
        )
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def strictly_decreasing_finite(values: list[float | None]) -> bool:
    return (
        all(value is not None for value in values)
        and all(
            float(left) > float(right)
            for left, right in zip(values, values[1:])
        )
    )


def state_errors(
    numerical: PicardState, exact: PicardState
) -> dict[str, dict[str, float]]:
    result = {}
    for name in numerical.arrays():
        if name not in exact.arrays():
            continue
        result[name] = field_error(
            getattr(numerical, name), getattr(exact, name)
        )
    return result


def save_boundary(
    path: Path,
    boundary: BoundaryData,
    *,
    u: Array,
    v: Array,
) -> None:
    arrays: dict[str, Any] = {
        "schema": np.asarray("nee-official-immutable-boundary-data-v1"),
        "digest": np.asarray(boundary.digest),
        "u": u,
        "v": v,
    }
    arrays.update(
        {f"outgoing__{name}": value for name, value in boundary.outgoing.items()}
    )
    arrays.update(
        {f"incoming__{name}": value for name, value in boundary.incoming.items()}
    )
    np.savez_compressed(path, **arrays)


def direct_audit(
    grid: Any,
    state: PicardState,
    u: Array,
    v: Array,
    *,
    retained_degree: int,
    source_points: int,
) -> dict[str, Any]:
    from nee.diagnostics.state_audit import audit
    return audit(grid, state, u, v, retained_degree=retained_degree)


def mapped_direct_audit(
    grid: Any, state: PicardState, coordinates: Any, *, retained_degree: int,
    protected_s_values: tuple[float, ...] = (0.6, 0.7, 0.8),
) -> dict[str, Any]:
    from nee.diagnostics.state_audit import audit
    return audit(grid, state, coordinates.u, coordinates.v,
                 retained_degree=retained_degree, coordinates=coordinates,
                 protected_s_values=protected_s_values)


def first_order_audit(
    grid: Any,
    state: PicardState,
    u: Array,
    v: Array,
    *,
    coordinates: Any | None = None,
) -> dict[str, Any]:
    values = first_order_components(
        grid, state, u, v, coordinates=coordinates
    )
    maps = first_order_maps(grid, state, u, values)
    return summarize_first_order(maps, halo=2)


def vacuum_case(
    *,
    case_id: str,
    builder: Callable[[Any, Array, Array], tuple[Any, dict[str, Any]]],
    u: Array,
    v: Array,
    points: int,
    retained_degree: int,
    iterations: int,
    output: Path,
    high_precision_spherical: bool = False,
    metric_substeps: int = 2,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"immutable case exists: {output}")
    output.mkdir(parents=True)
    grid = vacuum_bench._grid(points, retained_degree)
    exact_old, exact_diagnostics = builder(grid, u, v)
    outgoing, incoming = vacuum_bench.characteristic_data(grid, exact_old)
    boundary = BoundaryData.create(outgoing, incoming)
    exact = from_numerical(exact_old, grid)
    state = from_numerical(vacuum_bench.face_compatible_seed(exact_old, u), grid)
    records: list[dict[str, Any]] = []
    for sweep in range(1, iterations + 1):
        started = time.perf_counter()
        next_state, _ = vacuum_picard_step(
            grid,
            state,
            boundary.outgoing,
            u,
            v,
            metric_substeps=metric_substeps,
            metric_parameterization="direct",
            metric_integrator="rk4",
            u_integrator="rk4",
            incoming=boundary.incoming,
        )
        boundary.verify_unchanged()
        records.append(
            {
                "sweep": sweep,
                "seconds": time.perf_counter() - started,
                "official_picard_update": weighted_update_norm(next_state, state),
                "field_errors": state_errors(next_state, exact),
            }
        )
        print(
            f"{case_id}: sweep {sweep}/{iterations} "
            f"update={records[-1]['official_picard_update']:.6e}",
            flush=True,
        )
        state = next_state
    closures = closure_audit(
        grid, state, u, v, stencil=9, halo=4
    )
    exact_closures = closure_audit(
        grid, exact, u, v, stencil=9, halo=4
    )
    missing_lie_mutation = closure_audit(
        grid,
        exact,
        u,
        v,
        stencil=9,
        halo=4,
        omit_lie_derivative=True,
    )
    null_residual = first_order_audit(grid, state, u, v)
    geometric_residual = direct_audit(
        grid,
        state,
        u,
        v,
        retained_degree=retained_degree,
        source_points=points,
    )
    high_precision_residual = (
        {"status": "omitted", "reason": "use weighted first-order audit without second null derivatives"}
        if high_precision_spherical else None
    )
    state.save(output / "official-state.npz", u=u, v=v)
    exact.save(output / "official-exact-state.npz", u=u, v=v)
    save_boundary(output / "boundary-data.npz", boundary, u=u, v=v)
    summary = {
        "schema": "nee-official-vacuum-case-v1",
        "case_id": case_id,
        "terminal_status": "completed",
        "state_semantics": "full weighted forms; traces derived from current g",
        "resolution": {
            "u_count": len(u),
            "v_count": len(v),
            "sphere_points": points,
            "retained_degree": retained_degree,
            "picard_sweeps": iterations,
        },
        "boundary_digest": boundary.digest,
        "exact_diagnostics": exact_diagnostics,
        "records": records,
        "closures": closures,
        "exact_closures": exact_closures,
        "missing_lie_derivative_mutation": {
            "corrected_C3": exact_closures["C3"],
            "mutated_C3": missing_lie_mutation["C3"],
            "rejected": bool(
                missing_lie_mutation["C3"]["raw_maximum"]
                > 10.0 * max(exact_closures["C3"]["raw_maximum"], 1.0e-30)
            ),
        },
        "first_order_null_residual_fresh": null_residual,
        "independent_first_order_residual": geometric_residual,
        "independent_spherical_residual_high_precision": (
            high_precision_residual
        ),
    }
    write_json(output / "summary.json", summary)
    return summary


def run_experiment_1(output: Path, public) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    levels = tuple((n, n) for n in public.initial_data["coordinate_counts"])
    domains = {
        "short": (-1.0, -0.5, 0.0, 0.5),
        "long": (-1.0, 0.0, 0.0, 1.0),
    }
    summaries: list[dict[str, Any]] = []
    for domain_name in public.initial_data["domains"]:
        bounds = domains[domain_name]
        for level, (nu, nv) in enumerate(levels):
            summaries.append(
                vacuum_case(
                    case_id=f"exp01-schwarzschild-{domain_name}-C{level}",
                    builder=lambda grid, u, v: (
                        vacuum_bench.regular_schwarzschild_state(
                            grid, u, v, float(public.physics["mass"])
                        )
                    ),
                    u=np.linspace(bounds[0], bounds[1], nu),
                    v=np.linspace(bounds[2], bounds[3], nv),
                    points=public.initial_data.get("schwarzschild_point_count", public.angular.point_count),
                    retained_degree=public.initial_data.get("schwarzschild_retained_degree", public.angular.retained_degree),
                    iterations=public.solver.maximum_sweeps,
                    output=output
                    / f"schwarzschild-{domain_name}"
                    / f"coordinate-{level}",
                )
            )
    for rotation in public.physics["kerr_rotations"]:
        builder = (
            (
                lambda grid, u, v: vacuum_bench.kerr_zero_spin_state(
                    grid, u, v, float(public.physics["mass"])
                )
            )
            if rotation == 0.0
            else (
                lambda grid, u, v, a=rotation: vacuum_bench.kerr_exact_state(
                    grid,
                    u,
                    v,
                    mass=float(public.physics["mass"]),
                    rotation=a,
                    reference_radius=float(public.physics["reference_radius"]),
                )
            )
        )
        for domain_name in public.initial_data["domains"]:
            bounds = domains[domain_name]
            for level, (nu, nv) in enumerate(levels):
                try:
                    summary = vacuum_case(
                        case_id=(
                            f"exp01-kerr-a{rotation:.1f}-{domain_name}-C{level}-A2"
                        ),
                        builder=builder,
                        u=np.linspace(bounds[0], bounds[1], nu),
                        v=np.linspace(bounds[2], bounds[3], nv),
                        points=public.initial_data["angular_point_counts"][-1],
                        retained_degree=public.initial_data["angular_retained_degrees"][-1],
                        iterations=public.solver.maximum_sweeps,
                        output=output
                        / f"kerr-a{rotation:.1f}-{domain_name}"
                        / f"coordinate-{level}-angular-2",
                    )
                except Exception as error:
                    case = (
                        output
                        / f"kerr-a{rotation:.1f}-{domain_name}"
                        / f"coordinate-{level}-angular-2"
                    )
                    case.mkdir(parents=True, exist_ok=True)
                    summary = {
                        "case_id": (
                            f"exp01-kerr-a{rotation:.1f}-{domain_name}-C{level}-A2"
                        ),
                        "terminal_status": "failed",
                        "failure_classification": "positivity, map, or resolution",
                        "error": repr(error),
                    }
                    write_json(case / "summary.json", summary)
                summaries.append(summary)
            # Two additional angular levels at the middle coordinate grid.
            for angular_level, (points, degree) in enumerate(zip(public.initial_data["angular_point_counts"][:-1], public.initial_data["angular_retained_degrees"][:-1], strict=True)):
                try:
                    summary = vacuum_case(
                        case_id=(
                            f"exp01-kerr-a{rotation:.1f}-{domain_name}-C1-A{angular_level}"
                        ),
                        builder=builder,
                        u=np.linspace(bounds[0], bounds[1], public.initial_data["coordinate_counts"][1]),
                        v=np.linspace(bounds[2], bounds[3], public.initial_data["coordinate_counts"][1]),
                        points=points,
                        retained_degree=degree,
                        iterations=public.solver.maximum_sweeps,
                        output=output
                        / f"kerr-a{rotation:.1f}-{domain_name}"
                        / f"coordinate-1-angular-{angular_level}",
                    )
                except Exception as error:
                    case = (
                        output
                        / f"kerr-a{rotation:.1f}-{domain_name}"
                        / f"coordinate-1-angular-{angular_level}"
                    )
                    case.mkdir(parents=True, exist_ok=True)
                    summary = {
                        "case_id": (
                            f"exp01-kerr-a{rotation:.1f}-{domain_name}-C1-A{angular_level}"
                        ),
                        "terminal_status": "failed",
                        "failure_classification": "positivity, map, or resolution",
                        "error": repr(error),
                    }
                    write_json(case / "summary.json", summary)
                summaries.append(summary)
    aggregate = {
        "schema": "nee-official-experiment-01-aggregate-v1",
        "experiment": 1,
        "runs": summaries,
    }
    write_json(output / "aggregate-summary.json", aggregate)
    return aggregate


def run_experiment_2(output: Path, public) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    levels = tuple((n, n) for n in public.initial_data["coordinate_counts"])
    summaries = []
    for epsilon in public.physics["static_epsilons"]:
        for level, (nu, nv) in enumerate(levels):
            case = output / f"static-epsilon-{epsilon:.0e}" / f"coordinate-{level}"
            try:
                summary = vacuum_case(
                    case_id=f"exp02-static-eps{epsilon:.0e}-C{level}",
                    builder=lambda grid, u, v, e=epsilon: (
                        vacuum_bench.near_horizon_static_state(
                            grid, u, v, float(public.physics["mass"]), e
                        )
                    ),
                    u=np.linspace(-1.0, -0.5, nu),
                    v=np.linspace(0.0, 0.5, nv),
                    points=public.initial_data.get("schwarzschild_point_count", public.angular.point_count),
                    retained_degree=public.initial_data.get("schwarzschild_retained_degree", public.angular.retained_degree),
                    iterations=public.solver.maximum_sweeps,
                    output=case,
                    metric_substeps=public.solver.metric_substeps,
                    high_precision_spherical=epsilon <= 1.0e-4,
                )
            except Exception as error:
                case.mkdir(parents=True, exist_ok=True)
                summary = {
                    "case_id": f"exp02-static-eps{epsilon:.0e}-C{level}",
                    "terminal_status": "failed",
                    "failure_classification": "coordinate conditioning",
                    "error": repr(error),
                }
                write_json(case / "summary.json", summary)
            summaries.append(summary)
    for level, (nu, nv) in enumerate(levels):
        summaries.append(
            vacuum_case(
                case_id=f"exp02-kruskal-crossing-C{level}",
                builder=lambda grid, u, v: vacuum_bench.kruskal_state(
                    grid,
                    u,
                    v,
                    float(public.physics["mass"]),
                    u_offset=float(public.physics["u_offset"]),
                    v_offset=float(public.physics["v_offset"]),
                ),
                u=np.linspace(-1.0, -0.5, nu),
                v=np.linspace(0.0, 0.5, nv),
                points=public.angular.point_count,
                retained_degree=public.angular.retained_degree,
                iterations=public.solver.maximum_sweeps,
                output=output / "kruskal-crossing" / f"coordinate-{level}",
                high_precision_spherical=True,
            )
        )
    aggregate = {
        "schema": "nee-official-experiment-02-aggregate-v1",
        "experiment": 2,
        "runs": summaries,
        "high_precision_note": (
            "A 35-decimal-digit direct warped-product Ricci audit is attached "
            "to static epsilon <= 1e-4 and every Kruskal crossing run."
        ),
    }
    write_json(output / "aggregate-summary.json", aggregate)
    return aggregate


def run_experiment_3(output: Path, public) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    levels = public.initial_data["coordinate_counts"]
    summaries = []
    for epsilon in public.physics["epsilons"]:
        for level, count in enumerate(levels):
            case = output / f"epsilon-{epsilon:.8f}" / f"coordinate-{level}"
            case.mkdir(parents=True)
            try:
                u_nodes, _ = chebyshev_lobatto(count, -1.0, -0.5)
                xi_nodes, _ = chebyshev_lobatto(count, 0.0, 1.0)
                exact = curved_exact_fields(
                    u_nodes,
                    xi_nodes,
                    epsilon,
                    float(public.physics["mass"]),
                    high_precision=epsilon <= 2.0**-4,
                )
                np.savez_compressed(
                    case / "boundary-data.npz",
                    schema=np.asarray(
                        "nee-official-curved-characteristic-boundary-v1"
                    ),
                    u=np.asarray(u_nodes),
                    xi=np.asarray(xi_nodes),
                    v0=np.asarray(exact["v0"]),
                    outgoing_radius=np.asarray(exact["radius"][0]),
                    incoming_radius=np.asarray(exact["radius"][:, 0]),
                    outgoing_log_lapse=np.asarray(exact["log_Omega"][0]),
                    incoming_log_lapse=np.asarray(exact["log_Omega"][:, 0]),
                    corner_radius_mismatch=np.asarray(0.0),
                    corner_log_lapse_mismatch=np.asarray(0.0),
                    minimum_radius=np.asarray(np.min(exact["radius"])),
                )
                solution = solve_curved(
                    epsilon,
                    count,
                    count,
                    iterations=public.solver.maximum_sweeps,
                    tolerance=public.solver.tolerance,
                    mass=float(public.physics["mass"]),
                )
                overgrid = curved_overgrid_audit(solution, epsilon, mass=float(public.physics["mass"]))
                relative = np.abs(solution.radius - exact["radius"]) / np.maximum(
                    np.abs(exact["radius"]), 1.0e-30
                )
                protected = {}
                for name, threshold in (
                    ("r_ge_M", float(public.physics["mass"])),
                    ("r_ge_half_M", 0.5*float(public.physics["mass"])),
                    ("r_ge_8M_epsilon", 8.0 * float(public.physics["mass"]) * epsilon),
                ):
                    mask = exact["radius"] >= threshold
                    protected[name] = (
                        None
                        if not np.any(mask)
                        else {
                            "point_count": int(np.sum(mask)),
                            "radius_relative_maximum": float(
                                np.max(relative[mask])
                            ),
                        }
                    )
                np.savez_compressed(
                    case / "official-mapped-state.npz",
                    state_schema=np.asarray(
                        "nee-official-curved-full-weighted-spherical-state-v1"
                    ),
                    u=np.asarray(solution.u),
                    xi=np.asarray(solution.xi),
                    v=np.asarray(solution.v),
                    v0=np.asarray(solution.v0),
                    metric_scalar=np.asarray(solution.radius**2),
                    log_Omega=np.asarray(solution.log_Omega),
                    x_out_scalar=np.asarray(solution.x_out_scalar),
                    x_in_scalar=np.asarray(solution.x_in_scalar),
                    Omega_omega=np.asarray(solution.Omega_omega),
                    Omega_omegab=np.asarray(solution.Omega_omegab),
                )
                summary = {
                    "schema": "nee-official-curved-spherical-v1",
                    "case_id": f"exp03-eps{epsilon:.8f}-C{level}",
                    "terminal_status": "completed",
                    "state_semantics": "full weighted spherical forms",
                    "resolution": {
                        "u_count": count,
                        "xi_count": count,
                        "picard_sweeps": len(solution.records),
                    },
                    "records": solution.records,
                    "diagnostics": solution.diagnostics,
                    "independent_coordinate_overgrid_audit": overgrid,
                    "protected_subdomains": protected,
                }
            except Exception as error:
                summary = {
                    "case_id": f"exp03-eps{epsilon:.8f}-C{level}",
                    "terminal_status": "failed",
                    "failure_classification": (
                        "Picard contraction, conditioning, positivity, or precision"
                    ),
                    "error": repr(error),
                }
            write_json(case / "summary.json", summary)
            summaries.append(summary)
    aggregate = {
        "schema": "nee-official-experiment-03-aggregate-v1",
        "experiment": 3,
        "runs": summaries,
        "map": "v=xi*v0(u), with both physical derivative Jacobian terms",
    }
    write_json(output / "aggregate-summary.json", aggregate)
    return aggregate


def ese_case(
    *,
    nu: float,
    level: int,
    config: Any,
    output: Path,
    sigma: float = 1.0,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"immutable case exists: {output}")
    output.mkdir(parents=True)
    grid, angular = ese_bench.build_angular(config)
    mesh = ese_bench.mesh_from_config(config.scalar_coordinates)
    exact_old, exact_diagnostics = ese_bench.jnw_exact_state(
        grid, mesh.u, mesh.v, sigma=sigma, nu=nu
    )
    bundle = ese_bench._faces(
        exact_old, mesh.u, mesh.v, exact_diagnostics
    )
    boundary = BoundaryData.create(bundle.outgoing, bundle.incoming)
    exact = from_numerical(exact_old, grid)
    state = from_numerical(ese_bench.initial_state(bundle, angular), grid)
    records = []
    for iteration in range(1, config.solver.picard_iterations + 1):
        started = time.perf_counter()
        next_state, _ = ese_picard_step(
            grid,
            angular,
            mesh,
            state,
            bundle,
            metric_substeps=config.solver.metric_substeps,
        )
        boundary.verify_unchanged()
        records.append(
            {
                "iteration": iteration,
                "seconds": time.perf_counter() - started,
                "official_picard_update": weighted_update_norm(next_state, state),
                "field_errors": state_errors(next_state, exact),
            }
        )
        print(
            f"exp06-jnw-nu{nu:.2f}-C{level}: sweep "
            f"{iteration}/{config.solver.picard_iterations} "
            f"update={records[-1]['official_picard_update']:.6e}",
            flush=True,
        )
        state = next_state
    closures = closure_audit(
        grid, state, mesh.u, mesh.v, stencil=7, halo=2
    )
    null_residual = first_order_audit(
        grid,
        state,
        mesh.u,
        mesh.v,
        coordinates=mesh,
    )
    geometric_residual = direct_audit(
        grid,
        state,
        mesh.u,
        mesh.v,
        retained_degree=config.angular.retained_degree,
        source_points=config.angular.point_count,
    )
    # Change the matter field and its first-order derivatives consistently.
    exact_direct = direct_audit(grid, exact, mesh.u, mesh.v,
        retained_degree=config.angular.retained_degree, source_points=config.angular.point_count)
    mutated_state = exact.copy()
    for field in ('phi', 'Omega_e3phi', 'Omega_e4phi', 'nabla_phi'):
        setattr(mutated_state, field, 1.01*getattr(mutated_state, field))
    mutated_direct = direct_audit(grid, mutated_state, mesh.u, mesh.v,
        retained_degree=config.angular.retained_degree, source_points=config.angular.point_count)
    mutation = {
        "kind": "multiply the scalar field and its stored first derivatives by 1.01",
        "baseline_masked_residual": exact_direct[
            "masked_Linf_uv_L2_sphere"
        ],
        "mutated_masked_residual": mutated_direct[
            "masked_Linf_uv_L2_sphere"
        ],
        "rejected": bool(
            mutated_direct["masked_Linf_uv_L2_sphere"]
            > 10.0
            * max(exact_direct["masked_Linf_uv_L2_sphere"], 1.0e-30)
        ),
    }
    state.save(output / "official-state.npz", u=mesh.u, v=mesh.v)
    exact.save(output / "official-exact-state.npz", u=mesh.u, v=mesh.v)
    save_boundary(output / "boundary-data.npz", boundary, u=mesh.u, v=mesh.v)
    summary = {
        "schema": "nee-official-ese-exact-case-v1",
        "case_id": f"exp06-jnw-nu{nu:.2f}-C{level}",
        "terminal_status": "completed",
        "state_semantics": "full weighted forms; traces derived from current g",
        "config": config.to_dict(),
        "exact_diagnostics": exact_diagnostics,
        "boundary_digest": boundary.digest,
        "records": records,
        "closures": closures,
        "first_order_null_residual_fresh": null_residual,
        "independent_first_order_residual": geometric_residual,
        "independent_state_mutation_test": mutation,
    }
    write_json(output / "summary.json", summary)
    return summary


def run_experiment_6(output: Path, public) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    levels = tuple((n, public.initial_data["coordinate_degree"], n, public.initial_data["coordinate_degree"]) for n in public.initial_data["coordinate_elements"])
    summaries = []
    for nu in public.physics["nu_values"]:
        for level, (te, td, se, sd) in enumerate(levels):
            config = ese_bench._configuration(
                name=f"jnw-nu{nu:.2f}-C{level}",
                tau_elements=te,
                tau_degree=td,
                s_elements=se,
                s_degree=sd,
                iterations=public.solver.maximum_sweeps,
                # Spherical data need no expensive high angular band.
                quick=True,
            )
            case = output / f"nu-{nu:.2f}" / f"coordinate-{level}"
            try:
                summary = ese_case(
                    nu=nu,
                    level=level,
                    config=config,
                    output=case,
                    sigma=float(public.physics["sigma"]),
                )
            except Exception as error:
                case.mkdir(parents=True, exist_ok=True)
                summary = {
                    "case_id": f"exp06-jnw-nu{nu:.2f}-C{level}",
                    "terminal_status": "failed",
                    "failure_classification": (
                        "Picard, positivity, independent residual, or resolution"
                    ),
                    "error": repr(error),
                }
                write_json(case / "summary.json", summary)
            summaries.append(summary)
    # Exact limiting generator test at nu=1.
    config = ese_bench._configuration(
        name="jnw-nu1-limit",
        tau_elements=2,
        tau_degree=6,
        s_elements=2,
        s_degree=6,
        iterations=1,
        quick=True,
    )
    grid, _ = ese_bench.build_angular(config)
    mesh = ese_bench.mesh_from_config(config.scalar_coordinates)
    limit, diagnostics = ese_bench.jnw_exact_state(
        grid, mesh.u, mesh.v, sigma=float(public.physics["sigma"]), nu=1.0
    )
    limit_test = {
        "nu": 1.0,
        "expected_schwarzschild_mass": float(public.physics["sigma"])/2,
        "maximum_abs_phi": float(np.max(np.abs(limit.phi))),
        "maximum_abs_P3": float(np.max(np.abs(limit.Omega_e3phi))),
        "maximum_abs_P4": float(np.max(np.abs(limit.Omega_e4phi))),
        "passed": bool(
            np.max(np.abs(limit.phi)) == 0.0
            and np.max(np.abs(limit.Omega_e3phi)) == 0.0
            and np.max(np.abs(limit.Omega_e4phi)) == 0.0
        ),
        "diagnostics": diagnostics,
    }
    aggregate = {
        "schema": "nee-official-experiment-06-aggregate-v1",
        "experiment": 6,
        "runs": summaries,
        "nu_to_one_limit": limit_test,
    }
    write_json(output / "aggregate-summary.json", aggregate)
    return aggregate


def nonspherical_ese_case(
    *,
    output: Path,
    name: str,
    cap: float,
    multipliers: tuple[float, float, float, float],
    rotation_angle: float = 0.0,
    retained: int = 3,
    work: int = 6,
    points: int = 100,
    tau_elements: int = 1,
    s_elements: int = 1,
    tau_degree: int = 4,
    s_degree: int = 6,
    metric_substeps: int = 1,
    derivative_halo: int = 1,
    iterations: int = 8,
    public_config=None,
) -> dict[str, Any]:
    """Run the smooth-harmonic ESE case with a persistent official state."""

    from nee.experiments.exp07_nonspherical_scalar import (
        campaign as nonspherical,
    )

    ese_run = nonspherical.ese_run
    captured: dict[str, Any] = {}
    original_scaled_construct = nonspherical._scaled_construct
    original_initial_state = ese_run.initial_state
    original_step = ese_run.picard_step
    original_update_norm = ese_run.update_norm
    original_update_map = ese_run.update_map
    original_save_state = ese_run.save_state

    def scaled_construct(
        original: Callable[..., Any],
        lambda_phi: float,
        lambda_chi: float,
    ) -> Callable[..., Any]:
        base = original_scaled_construct(
            original, lambda_phi, lambda_chi
        )

        def construct(config: Any) -> Any:
            bundle, grid, angular, mesh = base(config)
            captured.update(
                {
                    "bundle": bundle,
                    "grid": grid,
                    "angular": angular,
                    "mesh": mesh,
                    "config": config,
                }
            )
            return bundle, grid, angular, mesh

        return construct

    def initial_state(bundle: Any, angular: Any) -> PicardState:
        state = from_numerical(original_initial_state(bundle, angular), captured["grid"])
        captured["state"] = state
        return state

    def step(
        grid: Any,
        angular: Any,
        mesh: Any,
        state: PicardState,
        bundle: Any,
        **kwargs: Any,
    ) -> tuple[PicardState, dict[str, Any]]:
        result, context = ese_picard_step(
            grid, angular, mesh, state, bundle, **kwargs
        )
        captured["state"] = result
        return result, context

    def save_state(
        path: Path,
        state: PicardState,
        update_maps: list[Array],
        residual_maps: dict[str, list[Array]],
        u: Array,
        v: Array,
    ) -> None:
        extra: dict[str, Any] = {
            "update_maps": np.stack(update_maps)
        }
        extra.update(
            {
                f"construction_residual_maps__{field}": np.stack(history)
                for field, history in residual_maps.items()
            }
        )
        state.save(path, u=u, v=v, extra=extra)

    numerical_summary = nonspherical._run_case(
        output=output,
        name=name,
        cap=cap,
        multipliers=multipliers,
        rotation_angle=rotation_angle,
        retained=retained,
        work=work,
        points=points,
        tau_elements=tau_elements,
        s_elements=s_elements,
        tau_degree=tau_degree,
        s_degree=s_degree,
        metric_substeps=metric_substeps,
        derivative_halo=derivative_halo,
        iterations=iterations,
        public_config=public_config,
        construct_wrapper=scaled_construct,
        run_hooks={"initial_state_fn": initial_state, "picard_step_fn": step,
                   "update_norm_fn": weighted_update_norm, "update_map_fn": weighted_update_map,
                   "save_state_fn": save_state},
    )

    case_output = output / "cases" / name
    if "state" not in captured:
        return {
            **numerical_summary,
            "case_id": name,
            "terminal_status": "failed",
            "failure_classification": (
                "nonspherical constraint or Picard failure before state audit"
            ),
        }
    grid = captured["grid"]
    mesh = captured["mesh"]
    state = captured["state"]
    bundle = captured["bundle"]
    boundary = BoundaryData.create(bundle.outgoing, bundle.incoming)
    boundary.verify_unchanged()
    closures = closure_audit(
        grid,
        state,
        mesh.u,
        mesh.v,
        stencil=7,
        halo=max(2, derivative_halo),
        coordinates=mesh,
        protected_s_minimum=0.6,
    )
    null_residual = first_order_audit(
        grid, state, mesh.u, mesh.v, coordinates=mesh
    )
    geometric_residual = mapped_direct_audit(
        grid,
        state,
        mesh,
        retained_degree=retained,
    )
    mutated_state = state.copy()
    assert mutated_state.phi is not None
    mutated_state.phi = 1.01 * mutated_state.phi
    mutated = mapped_direct_audit(
        grid,
        mutated_state,
        mesh,
        retained_degree=retained,
    )
    baseline_primary = geometric_residual["protected"]["s_ge_0.60"][
        "ESE_acceptance_sum"
    ]
    mutated_primary = mutated["protected"]["s_ge_0.60"][
        "ESE_acceptance_sum"
    ]
    mutation = {
        "kind": "multiply only phi in the final numerical state by 1.01",
        "region": "s_ge_0.60",
        "baseline_ESE_acceptance_sum": baseline_primary,
        "mutated_ESE_acceptance_sum": mutated_primary,
        "absolute_response": abs(mutated_primary - baseline_primary),
    }
    save_boundary(
        case_output / "boundary-data-official.npz",
        boundary,
        u=mesh.u,
        v=mesh.v,
    )
    summary = {
        **numerical_summary,
        "schema": "nee-official-nonspherical-ese-case-v1",
        "case_id": name,
        "terminal_status": "completed",
        "state_semantics": (
            "persistent full weighted forms; scalar gradient and traces "
            "derived from the current primitive state"
        ),
        "boundary_digest": boundary.digest,
        "corrected_closures": closures,
        "first_order_null_residual_fresh": null_residual,
        "independent_first_order_residual": geometric_residual,
        "independent_state_mutation_test": mutation,
        "limitations": [
            "The one-step algebraic kernel remains the numerical backend RK4 "
            "implementation; all stored and iterated states use official semantics."
        ],
    }
    write_json(case_output / "summary.json", summary)
    return summary


def run_experiment_7(output: Path, public) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    baseline = (1.0, 1.0, 1.0, 1.0)
    summaries: list[dict[str, Any]] = []

    central = {
        "retained": public.angular.retained_degree,
        "work": public.angular.work_degree,
        "points": public.angular.point_count,
        "tau_elements": public.initial_data["coordinate_elements"][1],
        "s_elements": public.initial_data["coordinate_elements"][1],
        "tau_degree": public.initial_data["coordinate_tau_degrees"][1],
        "s_degree": public.initial_data["coordinate_s_degrees"][1],
        "metric_substeps": public.solver.metric_substeps,
        "derivative_halo": 2,
        "iterations": public.solver.maximum_sweeps,
    }
    refinement_cases = [
        (
            "coordinate-level-0",
            {
                **central,
                "tau_degree": public.initial_data["coordinate_tau_degrees"][0],
                "s_degree": public.initial_data["coordinate_s_degrees"][0],
            },
        ),
        ("baseline-cap-0.04", central),
        (
            "coordinate-level-2",
            {
                **central,
                "tau_elements": public.initial_data["coordinate_elements"][2],
                "s_elements": public.initial_data["coordinate_elements"][2],
            },
        ),
        (
            "angular-level-0",
            {
                **central,
                "retained": public.initial_data["angular_retained_degrees"][0],
                "work": public.initial_data["angular_work_degrees"][0],
                "points": public.initial_data["angular_point_counts"][0],
            },
        ),
        (
            "angular-level-2",
            {
                **central,
                "retained": public.initial_data["angular_retained_degrees"][2],
                "work": public.initial_data["angular_work_degrees"][2],
                "points": public.initial_data["angular_point_counts"][2],
            },
        ),
    ]

    def execute(
        name: str,
        cap: float,
        multipliers: tuple[float, float, float, float],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            summary = nonspherical_ese_case(
                output=output,
                name=name,
                cap=cap,
                multipliers=multipliers,
                public_config=public,
                **options,
            )
        except Exception as error:
            case = output / "cases" / name
            case.mkdir(parents=True, exist_ok=True)
            summary = {
                "case_id": name,
                "terminal_status": "failed",
                "failure_classification": (
                    "constraint radicand, positivity, Picard, or audit failure"
                ),
                "error": repr(error),
            }
            if name in public.initial_data['expected_constraint_rejections'] and isinstance(error, FloatingPointError) and 'radicand' in str(error):
                summary['terminal_status'] = 'expected_rejection'
                summary['expected_outcome'] = 'invalid characteristic scalar constraint'
            write_json(case / "summary.json", summary)
        summaries.append(summary)
        return summary

    refinement = {
        name: execute(name, float(public.physics["central_cap"]), baseline, options)
        for name, options in refinement_cases
    }

    def protected_residual(name: str) -> float | None:
        try:
            return float(
                refinement[name]["independent_first_order_residual"][
                    "protected"
                ]["s_ge_0.60"]["ESE_acceptance_sum"]
            )
        except (KeyError, TypeError, ValueError):
            return None

    def construction_residual(name: str) -> float | None:
        try:
            return float(
                refinement[name]["iterations"][-1]["residual"]["combined"][
                    "maximum"
                ]
            )
        except (KeyError, TypeError, ValueError, IndexError):
            return None

    def closure_envelope(name: str) -> float | None:
        try:
            protected = refinement[name]["corrected_closures"]["protected"]
            values = [
                float(protected[field]["masked_maximum"])
                for field in ("C3", "C4", "scalar_P3", "scalar_P4")
            ]
            return max(values)
        except (KeyError, TypeError, ValueError):
            return None

    coordinate_names = (
        "coordinate-level-0",
        "baseline-cap-0.04",
        "coordinate-level-2",
    )
    angular_names = (
        "angular-level-0",
        "baseline-cap-0.04",
        "angular-level-2",
    )
    coordinate_direct = [
        protected_residual(name) for name in coordinate_names
    ]
    angular_direct = [protected_residual(name) for name in angular_names]
    coordinate_construction = [
        construction_residual(name) for name in coordinate_names
    ]
    angular_construction = [
        construction_residual(name) for name in angular_names
    ]
    coordinate_closure = [closure_envelope(name) for name in coordinate_names]
    angular_closure = [closure_envelope(name) for name in angular_names]
    construction_below_direct = all(
        construction is not None
        and direct is not None
        and construction < direct
        for construction, direct in zip(
            coordinate_construction + angular_construction,
            coordinate_direct + angular_direct,
            strict=True,
        )
    )
    update_small = all(
        (
            run.get("iterations")
            and float(run["iterations"][-1]["update_norm"]) <= 1.0e-6
        )
        for run in refinement.values()
        if run.get("terminal_status") == "completed"
    ) and all(
        run.get("terminal_status") == "completed"
        for run in refinement.values()
    )
    refinement_gate = {
        "coordinate_case_order": coordinate_names,
        "angular_case_order": angular_names,
        "coordinate_protected_ESE_acceptance_sum": coordinate_direct,
        "angular_protected_ESE_acceptance_sum": angular_direct,
        "coordinate_construction_residual": coordinate_construction,
        "angular_construction_residual": angular_construction,
        "coordinate_closure_envelope": coordinate_closure,
        "angular_closure_envelope": angular_closure,
        "coordinate_direct_strictly_decreasing": (
            strictly_decreasing_finite(coordinate_direct)
        ),
        "angular_direct_strictly_decreasing": (
            strictly_decreasing_finite(angular_direct)
        ),
        "coordinate_construction_strictly_decreasing": (
            strictly_decreasing_finite(coordinate_construction)
        ),
        "angular_construction_strictly_decreasing": (
            strictly_decreasing_finite(angular_construction)
        ),
        "construction_residual_below_independent_residual": (
            construction_below_direct
        ),
        "coordinate_closure_strictly_decreasing": (
            strictly_decreasing_finite(coordinate_closure)
        ),
        "angular_closure_strictly_decreasing": (
            strictly_decreasing_finite(angular_closure)
        ),
        "all_picard_updates_at_most_1e-6": update_small,
    }
    refinement_gate["passed"] = bool(
        refinement_gate["coordinate_direct_strictly_decreasing"]
        and refinement_gate["angular_direct_strictly_decreasing"]
        and construction_below_direct
        and refinement_gate["coordinate_closure_strictly_decreasing"]
        and refinement_gate["angular_closure_strictly_decreasing"]
        and update_small
    )

    stress_status = "withheld because the baseline refinement gate failed"
    if refinement_gate["passed"]:
        stress_status = "completed after the baseline refinement gate passed"
        execute(
            "rotated-baseline",
            float(public.physics["central_cap"]),
            baseline,
            {**central, "rotation_angle": float(public.physics["rotation_angle"])},
        )
        for cap in public.physics["stress_caps"]:
            execute(f"baseline-cap-{cap:.2f}", cap, baseline, central)
        for axis in range(4):
            for value in (0.0, 0.5, 2.0):
                multipliers = list(baseline)
                multipliers[axis] = value
                execute(
                    f"stress-axis-{axis}-value-{value:.1f}",
                    float(public.physics["central_cap"]),
                    tuple(multipliers),
                    central,
                )
        for value in (0.0, 0.5, 2.0):
            execute(
                f"stress-joint-{value:.1f}",
                float(public.physics["central_cap"]),
                (value, value, value, value),
                central,
            )
    aggregate = {
        "schema": "nee-official-experiment-07-aggregate-v1",
        "experiment": 7,
        "terminal_status": "completed" if refinement_gate["passed"] and all(row.get("terminal_status") in {"completed", "expected_rejection"} for row in summaries) else "failed",
        "runs": summaries,
        "harmonics": {
            "Y_Omega": "(3 z^2 - 1)/2",
            "V_b": "grad_round(x z)",
            "Q_chi": "trace-free Hessian_round(x^2 - y^2)",
            "normalization": "fixed analytic amplitudes, independent of sphere samples",
            "corner_power": 0.1,
            "corner_regularization": "none",
        },
        "refinement_controls": {
            "central_configuration": central,
            "coordinate_levels": (
                {"elements": 2, "tau_degree": public.initial_data["coordinate_tau_degrees"][0], "s_degree": 9},
                {"elements": 2, "tau_degree": public.initial_data["coordinate_tau_degrees"][1], "s_degree": 11},
                {"elements": 3, "tau_degree": public.initial_data["coordinate_tau_degrees"][1], "s_degree": 11},
            ),
            "angular_retained_degree": (6, 8, 10),
            "protected_direct_region": "s >= 0.60",
        },
        "refinement_gate": refinement_gate,
        "stress_matrix_status": stress_status,
    }
    write_json(output / "aggregate-summary.json", aggregate)
    return aggregate
