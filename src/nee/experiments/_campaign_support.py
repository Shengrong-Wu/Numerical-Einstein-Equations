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
    warped_product_rectangular_residual_mpmath,
)
from nee.diagnostics.ricci_components import (  # noqa: E402
    components as first_order_components,
    section_maps as first_order_maps,
    summarize as summarize_first_order,
)
from nee.diagnostics.independent_audit import (  # noqa: E402
    PrimitiveFields,
    evaluate,
    evaluate_blocked,
    summarize_result_on_mask,
)
from nee.solver.backend import (  # noqa: E402
    ese_picard_step,
    from_numerical,
    to_numerical,
    vacuum_picard_step,
    weighted_update_map,
    weighted_update_norm,
)
from nee.discretization.overgrid import (  # noqa: E402
    resample_primitives,
    resample_primitives_power,
)
from nee.state.boundary import BoundaryData
from nee.state.iterate import PicardState as WeightedState  # noqa: E402


Array = np.ndarray


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def direct_residual_value(summary: dict[str, Any]) -> float | None:
    try:
        value = float(
            summary["independent_four_metric_residual"][
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
    numerical: WeightedState, exact: WeightedState
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
    state: WeightedState,
    u: Array,
    v: Array,
    *,
    retained_degree: int,
    source_points: int,
) -> dict[str, Any]:
    point_count = max(
        source_points + 12, (retained_degree + 1) ** 2 + 12
    )
    overgrid = resample_primitives(
        grid,
        PrimitiveFields(
            metric=state.metric,
            log_omega=state.log_omega,
            shift=state.shift,
            phi=state.phi,
        ),
        u,
        v,
        u_count=len(u) + 4,
        v_count=len(v) + 4,
        point_count=point_count,
        harmonic_degree=retained_degree,
    )
    summary = evaluate_blocked(
        overgrid.grid,
        overgrid.fields,
        overgrid.u,
        overgrid.v,
        stencil=7,
        block_size=4,
        derivative_halo=6,
        mask_halo=4,
    )
    summary["overgrid"] = overgrid.diagnostics
    return summary


def mapped_direct_audit(
    grid: Any,
    state: WeightedState,
    coordinates: Any,
    *,
    retained_degree: int,
    protected_s_values: tuple[float, ...] = (0.6, 0.7, 0.8),
) -> dict[str, Any]:
    """Audit a power-coordinate state without reinterpolating in physical v."""

    if not protected_s_values:
        raise ValueError("at least one protected s threshold is required")
    if any(
        not math.isfinite(value) for value in protected_s_values
    ) or any(
        right <= left
        for left, right in zip(
            protected_s_values, protected_s_values[1:], strict=False
        )
    ):
        raise ValueError(
            "protected s thresholds must be finite and strictly increasing"
        )
    point_count = (retained_degree + 1) ** 2 + 8
    overgrid = resample_primitives_power(
        grid,
        PrimitiveFields(
            metric=state.metric,
            log_omega=state.log_omega,
            shift=state.shift,
            phi=state.phi,
        ),
        coordinates,
        u_count=len(coordinates.u) + 4,
        v_count=len(coordinates.v) + 4,
        point_count=point_count,
        harmonic_degree=retained_degree,
        stencil=7,
        spectral_degree_increment=3,
    )
    result = evaluate(
        overgrid.grid,
        overgrid.fields,
        overgrid.u,
        overgrid.v,
        stencil=7,
        mask_halo=3,
        coordinates=overgrid.coordinates,
    )
    target = overgrid.coordinates
    assert target is not None
    safe_u = np.ones(len(target.u), dtype=bool)
    safe_v_base = np.ones(len(target.v), dtype=bool)
    outer_halo = 3
    interface_halo = 3
    safe_u[:outer_halo] = False
    safe_u[-outer_halo:] = False
    safe_v_base[-outer_halo:] = False
    for source_mesh, target_nodes, safe in (
        (coordinates.tau, target.tau, safe_u),
        (coordinates.s, target.s, safe_v_base),
    ):
        for segment in source_mesh.segments[:-1]:
            center = int(np.argmin(np.abs(target_nodes - segment.right)))
            safe[
                max(0, center - interface_halo) : min(
                    len(safe), center + interface_halo + 1
                )
            ] = False
    protected = {}
    for minimum_s in protected_s_values:
        safe_v = safe_v_base & (target.s >= minimum_s)
        mask = safe_u[:, None] & safe_v[None, :]
        protected[f"s_ge_{minimum_s:.2f}"] = summarize_result_on_mask(
            result,
            mask,
            label=(
                f"s >= {minimum_s:.2f}, outer halo {outer_halo}, "
                f"interface halo {interface_halo}"
            ),
        )
    summary = dict(result.summary)
    summary.update(
        {
            "method": (
                "four-metric connection difference on an independent "
                "higher-degree (tau,s) LGL overgrid"
            ),
            "overgrid": overgrid.diagnostics,
            "protected": protected,
            "primary_protected_region": (
                f"s_ge_{protected_s_values[0]:.2f}"
            ),
            "section_L2_map": result.section_l2.tolist(),
            "einstein_section_L2_map": (
                result.einstein_section_l2.tolist()
            ),
            "wave_section_L2_map": (
                None
                if result.wave_section_l2 is None
                else result.wave_section_l2.tolist()
            ),
        }
    )
    return summary


def first_order_audit(
    grid: Any,
    state: WeightedState,
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


def spherical_high_precision_audit(
    grid: Any,
    state: WeightedState,
    u: Array,
    v: Array,
) -> dict[str, Any]:
    """Reconstruct the spherical warped product and audit it with mpmath."""

    radius_squared = 0.5 * np.einsum(
        "nij,nuvij->nuv", grid.projector, state.metric
    )
    radius = np.sqrt(np.mean(radius_squared, axis=0))
    log_omega = np.mean(state.log_omega, axis=0)
    result = warped_product_rectangular_residual_mpmath(
        radius,
        log_omega,
        u,
        v,
        halo=max(2, min(len(u), len(v)) // 8),
        decimal_digits=35,
    )
    result["spherical_reconstruction_angular_variation"] = {
        "radius_squared": float(
            np.max(
                np.abs(
                    radius_squared
                    - np.mean(radius_squared, axis=0, keepdims=True)
                )
            )
        ),
        "log_omega": float(
            np.max(
                np.abs(
                    state.log_omega
                    - np.mean(state.log_omega, axis=0, keepdims=True)
                )
            )
        ),
    }
    return result


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
            metric_substeps=2,
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
        spherical_high_precision_audit(grid, state, u, v)
        if high_precision_spherical
        else None
    )
    state.save(output / "official-state.npz", u=u, v=v)
    exact.save(output / "official-exact-state.npz", u=u, v=v)
    save_boundary(output / "boundary-data.npz", boundary, u=u, v=v)
    summary = {
        "schema": "nee-official-vacuum-case-v1",
        "case_id": case_id,
        "terminal_status": "completed",
        "state_semantics": "full weighted forms; traces derived from current metric",
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
        "independent_four_metric_residual": geometric_residual,
        "independent_spherical_residual_high_precision": (
            high_precision_residual
        ),
    }
    write_json(output / "summary.json", summary)
    return summary


def run_experiment_1(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    levels = ((17, 17), (33, 33), (65, 65))
    domains = {
        "short": (-1.0, -0.5, 0.0, 0.5),
        "long": (-1.0, 0.0, 0.0, 1.0),
    }
    summaries: list[dict[str, Any]] = []
    for domain_name, bounds in domains.items():
        for level, (nu, nv) in enumerate(levels):
            summaries.append(
                vacuum_case(
                    case_id=f"exp01-schwarzschild-{domain_name}-C{level}",
                    builder=lambda grid, u, v: (
                        vacuum_bench.regular_schwarzschild_state(
                            grid, u, v, 1.0
                        )
                    ),
                    u=np.linspace(bounds[0], bounds[1], nu),
                    v=np.linspace(bounds[2], bounds[3], nv),
                    points=50,
                    retained_degree=5,
                    iterations=8,
                    output=output
                    / f"schwarzschild-{domain_name}"
                    / f"coordinate-{level}",
                )
            )
    for rotation in (0.0, 0.3, 0.7, 0.9):
        builder = (
            (
                lambda grid, u, v: vacuum_bench.kerr_zero_spin_state(
                    grid, u, v, 1.0
                )
            )
            if rotation == 0.0
            else (
                lambda grid, u, v, a=rotation: vacuum_bench.kerr_exact_state(
                    grid,
                    u,
                    v,
                    mass=1.0,
                    rotation=a,
                    reference_radius=4.0,
                )
            )
        )
        for domain_name, bounds in domains.items():
            for level, (nu, nv) in enumerate(levels):
                try:
                    summary = vacuum_case(
                        case_id=(
                            f"exp01-kerr-a{rotation:.1f}-{domain_name}-C{level}-A2"
                        ),
                        builder=builder,
                        u=np.linspace(bounds[0], bounds[1], nu),
                        v=np.linspace(bounds[2], bounds[3], nv),
                        points=86,
                        retained_degree=7,
                        iterations=8,
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
            for angular_level, (points, degree) in enumerate(((30, 3), (50, 5))):
                try:
                    summary = vacuum_case(
                        case_id=(
                            f"exp01-kerr-a{rotation:.1f}-{domain_name}-C1-A{angular_level}"
                        ),
                        builder=builder,
                        u=np.linspace(bounds[0], bounds[1], 33),
                        v=np.linspace(bounds[2], bounds[3], 33),
                        points=points,
                        retained_degree=degree,
                        iterations=8,
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


def run_experiment_2(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    levels = ((17, 17), (33, 33), (65, 65))
    summaries = []
    for epsilon in (1.0e-1, 1.0e-2, 1.0e-4, 1.0e-6):
        for level, (nu, nv) in enumerate(levels):
            case = output / f"static-epsilon-{epsilon:.0e}" / f"coordinate-{level}"
            try:
                summary = vacuum_case(
                    case_id=f"exp02-static-eps{epsilon:.0e}-C{level}",
                    builder=lambda grid, u, v, e=epsilon: (
                        vacuum_bench.near_horizon_static_state(
                            grid, u, v, 1.0, e
                        )
                    ),
                    u=np.linspace(-1.0, -0.5, nu),
                    v=np.linspace(0.0, 0.1, nv),
                    points=50,
                    retained_degree=5,
                    iterations=8,
                    output=case,
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
                    1.0,
                    u_offset=0.75,
                    v_offset=1.0,
                ),
                u=np.linspace(-1.0, -0.5, nu),
                v=np.linspace(0.0, 0.1, nv),
                points=50,
                retained_degree=5,
                iterations=8,
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


def run_experiment_3(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    levels = (17, 33, 49)
    summaries = []
    for epsilon in (2.0**-power for power in range(1, 7)):
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
                    1.0,
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
                    outgoing_log_lapse=np.asarray(exact["log_omega"][0]),
                    incoming_log_lapse=np.asarray(exact["log_omega"][:, 0]),
                    corner_radius_mismatch=np.asarray(0.0),
                    corner_log_lapse_mismatch=np.asarray(0.0),
                    minimum_radius=np.asarray(np.min(exact["radius"])),
                )
                solution = solve_curved(
                    epsilon,
                    count,
                    count,
                    iterations=20,
                    tolerance=1.0e-12,
                )
                overgrid = curved_overgrid_audit(solution, epsilon)
                relative = np.abs(solution.radius - exact["radius"]) / np.maximum(
                    np.abs(exact["radius"]), 1.0e-30
                )
                protected = {}
                for name, threshold in (
                    ("r_ge_M", 1.0),
                    ("r_ge_half_M", 0.5),
                    ("r_ge_8M_epsilon", 8.0 * epsilon),
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
                    log_omega=np.asarray(solution.log_omega),
                    x_out_scalar=np.asarray(solution.x_out_scalar),
                    x_in_scalar=np.asarray(solution.x_in_scalar),
                    w_out=np.asarray(solution.w_out),
                    w_in=np.asarray(solution.w_in),
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
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"immutable case exists: {output}")
    output.mkdir(parents=True)
    grid, angular = ese_bench.build_angular(config)
    mesh = ese_bench.mesh_from_config(config.scalar_coordinates)
    exact_old, exact_diagnostics = ese_bench.jnw_exact_state(
        grid, mesh.u, mesh.v, sigma=1.0, nu=nu
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
    # The mutation changes the numerical scalar field presented to the same
    # independent geometric pipeline; it does not call an analytic identity.
    exact_overgrid = resample_primitives(
        grid,
        PrimitiveFields(
            metric=exact.metric,
            log_omega=exact.log_omega,
            shift=exact.shift,
            phi=exact.phi,
        ),
        mesh.u,
        mesh.v,
        u_count=len(mesh.u) + 4,
        v_count=len(mesh.v) + 4,
        point_count=max(config.angular.point_count + 12, 64),
        harmonic_degree=config.angular.retained_degree,
    )
    exact_direct = evaluate_blocked(
        exact_overgrid.grid,
        exact_overgrid.fields,
        exact_overgrid.u,
        exact_overgrid.v,
        stencil=7,
        derivative_halo=6,
        mask_halo=4,
        block_size=4,
    )
    mutated_fields = PrimitiveFields(
        metric=exact_overgrid.fields.metric,
        log_omega=exact_overgrid.fields.log_omega,
        shift=exact_overgrid.fields.shift,
        phi=1.01 * exact_overgrid.fields.phi,
    )
    mutated_direct = evaluate_blocked(
        exact_overgrid.grid,
        mutated_fields,
        exact_overgrid.u,
        exact_overgrid.v,
        stencil=7,
        derivative_halo=6,
        mask_halo=4,
        block_size=4,
    )
    mutation = {
        "kind": "multiply the scalar field in the exact numerical state by 1.01",
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
        "state_semantics": "full weighted forms; traces derived from current metric",
        "config": config.to_dict(),
        "exact_diagnostics": exact_diagnostics,
        "boundary_digest": boundary.digest,
        "records": records,
        "closures": closures,
        "first_order_null_residual_fresh": null_residual,
        "independent_four_metric_residual": geometric_residual,
        "independent_state_mutation_test": mutation,
    }
    write_json(output / "summary.json", summary)
    return summary


def run_experiment_6(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    levels = ((2, 6, 2, 6), (4, 6, 4, 6), (8, 6, 8, 6))
    summaries = []
    for nu in (0.99, 0.8, 0.5, 0.2):
        for level, (te, td, se, sd) in enumerate(levels):
            config = ese_bench._configuration(
                name=f"jnw-nu{nu:.2f}-C{level}",
                tau_elements=te,
                tau_degree=td,
                s_elements=se,
                s_degree=sd,
                iterations=8,
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
        grid, mesh.u, mesh.v, sigma=1.0, nu=1.0
    )
    limit_test = {
        "nu": 1.0,
        "expected_schwarzschild_mass": 0.5,
        "maximum_abs_phi": float(np.max(np.abs(limit.phi))),
        "maximum_abs_P3": float(np.max(np.abs(limit.incoming_scalar))),
        "maximum_abs_P4": float(np.max(np.abs(limit.scalar_p))),
        "passed": bool(
            np.max(np.abs(limit.phi)) == 0.0
            and np.max(np.abs(limit.incoming_scalar)) == 0.0
            and np.max(np.abs(limit.scalar_p)) == 0.0
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

    def initial_state(bundle: Any, angular: Any) -> WeightedState:
        state = from_numerical(original_initial_state(bundle, angular), captured["grid"])
        captured["state"] = state
        return state

    def step(
        grid: Any,
        angular: Any,
        mesh: Any,
        state: WeightedState,
        bundle: Any,
        **kwargs: Any,
    ) -> tuple[WeightedState, dict[str, Any]]:
        result, context = ese_picard_step(
            grid, angular, mesh, state, bundle, **kwargs
        )
        captured["state"] = result
        return result, context

    def save_state(
        path: Path,
        state: WeightedState,
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

    nonspherical._scaled_construct = scaled_construct
    ese_run.initial_state = initial_state
    ese_run.picard_step = step
    ese_run.update_norm = weighted_update_norm
    ese_run.update_map = weighted_update_map
    ese_run.save_state = save_state
    try:
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
        )
    finally:
        nonspherical._scaled_construct = original_scaled_construct
        ese_run.initial_state = original_initial_state
        ese_run.picard_step = original_step
        ese_run.update_norm = original_update_norm
        ese_run.update_map = original_update_map
        ese_run.save_state = original_save_state

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
    assert mutated_state.scalar is not None
    mutated_state.scalar = 1.01 * mutated_state.scalar
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
        "independent_four_metric_residual": geometric_residual,
        "independent_state_mutation_test": mutation,
        "limitations": [
            "The one-step algebraic kernel remains the numerical backend RK4 "
            "implementation; all stored and iterated states use official semantics."
        ],
    }
    write_json(case_output / "summary.json", summary)
    return summary


def run_experiment_7(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    baseline = (1.0, 1.0, 1.0, 1.0)
    summaries: list[dict[str, Any]] = []

    central = {
        "retained": 8,
        "work": 16,
        "points": 350,
        "tau_elements": 2,
        "s_elements": 2,
        "tau_degree": 8,
        "s_degree": 11,
        "metric_substeps": 2,
        "derivative_halo": 2,
        "iterations": 6,
    }
    refinement_cases = [
        (
            "coordinate-level-0",
            {
                **central,
                "tau_degree": 6,
                "s_degree": 9,
            },
        ),
        ("baseline-cap-0.04", central),
        (
            "coordinate-level-2",
            {
                **central,
                "tau_elements": 3,
                "s_elements": 3,
            },
        ),
        (
            "angular-level-0",
            {
                **central,
                "retained": 6,
                "work": 12,
                "points": 200,
            },
        ),
        (
            "angular-level-2",
            {
                **central,
                "retained": 10,
                "work": 20,
                "points": 550,
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
            write_json(case / "summary.json", summary)
        summaries.append(summary)
        return summary

    refinement = {
        name: execute(name, 0.04, baseline, options)
        for name, options in refinement_cases
    }

    def protected_residual(name: str) -> float | None:
        try:
            return float(
                refinement[name]["independent_four_metric_residual"][
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
            0.04,
            baseline,
            {**central, "rotation_angle": 0.731},
        )
        for cap in (0.01, 0.02, 0.10):
            execute(f"baseline-cap-{cap:.2f}", cap, baseline, central)
        for axis in range(4):
            for value in (0.0, 0.5, 2.0):
                multipliers = list(baseline)
                multipliers[axis] = value
                execute(
                    f"stress-axis-{axis}-value-{value:.1f}",
                    0.04,
                    tuple(multipliers),
                    central,
                )
        for value in (0.0, 0.5, 2.0):
            execute(
                f"stress-joint-{value:.1f}",
                0.04,
                (value, value, value, value),
                central,
            )
    aggregate = {
        "schema": "nee-official-experiment-07-aggregate-v1",
        "experiment": 7,
        "runs": summaries,
        "harmonics": {
            "Y_Omega": "normalized real Y_20",
            "V_b": "normalized grad(real Y_21)",
            "V_chi": "0.1 normalized grad(real Y_22)",
            "corner_power": 0.1,
            "corner_regularization": "none",
        },
        "refinement_controls": {
            "central_configuration": central,
            "coordinate_levels": (
                {"elements": 2, "tau_degree": 6, "s_degree": 9},
                {"elements": 2, "tau_degree": 8, "s_degree": 11},
                {"elements": 3, "tau_degree": 8, "s_degree": 11},
            ),
            "angular_retained_degree": (6, 8, 10),
            "protected_direct_region": "s >= 0.60",
        },
        "refinement_gate": refinement_gate,
        "stress_matrix_status": stress_status,
    }
    write_json(output / "aggregate-summary.json", aggregate)
    return aggregate
