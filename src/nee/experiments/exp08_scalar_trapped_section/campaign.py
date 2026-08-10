"""Curved-domain trapped-region and apparent-horizon campaign."""

from __future__ import annotations

import gc
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from nee.config import ExperimentConfig as PublicConfig
from nee.experiments._campaign_support import mapped_direct_audit
from nee.io.boundary_artifact import load_boundary_data, save_boundary_data
from nee.io.state_artifact import load_state, save_state
from nee.numerics.scalar_config import (
    AngularConfig,
    CoordinateConfig,
    ExperimentConfig,
    InitialDataConfig,
    SolverConfig,
)
from nee.numerics.scalar_initial_data import (
    InitialDataBundle,
    construct_initial_data,
)
from nee.numerics.scalar_iteration import (
    ESEState,
    initial_state,
    picard_step,
    update_map,
    update_norm,
)
from nee.numerics.scalar_residual import components, l2_maps, safe_summary
from nee.numerics.sphere import (
    PointSphereGrid,
    fibonacci_sphere,
    spherical_harmonic_collocation,
    tangent_inverse,
    tensor_trace,
)
from nee.solver.backend import from_numerical
from nee.state.boundary import BoundaryData
from nee.state.fields import PrimitiveFields

from .horizon import Patch, load_patch, save_surface, solve_mots, trace_horizon
from .mots_geometry import NullConeGeometry, composite_value_and_derivative


Array = np.ndarray
@dataclass(frozen=True)
class PatchSpec:
    name: str
    u_right: float
    v_cap: float
    tau_elements: int
    s_elements: int
    point_count: int
    retained_degree: int
    work_degree: int

def _tau_elements(u_right: float) -> int:
    base_width = math.log(50.0) / 12.0
    return max(2, math.ceil(-math.log(-u_right) / base_width))


def _cap(public: PublicConfig, u_right: float) -> float:
    return min(
        float(public.physics["curved_constant"])
        * (-u_right) ** float(public.physics["curved_exponent"]),
        float(public.physics["v_max"]),
    )


def _production_specs(public: PublicConfig) -> tuple[PatchSpec, ...]:
    values = public.initial_data
    rows = zip(
        values["atlas_names"],
        values["atlas_u_rights"],
        values["atlas_tau_elements"],
        values["atlas_s_elements"],
        values["atlas_point_counts"],
        values["atlas_retained_degrees"],
        values["atlas_work_degrees"],
        strict=True,
    )
    return tuple(
        PatchSpec(
            str(name),
            float(u_right),
            _cap(public, float(u_right)),
            int(tau_elements),
            int(s_elements),
            int(point_count),
            int(retained_degree),
            int(work_degree),
        )
        for (
            name,
            u_right,
            tau_elements,
            s_elements,
            point_count,
            retained_degree,
            work_degree,
        ) in rows
    )


def _mots_control_specs(public: PublicConfig) -> tuple[PatchSpec, PatchSpec]:
    values = public.initial_data
    u_right = float(values["mots_anchor_u"])
    point_count = int(values["mots_point_count"])
    retained = int(values["mots_retained_degree"])
    work = int(values["mots_work_degree"])
    return (
        PatchSpec(
            "mots-anchor-standard",
            u_right,
            _cap(public, u_right),
            int(values["mots_standard_tau_elements"]),
            int(values["mots_standard_s_elements"]),
            point_count,
            retained,
            work,
        ),
        PatchSpec(
            "mots-anchor-refined",
            u_right,
            _cap(public, u_right),
            int(values["mots_refined_tau_elements"]),
            int(values["mots_refined_s_elements"]),
            point_count,
            retained,
            work,
        ),
    )


def _support_specs(public: PublicConfig) -> tuple[PatchSpec, ...]:
    u_right_values = tuple(
        float(value) for value in public.initial_data["support_u_rights"]
    )
    return tuple(
        PatchSpec(
            f"support-v{int(round(_cap(public, u) * 1e6)):06d}",
            u,
            _cap(public, u),
            _tau_elements(u),
            3,
            90,
            4,
            7,
        )
        for u in u_right_values
    )


def numerical_config(spec: PatchSpec, public: PublicConfig) -> ExperimentConfig:
    """Translate one public atlas-patch specification to the numerical kernel."""

    kappa = float(public.physics["kappa"])
    delta = float(public.physics["delta"])
    corner_scalar = -math.sqrt(2.0 / kappa) / (1.0 + kappa)
    result = ExperimentConfig(
        name=spec.name,
        angular=AngularConfig(
            point_count=spec.point_count,
            neighbor_count=min(public.angular.neighbor_count, spec.point_count - 1),
            retained_degree=spec.retained_degree,
            work_degree=spec.work_degree,
        ),
        scalar_coordinates=CoordinateConfig(
            u_left=-1.0,
            u_right=spec.u_right,
            v_max=spec.v_cap,
            tau_elements=spec.tau_elements,
            tau_degree=int(public.initial_data["tau_degree"]),
            s_elements=spec.s_elements,
            s_degree=int(public.initial_data["s_degree"]),
            fractional_power=delta,
        ),
        scalar_initial_data=InitialDataConfig(
            lapse_radial_power=kappa,
            lapse_angular_amplitude=0.0,
            shift_amplitude=0.0,
            corner_outgoing_expansion=2.0 / (1.0 + kappa),
            corner_outgoing_scalar=corner_scalar,
            incoming_scalar_branch="negative",
            outgoing_scalar_power=delta,
            outgoing_scalar_amplitude=float(
                public.initial_data["scalar_amplitude"]
            ),
            outgoing_profile_scale=float(
                public.initial_data["profile_scale"]
            ),
            shear_vector_amplitude=float(
                public.initial_data["shear_amplitude"]
            ),
            shear_profile_power=delta,
            shear_profile=str(public.initial_data["shear_profile"]),
            boundary_substeps=public.solver.boundary_substeps,
            corner_zeta=0.0,
            corner_weighted_omega=0.0,
        ),
        solver=SolverConfig(
            picard_iterations=public.solver.maximum_sweeps,
            metric_substeps=public.solver.metric_substeps,
            derivative_halo_u=public.audit.derivative_halo_u,
            derivative_halo_v=public.audit.derivative_halo_v,
        ),
    )
    result.validate()
    return result


def _verified_data(
    output: Path, config: ExperimentConfig
) -> tuple[InitialDataBundle, Any, Any, Any, BoundaryData]:
    bundle, grid, angular, mesh = construct_initial_data(config)
    boundary = BoundaryData.create(
        bundle.outgoing,
        bundle.incoming,
        metadata={
            "experiment": "scalar-apparent-horizon",
            "configuration": config.name,
            "constraint_checks": bundle.metadata,
        },
    )
    output.mkdir(parents=True, exist_ok=False)
    path = output / "boundary-data.npz"
    save_boundary_data(path, boundary, u=mesh.u, v=mesh.v)
    verified, u, v = load_boundary_data(path)
    np.testing.assert_array_equal(u, mesh.u)
    np.testing.assert_array_equal(v, mesh.v)
    verified_bundle = InitialDataBundle(
        incoming=dict(verified.incoming),
        outgoing=dict(verified.outgoing),
        raw={name: np.asarray(value).copy() for name, value in bundle.raw.items()},
        metadata=dict(bundle.metadata),
    )
    return verified_bundle, grid, angular, mesh, verified


def _face_mismatches(state: ESEState, bundle: InitialDataBundle) -> dict[str, float]:
    mismatches: dict[str, float] = {}
    for state_name, data_name in (
        ("g", "g"),
        ("Omega_trchi", "Omega_trchi"),
        ("Omega_chih", "Omega_chih"),
        ("Omega", "Omega"),
        ("Omega_omega", "Omega_omega"),
        ("phi", "phi"),
        ("Omega_e4phi", "Omega_e4phi"),
    ):
        mismatches[f"outgoing.{state_name}"] = float(
            np.max(np.abs(getattr(state, state_name)[:, 0] - bundle.outgoing[data_name]))
        )
    for state_name, data_name in (
        ("g", "g"),
        ("Omega_trchi", "Omega_trchi"),
        ("Omega_chih", "Omega_chih"),
        ("phi", "phi"),
        ("zeta", "zeta"),
        ("b", "b"),
        ("Omega_chib", "Omega_chib"),
        ("Omega_omegab", "Omega_omegab"),
        ("Omega_e3phi", "Omega_e3phi"),
    ):
        mismatches[f"incoming.{state_name}"] = float(
            np.max(np.abs(getattr(state, state_name)[:, :, 0] - bundle.incoming[data_name]))
        )
    return mismatches


def _iterate(
    *,
    state: ESEState,
    bundle: InitialDataBundle,
    grid: Any,
    angular: Any,
    mesh: Any,
    config: ExperimentConfig,
    boundary: BoundaryData,
) -> tuple[ESEState, list[dict[str, Any]], list[Array], dict[str, list[Array]]]:
    records: list[dict[str, Any]] = []
    updates: list[Array] = []
    residual_history: dict[str, list[Array]] = {}
    for sweep in range(1, config.solver.picard_iterations + 1):
        previous = state
        started = time.perf_counter()
        state, context = picard_step(
            grid,
            angular,
            mesh,
            previous,
            bundle,
            metric_substeps=config.solver.metric_substeps,
        )
        boundary.verify_unchanged()
        face_mismatches = _face_mismatches(state, bundle)
        maximum_face_mismatch = max(face_mismatches.values(), default=0.0)
        if maximum_face_mismatch > 5.0e-12:
            raise FloatingPointError(
                "a fixed characteristic trace changed during the Picard sweep: "
                f"{maximum_face_mismatch:.6e}"
            )
        change_map = update_map(state, previous)
        values = components(grid, state, previous, context, mesh)
        maps = l2_maps(grid, state, mesh, values)
        residual = safe_summary(
            maps,
            config.solver.derivative_halo_u,
            config.solver.derivative_halo_v,
            mesh,
        )
        updates.append(change_map)
        for name, value in maps.items():
            residual_history.setdefault(name, []).append(value)
        record = {
            "sweep": sweep,
            "seconds": time.perf_counter() - started,
            "update": update_norm(state, previous),
            "maximum_update_map": float(np.max(change_map)),
            "construction_residual": residual,
            "minimum_lapse": float(np.min(state.Omega)),
            "maximum_fixed_face_mismatch": maximum_face_mismatch,
            "fixed_face_mismatches": face_mismatches,
            "boundary_hash_verified": True,
        }
        records.append(record)
        print(
            f"{config.name}: sweep {sweep}/{config.solver.picard_iterations} "
            f"update={record['update']:.6e}",
            flush=True,
        )
    return state, records, updates, residual_history


def _resample_scalar(
    points: Array, values: Array, degree: int, target_count: int
) -> Array:
    _, source_basis, _ = spherical_harmonic_collocation(points, degree)
    _, target_basis, _ = spherical_harmonic_collocation(
        fibonacci_sphere(target_count), degree
    )
    coefficients = np.linalg.pinv(source_basis, rcond=1.0e-13) @ values.reshape(
        len(points), -1
    )
    return (target_basis @ coefficients).reshape(
        (target_count, *values.shape[1:])
    )


def _trapped_scan(
    state: ESEState,
    grid: Any,
    degree: int,
    u: Array,
    v: Array,
    sphere_point_count: int,
) -> dict[str, Any]:
    inverse = tangent_inverse(grid, state.g)
    outgoing = _resample_scalar(
        grid.points, state.Omega_trchi, degree, sphere_point_count
    )
    incoming = _resample_scalar(
        grid.points,
        tensor_trace(state.Omega_chib, inverse) / state.Omega,
        degree,
        sphere_point_count,
    )
    outgoing_supremum = np.max(outgoing, axis=0)
    incoming_supremum = np.max(incoming, axis=0)
    trapped = (outgoing_supremum < 0.0) & (incoming_supremum < 0.0)
    protected = trapped.copy()
    protected[:3] = False
    protected[-3:] = False
    indices = np.argwhere(trapped)
    coordinates = [[float(u[i]), float(v[j])] for i, j in indices]
    bounds = None
    if coordinates:
        values = np.asarray(coordinates)
        bounds = {
            "u_min": float(values[:, 0].min()),
            "u_max": float(values[:, 0].max()),
            "v_min": float(values[:, 1].min()),
            "v_max": float(values[:, 1].max()),
        }
    return {
        "sphere_point_count": sphere_point_count,
        "raw_trapped_count": int(np.count_nonzero(trapped)),
        "u_protected_trapped_count": int(np.count_nonzero(protected)),
        "candidate_indices": indices.tolist(),
        "candidate_coordinates": coordinates,
        "coordinate_bounds": bounds,
        "outgoing_supremum": outgoing_supremum.tolist(),
        "incoming_supremum": incoming_supremum.tolist(),
    }


def _run_patch(output: Path, spec: PatchSpec, public: PublicConfig) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"immutable patch exists: {output}")
    output.mkdir(parents=True)
    config = numerical_config(spec, public)
    bundle, grid, angular, mesh, boundary = _verified_data(
        output / "boundary", config
    )
    state = initial_state(bundle, angular)
    state, records, update_maps, residual_history = _iterate(
        state=state,
        bundle=bundle,
        grid=grid,
        angular=angular,
        mesh=mesh,
        config=config,
        boundary=boundary,
    )
    public_state = from_numerical(state, grid)
    state_hash = save_state(
        output / "final-state.npz", public_state, u=mesh.u, v=mesh.v
    )
    np.savez_compressed(
        output / "residual-maps.npz",
        update_maps=np.asarray(update_maps),
        **{name: np.asarray(value) for name, value in residual_history.items()},
    )
    sign = _trapped_scan(
        state,
        grid,
        config.angular.retained_degree,
        mesh.u,
        mesh.v,
        public.audit.sphere_point_count,
    )
    independent = mapped_direct_audit(
        grid,
        public_state,
        mesh,
        retained_degree=config.angular.retained_degree,
        protected_s_values=(0.6,),
    )
    summary = {
        "schema": "nee-exp08-curved-patch-v1",
        "curved_region": {
            "constant": float(public.physics["curved_constant"]),
            "exponent": float(public.physics["curved_exponent"]),
            "u_right": spec.u_right,
            "v_cap": spec.v_cap,
            "rectangle_is_inside_region": True,
        },
        "config": config.to_dict(),
        "boundary_hash": boundary.content_hash,
        "sweeps": records,
        "trapped_scan": sign,
        "independent_Ric_minus_dphi_dphi": independent,
        "minimum_lapse": float(np.min(state.Omega)),
        "maximum_fixed_face_mismatch": float(
            max(record["maximum_fixed_face_mismatch"] for record in records)
        ),
        "final_state_hash": state_hash,
        "terminal_status": "completed",
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def _evaluate_uv(
    patch: Patch,
    u_value: float,
    v_value: float,
    target_grid: PointSphereGrid,
    degree: int,
) -> dict[str, Array]:
    tau = -math.log(-u_value)
    s = (v_value / patch.mesh.v1) ** patch.mesh.delta
    _, source_basis, _ = spherical_harmonic_collocation(
        patch.grid.points, degree
    )
    _, target_basis, _ = spherical_harmonic_collocation(
        target_grid.points, degree
    )
    angular_map = target_basis @ np.linalg.pinv(source_basis, rcond=1.0e-13)

    def transfer(values: Array) -> Array:
        at_u, _ = composite_value_and_derivative(
            patch.mesh.tau, values, tau, axis=1
        )
        at_uv, _ = composite_value_and_derivative(
            patch.mesh.s, at_u, s, axis=1
        )
        return np.tensordot(angular_map, at_uv, axes=(1, 0))

    return {
        "g": transfer(patch.state.g),
        "log_Omega": transfer(patch.state.log_Omega),
        "b": transfer(patch.state.b),
        "phi": transfer(patch.state.phi),
        "Omega_trchi": transfer(patch.state.Omega_trchi),
    }


def _overlap_comparison(
    left: Patch, right: Patch, target_grid: PointSphereGrid
) -> dict[str, Any]:
    intersection_u_right = min(left.u_right, right.u_right)
    intersection_v_cap = min(left.cap, right.cap)
    sample_u = -1.0 + 0.65 * (intersection_u_right + 1.0)
    sample_v = 0.55 * intersection_v_cap
    left_values = _evaluate_uv(left, sample_u, sample_v, target_grid, 4)
    right_values = _evaluate_uv(right, sample_u, sample_v, target_grid, 4)
    errors = {}
    for name in left_values:
        difference = left_values[name] - right_values[name]
        scale = max(
            float(np.max(np.abs(left_values[name]))),
            float(np.max(np.abs(right_values[name]))),
            1.0e-14,
        )
        errors[name] = {
            "linf": float(np.max(np.abs(difference))),
            "relative_linf": float(np.max(np.abs(difference)) / scale),
            "rms": float(np.sqrt(np.mean(difference**2))),
        }
    return {
        "left": left.name,
        "right": right.name,
        "sample_u": sample_u,
        "sample_v": sample_v,
        "errors": errors,
    }


def _coordinate_section(patch: Patch, u_value: float, v_value: float) -> dict[str, Any]:
    cone = NullConeGeometry.create_at_v(
        patch.grid, patch.fields, patch.mesh, v_value
    )
    graph = cone.expansion(np.full(patch.grid.count, u_value))
    return {
        "u": u_value,
        "v": v_value,
        "theta_out_min": float(np.min(graph.theta_out)),
        "theta_out_sup": float(np.max(graph.theta_out)),
        "theta_in_min": float(np.min(graph.theta_in)),
        "theta_in_sup": float(np.max(graph.theta_in)),
        "strictly_trapped": bool(
            np.max(graph.theta_out) < 0.0 and np.max(graph.theta_in) < 0.0
        ),
    }


def _promote_patch_artifacts(output: Path, patch: Path) -> None:
    shutil.copy2(patch / "final-state.npz", output / "final-state.npz")
    shutil.copy2(patch / "residual-maps.npz", output / "residual-maps.npz")
    shutil.copy2(
        patch / "boundary" / "boundary-data.npz",
        output / "boundary-data.npz",
    )


def _curved_area(public: PublicConfig) -> tuple[float, float, float]:
    constant = float(public.physics["curved_constant"])
    exponent = float(public.physics["curved_exponent"])
    v_max = float(public.physics["v_max"])
    u_right = float(public.physics["u_right"])
    flat_radius = (v_max / constant) ** (
        1.0 / exponent
    )
    exact = v_max * (1.0 - flat_radius) + constant / (
        exponent + 1.0
    ) * (
        flat_radius ** (exponent + 1.0)
        - (-u_right) ** (exponent + 1.0)
    )
    left = -1.0
    staircase = 0.0
    for spec in _production_specs(public):
        staircase += (spec.u_right - left) * spec.v_cap
        left = spec.u_right
    return exact, staircase, staircase / exact


def _summary_figure(output: Path, report: dict[str, Any]) -> None:
    figure_dir = output / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    region = report["region"]
    u_curve = np.linspace(region["u_interval"][0], region["u_interval"][1], 500)
    v_curve = np.minimum(
        region["constant"] * (-u_curve) ** region["exponent"], region["v_max"]
    )
    axes[0].plot(u_curve, v_curve, color="black", label="curved boundary")
    previous_u = -1.0
    for row in report["atlas"]["patches"]:
        axes[0].fill_between(
            [previous_u, row["u_right"]],
            0.0,
            row["v_cap"],
            alpha=0.18,
            step="post",
        )
        previous_u = row["u_right"]
        coordinates = np.asarray(row["trapped_coordinates"])
        if coordinates.size:
            axes[0].scatter(
                coordinates[:, 0], coordinates[:, 1], s=8, color="tab:red"
            )
    horizon = report["apparent_horizon"]["degree4"]["sections"]
    axes[0].plot(
        [row["surface"]["h_mean"] for row in horizon],
        [row["v"] for row in horizon],
        "o-",
        ms=3,
        color="tab:blue",
        label="apparent horizon",
    )
    axes[0].set(xlabel="u", ylabel="v", title="Curved domain and trapped region")
    axes[0].legend()

    axes[1].plot(
        [row["v"] for row in horizon],
        [row["surface"]["areal_radius"] for row in horizon],
        "o-",
    )
    axes[1].set(
        xlabel="v",
        ylabel="areal radius",
        title="Apparent-horizon area radius",
    )
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_dir / "curved-domain-and-horizon.png", dpi=180)
    plt.close(fig)


def _run_standard(output: Path, public: PublicConfig) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    production = _production_specs(public)
    controls = _mots_control_specs(public)
    supports = _support_specs(public)
    target_u = float(public.physics["target_u"])
    target_v = float(public.physics["target_v"])
    domain_v_max = float(public.physics["v_max"])
    summaries: dict[str, dict[str, Any]] = {}
    for spec in (*production, *controls, *supports):
        summaries[spec.name] = _run_patch(output / "patches" / spec.name, spec, public)

    production_patches = [
        load_patch(output / "patches", spec.name) for spec in production
    ]
    target_grid = PointSphereGrid.create(
        50, neighbor_count=24, spectral_degree=4
    )
    overlaps = [
        _overlap_comparison(left, right, target_grid)
        for left, right in zip(
            production_patches[:-1], production_patches[1:], strict=True
        )
    ]

    standard_anchor = load_patch(output / "patches", controls[0].name)
    refined_anchor = load_patch(output / "patches", controls[1].name)
    target = _coordinate_section(refined_anchor, target_u, target_v)
    standard_mots, standard_arrays = solve_mots(
        standard_anchor,
        target_v,
        degree=int(public.initial_data["mots_graph_degree"]),
    )
    refined_mots, refined_arrays = solve_mots(
        refined_anchor,
        target_v,
        degree=int(public.initial_data["mots_graph_degree"]),
    )
    save_surface(output / "mots-controls" / "standard", standard_mots, standard_arrays)
    save_surface(output / "mots-controls" / "refined", refined_mots, refined_arrays)
    mots_control = {
        "standard": standard_mots,
        "refined": refined_mots,
        "h_coordinate_refinement_Linf": float(
            np.max(np.abs(standard_arrays["h"] - refined_arrays["h"]))
        ),
        "area_coordinate_refinement_relative": float(
            abs(
                standard_mots["surface"]["area"]
                / refined_mots["surface"]["area"]
                - 1.0
            )
        ),
    }

    horizon_names = [spec.name for spec in production]
    horizon_names.append(controls[1].name)
    horizon_names.extend(spec.name for spec in supports)
    horizon_patches = [
        load_patch(output / "patches", name) for name in horizon_names
    ]
    caps = np.asarray([patch.cap for patch in horizon_patches])
    requested_v = np.unique(
        np.concatenate(
            (
                caps,
                np.array(
                    [
                        float(public.physics["horizon_v_min"]),
                        target_v,
                        domain_v_max,
                    ]
                ),
            )
        )
    )
    degree4 = trace_horizon(
        horizon_patches,
        output / "apparent-horizon" / "degree4",
        degree=int(public.initial_data["horizon_graph_degree"]),
        requested_v=requested_v,
    )
    degree3 = trace_horizon(
        horizon_patches,
        output / "apparent-horizon" / "degree3-control",
        degree=int(public.initial_data["horizon_control_degree"]),
        requested_v=requested_v,
    )
    degree3_by_v = {row["v"]: row for row in degree3["sections"]}
    h_degree_difference = max(
        abs(row["surface"]["h_mean"] - degree3_by_v[row["v"]]["surface"]["h_mean"])
        for row in degree4["sections"]
    )
    area_degree_difference = max(
        abs(
            row["surface"]["area"]
            / degree3_by_v[row["v"]]["surface"]["area"]
            - 1.0
        )
        for row in degree4["sections"]
    )
    v_fit = np.asarray([row["v"] for row in degree4["sections"]])
    h_fit = -np.asarray(
        [row["surface"]["h_mean"] for row in degree4["sections"]]
    )
    exponent, log_factor = np.polyfit(np.log(v_fit), np.log(h_fit), 1)

    patch_rows = []
    for spec in production:
        summary = summaries[spec.name]
        protected = summary["independent_Ric_minus_dphi_dphi"]["protected"][
            "s_ge_0.60"
        ]
        patch_rows.append(
            {
                "name": spec.name,
                "u_right": spec.u_right,
                "v_cap": spec.v_cap,
                "grid": {
                    "u_count": 1 + 8 * spec.tau_elements,
                    "v_count": 1 + 11 * spec.s_elements,
                    "sphere_point_count": spec.point_count,
                    "retained_degree": spec.retained_degree,
                },
                "last_update": summary["sweeps"][-1]["update"],
                "raw_trapped_count": summary["trapped_scan"]["raw_trapped_count"],
                "u_protected_trapped_count": summary["trapped_scan"][
                    "u_protected_trapped_count"
                ],
                "trapped_coordinates": summary["trapped_scan"][
                    "candidate_coordinates"
                ],
                "residual": protected,
            }
        )
    exact_area, staircase_area, coverage = _curved_area(public)
    maximum_overlap = max(
        error["linf"]
        for row in overlaps
        for error in row["errors"].values()
    )
    report = {
        "schema": "nee-exp08-curved-domain-apparent-horizon-v1",
        "region": {
            "u_interval": [-1.0, float(public.physics["u_right"])],
            "v_lower": 0.0,
            "v_upper": "min(0.1*(-u)^(25/24), 0.05)",
            "constant": float(public.physics["curved_constant"]),
            "exponent": float(public.physics["curved_exponent"]),
            "v_max": domain_v_max,
            "coordinate_area": exact_area,
        },
        "atlas": {
            "coordinate_area_staircase": staircase_area,
            "coverage_fraction": coverage,
            "patches": patch_rows,
            "overlap_comparisons": overlaps,
            "maximum_absolute_overlap_discrepancy": maximum_overlap,
        },
        "specified_coordinate_section": target,
        "mots_v_0p04": mots_control,
        "apparent_horizon": {
            "degree4": degree4,
            "degree3_control": degree3,
            "degree3_to_degree4_h_mean_Linf": h_degree_difference,
            "degree3_to_degree4_area_relative_Linf": area_degree_difference,
            "power_law_fit": {
                "factor": float(math.exp(log_factor)),
                "exponent": float(exponent),
                "interpretation": "descriptive fit; stored MOTSs define the horizon",
            },
            "verified": bool(
                degree4["failed_v_count"] == 0
                and all(
                    row["surface"]["theta_out"]["linf"] < 1.0e-5
                    and row["surface"]["theta_in"]["maximum"] < 0.0
                    and row["sign_check"]["bracketed"]
                    for row in degree4["sections"]
                )
            ),
        },
        "terminal_status": "completed",
    }
    (output / "aggregate-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    _summary_figure(output, report)
    _promote_patch_artifacts(
        output, output / "patches" / controls[1].name
    )
    del production_patches, horizon_patches
    gc.collect()
    return report


def _run_angular_control(output: Path, public: PublicConfig) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    u_right = float(public.initial_data["mots_anchor_u"])
    spec = PatchSpec(
        "angular-control",
        u_right,
        _cap(public, u_right),
        int(public.initial_data["mots_refined_tau_elements"]),
        int(public.initial_data["mots_refined_s_elements"]),
        public.angular.point_count,
        public.angular.retained_degree,
        public.angular.work_degree,
    )
    patch_summary = _run_patch(output / "patches" / spec.name, spec, public)
    patch = load_patch(output / "patches", spec.name)
    target_v = float(public.physics["target_v"])
    target = _coordinate_section(
        patch, float(public.physics["target_u"]), target_v
    )
    mots, arrays = solve_mots(
        patch,
        target_v,
        degree=int(public.initial_data["mots_graph_degree"]),
    )
    save_surface(output / "mots-v0.04", mots, arrays)
    report = {
        "schema": "nee-exp08-angular-control-v1",
        "patch": {
            "config": patch_summary["config"],
            "last_update": patch_summary["sweeps"][-1]["update"],
            "residual": patch_summary["independent_Ric_minus_dphi_dphi"][
                "protected"
            ]["s_ge_0.60"],
        },
        "specified_coordinate_section": target,
        "mots_v_0p04": mots,
        "terminal_status": "completed",
    }
    (output / "aggregate-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    _promote_patch_artifacts(output, output / "patches" / spec.name)
    return report


def _run_smoke(output: Path, public: PublicConfig) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    u_right = -0.52
    spec = PatchSpec(
        "smoke", u_right, min(_cap(public, u_right), 0.01), 2, 2, 40, 2, 3
    )
    summary = _run_patch(output / "patches" / spec.name, spec, public)
    report = {
        "schema": "nee-exp08-smoke-v1",
        "patch": summary,
        "terminal_status": "completed",
    }
    (output / "aggregate-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    _promote_patch_artifacts(output, output / "patches" / spec.name)
    return report


def run_configuration(output: Path, config: PublicConfig) -> dict[str, Any]:
    stem = config.source_path.stem if config.source_path is not None else "standard"
    if stem == "smoke":
        return _run_smoke(output, config)
    if stem == "angular-control":
        return _run_angular_control(output, config)
    return _run_standard(output, config)
