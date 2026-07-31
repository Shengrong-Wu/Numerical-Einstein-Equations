"""Scalar-pulse evolution, whole-element continuation, and sign audit."""

from __future__ import annotations

import json
import math
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np

from nee.diagnostics.trapped_sections import trapped_sections
from nee.experiments._reference_campaign import mapped_direct_audit
from nee.io.boundary_artifact import load_boundary_data, save_boundary_data
from nee.io.manifest import write_json
from nee.io.state_artifact import save_state
from nee.numerics.scalar_config import (
    AngularConfig,
    CoordinateConfig,
    ExperimentConfig,
    InitialDataConfig,
    SolverConfig,
)
from nee.numerics.scalar_coordinates import mesh_from_config
from nee.numerics.scalar_initial_data import InitialDataBundle, construct_initial_data
from nee.numerics.scalar_iteration import (
    ESEState,
    initial_state,
    picard_step,
    update_map,
    update_norm,
)
from nee.numerics.scalar_residual import components, l2_maps, safe_summary
from nee.numerics.sphere import (
    fibonacci_sphere,
    spherical_harmonic_collocation,
    tangent_inverse,
    tensor_trace,
)
from nee.numerics.scalar_continuation import prolong_appended_tau_elements
from nee.solver.backend import from_numerical
from nee.state.boundary import BoundaryData


Array = np.ndarray


def numerical_config(*, control: bool, continued: bool, quick: bool) -> ExperimentConfig:
    kappa = 0.25
    if quick:
        retained, work, points, neighbors = 2, 3, 40, 16
        tau_elements, tau_degree = (3 if continued else 2), 4
        s_elements, s_degree = 2, 4
        tau_right = (1.5 if continued else 1.0) * math.log(2.0)
        u_right = -math.exp(-tau_right)
        v_max = 0.01
        sweeps = 1
        boundary_substeps = 4
        metric_substeps = 1
    else:
        retained, work, points, neighbors = (
            (7, 14, 300, 40) if control else (5, 10, 170, 32)
        )
        tau_elements, tau_degree = (13 if continued else 12), 8
        s_elements, s_degree = 3, 11
        tau_right = (13.0 / 12.0 if continued else 1.0) * math.log(50.0)
        u_right = -math.exp(-tau_right)
        v_max = 0.04
        sweeps = 16 if continued else 14
        boundary_substeps = 16
        metric_substeps = 4
    corner_scalar = -math.sqrt(2.0 / kappa) / (1.0 + kappa)
    result = ExperimentConfig(
        name=("angular-control" if control else "standard") + ("-continued" if continued else "-base"),
        angular=AngularConfig(
            point_count=points,
            neighbor_count=neighbors,
            retained_degree=retained,
            work_degree=work,
        ),
        scalar_coordinates=CoordinateConfig(
            u_left=-1.0,
            u_right=u_right,
            v_max=v_max,
            tau_elements=tau_elements,
            tau_degree=tau_degree,
            s_elements=s_elements,
            s_degree=s_degree,
            fractional_power=0.1,
        ),
        scalar_initial_data=InitialDataConfig(
            lapse_radial_power=kappa,
            lapse_angular_amplitude=0.0,
            shift_amplitude=0.0,
            corner_outgoing_expansion=2.0 / (1.0 + kappa),
            corner_outgoing_scalar=corner_scalar,
            incoming_scalar_branch="negative",
            outgoing_scalar_power=0.1,
            outgoing_scalar_amplitude=2.0,
            outgoing_profile_scale=0.01,
            shear_vector_amplitude=0.1,
            shear_profile_power=0.1,
            shear_profile="quadrupole",
            boundary_substeps=boundary_substeps,
            corner_zeta=0.0,
            corner_weighted_omega=0.0,
        ),
        solver=SolverConfig(
            picard_iterations=sweeps,
            metric_substeps=metric_substeps,
            derivative_halo_u=1 if quick else 3,
            derivative_halo_v=1 if quick else 3,
        ),
    )
    result.validate()
    return result


def _verified_data(
    output: Path,
    config: ExperimentConfig,
) -> tuple[InitialDataBundle, Any, Any, Any, BoundaryData]:
    bundle, grid, angular, mesh = construct_initial_data(config)
    boundary = BoundaryData.create(
        bundle.outgoing,
        bundle.incoming,
        metadata={
            "experiment": "scalar-trapped-section",
            "configuration": config.name,
            "constraint_checks": bundle.metadata,
        },
    )
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


def _iterate(
    *,
    state: ESEState,
    bundle: InitialDataBundle,
    grid: Any,
    angular: Any,
    mesh: Any,
    config: ExperimentConfig,
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
            "minimum_lapse": float(np.min(state.omega)),
        }
        records.append(record)
        print(
            f"{config.name}: sweep {sweep}/{config.solver.picard_iterations} "
            f"update={record['update']:.6e}",
            flush=True,
        )
    return state, records, updates, residual_history


def _resample_scalar(points: Array, values: Array, degree: int, target_count: int) -> Array:
    _, source_basis, _ = spherical_harmonic_collocation(points, degree)
    _, target_basis, _ = spherical_harmonic_collocation(
        fibonacci_sphere(target_count), degree
    )
    coefficients = np.linalg.pinv(source_basis, rcond=1.0e-13) @ values.reshape(len(points), -1)
    return (target_basis @ coefficients).reshape((target_count, *values.shape[1:]))


def _sign_audit(state: ESEState, grid: Any, degree: int) -> dict[str, Any]:
    inverse = tangent_inverse(grid, state.metric)
    theta_plus = state.q
    theta_minus = tensor_trace(state.weighted_chib, inverse) / state.omega
    plus = _resample_scalar(grid.points, theta_plus, degree, 1000)
    minus = _resample_scalar(grid.points, theta_minus, degree, 1000)
    plus_supremum = np.max(plus, axis=0)
    minus_supremum = np.max(minus, axis=0)
    trapped, protected = trapped_sections(
        plus_supremum,
        minus_supremum,
        u_endpoint_halo=3,
    )
    candidates = np.argwhere(protected)
    return {
        "sphere_point_count": 1000,
        "outgoing_supremum": plus_supremum.tolist(),
        "incoming_supremum": minus_supremum.tolist(),
        "trapped_count": int(np.count_nonzero(trapped)),
        "protected_trapped_count": int(np.count_nonzero(protected)),
        "candidate_indices": candidates.tolist(),
        "minimum_outgoing_margin": float(-np.min(plus_supremum)),
        "minimum_incoming_margin": float(-np.min(minus_supremum)),
    }


def run_configuration(output: Path, *, control: bool, quick: bool) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"immutable run directory exists: {output}")
    output.mkdir(parents=True)
    base_config = numerical_config(control=control, continued=False, quick=quick)
    bundle, grid, angular, mesh, boundary = _verified_data(output / "base", base_config)
    state = initial_state(bundle, angular)
    state, base_records, _, _ = _iterate(
        state=state,
        bundle=bundle,
        grid=grid,
        angular=angular,
        mesh=mesh,
        config=base_config,
    )

    continued_config = numerical_config(control=control, continued=True, quick=quick)
    extended_bundle, extended_grid, extended_angular, extended_mesh, extended_boundary = _verified_data(
        output / "continued", continued_config
    )
    seed = initial_state(extended_bundle, extended_angular)
    extended, prolongation = prolong_appended_tau_elements(
        state,
        mesh.u,
        mesh.v,
        seed,
        extended_mesh.u,
        extended_mesh.v,
        extended_bundle,
    )
    extended, continued_records, update_maps, residual_history = _iterate(
        state=extended,
        bundle=extended_bundle,
        grid=extended_grid,
        angular=extended_angular,
        mesh=extended_mesh,
        config=continued_config,
    )
    public_state = from_numerical(extended, extended_grid)
    state_hash = save_state(
        output / "final-state.npz",
        public_state,
        u=extended_mesh.u,
        v=extended_mesh.v,
    )
    np.savez_compressed(
        output / "residual-maps.npz",
        update_maps=np.asarray(update_maps),
        **{name: np.asarray(values) for name, values in residual_history.items()},
    )
    sign = _sign_audit(
        extended,
        extended_grid,
        continued_config.angular.retained_degree,
    )
    independent = mapped_direct_audit(
        extended_grid,
        public_state,
        extended_mesh,
        retained_degree=continued_config.angular.retained_degree,
        protected_s_values=(0.6,),
    )
    summary = {
        "schema": "nee-scalar-trapped-section-1",
        "configuration": "angular-control" if control else "standard",
        "base_config": base_config.to_dict(),
        "continued_config": continued_config.to_dict(),
        "base_boundary_hash": boundary.content_hash,
        "continued_boundary_hash": extended_boundary.content_hash,
        "base_sweeps": base_records,
        "continued_sweeps": continued_records,
        "prolongation": prolongation,
        "independent_audit": independent,
        "sign_audit": sign,
        "candidate": bool(
            sign["protected_trapped_count"] > 0
            and continued_records[-1]["update"] < 1.0e-4
        ),
        "certificate": False,
        "certificate_limitation": "coordinate-refinement certificate gates are not part of this two-band sign run",
        "final_state_hash": state_hash,
    }
    write_json(output / "summary.json", summary)
    return summary
