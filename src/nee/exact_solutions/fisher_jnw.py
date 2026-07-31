"""JNW/Fisher Einstein--scalar characteristic benchmarks.

The exact characteristic state is regenerated from the official plan and supplied
only on the two initial null faces. The Einstein--scalar Picard map
starts from its non-exact face-compatible seed and must recover the interior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp



from nee.numerics.scalar_config import (  # noqa: E402
    AngularConfig,
    CoordinateConfig,
    ExperimentConfig,
    InitialDataConfig,
    SolverConfig,
)
from nee.numerics.scalar_coordinates import mesh_from_config  # noqa: E402
from nee.numerics.scalar_iteration import (  # noqa: E402
    ESEState,
    initial_state,
    picard_step,
    update_norm,
)
from nee.numerics.scalar_residual import components, l2_maps, safe_summary  # noqa: E402
from nee.numerics.scalar_initial_data import InitialDataBundle, build_angular  # noqa: E402


Array = np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def provenance() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "numerics" / "scalar_iteration.py",
        Path(__file__).resolve().parents[1] / "numerics" / "scalar_residual.py",
        Path(__file__).resolve().parents[1] / "numerics" / "scalar_coordinates.py",
        Path(__file__).resolve().parents[1] / "numerics" / "scalar_initial_data.py",
    ]
    return {str(path): _sha256(path) for path in paths}


def _configuration(
    *,
    name: str,
    tau_elements: int,
    tau_degree: int,
    s_elements: int,
    s_degree: int,
    iterations: int,
    quick: bool,
) -> ExperimentConfig:
    config = ExperimentConfig(
        name=name,
        angular=AngularConfig(
            point_count=100 if quick else 350,
            neighbor_count=24 if quick else 40,
            retained_degree=3 if quick else 8,
            work_degree=6 if quick else 16,
        ),
        scalar_coordinates=CoordinateConfig(
            u_left=-1.0,
            u_right=-0.5,
            v_max=0.5,
            tau_elements=tau_elements,
            tau_degree=tau_degree,
            s_elements=s_elements,
            s_degree=s_degree,
            fractional_power=1.0,
        ),
        # These values are unused by the exact-data generator, but the numerical
        # configuration validator requires the coordinate/profile powers to
        # agree.
        scalar_initial_data=InitialDataConfig(
            outgoing_scalar_power=1.0,
            shear_profile_power=1.0,
        ),
        solver=SolverConfig(
            picard_iterations=iterations,
            metric_substeps=2,
            derivative_halo_u=1,
            derivative_halo_v=1,
        ),
    )
    config.validate()
    return config


def _jnw_radius(x: Array, sigma: float, nu: float) -> Array:
    """Solve dr/d(r*) = (1-sigma/r)^nu with r(0)=2 sigma."""

    values = np.asarray(x, dtype=float)
    lower = float(np.min(values))
    upper = float(np.max(values))
    if lower < -1.0e-14:
        raise ValueError("the configured JNW domain extends below r*=0")

    def rhs(_: float, y: Array) -> Array:
        return np.asarray([(1.0 - sigma / y[0]) ** nu])

    solution = solve_ivp(
        rhs,
        (0.0, max(upper, 1.0e-14)),
        np.asarray([2.0 * sigma]),
        rtol=2.0e-13,
        atol=2.0e-14,
        dense_output=True,
        method="DOP853",
    )
    if not solution.success or solution.sol is None:
        raise RuntimeError(f"JNW optical-map integration failed: {solution.message}")
    return solution.sol(values.ravel())[0].reshape(values.shape)


def jnw_exact_state(
    grid: Any,
    u: Array,
    v: Array,
    sigma: float,
    nu: float,
) -> tuple[ESEState, dict[str, float]]:
    uu, vv = np.meshgrid(u, v, indexing="ij")
    optical = vv - uu - 0.5
    radius = _jnw_radius(optical, sigma, nu)
    f = 1.0 - sigma / radius
    if float(np.min(f)) <= 0.0:
        raise ValueError("JNW grid intersects the curvature singularity")
    c_nu = math.sqrt((1.0 - nu**2) / 2.0)
    areal_radius = radius * f ** ((1.0 - nu) / 2.0)
    lambda_r = (
        1.0 / radius
        + 0.5 * (1.0 - nu) * sigma / (radius**2 * f)
    )
    omega_sq = f**nu
    projector = grid.projector[:, None, None]
    metric = areal_radius[None, ..., None, None] ** 2 * projector
    scalar_shape = (grid.count, len(u), len(v))
    scalar = np.ones(scalar_shape)
    omega = scalar * np.sqrt(omega_sq)[None]
    q = scalar * (2.0 * lambda_r)[None]
    weighted_chib = (
        -(omega_sq * lambda_r)[None, ..., None, None] * metric
    )
    weighted_omega = scalar * (
        -nu * sigma / (4.0 * radius**2) * f ** (nu - 1.0)
    )[None]
    weighted_omegab = -weighted_omega
    phi = scalar * (c_nu * np.log(f))[None]
    scalar_p = scalar * (
        c_nu * sigma / radius**2 * f ** (nu - 1.0)
    )[None]
    state = ESEState(
        metric=metric.copy(),
        omega=omega.copy(),
        zeta_up=np.zeros((*scalar_shape, 3)),
        shift=np.zeros((*scalar_shape, 3)),
        q=q.copy(),
        shear=np.zeros_like(metric),
        weighted_chib=weighted_chib.copy(),
        weighted_omega=weighted_omega.copy(),
        weighted_omegab=weighted_omegab.copy(),
        phi=phi.copy(),
        scalar_p=scalar_p.copy(),
        incoming_scalar=-scalar_p.copy(),
    )
    return state, {
        "sigma": sigma,
        "nu": nu,
        "c_nu": c_nu,
        "minimum_radius": float(np.min(radius)),
        "maximum_radius": float(np.max(radius)),
        "minimum_omega_squared": float(np.min(omega_sq)),
        "minimum_areal_radius": float(np.min(areal_radius)),
    }


def _faces(
    exact: ESEState, u: Array, v: Array, diagnostics: dict[str, float]
) -> InitialDataBundle:
    common = (
        "metric",
        "omega",
        "shift",
        "zeta_up",
        "q",
        "shear",
        "weighted_chib",
        "weighted_omega",
        "weighted_omegab",
        "phi",
        "scalar_p",
        "incoming_scalar",
    )
    incoming = {"u": u.copy()}
    outgoing = {"v": v.copy()}
    for name in common:
        value = np.asarray(getattr(exact, name))
        incoming[name] = value[:, :, 0].copy()
        outgoing[name] = value[:, 0].copy()
    return InitialDataBundle(
        incoming=incoming,
        outgoing=outgoing,
        raw={},
        metadata={
            "schema": "nee-official-exact-jnw-boundary-v1",
            "formula": "experiments-plan:eq:jnw-state",
            "diagnostics": diagnostics,
            "corner_mismatches": {
                name: float(
                    np.max(np.abs(outgoing[name][:, 0] - incoming[name][:, 0]))
                )
                for name in common
            },
        },
    )


def _field_error(numerical: ESEState, exact: ESEState, name: str) -> dict[str, float]:
    value = np.asarray(getattr(numerical, name))
    target = np.asarray(getattr(exact, name))
    difference = value - target
    numerator = float(np.sqrt(np.mean(difference**2)))
    denominator = float(np.sqrt(np.mean(target**2)))
    return {
        "absolute_rms": numerator,
        "absolute_max": float(np.max(np.abs(difference))),
        "relative_rms": numerator / max(denominator, 1.0e-14),
    }


def scalar_normalization_mutation(nu: float, sigma: float = 1.0) -> dict[str, float]:
    """Independent pointwise identity and deliberate coefficient mutation."""

    radius = np.linspace(2.0 * sigma, 4.0 * sigma, 257)
    f = 1.0 - sigma / radius
    dlogf = sigma / (radius**2 * f)
    ricci_rr = 0.5 * (1.0 - nu**2) * dlogf**2
    c = math.sqrt((1.0 - nu**2) / 2.0)
    correct = ricci_rr - (c * dlogf) ** 2
    mutated = ricci_rr - (1.01 * c * dlogf) ** 2
    return {
        "correct_max_abs": float(np.max(np.abs(correct))),
        "mutated_max_abs": float(np.max(np.abs(mutated))),
        "mutation_detected": bool(
            np.max(np.abs(mutated)) > 1.0e8 * max(np.max(np.abs(correct)), 1.0e-30)
        ),
    }


def run_case(
    *,
    nu: float,
    level: int,
    config: ExperimentConfig,
    output: Path,
) -> dict[str, Any]:
    if (output / "summary.json").exists():
        return json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(f"incomplete immutable output exists: {output}")
    grid, angular = build_angular(config)
    mesh = mesh_from_config(config.scalar_coordinates)
    exact, exact_diagnostics = jnw_exact_state(
        grid, mesh.u, mesh.v, sigma=1.0, nu=nu
    )
    bundle = _faces(exact, mesh.u, mesh.v, exact_diagnostics)
    state = initial_state(bundle, angular)
    records: list[dict[str, Any]] = []
    residual_histories: dict[str, list[Array]] = {}
    for iteration in range(1, config.solver.picard_iterations + 1):
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
        residual_values = components(grid, state, previous, context, mesh)
        maps = l2_maps(grid, state, mesh, residual_values)
        residual_summary = safe_summary(
            maps,
            config.solver.derivative_halo_u,
            config.solver.derivative_halo_v,
            mesh,
        )
        for name, value in maps.items():
            residual_histories.setdefault(name, []).append(value)
        record = {
            "iteration": iteration,
            "seconds": time.perf_counter() - started,
            "picard_update": update_norm(state, previous),
            "metric_error": _field_error(state, exact, "metric"),
            "lapse_error": _field_error(state, exact, "omega"),
            "scalar_error": _field_error(state, exact, "phi"),
            "scalar_p_error": _field_error(state, exact, "scalar_p"),
            "incoming_scalar_error": _field_error(
                state, exact, "incoming_scalar"
            ),
            "residual": residual_summary,
        }
        print(json.dumps({"nu": nu, "level": level, **record}), flush=True)
        records.append(record)

    output.mkdir(parents=True, exist_ok=False)
    arrays = {
        item.name: np.asarray(getattr(state, item.name))
        for item in fields(ESEState)
    }
    arrays.update(
        {
            f"exact_{item.name}": np.asarray(getattr(exact, item.name))
            for item in fields(ESEState)
        }
    )
    arrays.update({"u": mesh.u, "v": mesh.v})
    for name, values in residual_histories.items():
        arrays[f"residual__{name}"] = np.stack(values)
    np.savez_compressed(output / "state-and-exact.npz", **arrays)
    bundle.save(output / "boundary-data.npz")
    summary: dict[str, Any] = {
        "schema": "nee-official-exact-jnw-run-v1",
        "case_id": f"exp06-jnw-nu{nu:.2f}-L{level}",
        "config": config.to_dict(),
        "exact_diagnostics": exact_diagnostics,
        "normalization_mutation_test": scalar_normalization_mutation(nu),
        "records": records,
        "terminal_status": "finite",
        "provenance": {
            "sources": provenance(),
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "boundary_data_regenerated": True,
            "seed": "non-exact face-compatible quintic blend",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_experiment(output: Path, quick: bool) -> dict[str, Any]:
    levels = (
        [(1, 4, 1, 4), (2, 4, 2, 4), (4, 4, 4, 4)]
        if quick
        else [(2, 6, 2, 6), (4, 6, 4, 6), (8, 6, 8, 6)]
    )
    summaries: list[dict[str, Any]] = []
    for nu in (0.99, 0.8, 0.5, 0.2):
        for level, (te, td, se, sd) in enumerate(levels):
            config = _configuration(
                name=f"jnw-nu{nu:.2f}-L{level}",
                tau_elements=te,
                tau_degree=td,
                s_elements=se,
                s_degree=sd,
                iterations=5 if quick else 8,
                quick=quick,
            )
            try:
                summary = run_case(
                    nu=nu,
                    level=level,
                    config=config,
                    output=output / f"nu-{nu:.2f}" / f"coordinate-level-{level}",
                )
            except Exception as error:
                failed = output / f"nu-{nu:.2f}" / f"coordinate-level-{level}"
                failed.mkdir(parents=True, exist_ok=False)
                summary = {
                    "case_id": f"exp06-jnw-nu{nu:.2f}-L{level}",
                    "terminal_status": "failed",
                    "failure_classification": "ESE solver or discretization",
                    "error": repr(error),
                }
                (failed / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n",
                    encoding="utf-8",
                )
            summaries.append(summary)
    aggregate = {
        "schema": "nee-official-experiment-06-aggregate-v1",
        "experiment": 6,
        "runs": summaries,
        "provenance": provenance(),
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
