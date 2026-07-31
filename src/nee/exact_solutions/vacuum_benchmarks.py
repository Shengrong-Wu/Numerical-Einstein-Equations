"""Exact-vacuum characteristic benchmarks.

All exact states and characteristic faces are regenerated from the documented
Schwarzschild and Kerr formulae.

The numerical kernel derives ``q = Omega**(-1) tr(chi)`` and the trace-free
part of ``Omega chi``. Summaries additionally report the full
weighted forms

    Xout = Omega chi = shear + 0.5 * Omega**2 * q * metric,
    Xin  = Omega chib,

to make the stored full-form state explicit.
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
from typing import Any, Callable

import numpy as np
from scipy.special import lambertw



from nee.numerics.vacuum_exact import (  # noqa: E402
    characteristic_data,
    face_compatible_seed,
    relative_field_error,
)
from nee.numerics.vacuum_iteration import (  # noqa: E402
    FirstOrderState,
    picard_step,
    sphere_broadcast,
    update_norm,
)
from nee.numerics.sphere import PointSphereGrid  # noqa: E402
from nee.numerics.kerr import (  # noqa: E402
    kerr_exact_state,
)


Array = np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def source_provenance() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parents[1] / "numerics" / "vacuum_iteration.py",
        Path(__file__).resolve().parents[1] / "numerics" / "vacuum_exact.py",
        Path(__file__).resolve().parents[1] / "numerics" / "kerr.py",
        Path(__file__).resolve().parents[1] / "numerics" / "sphere.py",
    ]
    return {str(path): _sha256(path) for path in paths}


def _grid(points: int, retained_degree: int) -> PointSphereGrid:
    return PointSphereGrid.create(
        points,
        neighbor_count=min(max(20, 3 * retained_degree), points - 1),
        degree=min(4, retained_degree),
        spectral_degree=retained_degree,
    )


def _static_tortoise(radius: Array, mass: float) -> Array:
    return radius - 6.0 * mass + 2.0 * mass * np.log(
        (radius - 2.0 * mass) / (4.0 * mass)
    )


def _invert_static_tortoise(target: Array, mass: float) -> Array:
    """Invert the normalized Experiment-1 tortoise map."""

    target = np.asarray(target, dtype=float)
    horizon = 2.0 * mass
    radius = np.maximum(6.0 * mass + target, horizon * (1.0 + 1.0e-8))
    for _ in range(80):
        residual = _static_tortoise(radius, mass) - target
        lapse_sq = 1.0 - horizon / radius
        radius = np.maximum(
            radius - residual * lapse_sq,
            horizon * (1.0 + 4.0 * np.finfo(float).eps),
        )
        if float(np.max(np.abs(residual))) < 8.0e-14:
            break
    final = _static_tortoise(radius, mass) - target
    if float(np.max(np.abs(final))) > 5.0e-11:
        raise RuntimeError(
            f"static tortoise inversion failed: {np.max(np.abs(final)):.3e}"
        )
    return radius


def _invert_near_horizon_tortoise(
    target: Array, mass: float, epsilon: float
) -> Array:
    """Invert the normalized near-horizon map with its Lambert-W form.

    With ``y=(r-2M)/(2M)`` the defining equation becomes

        y + log(y) = eps + log(eps) + (target-1/2)/(2M),

    hence ``y=W(eps*exp(eps+(target-1/2)/(2M)))``.  This avoids the
    precision-sensitive Newton stopping test precisely in the small-epsilon
    regime that Experiment 2 is designed to probe.
    """

    target = np.asarray(target, dtype=float)
    argument = epsilon * np.exp(
        epsilon + (target - 0.5) / (2.0 * mass)
    )
    y = np.real(lambertw(argument, k=0))
    if float(np.max(np.abs(np.imag(lambertw(argument, k=0))))) > 1.0e-14:
        raise RuntimeError("near-horizon Lambert-W inversion became complex")
    return 2.0 * mass * (1.0 + y)


def _spherical_state_from_radius(
    grid: PointSphereGrid,
    radius: Array,
    omega_sq: Array,
    radius_u: Array,
    radius_v: Array,
    weighted_omega: Array,
    weighted_omegab: Array,
) -> FirstOrderState:
    scalar = np.broadcast_to(
        np.ones((grid.count, 1, 1)),
        (grid.count, radius.shape[0], radius.shape[1]),
    )
    projector = sphere_broadcast(grid, 2)
    metric = radius[None, ..., None, None] ** 2 * projector
    omega = scalar * np.sqrt(omega_sq)[None]
    weighted_expansion = scalar * (2.0 * radius_v / radius)[None]
    q = weighted_expansion / omega**2
    weighted_chib = (
        radius[None, ..., None, None]
        * radius_u[None, ..., None, None]
        * projector
    )
    return FirstOrderState(
        metric=metric.copy(),
        omega=omega.copy(),
        zeta_up=np.zeros((*omega.shape, 3)),
        shift=np.zeros((*omega.shape, 3)),
        q=q.copy(),
        shear=np.zeros_like(metric),
        weighted_chib=weighted_chib.copy(),
        weighted_omega=scalar * weighted_omega[None],
        weighted_omegab=scalar * weighted_omegab[None],
    )


def regular_schwarzschild_state(
    grid: PointSphereGrid, u: Array, v: Array, mass: float
) -> tuple[FirstOrderState, dict[str, float]]:
    uu, vv = np.meshgrid(u, v, indexing="ij")
    target = vv - uu - 0.5
    radius = _invert_static_tortoise(target, mass)
    lapse_sq = 1.0 - 2.0 * mass / radius
    state = _spherical_state_from_radius(
        grid,
        radius,
        lapse_sq,
        -lapse_sq,
        lapse_sq,
        -mass / (2.0 * radius**2),
        mass / (2.0 * radius**2),
    )
    return state, {
        "minimum_radius": float(np.min(radius)),
        "maximum_radius": float(np.max(radius)),
        "minimum_omega_squared": float(np.min(lapse_sq)),
        "maximum_kretschmann": float(np.max(48.0 * mass**2 / radius**6)),
    }


def kerr_zero_spin_state(
    grid: PointSphereGrid, u: Array, v: Array, mass: float
) -> tuple[FirstOrderState, dict[str, float]]:
    """Return the a=0 limit of the Kerr optical map.

    The numerical Kerr generator intentionally rejects ``a=0``.  In the plan's
    normalization the zero-spin optical coordinate is ``s=u-v``, with
    ``r(0)=4M`` and ``dr/ds=1-2M/r``.  This is an independently generated
    Schwarzschild state in the Kerr orientation and therefore supplies the
    required cross-implementation control.
    """

    uu, vv = np.meshgrid(u, v, indexing="ij")
    target = uu - vv

    def mapping(radius: Array) -> Array:
        return radius - 4.0 * mass + 2.0 * mass * np.log(
            (radius - 2.0 * mass) / (2.0 * mass)
        )

    horizon = 2.0 * mass
    radius = np.maximum(4.0 * mass + target, horizon * (1.0 + 1.0e-8))
    for _ in range(80):
        residual = mapping(radius) - target
        lapse_sq = 1.0 - horizon / radius
        radius = np.maximum(
            radius - residual * lapse_sq,
            horizon * (1.0 + 4.0 * np.finfo(float).eps),
        )
        if float(np.max(np.abs(residual))) < 8.0e-14:
            break
    lapse_sq = 1.0 - horizon / radius
    state = _spherical_state_from_radius(
        grid,
        radius,
        lapse_sq,
        lapse_sq,
        -lapse_sq,
        mass / (2.0 * radius**2),
        -mass / (2.0 * radius**2),
    )
    return state, {
        "minimum_radius": float(np.min(radius)),
        "maximum_radius": float(np.max(radius)),
        "minimum_omega_squared": float(np.min(lapse_sq)),
        "maximum_kretschmann": float(np.max(48.0 * mass**2 / radius**6)),
        "zero_spin_limit": True,
    }


def near_horizon_static_state(
    grid: PointSphereGrid,
    u: Array,
    v: Array,
    mass: float,
    epsilon: float,
) -> tuple[FirstOrderState, dict[str, float]]:
    uu, vv = np.meshgrid(u, v, indexing="ij")
    target = vv - uu
    radius = _invert_near_horizon_tortoise(target, mass, epsilon)
    lapse_sq = 1.0 - 2.0 * mass / radius
    state = _spherical_state_from_radius(
        grid,
        radius,
        lapse_sq,
        -lapse_sq,
        lapse_sq,
        -mass / (2.0 * radius**2),
        mass / (2.0 * radius**2),
    )
    return state, {
        "epsilon": epsilon,
        "minimum_radius": float(np.min(radius)),
        "minimum_omega_squared": float(np.min(lapse_sq)),
        "declared_minimum_omega_squared": epsilon / (1.0 + epsilon),
        "maximum_kretschmann": float(np.max(48.0 * mass**2 / radius**6)),
    }


def _kruskal_radius(product: Array, mass: float) -> Array:
    argument = -np.asarray(product, dtype=float) / math.e
    radius = 2.0 * mass * (1.0 + np.real(lambertw(argument, k=0)))
    if np.max(np.abs(np.imag(lambertw(argument, k=0)))) > 1.0e-12:
        raise ValueError("Kruskal product left the real principal branch")
    if float(np.min(radius)) <= 0.0:
        raise ValueError("Kruskal grid intersects r=0")
    return radius


def kruskal_state(
    grid: PointSphereGrid,
    u: Array,
    v: Array,
    mass: float,
    *,
    u_offset: float,
    v_offset: float,
) -> tuple[FirstOrderState, dict[str, float]]:
    uu, vv = np.meshgrid(u, v, indexing="ij")
    U = uu + u_offset
    V = vv + v_offset
    radius = _kruskal_radius(U * V, mass)
    y = radius / (2.0 * mass)
    coefficient = 4.0 * mass**2 / radius * np.exp(-y)
    radius_u = -coefficient * V
    radius_v = -coefficient * U
    omega_sq = 8.0 * mass**3 / radius * np.exp(-y)
    weighted_omega = -coefficient * U * (radius + 2.0 * mass) / (
        8.0 * mass * radius
    )
    weighted_omegab = -coefficient * V * (radius + 2.0 * mass) / (
        8.0 * mass * radius
    )
    state = _spherical_state_from_radius(
        grid,
        radius,
        omega_sq,
        radius_u,
        radius_v,
        weighted_omega,
        weighted_omegab,
    )
    return state, {
        "minimum_radius": float(np.min(radius)),
        "maximum_radius": float(np.max(radius)),
        "minimum_omega_squared": float(np.min(omega_sq)),
        "maximum_kretschmann": float(np.max(48.0 * mass**2 / radius**6)),
        "horizon_grid_distance": float(np.min(np.abs(U))),
        "crosses_horizon": bool(np.min(U) < 0.0 < np.max(U)),
    }


def _full_weighted_outgoing(state: FirstOrderState) -> Array:
    return state.shear + 0.5 * (
        state.omega**2 * state.q
    )[..., None, None] * state.metric


def _state_arrays(state: FirstOrderState) -> dict[str, Array]:
    values = {
        item.name: np.asarray(getattr(state, item.name))
        for item in fields(FirstOrderState)
    }
    values["weighted_chi"] = _full_weighted_outgoing(state)
    return values


def finite_difference_closures(
    state: FirstOrderState, u: Array, v: Array
) -> dict[str, float]:
    edge_order = 2 if min(len(u), len(v)) >= 3 else 1
    metric_v = np.gradient(state.metric, v, axis=2, edge_order=edge_order)
    metric_u = np.gradient(state.metric, u, axis=1, edge_order=edge_order)
    outgoing = 2.0 * _full_weighted_outgoing(state)
    incoming = 2.0 * state.weighted_chib
    scale = np.maximum(np.abs(state.metric), 1.0)
    return {
        "metric_C4_max_relative": float(
            np.max(np.abs(metric_v - outgoing) / scale)
        ),
        "metric_C3_max_relative": float(
            np.max(np.abs(metric_u - incoming) / scale)
        ),
    }


def run_picard_case(
    *,
    case_id: str,
    exact_builder: Callable[
        [PointSphereGrid, Array, Array], tuple[FirstOrderState, dict[str, float]]
    ],
    u: Array,
    v: Array,
    points: int,
    retained_degree: int,
    iterations: int,
    metric_substeps: int,
    output: Path,
) -> dict[str, Any]:
    existing_summary = output / "summary.json"
    if existing_summary.exists():
        return json.loads(existing_summary.read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(
            f"incomplete immutable case directory already exists: {output}"
        )
    grid = _grid(points, retained_degree)
    exact, exact_diagnostics = exact_builder(grid, u, v)
    outgoing, incoming = characteristic_data(grid, exact)
    state = face_compatible_seed(exact, u)
    records: list[dict[str, Any]] = []
    contexts: list[dict[str, Any]] = []
    for sweep in range(1, iterations + 1):
        started = time.perf_counter()
        next_state, context = picard_step(
            grid,
            state,
            outgoing,
            u,
            v,
            metric_substeps=metric_substeps,
            metric_parameterization="direct",
            metric_integrator="rk4",
            u_integrator="rk4",
            incoming=incoming,
        )
        record = {
            "sweep": sweep,
            "seconds": time.perf_counter() - started,
            "picard_update": update_norm(next_state, state),
            "metric_error": relative_field_error(
                next_state, exact, "metric"
            ),
            "lapse_error": relative_field_error(
                next_state, exact, "omega"
            ),
            "outgoing_form_error": {
                "absolute_max": float(
                    np.max(
                        np.abs(
                            _full_weighted_outgoing(next_state)
                            - _full_weighted_outgoing(exact)
                        )
                    )
                )
            },
            "incoming_form_error": relative_field_error(
                next_state, exact, "weighted_chib"
            ),
        }
        print(json.dumps({"case": case_id, **record}), flush=True)
        records.append(record)
        contexts.append(context)
        state = next_state

    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "state-and-exact.npz",
        u=u,
        v=v,
        **_state_arrays(state),
        **{
            f"exact_{name}": value
            for name, value in _state_arrays(exact).items()
        },
    )
    np.savez_compressed(
        output / "characteristic-data.npz",
        u=u,
        v=v,
        **{f"outgoing_{name}": value for name, value in outgoing.items()},
        **{f"incoming_{name}": value for name, value in incoming.items()},
    )
    summary: dict[str, Any] = {
        "schema": "nee-official-exact-vacuum-run-v1",
        "case_id": case_id,
        "domain": {
            "u": [float(u[0]), float(u[-1])],
            "v": [float(v[0]), float(v[-1])],
        },
        "resolution": {
            "sphere_points": points,
            "retained_degree": retained_degree,
            "u_count": len(u),
            "v_count": len(v),
            "metric_substeps": metric_substeps,
        },
        "iterations": iterations,
        "exact_diagnostics": exact_diagnostics,
        "exact_state_fd_closures": finite_difference_closures(exact, u, v),
        "records": records,
        "terminal_status": (
            "finite"
            if all(
                np.all(np.isfinite(value))
                for value in _state_arrays(state).values()
            )
            else "nonfinite"
        ),
        "provenance": {
            "runtime": {
                "python": platform.python_version(),
                "numpy": np.__version__,
            },
            "sources": source_provenance(),
            "boundary_data_regenerated": True,
            "seed": "non-exact face-compatible quintic blend",
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _coordinate_levels(quick: bool) -> list[tuple[int, int]]:
    return [(9, 9), (17, 17), (33, 33)] if quick else [
        (17, 17),
        (33, 33),
        (65, 65),
    ]


def run_experiment_1(root: Path, quick: bool) -> dict[str, Any]:
    mass = 1.0
    levels = _coordinate_levels(quick)
    summaries: list[dict[str, Any]] = []
    domains = {
        "short": (-1.0, -0.5, 0.0, 0.5),
        "long": (-1.0, 0.0, 0.0, 1.0),
    }
    for domain_name, bounds in domains.items():
        u0, u1, v0, v1 = bounds
        for level, (nu, nv) in enumerate(levels):
            summaries.append(
                run_picard_case(
                    case_id=f"exp01-schwarzschild-{domain_name}-L{level}",
                    exact_builder=lambda grid, u, v, m=mass: (
                        regular_schwarzschild_state(grid, u, v, m)
                    ),
                    u=np.linspace(u0, u1, nu),
                    v=np.linspace(v0, v1, nv),
                    points=50,
                    retained_degree=5,
                    iterations=5 if quick else 8,
                    metric_substeps=2,
                    output=(
                        root
                        / f"schwarzschild-{domain_name}"
                        / f"coordinate-level-{level}"
                    ),
                )
            )

    # Kerr is substantially more expensive.  The complete parameter/domain
    # matrix is run once at the middle coordinate level; the a=0.3 short
    # baseline receives all three coordinate levels.
    kerr_levels = list(enumerate(levels))
    for rotation in (0.0, 0.3, 0.7, 0.9):
        for domain_name, bounds in domains.items():
            selected = (
                kerr_levels
                if rotation == 0.3 and domain_name == "short"
                else [(1, levels[1])]
            )
            for level, (nu, nv) in selected:
                u0, u1, v0, v1 = bounds
                case_id = (
                    f"exp01-kerr-a{rotation:.1f}-{domain_name}-L{level}"
                )
                case_output = (
                    root
                    / f"kerr-a{rotation:.1f}-{domain_name}"
                    / f"coordinate-level-{level}"
                )
                try:
                    if rotation == 0.0:
                        builder = lambda grid, u, v: kerr_zero_spin_state(
                            grid, u, v, 1.0
                        )
                    else:
                        builder = lambda grid, u, v, a=rotation: (
                            kerr_exact_state(
                                grid,
                                u,
                                v,
                                mass=1.0,
                                rotation=a,
                                reference_radius=4.0,
                            )
                        )
                    summary = run_picard_case(
                        case_id=(
                            case_id
                        ),
                        exact_builder=builder,
                        u=np.linspace(u0, u1, nu),
                        v=np.linspace(v0, v1, nv),
                        points=50 if quick else 86,
                        retained_degree=5 if quick else 7,
                        iterations=4 if quick else 6,
                        metric_substeps=2,
                        output=case_output,
                    )
                except Exception as error:
                    case_output.mkdir(parents=True, exist_ok=False)
                    summary = {
                        "case_id": case_id,
                        "terminal_status": "failed",
                        "failure_classification": (
                            "coordinate conditioning or optical-map failure"
                        ),
                        "error": repr(error),
                    }
                    (case_output / "summary.json").write_text(
                        json.dumps(summary, indent=2) + "\n",
                        encoding="utf-8",
                    )
                summaries.append(summary)
    return {"experiment": 1, "runs": summaries}


def run_experiment_2(root: Path, quick: bool) -> dict[str, Any]:
    mass = 1.0
    levels = _coordinate_levels(quick)
    summaries: list[dict[str, Any]] = []
    for epsilon in (1.0e-1, 1.0e-2, 1.0e-4, 1.0e-6):
        for level, (nu, nv) in enumerate(levels):
            try:
                summary = run_picard_case(
                    case_id=f"exp02-static-eps{epsilon:.0e}-L{level}",
                    exact_builder=lambda grid, u, v, e=epsilon: (
                        near_horizon_static_state(grid, u, v, mass, e)
                    ),
                    u=np.linspace(-1.0, -0.5, nu),
                    v=np.linspace(0.0, 0.1, nv),
                    points=50,
                    retained_degree=5,
                    iterations=5 if quick else 8,
                    metric_substeps=2,
                    output=(
                        root
                        / f"static-epsilon-{epsilon:.0e}"
                        / f"coordinate-level-{level}"
                    ),
                )
            except Exception as error:  # preserve failed stress cases
                failed = (
                    root
                    / f"static-epsilon-{epsilon:.0e}"
                    / f"coordinate-level-{level}"
                )
                failed.mkdir(parents=True, exist_ok=False)
                summary = {
                    "case_id": f"exp02-static-eps{epsilon:.0e}-L{level}",
                    "terminal_status": "failed",
                    "failure_classification": "coordinate conditioning",
                    "error": repr(error),
                }
                (failed / "summary.json").write_text(
                    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
                )
            summaries.append(summary)

    for level, (nu, nv) in enumerate(levels):
        summaries.append(
            run_picard_case(
                case_id=f"exp02-kruskal-crossing-L{level}",
                exact_builder=lambda grid, u, v: kruskal_state(
                    grid,
                    u,
                    v,
                    mass,
                    u_offset=0.75,
                    v_offset=1.0,
                ),
                u=np.linspace(-1.0, -0.5, nu),
                v=np.linspace(0.0, 0.1, nv),
                points=50,
                retained_degree=5,
                iterations=5 if quick else 8,
                metric_substeps=2,
                output=root / "kruskal-crossing" / f"coordinate-level-{level}",
            )
        )
    return {"experiment": 2, "runs": summaries}


def singularity_map_audit(
    *,
    epsilon: float,
    mass: float,
    nu: int,
    nxi: int,
    output: Path,
) -> dict[str, Any]:
    """Audit the Experiment-3 curved-coordinate Jacobian identities.

    The inherited rectangular Picard driver cannot evolve a u-dependent
    physical v boundary.  This audit therefore regenerates the exact Kruskal
    state on the mapped grid and independently verifies both physical metric
    closures using the declared Jacobian.  The limitation is explicit in the
    summary and is not promoted to a completed Picard certification.
    """

    u = np.linspace(-1.0, -0.5, nu)
    xi = np.linspace(0.0, 1.0, nxi)
    x_s = (1.0 - epsilon) * math.exp(epsilon)
    v0 = x_s / (u + 2.0) - 0.2
    v0_prime = -x_s / (u + 2.0) ** 2
    vv = xi[None, :] * v0[:, None]
    uu = np.broadcast_to(u[:, None], vv.shape)
    U = uu + 2.0
    V = vv + 0.2
    radius = _kruskal_radius(U * V, mass)
    y = radius / (2.0 * mass)
    coefficient = 4.0 * mass**2 / radius * np.exp(-y)
    radius_u = -coefficient * V
    radius_v = -coefficient * U
    omega_sq = 8.0 * mass**3 / radius * np.exp(-y)

    metric_scalar = radius**2
    xout_scalar = radius * radius_v
    xin_scalar = radius * radius_u
    edge_order = 2
    dxi_metric = np.gradient(
        metric_scalar, xi, axis=1, edge_order=edge_order
    )
    du_at_xi = np.gradient(
        metric_scalar, u, axis=0, edge_order=edge_order
    )
    dv_metric = dxi_metric / v0[:, None]
    du_at_v = du_at_xi - (
        xi[None, :] * v0_prime[:, None] / v0[:, None] * dxi_metric
    )
    scale = np.maximum(metric_scalar, 1.0)
    c4 = np.abs(dv_metric - 2.0 * xout_scalar) / scale
    c3 = np.abs(du_at_v - 2.0 * xin_scalar) / scale
    boundary_radius_error = float(
        np.max(np.abs(radius[:, -1] - 2.0 * mass * epsilon))
    )
    summary = {
        "schema": "nee-official-singularity-map-audit-v1",
        "epsilon": epsilon,
        "resolution": {"u_count": nu, "xi_count": nxi},
        "minimum_radius": float(np.min(radius)),
        "maximum_kretschmann": float(
            np.max(48.0 * mass**2 / radius**6)
        ),
        "declared_maximum_kretschmann": (
            3.0 / (4.0 * mass**4 * epsilon**6)
        ),
        "future_boundary_radius_error": boundary_radius_error,
        "metric_C4_max_relative": float(np.max(c4)),
        "metric_C3_max_relative": float(np.max(c3)),
        "minimum_omega_squared": float(np.min(omega_sq)),
        "terminal_status": "map-audit-complete",
        "certification_status": "not-certified",
        "limitation": (
            "The inherited numerical backend Picard implementation assumes a "
            "rectangular physical (u,v) grid.  This run verifies the exact "
            "curved-map data and both Jacobian-corrected metric closures, "
            "but it does not claim an interior Picard solve."
        ),
        "provenance": {
            "sources": source_provenance(),
            "boundary_data_regenerated": True,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(
        output / "mapped-exact-state.npz",
        u=u,
        xi=xi,
        v=vv,
        v0=v0,
        radius=radius,
        omega_squared=omega_sq,
        metric_scalar=metric_scalar,
        weighted_chi_scalar=xout_scalar,
        weighted_chib_scalar=xin_scalar,
        C4_relative=c4,
        C3_relative=c3,
    )
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def run_experiment_3(root: Path, quick: bool) -> dict[str, Any]:
    levels = _coordinate_levels(quick)
    summaries: list[dict[str, Any]] = []
    for epsilon in (2.0**-power for power in range(1, 7)):
        for level, (nu, nxi) in enumerate(levels):
            summaries.append(
                singularity_map_audit(
                    epsilon=epsilon,
                    mass=1.0,
                    nu=nu,
                    nxi=nxi,
                    output=(
                        root
                        / f"epsilon-{epsilon:.8f}"
                        / f"coordinate-level-{level}"
                    ),
                )
            )
    return {"experiment": 3, "runs": summaries}


def _write_aggregate(root: Path, aggregate: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    aggregate["created_unix"] = time.time()
    aggregate["provenance"] = {
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "sources": source_provenance(),
    }
    (root / "aggregate-summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse completed immutable case directories in an existing run",
    )
    args = parser.parse_args()
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"immutable run directory already exists: {args.output}")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    if args.experiment == 1:
        aggregate = run_experiment_1(args.output, args.quick)
    elif args.experiment == 2:
        aggregate = run_experiment_2(args.output, args.quick)
    else:
        aggregate = run_experiment_3(args.output, args.quick)
    _write_aggregate(args.output, aggregate)


if __name__ == "__main__":
    main()
