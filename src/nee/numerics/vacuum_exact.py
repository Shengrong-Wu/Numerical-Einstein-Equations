"""Exact-vacuum characteristic benchmarks for the numerical Picard map.

The production Q1 driver uses ``tau=-log(-u)`` and fixed Minkowski data on
``v=0``.  This independent harness instead uses a direct, uniformly sampled
``u`` coordinate and supplies both characteristic faces explicitly.  It is
therefore suitable for exact solutions on intervals that cross ``u=0``.

The first certification family consists of three non-isometric exterior
Schwarzschild metrics.  In the solver convention

    g_4 = -2 Omega^2 (du dv + dv du) + g_AB dtheta^A dtheta^B,

write ``r_star=v-u+c`` and invert

    r_star = r + 2 M log(r/(2M)-1).

Then ``Omega^2=1-2M/r`` and the complete first-order state is analytic.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent

from .vacuum_iteration import (  # noqa: E402
    FirstOrderState,
    picard_step,
    sphere_broadcast,
    tangent_inverse,
    update_norm,
)
from .sphere import PointSphereGrid  # noqa: E402


Array = np.ndarray
DEFAULT_MASSES = (0.05, 0.10, 0.20)


def tortoise_radius(radius: Array | float, mass: float) -> Array:
    value = np.asarray(radius, dtype=float)
    return value + 2.0 * mass * np.log(value / (2.0 * mass) - 1.0)


def invert_tortoise(
    target: Array,
    mass: float,
    *,
    reference_radius: float,
    reference_coordinate: float,
) -> Array:
    """Invert the Schwarzschild tortoise map by safeguarded Newton steps."""

    if mass <= 0.0 or reference_radius <= 2.0 * mass:
        raise ValueError("require M>0 and reference_radius>2M")
    desired = (
        np.asarray(target, dtype=float)
        - float(reference_coordinate)
        + float(tortoise_radius(reference_radius, mass))
    )
    horizon = 2.0 * mass
    radius = np.maximum(
        reference_radius + np.asarray(target) - reference_coordinate,
        horizon * (1.0 + 1.0e-6),
    )
    for _ in range(30):
        residual = tortoise_radius(radius, mass) - desired
        lapse_sq = 1.0 - horizon / radius
        correction = residual * lapse_sq
        candidate = radius - correction
        candidate = np.maximum(candidate, horizon * (1.0 + 1.0e-12))
        radius = candidate
        if float(np.max(np.abs(residual))) < 2.0e-14:
            break
    final = tortoise_radius(radius, mass) - desired
    if float(np.max(np.abs(final))) > 2.0e-12:
        raise RuntimeError(
            "Schwarzschild tortoise inversion failed: "
            f"max residual={float(np.max(np.abs(final))):.6g}"
        )
    return radius


def schwarzschild_state(
    grid: PointSphereGrid,
    u: Array,
    v: Array,
    mass: float,
    *,
    reference_radius: float = 3.0,
) -> FirstOrderState:
    """Return the analytic Schwarzschild state on the full tensor grid."""

    uu, vv = np.meshgrid(np.asarray(u), np.asarray(v), indexing="ij")
    optical = vv - uu
    radius = invert_tortoise(
        optical,
        mass,
        reference_radius=reference_radius,
        reference_coordinate=1.0,
    )
    lapse_sq = 1.0 - 2.0 * mass / radius
    if float(np.min(lapse_sq)) <= 0.0:
        raise ValueError("the requested domain intersects the horizon")

    scalar = np.broadcast_to(
        np.ones((grid.count, 1, 1)), (grid.count, len(u), len(v))
    )
    projector = sphere_broadcast(grid, 2)
    g = radius[None, :, :, None, None] ** 2 * projector
    Omega = scalar * np.sqrt(lapse_sq)[None]
    Omega_trchi = scalar * (2.0 * lapse_sq / radius)[None]
    Omega_chib = (
        -(lapse_sq / radius)[None, :, :, None, None] * g
    )
    Omega_omega = scalar * (-mass / (2.0 * radius**2))[None]
    Omega_omegab = scalar * (mass / (2.0 * radius**2))[None]
    return FirstOrderState(
        g=g.copy(),
        Omega=Omega.copy(),
        zeta=np.zeros((*Omega.shape, 3)),
        b=np.zeros((*Omega.shape, 3)),
        Omega_trchi=Omega_trchi.copy(),
        Omega_chih=np.zeros_like(g),
        Omega_chib=Omega_chib.copy(),
        Omega_omega=Omega_omega.copy(),
        Omega_omegab=Omega_omegab.copy(),
    )


def characteristic_data(
    grid: PointSphereGrid, exact: FirstOrderState
) -> tuple[dict[str, Array], dict[str, Array]]:
    """Extract compatible outgoing and incoming characteristic faces."""

    outgoing = {
        "g": exact.g[:, 0].copy(),
        "inverse_g": tangent_inverse(grid, exact.g[:, 0]),
        "Omega_trchi": exact.Omega_trchi[:, 0].copy(),
        "Omega_chih": exact.Omega_chih[:, 0].copy(),
        "Omega": exact.Omega[:, 0].copy(),
        "Omega_omega": exact.Omega_omega[:, 0].copy(),
        "Omega_omegab": exact.Omega_omegab[:, 0].copy(),
        "Omega_chib": exact.Omega_chib[:, 0].copy(),
        "zeta": exact.zeta[:, 0].copy(),
        "b": exact.b[:, 0].copy(),
    }
    incoming = {
        "g": exact.g[:, :, 0].copy(),
        "Omega_trchi": exact.Omega_trchi[:, :, 0].copy(),
        "Omega_omegab": exact.Omega_omegab[:, :, 0].copy(),
        "Omega_chib": exact.Omega_chib[:, :, 0].copy(),
        "zeta": exact.zeta[:, :, 0].copy(),
        "b": exact.b[:, :, 0].copy(),
    }
    return outgoing, incoming


def face_compatible_seed(
    exact: FirstOrderState, u: Array
) -> FirstOrderState:
    """Construct a non-exact interior seed matching both exact null faces."""

    coordinate = (float(u[-1]) - np.asarray(u)) / float(u[-1] - u[0])
    weight = coordinate**3 * (10.0 + coordinate * (-15.0 + 6.0 * coordinate))
    values: dict[str, Array] = {}
    for description in fields(FirstOrderState):
        name = description.name
        target = np.asarray(getattr(exact, name))
        extra = target.ndim - 3
        blend = weight.reshape((1, len(u), 1, *(1,) * extra))
        incoming_extension = np.broadcast_to(
            target[:, :, :1], target.shape
        )
        outgoing_increment = target[:, :1] - target[:, :1, :1]
        seed = incoming_extension + blend * outgoing_increment
        seed[:, 0] = target[:, 0]
        seed[:, :, 0] = target[:, :, 0]
        values[name] = seed.copy()
    return FirstOrderState(**values)


def relative_field_error(
    numerical: FirstOrderState, exact: FirstOrderState, name: str
) -> dict[str, float]:
    value = np.asarray(getattr(numerical, name))
    target = np.asarray(getattr(exact, name))
    difference = value - target
    pointwise_axes = (0, *range(3, value.ndim))
    numerator = np.sqrt(np.mean(difference**2, axis=pointwise_axes))
    denominator = np.sqrt(np.mean(target**2, axis=pointwise_axes))
    relative = numerator / np.maximum(denominator, 1.0e-14)
    absolute = np.sqrt(np.mean(difference**2, axis=pointwise_axes))
    return {
        "relative_rms": float(np.sqrt(np.mean(relative**2))),
        "relative_max": float(np.max(relative)),
        "absolute_rms": float(np.sqrt(np.mean(absolute**2))),
        "absolute_max": float(np.max(absolute)),
    }


def kinematic_exactness(
    grid: PointSphereGrid,
    exact: FirstOrderState,
    u: Array,
    v: Array,
) -> dict[str, float]:
    """Check analytic state identities using independent finite differences."""

    edge_order = 2 if min(len(u), len(v)) >= 3 else 1
    metric_v = np.gradient(exact.g, v, axis=2, edge_order=edge_order)
    metric_u = np.gradient(exact.g, u, axis=1, edge_order=edge_order)
    outgoing_rhs = (
        exact.Omega_trchi[..., None, None] * exact.g
        + 2.0 * exact.Omega_chih
    )
    incoming_rhs = 2.0 * exact.Omega_chib
    scale = np.maximum(np.abs(exact.g), 1.0)
    return {
        "outgoing_metric_fd_max": float(
            np.max(np.abs(metric_v - outgoing_rhs) / scale)
        ),
        "incoming_metric_fd_max": float(
            np.max(np.abs(metric_u - incoming_rhs) / scale)
        ),
        "minimum_lapse": float(np.min(exact.Omega)),
        "minimum_tangent_metric_eigenvalue": float(
            np.min(
                np.linalg.eigvalsh(
                    np.einsum(
                        "nia,n...ij,njb->n...ab",
                        grid.frames,
                        exact.g,
                        grid.frames,
                    )
                )
            )
        ),
    }


def run_case(
    *,
    mass: float,
    points: int,
    u_min: float,
    u_max: float,
    v_min: float,
    v_max: float,
    u_count: int,
    v_count: int,
    iterations: int,
    metric_substeps: int,
    output: Path,
) -> dict[str, object]:
    grid = PointSphereGrid.create(
        points,
        neighbor_count=min(28, points - 1),
        degree=3,
        spectral_degree=min(7, int(math.sqrt(points)) - 2),
    )
    u = np.linspace(u_min, u_max, u_count)
    v = np.linspace(v_min, v_max, v_count)
    exact = schwarzschild_state(grid, u, v, mass)
    outgoing, incoming = characteristic_data(grid, exact)
    state = face_compatible_seed(exact, u)
    records: list[dict[str, object]] = []
    contexts: list[dict[str, object]] = []

    for sweep in range(1, iterations + 1):
        started = time.perf_counter()
        new_state, context = picard_step(
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
            "picard_update": update_norm(new_state, state),
            "metric_error": relative_field_error(new_state, exact, "g"),
            "lapse_error": relative_field_error(new_state, exact, "Omega"),
            "expansion_error": relative_field_error(new_state, exact, "Omega_trchi"),
            "incoming_form_error": relative_field_error(
                new_state, exact, "Omega_chib"
            ),
        }
        print(json.dumps({"mass": mass, **record}), flush=True)
        records.append(record)
        contexts.append(context)
        state = new_state

    case_output = output / f"schwarzschild-M{mass:.3f}"
    case_output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        case_output / "state-and-exact.npz",
        u=u,
        v=v,
        **{
            name: np.asarray(getattr(state, name))
            for name in (item.name for item in fields(FirstOrderState))
        },
        **{
            f"exact_{name}": np.asarray(getattr(exact, name))
            for name in (item.name for item in fields(FirstOrderState))
        },
    )
    np.savez_compressed(
        case_output / "characteristic-data.npz",
        u=u,
        v=v,
        **{f"outgoing_{name}": value for name, value in outgoing.items()},
        **{f"incoming_{name}": value for name, value in incoming.items()},
    )
    summary = {
        "g": "Schwarzschild",
        "mass": mass,
        "domain": {"u": [u_min, u_max], "v": [v_min, v_max]},
        "resolution": {
            "sphere_points": points,
            "u_count": u_count,
            "v_count": v_count,
            "metric_substeps": metric_substeps,
        },
        "iterations": iterations,
        "kinematic_check": kinematic_exactness(grid, exact, u, v),
        "records": records,
    }
    (case_output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masses", nargs="+", type=float, default=DEFAULT_MASSES)
    parser.add_argument("--points", type=int, default=50)
    parser.add_argument("--u-count", type=int, default=17)
    parser.add_argument("--v-count", type=int, default=17)
    parser.add_argument("--u-min", type=float, default=-1.0)
    parser.add_argument("--u-max", type=float, default=0.5)
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--g-substeps", type=int, default=2)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE.parent / "results" / "exact-vacuum-benchmarks" / "low",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = [
        run_case(
            mass=mass,
            points=args.points,
            u_min=args.u_min,
            u_max=args.u_max,
            v_min=args.v_min,
            v_max=args.v_max,
            u_count=args.u_count,
            v_count=args.v_count,
            iterations=args.iterations,
            metric_substeps=args.metric_substeps,
            output=args.output,
        )
        for mass in args.masses
    ]
    aggregate = {
        "benchmark": "three non-isometric Schwarzschild exterior metrics",
        "summaries": summaries,
    }
    (args.output / "summary.json").write_text(
        json.dumps(aggregate, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
