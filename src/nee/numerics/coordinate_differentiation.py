"""Direct four-dimensional Ricci audit for axisymmetric double-null metrics."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


Array = np.ndarray


@dataclass
class AuditSummary:
    case: str
    n_u: int
    n_v: int
    n_theta: int
    du: float
    dv: float
    dtheta: float
    ricci_l2: float
    ricci_linf: float
    center_ricci: float


def differentiate(values: Array, coordinate: Array, axis: int) -> Array:
    return np.gradient(values, coordinate, axis=axis, edge_order=2)


def high_order_differentiate(
    values: Array, coordinate: Array, axis: int, stencil: int = 11
) -> Array:
    """Differentiate on a nonuniform grid with local polynomial weights."""

    count = len(coordinate)
    width = min(stencil, count)
    if width < 5:
        return differentiate(values, coordinate, axis)
    if width % 2 == 0:
        width -= 1
    matrix = np.zeros((count, count), dtype=float)
    half = width // 2
    for i in range(count):
        start = min(max(i - half, 0), count - width)
        indices = np.arange(start, start + width)
        offsets = coordinate[indices] - coordinate[i]
        scale = float(np.max(np.abs(offsets)))
        normalized = offsets / scale
        vandermonde = np.vstack([normalized**power for power in range(width)])
        target = np.zeros(width)
        target[1] = 1.0 / scale
        matrix[i, indices] = np.linalg.solve(vandermonde, target)
    moved = np.moveaxis(values, axis, 0)
    result = np.tensordot(matrix, moved, axes=(1, 0))
    return np.moveaxis(result, 0, axis)


def build_metric(u: Array, v: Array, theta: Array, perturbation: float = 0.0) -> Array:
    """Build Minkowski or a trace-free angular perturbation in double-null gauge."""

    uu, vv, tt = np.meshgrid(u, v, theta, indexing="ij")
    radius = vv - uu
    Omega_trchi = perturbation * (uu + 1.0) ** 2 * vv**2 * np.sin(tt) ** 2
    g = np.zeros(uu.shape + (4, 4), dtype=float)
    g[..., 0, 1] = -2.0
    g[..., 1, 0] = -2.0
    g[..., 2, 2] = radius**2 * np.exp(2.0 * Omega_trchi)
    g[..., 3, 3] = radius**2 * np.exp(-2.0 * Omega_trchi) * np.sin(tt) ** 2
    return g


def christoffel_symbols(
    g: Array, scalar_coordinates: list[Array], high_order: bool = False
) -> tuple[Array, Array]:
    inverse_g = np.linalg.inv(g)
    diff = high_order_differentiate if high_order else differentiate
    derivatives = []
    for mu in range(4):
        if mu < 3:
            derivatives.append(diff(g, scalar_coordinates[mu], axis=mu))
        else:
            derivatives.append(np.zeros_like(g))

    grid_shape = g.shape[:-2]
    gamma = np.zeros(grid_shape + (4, 4, 4), dtype=float)
    for a in range(4):
        for b in range(4):
            for c in range(4):
                covector = np.empty(grid_shape + (4,), dtype=float)
                for d in range(4):
                    covector[..., d] = (
                        derivatives[b][..., d, c]
                        + derivatives[c][..., d, b]
                        - derivatives[d][..., b, c]
                    )
                gamma[..., a, b, c] = 0.5 * np.einsum(
                    "...d,...d->...", inverse_g[..., a, :], covector
                )
    return gamma, inverse_g


def direct_ricci(
    g: Array, scalar_coordinates: list[Array], high_order: bool = False
) -> tuple[Array, Array]:
    gamma, inverse_g = christoffel_symbols(g, scalar_coordinates, high_order=high_order)
    diff = high_order_differentiate if high_order else differentiate
    grid_shape = g.shape[:-2]
    ricci = np.zeros(grid_shape + (4, 4), dtype=float)

    trace_connection = np.zeros(grid_shape + (4,), dtype=float)
    for d in range(4):
        for c in range(4):
            trace_connection[..., d] += gamma[..., c, c, d]

    for a in range(4):
        trace_a = np.zeros(grid_shape, dtype=float)
        for c in range(4):
            trace_a += gamma[..., c, a, c]
        for b in range(4):
            term = np.zeros(grid_shape, dtype=float)
            for c in range(4):
                if c < 3:
                    term += diff(gamma[..., c, a, b], scalar_coordinates[c], axis=c)
                term += trace_connection[..., c] * gamma[..., c, a, b]
                for d in range(4):
                    term -= gamma[..., c, b, d] * gamma[..., d, a, c]
            if b < 3:
                term -= diff(trace_a, scalar_coordinates[b], axis=b)
            ricci[..., a, b] = term
    return ricci, inverse_g


def positive_frame_norm(ricci: Array, g: Array) -> Array:
    """Positive norm of Ricci components in the b=0 adapted null frame.

    The coordinate components with a u or v index acquire one factor of
    Omega^{-1} for each null-frame slot.  The angular block is contracted with
    the inverse_g section g.  This is a diagnostic positive norm, not the
    indefinite Lorentzian contraction Ric_{ab} Ric^{ab}.
    """

    gamma = g[..., 2:4, 2:4]
    gamma_inverse = np.linalg.inv(gamma)
    omega_sq = -0.5 * g[..., 0, 1]
    if np.any(omega_sq <= 0.0):
        raise FloatingPointError("non-positive Omega^2 in frame norm")
    value = (
        ricci[..., 0, 0] ** 2
        + ricci[..., 1, 1] ** 2
        + 2.0 * ricci[..., 0, 1] ** 2
    ) / omega_sq**2
    for null_index in [0, 1]:
        covector = ricci[..., null_index, 2:4]
        value += np.einsum(
            "...a,...ab,...b->...", covector, gamma_inverse, covector
        ) / omega_sq
    sphere = ricci[..., 2:4, 2:4]
    value += np.einsum(
        "...ac,...bd,...ab,...cd->...",
        gamma_inverse,
        gamma_inverse,
        sphere,
        sphere,
    )
    return np.sqrt(np.maximum(value, 0.0))


def audit_case(
    n_u: int,
    n_v: int,
    n_theta: int,
    perturbation: float,
) -> tuple[AuditSummary, Array, Array, Array, Array]:
    # u=0 and the coordinate poles are excluded.  The endpoint sequence toward
    # u=0 is handled separately after the baseline convergence test.
    u = np.linspace(-1.0, -0.1, n_u)
    v = np.linspace(0.0, 0.5, n_v)
    # A single spherical coordinate chart is not regular at the poles.  Keep a
    # fixed equatorial chart for this finite-difference auditor.  A full-sphere
    # experiment must use overlapping regular patches or spin-weighted fields.
    theta = np.linspace(0.4, math.pi - 0.4, n_theta)
    g = build_metric(u, v, theta, perturbation)
    ricci, _ = direct_ricci(g, [u, v, theta, np.array([0.0])])
    rho = positive_frame_norm(ricci, g)
    uu, vv, tt = np.meshgrid(u, v, theta, indexing="ij")
    core = (
        (uu > -0.9)
        & (uu < -0.2)
        & (vv > 0.05)
        & (vv < 0.45)
        & (tt > 0.5)
        & (tt < math.pi - 0.5)
    )
    interior = rho[core]
    summary = AuditSummary(
        case="minkowski" if perturbation == 0.0 else "angular-perturbation",
        n_u=n_u,
        n_v=n_v,
        n_theta=n_theta,
        du=float(u[1] - u[0]),
        dv=float(v[1] - v[0]),
        dtheta=float(theta[1] - theta[0]),
        ricci_l2=float(np.sqrt(np.mean(interior**2))),
        ricci_linf=float(np.max(interior)),
        center_ricci=float(rho[n_u // 2, n_v // 2, n_theta // 2]),
    )
    return summary, u, v, theta, rho


def convergence_orders(summaries: list[AuditSummary]) -> list[float]:
    orders = []
    for coarse, fine in zip(summaries, summaries[1:]):
        h_coarse = max(coarse.du, coarse.dv, coarse.dtheta)
        h_fine = max(fine.du, fine.dv, fine.dtheta)
        orders.append(
            math.log(coarse.ricci_l2 / fine.ricci_l2)
            / math.log(h_coarse / h_fine)
        )
    return orders


def write_uv_csv(path: Path, u: Array, v: Array, rho: Array) -> None:
    angular_rms = np.sqrt(np.mean(rho**2, axis=2))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["u", "v", "angular_rms_ricci"])
        for i, uu in enumerate(u):
            for j, vv in enumerate(v):
                writer.writerow([float(uu), float(vv), float(angular_rms[i, j])])


def run_axisymmetric_audit_suite(results_dir: Path, log_path: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    resolutions = [(17, 17, 33), (25, 25, 49), (33, 33, 65), (41, 41, 81)]
    minkowski: list[AuditSummary] = []
    perturbed: list[AuditSummary] = []
    finest_perturbed = None

    for n_u, n_v, n_theta in resolutions:
        summary, *_ = audit_case(n_u, n_v, n_theta, perturbation=0.0)
        minkowski.append(summary)
        result = audit_case(n_u, n_v, n_theta, perturbation=0.08)
        perturbed.append(result[0])
        finest_perturbed = result

    if finest_perturbed is not None:
        _, u, v, theta, rho = finest_perturbed
        write_uv_csv(results_dir / "axisymmetric-perturbed-rho-uv.csv", u, v, rho)

    report = {
        "experiment": "direct four-dimensional axisymmetric Ricci audit",
        "minkowski": {
            "summaries": [asdict(item) for item in minkowski],
            "observed_orders": convergence_orders(minkowski),
            "scope": "fixed regular chart and fixed interior subdomain",
        },
        "angular_perturbation": {
            "amplitude": 0.08,
            "summaries": [asdict(item) for item in perturbed],
            "interpretation": "non-vacuum negative control; residual should not approach zero",
        },
    }
    (results_dir / "axisymmetric-ricci-summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    log_lines = [
        "",
        "## 2026-07-15 — Full four-coordinate Ricci auditor",
        "",
        "The generic coordinate formula `g -> Gamma -> Ric` was evaluated on",
        "axisymmetric grids.  Exact spherical Minkowski data used",
        "`gamma=(v-u)^2 diag(1,sin(theta)^2)`, `Omega=1`, and `b=0`.",
        "",
        "| case | N_u | N_v | N_theta | Ricci L2 | Ricci Linf | center Ricci |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in minkowski + perturbed:
        log_lines.append(
            f"| {item.case} | {item.n_u} | {item.n_v} | {item.n_theta} | "
            f"{item.ricci_l2:.6e} | {item.ricci_linf:.6e} | {item.center_ricci:.6e} |"
        )
    log_lines.extend(
        [
            "",
            "Minkowski observed orders: "
            + ", ".join(f"{value:.3f}" for value in report["minkowski"]["observed_orders"])
            + ".",
            "",
            "The angular perturbation is a negative control and should converge",
            "to a nonzero continuum Ricci norm rather than to zero.",
            "",
            "Diagnostic correction: the first audit let the coordinate grid",
            "approach the polar singularities as resolution increased.  The",
            "inverse_g-g norm then amplified truncation error and falsely",
            "looked divergent.  These replacement numbers use a fixed regular",
            "equatorial chart and a fixed interior subdomain.  Full-sphere work",
            "will require regular overlapping patches or spin-weighted variables.",
            "",
        ]
    )
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(log_lines))
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(json.dumps(run_axisymmetric_audit_suite(root / "results", root / "results" / "run-log.md"), indent=2))
