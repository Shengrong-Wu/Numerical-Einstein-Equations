"""Smooth global spin-2 pulse design with uniform integrated generator energy.

An everywhere nonzero real trace-free symmetric tensor of constant pointwise
norm cannot exist on S2.  This module constructs the closest smooth global
replacement: its sphere-RMS norm is exactly C v^delta at every v, while its
v-integrated pointwise energy is independent of angle.  It is a data-design
and topology diagnostic, not yet a global EVE evolution.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


Array = np.ndarray


def ambient_stf_basis() -> Array:
    basis = np.zeros((5, 3, 3), dtype=float)
    basis[0] = np.diag([1.0, -1.0, 0.0]) / math.sqrt(2.0)
    basis[1, 0, 1] = basis[1, 1, 0] = 1.0 / math.sqrt(2.0)
    basis[2, 0, 2] = basis[2, 2, 0] = 1.0 / math.sqrt(2.0)
    basis[3, 1, 2] = basis[3, 2, 1] = 1.0 / math.sqrt(2.0)
    basis[4] = np.diag([1.0, 1.0, -2.0]) / math.sqrt(6.0)
    return basis


def tangent_spin2_basis(points: Array, trace_coefficient: float = 0.5) -> Array:
    """Project an ambient STF basis to tangent trace-free tensors on S2."""

    identity = np.eye(3)
    projector = identity - np.einsum("...i,...j->...ij", points, points)
    basis = ambient_stf_basis()
    projected = np.einsum(
        "...ia,kab,...bj->...kij", projector, basis, projector
    )
    tangent_trace = np.einsum("...kij,...ij->...k", projected, projector)
    tensors = projected - trace_coefficient * tangent_trace[..., None, None] * projector[..., None, :, :]
    # With the correct trace coefficient, equivariance gives
    # sum_k |raw T_k(n)|^2=2 and sphere mean |raw T_k|^2=2/5.
    return math.sqrt(5.0 / 2.0) * tensors


def time_coefficients(t: Array) -> Array:
    """Unit S4 curve with covariance integral p_k p_l dt = delta_kl/5."""

    return np.stack(
        [
            np.full_like(t, 1.0 / math.sqrt(5.0)),
            math.sqrt(2.0 / 5.0) * np.cos(2.0 * math.pi * t),
            math.sqrt(2.0 / 5.0) * np.sin(2.0 * math.pi * t),
            math.sqrt(2.0 / 5.0) * np.cos(4.0 * math.pi * t),
            math.sqrt(2.0 / 5.0) * np.sin(4.0 * math.pi * t),
        ],
        axis=-1,
    )


def fibonacci_sphere(count: int) -> Array:
    index = np.arange(count, dtype=float)
    z = 1.0 - 2.0 * (index + 0.5) / count
    longitude = math.pi * (3.0 - math.sqrt(5.0)) * index
    radius = np.sqrt(np.maximum(1.0 - z**2, 0.0))
    return np.column_stack([radius * np.cos(longitude), radius * np.sin(longitude), z])


def focusing_run(
    delta: float,
    point_count: int,
    step_count: int,
    v_max: float = 0.005,
    moving_zero: bool = True,
    record_curve: bool = False,
) -> dict:
    """Solve the outgoing Raychaudhuri constraint on every sphere generator."""

    points = fibonacci_sphere(point_count)
    tensors = tangent_spin2_basis(points)
    gram = np.einsum("...kij,...lij->...kl", tensors, tensors)
    static_norm_sq = gram[:, 0, 0]
    v = np.linspace(0.0, v_max, step_count)
    area_factor = np.ones(point_count)
    area_derivative = np.ones(point_count)
    first_trapped = None
    first_values = None
    curve = []
    stride = max((step_count - 1) // 100, 1)

    def shear_norm_sq(value: float) -> Array:
        if moving_zero:
            t = (value / v_max) ** (2.0 * delta + 1.0) if value > 0.0 else 0.0
            coefficients = time_coefficients(np.array([t]))[0]
            angular = np.einsum("k,...kl,l->...", coefficients, gram, coefficients)
        else:
            angular = static_norm_sq
        return 10000.0 * value ** (2.0 * delta) * angular

    def acceleration(value: float, factor: Array) -> Array:
        return -0.5 * shear_norm_sq(value) * factor

    for j in range(step_count - 1):
        value = float(v[j])
        step = float(v[j + 1] - v[j])
        k1_factor = area_derivative
        k1_derivative = acceleration(value, area_factor)
        k2_factor = area_derivative + 0.5 * step * k1_derivative
        k2_derivative = acceleration(
            value + 0.5 * step, area_factor + 0.5 * step * k1_factor
        )
        k3_factor = area_derivative + 0.5 * step * k2_derivative
        k3_derivative = acceleration(
            value + 0.5 * step, area_factor + 0.5 * step * k2_factor
        )
        k4_factor = area_derivative + step * k3_derivative
        k4_derivative = acceleration(value + step, area_factor + step * k3_factor)
        area_factor = area_factor + step * (
            k1_factor + 2.0 * k2_factor + 2.0 * k3_factor + k4_factor
        ) / 6.0
        area_derivative = area_derivative + step * (
            k1_derivative
            + 2.0 * k2_derivative
            + 2.0 * k3_derivative
            + k4_derivative
        ) / 6.0
        expansion = 2.0 * area_derivative / area_factor
        if first_trapped is None and float(np.max(expansion)) <= 0.0:
            first_trapped = float(v[j + 1])
            first_values = {
                "min_expansion": float(np.min(expansion)),
                "max_expansion": float(np.max(expansion)),
                "min_area_factor": float(np.min(area_factor)),
                "max_area_factor": float(np.max(area_factor)),
            }
        if record_curve and ((j + 1) % stride == 0 or j == step_count - 2):
            curve.append(
                {
                    "v": float(v[j + 1]),
                    "min_expansion": float(np.min(expansion)),
                    "max_expansion": float(np.max(expansion)),
                }
            )

    expansion = 2.0 * area_derivative / area_factor
    future_caustics = np.where(
        area_derivative < 0.0,
        v_max - area_factor / area_derivative,
        np.inf,
    )
    return {
        "delta": delta,
        "point_count": point_count,
        "step_count": step_count,
        "moving_zero": moving_zero,
        "first_all_generators_nonpositive": first_trapped,
        "first_surface": first_values,
        "final_expansion_min": float(np.min(expansion)),
        "final_expansion_max": float(np.max(expansion)),
        "final_area_factor_min": float(np.min(area_factor)),
        "final_area_factor_max": float(np.max(area_factor)),
        "earliest_post_pulse_caustic": float(np.min(future_caustics)),
        "curve": curve,
    }


def global_focusing_suite() -> dict:
    resolutions = [(512, 1001), (1024, 2001), (2048, 4001), (4096, 8001)]
    refinement = {}
    final_runs = []
    static_controls = []
    for delta in [0.1, 0.01]:
        rows = [
            focusing_run(delta, point_count, step_count)
            for point_count, step_count in resolutions
        ]
        refinement[str(delta)] = rows
        final_runs.append(
            focusing_run(delta, 4096, 8001, record_curve=True)
        )
        static_controls.append(
            focusing_run(delta, 4096, 4001, moving_zero=False)
        )
    return {
        "v_max": 0.005,
        "refinement": refinement,
        "moving_zero_runs": final_runs,
        "static_mode_controls": static_controls,
    }


def pulse_diagnostics(point_count: int = 8192, time_count: int = 1001) -> dict:
    points = fibonacci_sphere(point_count)
    tensors = tangent_spin2_basis(points)
    projector = np.eye(3) - np.einsum("...i,...j->...ij", points, points)
    tangency = np.einsum("...kij,...j->...ki", tensors, points)
    traces = np.einsum("...kij,...ij->...k", tensors, projector)
    pointwise_sum = np.einsum("...kij,...kij->...", tensors, tensors)
    gram = np.mean(np.einsum("...kij,...lij->...kl", tensors, tensors), axis=0)

    t = np.linspace(0.0, 1.0, time_count)
    coefficients = time_coefficients(t)
    coefficient_norm = np.sum(coefficients**2, axis=1)
    covariance = np.trapezoid(
        np.einsum("tk,tl->tkl", coefficients, coefficients), t, axis=0
    )
    # The time-integrated energy is computed from the 5x5 covariance instead
    # of materializing time x sphere x tensor arrays.
    integrated = np.einsum("kl,...kij,...lij->...", covariance, tensors, tensors)
    normalized_integrated = integrated / np.mean(integrated)

    sample_times = np.array([0.0, 0.137, 0.311, 0.529, 0.811])
    sample_coefficients = time_coefficients(sample_times)
    sample_fields = np.einsum("tk,...kij->t...ij", sample_coefficients, tensors)
    sample_norm_sq = np.einsum("t...ij,t...ij->t...", sample_fields, sample_fields)
    sphere_rms_sq = np.mean(sample_norm_sq, axis=1)

    static_energy = np.einsum("...ij,...ij->...", tensors[:, 0], tensors[:, 0])
    static_normalized = static_energy / np.mean(static_energy)
    mutated = tangent_spin2_basis(points, trace_coefficient=0.45)
    mutated_trace = np.einsum("...kij,...ij->...k", mutated, projector)

    return {
        "construction": {
            "interpretation": "sphere-RMS norm C v^delta and uniform v-integrated generator energy",
            "time_coordinate": "t=(v/v_max)^(2 delta+1)",
            "basis_dimension": 5,
        },
        "exact_identity_errors": {
            "max_tangency": float(np.max(np.abs(tangency))),
            "max_trace": float(np.max(np.abs(traces))),
            "max_pointwise_addition_theorem": float(np.max(np.abs(pointwise_sum - 5.0))),
            "max_time_coefficient_norm": float(np.max(np.abs(coefficient_norm - 1.0))),
            "max_time_covariance": float(
                np.max(np.abs(covariance - np.eye(5) / 5.0))
            ),
        },
        "quadrature_checks": {
            "max_sphere_gram_error": float(np.max(np.abs(gram - np.eye(5)))),
            "sample_sphere_rms_norm_errors": [
                float(abs(math.sqrt(value) - 1.0)) for value in sphere_rms_sq
            ],
            "integrated_energy_min": float(np.min(normalized_integrated)),
            "integrated_energy_max": float(np.max(normalized_integrated)),
            "integrated_energy_relative_spread": float(
                np.max(normalized_integrated) - np.min(normalized_integrated)
            ),
            "instantaneous_norm_min": float(np.sqrt(np.min(sample_norm_sq))),
            "instantaneous_norm_max": float(np.sqrt(np.max(sample_norm_sq))),
        },
        "negative_controls": {
            "static_mode_integrated_energy_relative_spread": float(
                np.max(static_normalized) - np.min(static_normalized)
            ),
            "trace_projection_coefficient_0p45_max_trace": float(
                np.max(np.abs(mutated_trace))
            ),
        },
    }


def grid_field_values(
    n_latitude: int = 61, n_longitude: int = 121
) -> tuple[Array, Array, Array, Array]:
    latitude = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_latitude)
    longitude = np.linspace(-math.pi, math.pi, n_longitude)
    lon, lat = np.meshgrid(longitude, latitude)
    points = np.stack(
        [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)],
        axis=-1,
    )
    tensors = tangent_spin2_basis(points)
    coefficients = time_coefficients(np.array([0.0]))[0]
    field = np.einsum("k,...kij->...ij", coefficients, tensors)
    instantaneous = np.sqrt(np.einsum("...ij,...ij->...", field, field))
    addition = np.einsum("...kij,...kij->...", tensors, tensors) / 5.0
    return latitude, longitude, instantaneous, addition


def write_global_pulse_svg(path: Path) -> None:
    latitude, longitude, instantaneous, integrated = grid_field_values()
    width, height = 800, 540
    left, right, top, gap, bottom = 78, 35, 48, 58, 62
    panel_h = (height - top - bottom - gap) / 2.0
    plot_w = width - left - right

    def xcoord(value: float) -> float:
        return left + (value + math.pi) / (2.0 * math.pi) * plot_w

    def panel_y(value: float, panel_top: float) -> float:
        return panel_top + (0.5 * math.pi - value) / math.pi * panel_h

    def color(value: float, minimum: float, maximum: float) -> str:
        ratio = 0.5 if maximum == minimum else (value - minimum) / (maximum - minimum)
        red = int(35 + 190 * ratio)
        green = int(85 + 95 * (1.0 - abs(2.0 * ratio - 1.0)))
        blue = int(210 - 165 * ratio)
        return f"rgb({red},{green},{blue})"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="400" y="25" text-anchor="middle" font-family="sans-serif" font-size="16">Smooth global spin-2 pulse design</text>',
    ]
    panels = [
        (instantaneous, top, "instantaneous |T(t=0,theta,phi)| (zeros required)"),
        (integrated, top + panel_h + gap, "normalized integrated generator energy (uniform)"),
    ]
    for values, panel_top, title in panels:
        data_minimum = float(np.min(values))
        data_maximum = float(np.max(values))
        if data_maximum - data_minimum < 1.0e-8:
            minimum, maximum = 0.999, 1.001
        else:
            minimum, maximum = data_minimum, data_maximum
        for i in range(len(latitude) - 1):
            for j in range(len(longitude) - 1):
                x0, x1 = xcoord(float(longitude[j])), xcoord(float(longitude[j + 1]))
                y0 = panel_y(float(latitude[i + 1]), panel_top)
                y1 = panel_y(float(latitude[i]), panel_top)
                lines.append(
                    f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1-x0+0.2:.2f}" height="{y1-y0+0.2:.2f}" fill="{color(float(values[i,j]), minimum, maximum)}"/>'
                )
        lines.extend(
            [
                f'<rect x="{left}" y="{panel_top}" width="{plot_w}" height="{panel_h}" fill="none" stroke="#222"/>',
                f'<text x="{left}" y="{panel_top-8}" font-family="sans-serif" font-size="12">{title}</text>',
                f'<text x="{left+plot_w}" y="{panel_top-8}" text-anchor="end" font-family="monospace" font-size="10">min={data_minimum:.6f}, max={data_maximum:.6f}</text>',
            ]
        )
        for value in [-math.pi, 0.0, math.pi]:
            x = xcoord(value)
            lines.append(
                f'<text x="{x:.2f}" y="{panel_top+panel_h+17}" text-anchor="middle" font-family="monospace" font-size="9">{value/math.pi:.0f}pi</text>'
            )
    lines.extend(
        [
            f'<text x="{left+plot_w/2:.1f}" y="{height-14}" text-anchor="middle" font-family="sans-serif" font-size="12">longitude phi; latitude runs from -pi/2 to pi/2</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_global_focusing_svg(focusing: dict, path: Path) -> None:
    width, height = 780, 470
    left, right, top, bottom = 84, 32, 42, 68
    plot_w, plot_h = width - left - right, height - top - bottom
    runs = focusing["moving_zero_runs"]
    all_values = [
        point["max_expansion"] for run in runs for point in run["curve"]
    ]
    y_min = min(all_values) - 2.0
    y_max = 3.0
    colors = ["#2563eb", "#dc2626"]

    def xcoord(value: float) -> float:
        return left + value / focusing["v_max"] * plot_w

    def ycoord(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="390" y="25" text-anchor="middle" font-family="sans-serif" font-size="16">Global outgoing focusing with the moving-zero pulse</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{ycoord(0.0):.2f}" x2="{left+plot_w}" y2="{ycoord(0.0):.2f}" stroke="#64748b" stroke-dasharray="5 4"/>',
    ]
    for color, run in zip(colors, runs):
        points = [(0.0, 2.0)] + [
            (point["v"], point["max_expansion"]) for point in run["curve"]
        ]
        lines.append(
            f'<polyline points="{" ".join(f"{xcoord(x):.2f},{ycoord(y):.2f}" for x,y in points)}" fill="none" stroke="{color}" stroke-width="2.5"/>'
        )
        trapped_v = run["first_all_generators_nonpositive"]
        lines.append(
            f'<circle cx="{xcoord(trapped_v):.2f}" cy="{ycoord(0.0):.2f}" r="4" fill="{color}"/>'
        )
        lines.append(
            f'<text x="{xcoord(trapped_v)+7:.2f}" y="{ycoord(0.0)-7:.2f}" font-family="monospace" font-size="10">v={trapped_v:.6f}</text>'
        )
    for value in np.linspace(0.0, focusing["v_max"], 6):
        x = xcoord(float(value))
        lines.append(
            f'<text x="{x:.2f}" y="{top+plot_h+22}" text-anchor="middle" font-family="monospace" font-size="10">{value:.3f}</text>'
        )
    for value in np.linspace(y_min, y_max, 6):
        y = ycoord(float(value))
        lines.append(
            f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-family="monospace" font-size="10">{value:.1f}</text>'
        )
    lines.extend(
        [
            f'<text x="{left+plot_w/2:.1f}" y="{height-20}" text-anchor="middle" font-family="sans-serif" font-size="13">v</text>',
            f'<text x="19" y="{top+plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 19 {top+plot_h/2:.1f})" font-family="sans-serif" font-size="13">sup over S2 of tr(chi)</text>',
            f'<line x1="{left+18}" y1="{top+16}" x2="{left+40}" y2="{top+16}" stroke="{colors[0]}" stroke-width="2"/><text x="{left+47}" y="{top+20}" font-family="sans-serif" font-size="11">delta=0.1</text>',
            f'<line x1="{left+18}" y1="{top+36}" x2="{left+40}" y2="{top+36}" stroke="{colors[1]}" stroke-width="2"/><text x="{left+47}" y="{top+40}" font-family="sans-serif" font-size="11">delta=0.01</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_global_pulse_suite(results_dir: Path, log_path: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    report = pulse_diagnostics()
    report["global_outgoing_focusing"] = global_focusing_suite()
    report["scope"] = (
        "smooth global initial-data replacement study; relaxes instantaneous "
        "pointwise norm to exact sphere-RMS norm; no global EVE evolution yet"
    )
    (results_dir / "global-pulse-design-summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_global_pulse_svg(results_dir / "global-pulse-design.svg")
    write_global_focusing_svg(
        report["global_outgoing_focusing"],
        results_dir / "global-pulse-focusing.svg",
    )
    checks = report["exact_identity_errors"]
    quadrature = report["quadrature_checks"]
    controls = report["negative_controls"]
    focusing = report["global_outgoing_focusing"]
    lines = [
        "",
        "## 2026-07-15 — Smooth global moving-zero pulse design",
        "",
        "Projected an orthonormal five-dimensional ambient STF basis to `S^2`",
        "and moved through it on a unit `S^4` coefficient curve.  The pulse has",
        "exact sphere-RMS norm `C v^delta` and uniform generator-integrated",
        "energy, while respecting the unavoidable instantaneous zeros.",
        "",
        f"- maximum tangency error: `{checks['max_tangency']:.3e}`",
        f"- maximum trace error: `{checks['max_trace']:.3e}`",
        f"- addition-theorem error: `{checks['max_pointwise_addition_theorem']:.3e}`",
        f"- integrated-energy relative spread: `{quadrature['integrated_energy_relative_spread']:.3e}`",
        f"- static-mode negative-control spread: `{controls['static_mode_integrated_energy_relative_spread']:.3e}`",
        f"- trace-coefficient mutation residual: `{controls['trace_projection_coefficient_0p45_max_trace']:.3e}`",
        f"- all-generator trapped v for delta=0.1: `{focusing['moving_zero_runs'][0]['first_all_generators_nonpositive']:.6e}`",
        f"- all-generator trapped v for delta=0.01: `{focusing['moving_zero_runs'][1]['first_all_generators_nonpositive']:.6e}`",
        "- static-mode controls do not make every generator outgoing trapped",
        "",
        "This is a globally smooth replacement study, not the exact pointwise",
        "constant-norm datum and not yet a global angular EVE evolution.",
        "",
    ]
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return report


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            run_global_pulse_suite(root / "results", root / "results" / "run-log.md"),
            indent=2,
        )
    )
