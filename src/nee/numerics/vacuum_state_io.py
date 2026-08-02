"""Continue the numerical vacuum iteration through dyadic slabs Q1--Q8."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import uuid
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE

from .vacuum_iteration import (  # noqa: E402
    FirstOrderState,
    impose_outgoing_boundary,
    initial_state,
    metric_closure,
    picard_step,
    update_norm,
)
from .vacuum_residual import components  # noqa: E402
from .sphere import (  # noqa: E402
    PointSphereGrid,
    tangent_inverse,
    tensor_norm_sq,
)
from .smooth_pulse import (  # noqa: E402
    calibrate_profiles,
    scaled_calibration,
    solve_outgoing_boundary,
)


Array = np.ndarray
STATE_NAMES = (
    "g",
    "Omega",
    "zeta",
    "b",
    "Omega_trchi",
    "Omega_chih",
    "Omega_chib",
    "Omega_omega",
    "Omega_omegab",
)
STATE_SCHEMA_VERSION = 2
OMEGAB_SEMANTICS = "integer-step"


def interpolation_matrix(source: Array, target: Array, stencil: int = 11) -> Array:
    """Local polynomial interpolation from one monotone grid to another."""

    if np.any(np.diff(source) <= 0.0) or np.any(np.diff(target) < 0.0):
        raise ValueError("interpolation grids must be monotone")
    tolerance = 1.0e-13 * max(1.0, float(source[-1] - source[0]))
    if float(target[0]) < float(source[0]) - tolerance or float(target[-1]) > float(
        source[-1]
    ) + tolerance:
        raise ValueError("target grid lies outside source grid")
    width = min(stencil, len(source))
    if width % 2 == 0:
        width -= 1
    result = np.zeros((len(target), len(source)), dtype=float)
    half = width // 2
    for row, value in enumerate(target):
        center = int(np.searchsorted(source, value))
        start = min(max(center - half, 0), len(source) - width)
        indices = np.arange(start, start + width)
        offsets = source[indices] - value
        scale = max(float(np.max(np.abs(offsets))), 1.0e-300)
        normalized = offsets / scale
        vandermonde = np.vstack(
            [normalized**power for power in range(width)]
        )
        target_moments = np.zeros(width)
        target_moments[0] = 1.0
        result[row, indices] = np.linalg.solve(vandermonde, target_moments)
    return result


def interpolate(values: Array, matrix: Array, axis: int = 1) -> Array:
    moved = np.moveaxis(values, axis, 0)
    interpolated = np.tensordot(matrix, moved, axes=(1, 0))
    return np.moveaxis(interpolated, 0, axis)


def continued_boundary(
    grid: PointSphereGrid,
    state: FirstOrderState,
    old_v: Array,
    new_v: Array,
) -> dict[str, Array]:
    """Interpolate the terminal u-slice of one slab onto the next slab."""

    matrix = interpolation_matrix(old_v, new_v)
    terminal = {
        name: interpolate(getattr(state, name)[:, -1], matrix, axis=1)
        for name in STATE_NAMES
    }
    return {
        "g": terminal["g"],
        "inverse_g": tangent_inverse(grid, terminal["g"]),
        "Omega_trchi": terminal["Omega_trchi"],
        "Omega_chih": terminal["Omega_chih"],
        "Omega": terminal["Omega"],
        "zeta": terminal["zeta"],
        "b": terminal["b"],
        "Omega_chib": terminal["Omega_chib"],
        "Omega_omega": terminal["Omega_omega"],
        "Omega_omegab": terminal["Omega_omegab"],
    }


def full_ricci_map(
    grid: PointSphereGrid,
    state: FirstOrderState,
    u: Array,
    values: dict[str, Array],
) -> Array:
    """Return (-u)||Ric||_L2 using the positive null-component norm."""

    inverse_g = tangent_inverse(grid, state.g)
    Omega = state.Omega
    omega_sq = Omega**2
    ric33 = values["Omega2_Ric33"] / omega_sq
    ric44 = values.get("Ric44", np.zeros_like(ric33))
    ric34 = values["Omega2_Ric34"] / omega_sq
    ric3 = values["Omega_Ric3A"] / Omega[..., None]
    ric4 = values["Omega_Ric4A"] / Omega[..., None]
    hat_ab = values["Omega2_hat_RicAB"] / omega_sq[..., None, None]
    trace_ab = values["Omega2_R_plus_Ric34"] / omega_sq
    ric_ab = hat_ab + 0.5 * trace_ab[..., None, None] * state.g
    density_sq = ric33**2 + ric44**2 + 2.0 * ric34**2
    density_sq += np.einsum(
        "n...i,n...ij,n...j->n...", ric3, inverse_g, ric3
    )
    density_sq += np.einsum(
        "n...i,n...ij,n...j->n...", ric4, inverse_g, ric4
    )
    density_sq += tensor_norm_sq(ric_ab, inverse_g)
    local = np.einsum(
        "nia,n...ij,njb->n...ab", grid.frames, state.g, grid.frames
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local), 0.0))
    l2 = np.sqrt(
        np.maximum(4.0 * math.pi * np.mean(density_sq * area_ratio, axis=0), 0.0)
    )
    return (-u[:, None]) * l2


def save_state(
    path: Path,
    state: FirstOrderState,
    u: Array,
    v: Array,
    extra: dict[str, Array] | None = None,
) -> None:
    arrays: dict[str, Array] = {
        "u": u,
        "v": v,
        "state_schema_version": np.asarray(STATE_SCHEMA_VERSION),
        "weighted_omegab_semantics": np.asarray(OMEGAB_SEMANTICS),
    }
    arrays.update({name: getattr(state, name) for name in STATE_NAMES})
    if extra is not None:
        overlap = set(arrays).intersection(extra)
        if overlap:
            raise ValueError(f"extra checkpoint arrays overlap state keys: {overlap}")
        arrays.update(extra)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        # Passing an open stream avoids NumPy appending an unexpected ``.npz``
        # suffix to the temporary filename.  The destination is replaced only
        # after the complete ZIP archive is flushed, so a killed writer cannot
        # corrupt an older valid checkpoint.
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        try:
            descriptor = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            descriptor = None
        if descriptor is not None:
            try:
                os.fsync(descriptor)
            except OSError:
                pass
            finally:
                os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def load_state(
    path: Path, *, allow_established_half_step: bool = False
) -> tuple[FirstOrderState, Array, Array]:
    data = np.load(path)
    version = (
        int(data["state_schema_version"])
        if "state_schema_version" in data.files
        else 1
    )
    semantics = (
        str(data["weighted_omegab_semantics"])
        if "weighted_omegab_semantics" in data.files
        else "half-step"
    )
    if version != STATE_SCHEMA_VERSION or semantics != OMEGAB_SEMANTICS:
        if not allow_established_half_step:
            raise ValueError(
                "established numerical checkpoint stores half-step Omega*omegab; "
                "regenerate slabs sequentially or pass allow_established_half_step=True "
                "for a read-only diagnostic sweep"
            )
    state = FirstOrderState(**{name: data[name] for name in STATE_NAMES})
    return state, data["u"], data["v"]


def region_code(f: Array) -> Array:
    """Codes 0..4 for R>=5, R4, R3, R2, R<=1 respectively."""

    code = np.full(f.shape, 4, dtype=np.int8)
    code[f < 1.0e-1] = 3
    code[f < 1.0e-2] = 2
    code[f < 1.0e-3] = 1
    code[f < 1.0e-4] = 0
    return code


def plot_regions(slabs: list[dict], output: Path) -> None:
    """Write dependency-free SVG maps with u vertical and v horizontal."""

    labels = ["R≥5", "R4", "R3", "R2", "R≤1"]
    colors = ["#166534", "#2563eb", "#eab308", "#f97316", "#b91c1c"]

    def edges(values: Array) -> Array:
        result = np.empty(len(values) + 1)
        result[1:-1] = 0.5 * (values[:-1] + values[1:])
        result[0] = values[0]
        result[-1] = values[-1]
        return result

    def marks(
        left: float,
        top: float,
        width: float,
        height: float,
        category_filter: int | None,
    ) -> list[str]:
        u_min = -1.0
        u_max = max(float(slab["u"][-1]) for slab in slabs)

        def px(value: float) -> float:
            return left + width * value / 0.5

        def py(value: float) -> float:
            return top + height * (u_max - value) / (u_max - u_min)

        result: list[str] = []
        for slab in slabs:
            ue = edges(slab["u"])
            ve = edges(slab["v"])
            for i, value_u in enumerate(slab["u"]):
                for j, value_v in enumerate(slab["v"]):
                    if not (value_v > 0.0 and value_v < -0.5 * value_u):
                        continue
                    category = int(slab["regions"][i, j])
                    if category_filter is not None and category != category_filter:
                        continue
                    x0, x1 = px(max(0.0, ve[j])), px(min(0.5, ve[j + 1]))
                    y0, y1 = py(max(u_min, ue[i])), py(min(u_max, ue[i + 1]))
                    result.append(
                        f'<rect x="{x0:.2f}" y="{min(y0,y1):.2f}" '
                        f'width="{max(x1-x0,.15):.2f}" height="{max(abs(y1-y0),.15):.2f}" '
                        f'fill="{colors[category]}"/>'
                    )
        result.append(
            f'<path d="M {px(.5):.2f} {py(-1):.2f} L {px(-0.5*u_max):.2f} '
            f'{py(u_max):.2f}" fill="none" stroke="#111827" stroke-width="1.2"/>'
        )
        result.append(
            f'<path d="M {left:.2f} {top:.2f} V {top+height:.2f} H '
            f'{left+width:.2f}" fill="none" stroke="#111827" stroke-width="1"/>'
        )
        return result

    width, height = 1000, 860
    left, top, plot_width, plot_height = 78, 58, 850, 735
    combined = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="500" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">Vacuum residual regions on Q1–Q{len(slabs)}</text>',
        *marks(left, top, plot_width, plot_height, None),
        f'<text x="{left+plot_width/2:.1f}" y="835" text-anchor="middle" font-family="sans-serif" font-size="16">v</text>',
        f'<text x="24" y="{top+plot_height/2:.1f}" text-anchor="middle" transform="rotate(-90 24 {top+plot_height/2:.1f})" font-family="sans-serif" font-size="16">u</text>',
    ]
    for index, (label, color) in enumerate(zip(labels, colors)):
        x = 690 + (index % 2) * 115
        y = 84 + (index // 2) * 24
        combined.extend(
            [
                f'<rect x="{x}" y="{y-12}" width="15" height="15" fill="{color}"/>',
                f'<text x="{x+22}" y="{y}" font-family="sans-serif" font-size="13">{label}</text>',
            ]
        )
    combined.append("</svg>")
    (output / "convergence-regions-combined.svg").write_text("\n".join(combined) + "\n")

    panel_width, panel_height = 470, 390
    panels = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1500" height="920" viewBox="0 0 1500 920">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="750" y="28" text-anchor="middle" font-family="sans-serif" font-size="20">Individual convergence regions in the (v,u) plane</text>',
    ]
    for category in range(5):
        column, row = category % 3, category // 3
        panel_left = 48 + column * 495
        panel_top = 72 + row * 420
        panels.append(
            f'<text x="{panel_left+panel_width/2}" y="{panel_top-12}" text-anchor="middle" font-family="sans-serif" font-size="17">{labels[category]}</text>'
        )
        panels.extend(
            marks(panel_left, panel_top, panel_width, panel_height, category)
        )
        panels.append(
            f'<text x="{panel_left+panel_width/2}" y="{panel_top+panel_height+24}" text-anchor="middle" font-family="sans-serif" font-size="13">v</text>'
        )
    panels.append("</svg>")
    (output / "convergence-regions-panels.svg").write_text("\n".join(panels) + "\n")


def run(args: argparse.Namespace) -> dict:
    output = ROOT / "results" / "Q1-Q8" / args.output_label
    output.mkdir(parents=True, exist_ok=True)
    grid = PointSphereGrid.create(
        args.points,
        neighbor_count=args.neighbors,
        degree=args.angular_degree,
        spectral_degree=args.spectral_degree,
    )
    calibration = scaled_calibration(
        calibrate_profiles(args.v1, args.c), args.Omega_chih_divisor
    )
    slabs: list[dict] = []
    summaries: list[dict] = []
    previous_state: FirstOrderState | None = None
    previous_v: Array | None = None

    if args.start_slab > 1:
        for completed in range(1, args.start_slab):
            completed_output = output / f"Q{completed}"
            saved = np.load(completed_output / "residual-data.npz")
            slabs.append(
                {
                    "u": saved["u"],
                    "v": saved["v"],
                    "residual": saved["residual"],
                    "regions": saved["regions"],
                }
            )
            summaries.append(
                json.loads((completed_output / "summary.json").read_text())
            )
        previous_state, _, previous_v = load_state(
            output / f"Q{args.start_slab - 1}" / "final-state.npz"
        )

    for slab_index in range(args.start_slab, args.slab_count + 1):
        left_radius = 2.0 ** (-(slab_index - 1))
        right_radius = 2.0**(-slab_index)
        u = -np.exp(
            np.linspace(math.log(left_radius), math.log(right_radius), args.u_count)
        )
        v = np.linspace(0.0, right_radius, args.v_count)
        if slab_index == 1:
            boundary = solve_outgoing_boundary(grid, v, calibration)
        else:
            assert previous_state is not None and previous_v is not None
            boundary = continued_boundary(grid, previous_state, previous_v, v)

        state = impose_outgoing_boundary(initial_state(grid, u, v), boundary)
        records = []
        last_closure = None
        last_context = None
        for iteration in range(1, args.iterations + 1):
            started = time.perf_counter()
            new_state, context = picard_step(
                grid,
                state,
                boundary,
                u,
                v,
                metric_substeps=args.metric_substeps,
            )
            closure = metric_closure(
                grid, new_state, context["incoming_metric"], u
            )
            update = update_norm(new_state, state)
            record = {
                "slab": slab_index,
                "iteration": iteration,
                "seconds": time.perf_counter() - started,
                "picard_update": update,
                "metric_closure_maximum": closure["maximum"],
                "differential_closure_maximum": closure[
                    "differential_maximum"
                ],
            }
            records.append(record)
            print(json.dumps(record), flush=True)
            state = new_state
            last_closure = closure
            last_context = context

        values = components(grid, state, u, v)
        residual = full_ricci_map(grid, state, u, values)
        regions = region_code(residual)
        domain = (v[None, :] > 0.0) & (v[None, :] < -0.5 * u[:, None])
        counts = {
            name: int(np.count_nonzero(domain & (regions == code)))
            for code, name in enumerate(("R_ge_5", "R4", "R3", "R2", "R_le_1"))
        }
        slab_summary = {
            "slab": slab_index,
            "u_left": float(u[0]),
            "u_right": float(u[-1]),
            "v_max": float(v[-1]),
            "records": records,
            "residual_minimum_on_plot_domain": float(np.min(residual[domain])),
            "residual_median_on_plot_domain": float(np.median(residual[domain])),
            "residual_maximum_on_plot_domain": float(np.max(residual[domain])),
            "region_counts": counts,
        }
        summaries.append(slab_summary)
        slab_output = output / f"Q{slab_index}"
        slab_output.mkdir(exist_ok=True)
        np.savez_compressed(
            slab_output / "residual-data.npz",
            u=u,
            v=v,
            residual=residual,
            regions=regions,
            metric_closure=last_closure["pointwise"],
            differential_metric_closure=last_closure["differential_pointwise"],
        )
        if args.save_states:
            save_state(slab_output / "final-state.npz", state, u, v)
        (slab_output / "summary.json").write_text(
            json.dumps(slab_summary, indent=2) + "\n"
        )
        slabs.append({"u": u, "v": v, "residual": residual, "regions": regions})
        previous_state = state
        previous_v = v

    plot_regions(slabs, output)
    result = {
        "experiment": "numerical continued dyadic vacuum Q1--Q8",
        "status": (
            "stopped_after_Q5_bad_approximation"
            if args.slab_count == 5
            else "completed_requested_slabs"
        ),
        "parameters": {
            "c": args.c,
            "Omega_chih_divisor": args.Omega_chih_divisor,
            "v1": args.v1,
            "points": args.points,
            "neighbors": args.neighbors,
            "angular_degree": args.angular_degree,
            "spectral_degree": args.spectral_degree,
            "u_count": args.u_count,
            "v_count": args.v_count,
            "metric_substeps": args.metric_substeps,
            "computed_slab_count": len(summaries),
            "iterations_per_slab": {
                f"Q{slab['slab']}": len(slab["records"])
                for slab in summaries
            },
        },
        "region_definition": {
            "R_ge_5": "f < 1e-4",
            "R4": "1e-4 <= f < 1e-3",
            "R3": "1e-3 <= f < 1e-2",
            "R2": "1e-2 <= f < 1e-1",
            "R_le_1": "f >= 1e-1",
            "note": "uses the non-overlapping convention 10^-i <= f < 10^(-i+1)",
        },
        "plot_domain": (
            f"-1<u<-2^-{args.slab_count} and 0<v<(-u)/2"
        ),
        "slabs": summaries,
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--c", type=float, default=1.0)
    result.add_argument("--Omega_chih-divisor", type=float, default=3.2)
    result.add_argument("--v1", type=float, default=0.5)
    result.add_argument("--points", type=int, default=50)
    result.add_argument("--neighbors", type=int, default=28)
    result.add_argument("--angular-degree", type=int, default=4)
    result.add_argument("--spectral-degree", type=int, default=3)
    result.add_argument("--u-count", type=int, default=20)
    result.add_argument("--v-count", type=int, default=200)
    result.add_argument("--iterations", type=int, default=20)
    result.add_argument(
        "--g-substeps", dest="metric_substeps", type=int, default=2
    )
    result.add_argument("--slab-count", type=int, default=8)
    result.add_argument("--start-slab", type=int, default=1)
    result.add_argument("--save-states", action="store_true")
    result.add_argument("--output-label", default="smooth-hemisphere-50x20x200-l3")
    return result


if __name__ == "__main__":
    run(parser().parse_args())
