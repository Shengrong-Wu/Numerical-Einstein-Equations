"""Vacuum data with shear on both characteristic faces.

The public state is the official full weighted state.  The repository-local
first-order transport kernel is used only inside one Picard map
``U^(i) -> U^(i+1)``.  Characteristic data are generated and saved
independently of every iterate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[4]

from nee.io.boundary_artifact import load_boundary_data, save_boundary_data  # noqa: E402
from nee.discretization.double_null_mesh import DoubleSqrtLGLMesh
from nee.discretization.lgl import CompositeLGLMesh
from nee.initial_data.crossed_shears import construct_boundary_data
from nee.solver.backend import weighted_update_map, weighted_update_norm  # noqa: E402
from nee.state.boundary import BoundaryData
from nee.state.iterate import PicardState as WeightedState  # noqa: E402
from nee.solver.picard import (  # noqa: E402
    assert_characteristic_traces,
    incoming_trace_errors,
    initial_iterate,
    raw_picard_candidate,
    relaxed_next_iterate,
)
from nee.diagnostics.convergence_regions import composite_uv_mask  # noqa: E402
from nee.solver.slab_continuation import (  # noqa: E402
    concatenate_states,
    outgoing_segment,
    slab_boundary,
    terminal_incoming_data,
)


from nee.numerics.spherical_harmonics import AngularGalerkin  # noqa: E402
from nee.numerics.vacuum_residual import components  # noqa: E402
from nee.numerics.sphere import (  # noqa: E402
    PointSphereGrid,
    connection_difference,
    gaussian_curvature,
    lie_covariant_tensor,
    scalar_gradient,
    tangent_inverse,
    tensor_divergence,
    tensor_norm_sq,
    tensor_tracefree,
    vector_divergence,
)


Array = np.ndarray


def source_revision() -> str:
    """Hash every maintained Python module in the installed package."""

    digest = hashlib.sha256()
    package_root = ROOT / "src" / "nee"
    for path in sorted(package_root.rglob("*.py")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class Resolution:
    name: str
    elements: int
    degree: int
    retained: int
    work: int
    points: int
    neighbors: int
    sweeps: int
    boundary_substeps: int
    metric_substeps: int
    relaxation: float
    tolerance: float


def _l2_maps(
    grid: PointSphereGrid,
    state: WeightedState,
    values: dict[str, Array],
) -> dict[str, Array]:
    inverse = tangent_inverse(grid, state.metric)
    local_metric = np.einsum(
        "nia,n...ij,njb->n...ab",
        grid.frames,
        state.metric,
        grid.frames,
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local_metric), 0.0))

    def integrate(density_sq: Array) -> Array:
        return np.sqrt(
            np.maximum(
                4.0 * math.pi * np.mean(density_sq * area_ratio, axis=0),
                0.0,
            )
        )

    def form_sq(value: Array) -> Array:
        return np.maximum(
            np.einsum(
                "n...i,n...ij,n...j->n...",
                value,
                inverse,
                value,
            ),
            0.0,
        )

    def tensor_sq(value: Array) -> Array:
        return np.maximum(tensor_norm_sq(value, inverse), 0.0)

    return {
        "r33": integrate(values["Omega2_Ric33"] ** 2),
        "r34": integrate(values["Omega2_Ric34"] ** 2),
        "r3": integrate(form_sq(values["Omega_Ric3A"])),
        "r4": integrate(form_sq(values["Omega_Ric4A"])),
        "rhat": integrate(tensor_sq(values["Omega2_hat_RicAB"])),
        "rR": integrate(values["Omega2_R"] ** 2),
        "r44": integrate(
            (state.omega**2 * values["Ric44_fresh"]) ** 2
        ),
    }


def residual_regions(total: Array) -> Array:
    """Return category indices for R_{<=1}, R_2, R_3, R_4, R_{>=5}."""

    result = np.zeros_like(total, dtype=np.int8)
    result[(total >= 1.0e-2) & (total < 1.0e-1)] = 1
    result[(total >= 1.0e-3) & (total < 1.0e-2)] = 2
    result[(total >= 1.0e-4) & (total < 1.0e-3)] = 3
    result[total < 1.0e-4] = 4
    return result


def plot_regions(
    path: Path,
    mesh: DoubleSqrtLGLMesh,
    total: Array,
    reliable: Array,
) -> dict[str, int]:
    categories = residual_regions(total)
    colors = ["#2b283d", "#6f6689", "#9a89b3", "#ccb8dc", "#f3e8f5"]
    width, height = 1700, 980
    left, right, top, bottom = 150, 570, 100, 130
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    small = ImageFont.load_default(size=23)
    u_bounds = np.concatenate(
        (
            [mesh.u[0]],
            0.5 * (mesh.u[:-1] + mesh.u[1:]),
            [mesh.u[-1]],
        )
    )
    v_bounds = np.concatenate(
        (
            [mesh.v[0]],
            0.5 * (mesh.v[:-1] + mesh.v[1:]),
            [mesh.v[-1]],
        )
    )

    def pixel_x(value: float) -> int:
        return left + round(plot_width * 2.0 * (value + 1.0))

    def pixel_y(value: float) -> int:
        return top + plot_height - round(plot_height * 2.0 * value)

    # Q is open.  The characteristic-face and terminal-face nodes are not
    # classified as interior convergence regions.
    for i in range(1, len(mesh.u) - 1):
        x0 = pixel_x(float(u_bounds[i]))
        x1 = pixel_x(float(u_bounds[i + 1]))
        for j in range(1, len(mesh.v) - 1):
            y0 = pixel_y(float(v_bounds[j + 1]))
            y1 = pixel_y(float(v_bounds[j]))
            draw.rectangle(
                (x0, y0, x1, y1),
                fill=(
                    colors[int(categories[i, j])]
                    if reliable[i, j]
                    else "#d9d9d9"
                ),
            )
    draw.rectangle(
        (left, top, left + plot_width, top + plot_height),
        outline="black",
        width=3,
    )
    draw.text(
        (left + 70, 30),
        "Six-component vacuum Ricci residual regions",
        fill="black",
        font=font,
    )
    draw.text(
        (left + plot_width // 2 - 10, height - 75),
        "u",
        fill="black",
        font=font,
    )
    draw.text((55, top + plot_height // 2), "v", fill="black", font=font)
    for fraction, label in (
        (0.0, "-1"),
        (0.5, "-0.75"),
        (1.0, "-0.5"),
    ):
        x = left + round(plot_width * fraction)
        draw.line((x, top + plot_height, x, top + plot_height + 12), fill="black")
        draw.text(
            (x - 30, top + plot_height + 20), label, fill="black", font=small
        )
    for fraction, label in ((0.0, "0"), (0.5, "0.25"), (1.0, "0.5")):
        y = top + plot_height - round(plot_height * fraction)
        draw.line((left - 12, y, left, y), fill="black")
        draw.text((55, y - 12), label, fill="black", font=small)
    legend_labels = (
        "R <= 1: r >= 1e-1",
        "R2: 1e-2 <= r < 1e-1",
        "R3: 1e-3 <= r < 1e-2",
        "R4: 1e-4 <= r < 1e-3",
        "R >= 5: r < 1e-4",
        "excluded diagnostic stencil",
    )
    legend_colors = (*colors, "#d9d9d9")
    legend_x = left + plot_width + 55
    for index, (color, label) in enumerate(
        zip(legend_colors, legend_labels, strict=True)
    ):
        y = top + 35 + 80 * index
        draw.rectangle((legend_x, y, legend_x + 48, y + 48), fill=color)
        draw.rectangle(
            (legend_x, y, legend_x + 48, y + 48), outline="black", width=1
        )
        draw.text((legend_x + 65, y + 8), label, fill="black", font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(220, 220))
    interior = categories[reliable]
    return {
        label: int(np.count_nonzero(interior == index))
        for index, label in enumerate(
            ("R_le_1", "R_2", "R_3", "R_4", "R_ge_5")
        )
    }


def run_level(
    output: Path,
    resolution: Resolution,
) -> dict[str, Any]:
    breakpoints = np.linspace(0.0, 1.0, resolution.elements + 1)
    mesh = DoubleSqrtLGLMesh.create(
        breakpoints,
        resolution.degree,
        breakpoints,
        resolution.degree,
    )
    grid = PointSphereGrid.create(
        resolution.points,
        neighbor_count=resolution.neighbors,
        degree=min(5, resolution.retained + 1),
        spectral_degree=resolution.work + 1,
    )
    angular = AngularGalerkin(
        grid,
        retained_degree=resolution.retained,
        work_degree=resolution.work,
    )
    boundary, certificates = construct_boundary_data(
        grid,
        mesh,
        substeps=resolution.boundary_substeps,
    )
    boundary_path = output / "boundary-data.npz"
    save_boundary_data(
        boundary_path,
        boundary,
        u=mesh.u,
        v=mesh.v,
    )
    # The iteration consumes the serialized, content-verified artifact rather
    # than retaining an implicit dependency on the generator's live objects.
    boundary, artifact_u, artifact_v = load_boundary_data(boundary_path)
    np.testing.assert_array_equal(artifact_u, mesh.u)
    np.testing.assert_array_equal(artifact_v, mesh.v)
    records: list[dict[str, float | int]] = []
    context: dict[str, Any] | None = None
    slab_states: list[WeightedState] = []
    slab_summaries: list[dict[str, Any]] = []
    incoming = dict(boundary.incoming)
    global_s_breakpoints = np.asarray(
        [
            mesh.s.segments[0].left,
            *[segment.right for segment in mesh.s.segments],
        ]
    )
    t_breakpoints = np.asarray(
        [
            mesh.t.segments[0].left,
            *[segment.right for segment in mesh.t.segments],
        ]
    )
    global_sweep = 0
    for slab_index in range(len(global_s_breakpoints) - 1):
        local_mesh = DoubleSqrtLGLMesh.create(
            t_breakpoints,
            resolution.degree,
            global_s_breakpoints[slab_index : slab_index + 2],
            resolution.degree,
        )
        local_outgoing = outgoing_segment(
            boundary.outgoing,
            mesh.v,
            local_mesh.v,
        )
        local_boundary = slab_boundary(local_outgoing, incoming)
        state = initial_iterate(grid, local_mesh.v, local_boundary)
        slab_records: list[dict[str, float | int]] = []
        for sweep in range(1, resolution.sweeps + 1):
            previous = state
            candidate, context = raw_picard_candidate(
                grid,
                local_mesh,
                angular,
                state,
                local_boundary,
                metric_substeps=resolution.metric_substeps,
            )
            candidate_update = weighted_update_norm(candidate, previous)
            state = relaxed_next_iterate(
                previous,
                candidate,
                local_boundary,
                resolution.relaxation,
            )
            state.validate(grid.frames)
            local_boundary.verify_unchanged()
            assert_characteristic_traces(state, local_boundary)
            global_sweep += 1
            record = {
                "slab": slab_index + 1,
                "slab_sweep": sweep,
                "sweep": global_sweep,
                "unrelaxed_candidate_update": candidate_update,
                "weighted_update": weighted_update_norm(state, previous),
                "maximum_update_map": float(
                    np.max(weighted_update_map(state, previous))
                ),
            }
            records.append(record)
            slab_records.append(record)
            if record["weighted_update"] <= resolution.tolerance:
                break
        if context is None:
            raise AssertionError("at least one Picard sweep is required")
        terminal_update = float(slab_records[-1]["weighted_update"])
        if terminal_update > resolution.tolerance:
            raise RuntimeError(
                f"slab {slab_index + 1} did not settle: "
                f"update={terminal_update:.6e}, "
                f"tolerance={resolution.tolerance:.6e}"
            )
        slab_states.append(state)
        slab_summaries.append(
            {
                "slab": slab_index + 1,
                "v_interval": [
                    float(local_mesh.v[0]),
                    float(local_mesh.v[-1]),
                ],
                "boundary_digest": local_boundary.digest,
                "terminal_weighted_update": terminal_update,
                "settled": True,
                "incoming_trace_errors": incoming_trace_errors(
                    state, local_boundary
                ),
                "last_raw_trace_errors_before_restore": context[
                    "trace_errors_before_restore"
                ],
                "last_raw_trace_errors_after_restore": context[
                    "trace_errors_after_restore"
                ],
            }
        )
        print(
            f"{resolution.name}: settled slab {slab_index + 1}/"
            f"{len(global_s_breakpoints) - 1} at update "
            f"{terminal_update:.3e}",
            flush=True,
        )
        incoming = terminal_incoming_data(state)

    state = concatenate_states(slab_states)
    state.validate(grid.frames)
    assert_characteristic_traces(state, boundary)
    np.testing.assert_allclose(
        np.concatenate(
            [
                (
                    0.5
                    * CompositeLGLMesh.create(
                        global_s_breakpoints[index : index + 2],
                        resolution.degree,
                    ).nodes**2
                )
                if index == 0
                else (
                    0.5
                    * CompositeLGLMesh.create(
                        global_s_breakpoints[index : index + 2],
                        resolution.degree,
                    ).nodes[1:] ** 2
                )
                for index in range(len(global_s_breakpoints) - 1)
            ]
        ),
        mesh.v,
        atol=5.0e-15,
        rtol=0.0,
    )
    values = components(
        grid,
        state,
        mesh.u,
        mesh.v,
        mode="fresh",
        scalar_coordinates=mesh,
        include_gauss_curvature=True,
    )
    maps = _l2_maps(grid, state, values)
    requested_names = ("r33", "r34", "r3", "r4", "rhat", "rR")
    total = sum(maps[name] for name in requested_names)
    full_total = total + maps["r44"]
    halo = max(1, resolution.degree // 3)
    protected = composite_uv_mask(
        total.shape[0],
        total.shape[1],
        element_degree=resolution.degree,
        interface_halo=halo,
    )
    open_grid = np.zeros(total.shape, dtype=bool)
    open_grid[1:-1, 1:-1] = True
    maxima = {
        name: {
            "all_nodes_including_characteristic_faces": float(np.max(value)),
            "open_grid": float(np.max(value[open_grid])),
            "protected_interior": float(np.max(value[protected])),
        }
        for name, value in {**maps, "r": total, "r_with_r44": full_total}.items()
    }
    state.save(
        output / "final-state.npz",
        u=mesh.u,
        v=mesh.v,
        extra={
            **{f"residual_{name}": value for name, value in maps.items()},
            "residual_requested_sum": total,
            "residual_sum_with_r44": full_total,
            "residual_reliability_mask": protected,
        },
    )
    np.savez_compressed(
        output / "residual-maps.npz",
        u=mesh.u,
        v=mesh.v,
        **maps,
        r=total,
        r_with_r44=full_total,
        reliability_mask=protected,
    )
    region_counts = plot_regions(
        output / "convergence-regions.png", mesh, total, protected
    )
    summary = {
        "schema": "nee-official-experiment-05-level-v1",
        "case_id": resolution.name,
        "resolution": resolution.__dict__,
        "mesh": mesh.diagnostics(),
        "angular": angular.diagnostics(),
        "tensor_certificates": certificates,
        "boundary_digest": boundary.digest,
        "incoming_trace_errors": incoming_trace_errors(state, boundary),
        "slabs": slab_summaries,
        "last_raw_trace_errors_before_restore": context[
            "trace_errors_before_restore"
        ],
        "last_raw_trace_errors_after_restore": context[
            "trace_errors_after_restore"
        ],
        "picard": records,
        "residual_semantics": {
            "r": "r33+r34+r3+r4+rhat+rR",
            "r4_weight": "Omega Ric_4A (the displayed identity's weight)",
            "r44": (
                "fresh independent Raychaudhuri residual, reported separately "
                "and not included in the requested six-term region plot"
            ),
            "protected_halo_nodes": halo,
            "reliability_mask": (
                "exclude the complete first and last coordinate elements "
                "and the stated halo on both sides of every remaining "
                "composite-element interface"
            ),
            "reliability_node_count": int(np.count_nonzero(protected)),
        },
        "residual_maxima": maxima,
        "region_counts": region_counts,
        "minimum_metric_eigenvalue": state.validate(grid.frames)[
            "minimum_metric_eigenvalue"
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def production_resolution() -> Resolution:
    """Return the reported 61x61x300, retained-L=8 configuration."""

    return Resolution(
        "level-3",
        6,
        10,
        8,
        14,
        300,
        40,
        30,
        12,
        6,
        1.0,
        1.0e-8,
    )


def run(
    output: Path,
    *,
    quick: bool,
    reported_only: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"immutable run directory exists: {output}")
    output.mkdir(parents=True)
    revision = source_revision()
    campaign_levels = (
        Resolution(
            "level-1", 6, 4, 3, 6, 100, 24, 20, 4, 2, 1.0, 1.0e-8
        ),
        Resolution(
            "level-2", 6, 5, 4, 8, 180, 32, 20, 6, 3, 1.0, 1.0e-8
        ),
        (
            Resolution(
                "level-3", 6, 7, 5, 10, 260, 36, 20, 8, 4, 1.0, 1.0e-8
            )
            if quick
            else production_resolution()
        ),
    )
    levels = (
        (production_resolution(),)
        if reported_only
        else campaign_levels
    )
    summaries = []
    for resolution in levels:
        summaries.append(run_level(output / resolution.name, resolution))
    high = summaries[-1]
    aggregate = {
        "schema": "nee-vacuum-crossed-pulses-aggregate-1",
        "experiment": 5,
        "configuration": "crossed characteristic shears",
        "domain": {
            "u": [-1.0, -0.5],
            "v": [0.0, 0.5],
        },
        "levels": summaries,
        "reported_level": high["case_id"],
        "reported_summary": high,
        "source_revision": revision,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
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
    parser.add_argument(
        "--reported-only",
        action="store_true",
        help="run only the reported 61x61x300, retained-L=8 configuration",
    )
    args = parser.parse_args()
    run(
        args.output,
        quick=args.quick,
        reported_only=args.reported_only,
    )


if __name__ == "__main__":
    main()
