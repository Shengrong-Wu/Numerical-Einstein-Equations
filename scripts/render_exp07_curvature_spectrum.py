"""Render the Experiment 7 Einstein--scalar curvature-residual spectrum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from nee.numerics.lgl import CompositeLGLMesh


Array = np.ndarray


def dense_axis(
    mesh: CompositeLGLMesh,
    samples_per_element: int,
) -> tuple[Array, list[Array]]:
    pieces: list[Array] = []
    matrices: list[Array] = []
    for element, segment in enumerate(mesh.segments):
        targets = np.linspace(
            segment.left,
            segment.right,
            samples_per_element,
            endpoint=(element == len(mesh.segments) - 1),
        )
        pieces.append(targets)
        matrices.append(segment.interpolation_matrix(targets))
    return np.concatenate(pieces), matrices


def dense_log_residual(
    residual: Array,
    tau_mesh: CompositeLGLMesh,
    s_mesh: CompositeLGLMesh,
    samples_per_element: int,
) -> tuple[Array, Array, Array]:
    dense_tau, tau_matrices = dense_axis(
        tau_mesh, samples_per_element
    )
    dense_s, s_matrices = dense_axis(s_mesh, samples_per_element)
    log_nodes = np.log10(np.maximum(residual, 1.0e-7))
    dense = np.empty((len(dense_tau), len(dense_s)))
    tau_offset = 0
    for tau_index, tau_matrix in zip(
        tau_mesh.indices, tau_matrices, strict=True
    ):
        tau_count = len(tau_matrix)
        s_offset = 0
        for s_index, s_matrix in zip(
            s_mesh.indices, s_matrices, strict=True
        ):
            s_count = len(s_matrix)
            local = log_nodes[np.ix_(tau_index, s_index)]
            dense[
                tau_offset : tau_offset + tau_count,
                s_offset : s_offset + s_count,
            ] = tau_matrix @ local @ s_matrix.T
            s_offset += s_count
        tau_offset += tau_count
    return dense_tau, dense_s, dense


def render(
    summary_path: Path,
    output: Path,
    *,
    samples_per_element: int,
) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    audit = summary["independent_first_order_residual"]
    coordinates = summary["scalar_coordinates"]
    residual = np.asarray(audit["einstein_section_L2_map"])

    increment = 3
    tau_breakpoints = np.asarray(coordinates["tau_breakpoints"])
    s_breakpoints = np.asarray(coordinates["s_breakpoints"])
    tau_degrees = np.asarray(coordinates["tau_degrees"]) + increment
    s_degrees = np.asarray(coordinates["s_degrees"]) + increment
    tau_mesh = CompositeLGLMesh.create(tau_breakpoints, tau_degrees)
    s_mesh = CompositeLGLMesh.create(s_breakpoints, s_degrees)
    if residual.shape != (len(tau_mesh.nodes), len(s_mesh.nodes)):
        raise ValueError(
            "the residual map does not match the independent audit overgrid"
        )

    dense_tau, dense_s, dense_log = dense_log_residual(
        residual,
        tau_mesh,
        s_mesh,
        samples_per_element,
    )
    dense_u = -np.exp(-dense_tau)
    v_max = float(coordinates["v_max"])
    delta = float(coordinates["fractional_power"])
    dense_v = v_max * dense_s ** (1.0 / delta)

    figure, axis = plt.subplots(
        figsize=(10.8, 7.0), constrained_layout=True
    )
    spectrum = axis.pcolormesh(
        dense_v,
        dense_u,
        dense_log,
        shading="auto",
        cmap="turbo",
        vmin=-5.0,
        vmax=3.0,
        rasterized=True,
    )
    axis.contour(
        dense_v,
        dense_u,
        dense_log,
        levels=(-4.0, -3.0, -2.0, -1.0),
        colors="white",
        linewidths=0.75,
        alpha=0.9,
    )
    for value in v_max * s_breakpoints[1:-1] ** (1.0 / delta):
        axis.axvline(value, color="white", linewidth=0.45, alpha=0.55)
    for value in -np.exp(-tau_breakpoints[1:-1]):
        axis.axhline(value, color="white", linewidth=0.45, alpha=0.55)

    colorbar = figure.colorbar(spectrum, ax=axis, pad=0.025)
    colorbar.set_label(
        r"$E_{\mathrm{Ric}}(u,v)$"
    )
    colorbar.set_ticks(np.arange(-5.0, 4.0))
    colorbar.set_ticklabels(
        [rf"$10^{{{power}}}$" for power in range(-5, 4)]
    )
    retained = int(summary["angular"]["retained_degree"])
    axis.set(
        xlim=(0.0, v_max),
        ylim=(-1.0, -0.5),
        xlabel=r"$v$",
        ylabel=r"$u$",
        title=(
            rf"Einstein--scalar curvature-residual spectrum ($L={retained}$)"
            "\n"
            r"white contours: $10^{-4},10^{-3},10^{-2},10^{-1}$; "
            "thin lines: source-element interfaces"
        ),
    )
    axis.text(
        0.01,
        0.99,
        (
            f"{samples_per_element} samples/element/axis; "
            "independent four-metric audit"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        color="white",
        bbox={
            "facecolor": (0.0, 0.0, 0.0, 0.48),
            "edgecolor": "none",
            "pad": 4.0,
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-element", type=int, default=96)
    arguments = parser.parse_args()
    if arguments.samples_per_element < 16:
        parser.error("--samples-per-element must be at least 16")
    render(
        arguments.summary,
        arguments.output,
        samples_per_element=arguments.samples_per_element,
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
