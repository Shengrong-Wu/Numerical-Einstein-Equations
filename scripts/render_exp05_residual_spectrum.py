"""Render the dense Experiment 5 six-component Ricci residual spectrum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


Array = np.ndarray


from residual_plotting import dense_open_log_residual


def render(
    result_directory: Path,
    output: Path,
    *,
    samples_per_element: int,
) -> None:
    summary = json.loads(
        (result_directory / "summary.json").read_text(encoding="utf-8")
    )
    with np.load(result_directory / "residual-maps.npz") as artifact:
        u = np.asarray(artifact["u"])
        v = np.asarray(artifact["v"])
        residual = np.asarray(artifact["r"])

    resolution = summary["resolution"]
    degree = int(resolution["degree"])
    elements = int(resolution["elements"])
    t = np.sqrt(np.maximum(2.0 * (u + 1.0), 0.0))
    s = np.sqrt(np.maximum(2.0 * v, 0.0))
    indices = [np.arange(i*degree, (i+1)*degree+1) for i in range(elements)]
    dense_t, dense_s, dense_log = dense_open_log_residual(
        residual, t, s, indices, indices, samples_per_element)
    dense_u = -1.0 + 0.5 * dense_t**2
    dense_v = 0.5 * dense_s**2

    figure, axis = plt.subplots(figsize=(10.8, 7.0), constrained_layout=True)
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
    square_root_breakpoints = np.linspace(0.0, 1.0, elements + 1)
    u_breakpoints = -1.0 + 0.5 * square_root_breakpoints**2
    v_breakpoints = 0.5 * square_root_breakpoints**2
    for value in v_breakpoints[1:-1]:
        axis.axvline(value, color="white", linewidth=0.45, alpha=0.55)
    for value in u_breakpoints[1:-1]:
        axis.axhline(value, color="white", linewidth=0.45, alpha=0.55)

    colorbar = figure.colorbar(spectrum, ax=axis, pad=0.025)
    colorbar.set_label(r"$E_{\mathrm{Ric}}(u,v)$")
    colorbar.set_ticks(np.arange(-5.0, 4.0))
    colorbar.set_ticklabels(
        [rf"$10^{{{power}}}$" for power in range(-5, 4)]
    )
    axis.set(
        xlim=(0.0, 0.5),
        ylim=(-1.0, -0.5),
        xlabel=r"$v$",
        ylabel=r"$u$",
        title=(
            rf"Aggregate Ricci residual spectrum "
            rf"($L={resolution['retained']}$)"
            "\n"
            r"white contours: $10^{-4},10^{-3},10^{-2},10^{-1}$; "
            "thin lines: element interfaces"
        ),
    )
    axis.text(
        0.01,
        0.99,
        (
            f"{samples_per_element} samples/element/axis; "
            "full open-grid reconstruction"
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
    parser.add_argument("--result-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-element", type=int, default=96)
    arguments = parser.parse_args()
    if arguments.samples_per_element < 16:
        parser.error("--samples-per-element must be at least 16")
    render(
        arguments.result_directory,
        arguments.output,
        samples_per_element=arguments.samples_per_element,
    )
    print(arguments.output)


if __name__ == "__main__":
    main()
