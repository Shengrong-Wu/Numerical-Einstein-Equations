"""Generate compact documentation figures from completed official runs."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from nee.diagnostics.trapped_sections import trapped_sections
from nee.experiments.exp05_vacuum_crossed_pulses.campaign import plot_regions


ROOT = Path(__file__).resolve().parents[1]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _exp08_completed() -> bool:
    paths = (
        ROOT / "results/exp08/summary.json",
        ROOT / "results/exp08-angular-control/summary.json",
    )
    return all(
        path.exists() and _json(path).get("terminal_status") == "completed"
        for path in paths
    )


def crossed_pulse_regions() -> None:
    source = ROOT / "results/exp05/data/level-3/residual-maps.npz"
    destination = ROOT / "docs/experiments/exp05/results/convergence-regions.png"
    with np.load(source) as archive:
        mesh = SimpleNamespace(u=archive["u"], v=archive["v"])
        total = archive["r"]
        reliable = archive["reliability_mask"].astype(bool)
    plot_regions(destination, mesh, total, reliable)


def pulse_updates() -> None:
    standard = _json(ROOT / "results/exp08/summary.json")
    control = _json(ROOT / "results/exp08-angular-control/summary.json")
    destination = ROOT / "docs/experiments/exp08/results/picard-updates.png"
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    for axis, key, title in zip(
        axes,
        ("base_sweeps", "continued_sweeps"),
        ("Base domain", "One-element continuation"),
        strict=True,
    ):
        for payload, label, marker in (
            (standard, "L=5, W=10, 170 points", "o"),
            (control, "L=7, W=14, 300 points", "s"),
        ):
            values = payload[key]
            axis.semilogy(
                [record["sweep"] for record in values],
                [record["update"] for record in values],
                marker=marker,
                markersize=3.5,
                linewidth=1.5,
                label=label,
            )
        axis.axhline(1.0e-4, color="0.35", linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_xlabel("Picard sweep")
        axis.grid(True, which="both", color="0.88", linewidth=0.6)
    axes[0].set_ylabel("weighted update norm")
    axes[1].legend(frameon=False, fontsize=8)
    figure.suptitle("Scalar-pulse Picard updates")
    figure.tight_layout()
    figure.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(figure)


def trapped_sign_comparison() -> None:
    standard = _json(ROOT / "results/exp08/summary.json")["sign_audit"]
    control = _json(ROOT / "results/exp08-angular-control/summary.json")["sign_audit"]
    with np.load(ROOT / "results/exp08/final-state.npz") as archive:
        u = archive["u"].copy()
        v = archive["v"].copy()

    def protected(payload: dict) -> np.ndarray:
        _, mask = trapped_sections(
            np.asarray(payload["outgoing_supremum"]),
            np.asarray(payload["incoming_supremum"]),
            u_endpoint_halo=3,
        )
        return mask

    low = protected(standard)
    high = protected(control)
    categories = low.astype(np.int8) + 2 * high.astype(np.int8)
    colors = ["#e7e7e7", "#d95f02", "#1b9e77", "#4c78a8"]
    labels = (
        "neither band",
        "standard only",
        "control only",
        "both bands",
    )
    u_slice = slice(88, None)
    v_slice = slice(25, None)
    figure, axis = plt.subplots(figsize=(9.2, 4.8))
    axis.pcolormesh(
        u[u_slice],
        v[v_slice],
        categories[u_slice, v_slice].T,
        cmap=ListedColormap(colors),
        shading="nearest",
        vmin=0,
        vmax=3,
    )
    axis.set_xlabel("u")
    axis.set_ylabel("v")
    axis.set_title("Protected trapped-section sign masks near the continued endpoint")
    axis.legend(
        handles=[Patch(facecolor=color, label=label) for color, label in zip(colors, labels, strict=True)],
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    figure.tight_layout()
    destination = ROOT / "docs/experiments/exp08/results/trapped-sign-comparison.png"
    figure.savefig(destination, dpi=220, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    crossed_pulse_regions()
    if _exp08_completed():
        pulse_updates()
        trapped_sign_comparison()
    else:
        for name in ("picard-updates.png", "trapped-sign-comparison.png"):
            (ROOT / "docs/experiments/exp08/results" / name).unlink(
                missing_ok=True
            )
        print("skipping Experiment 8 figures: production runs are not completed")


if __name__ == "__main__":
    main()
