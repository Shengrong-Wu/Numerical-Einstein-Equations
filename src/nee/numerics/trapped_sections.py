"""Independent angular sign audit for the outgoing null expansion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .scalar_config import ExperimentConfig
from .sphere import (
    fibonacci_sphere,
    spherical_harmonic_collocation,
    tangent_inverse,
    tensor_trace,
)
from .scalar_initial_data import build_angular


def run(
    state_path: Path,
    config_path: Path,
    output_directory: Path,
    target_count: int,
) -> dict[str, object]:
    scalar_config = ExperimentConfig.load(config_path)
    source_grid, angular = build_angular(scalar_config)
    with np.load(state_path, allow_pickle=False) as archive:
        u = archive["u"]
        v = archive["v"]
        Omega_trchi = archive["Omega_trchi"]
        Omega = archive["Omega"]
        g = archive["g"]
        Omega_chib = archive["Omega_chib"]
        terminal_update = (
            archive["update_maps"][-1]
            if "update_maps" in archive.files
            else None
        )
        terminal_residual = (
            archive["residual_maps__combined"][-1]
            if "residual_maps__combined" in archive.files
            else None
        )

    coefficients = angular.scalar.analyze_retained(Omega_trchi)
    target_points = fibonacci_sphere(target_count)
    _, target_basis, target_condition = spherical_harmonic_collocation(
        target_points, scalar_config.angular.retained_degree
    )
    target_Omega_trchi = np.einsum(
        "nm,muv->nuv", target_basis, coefficients, optimize=True
    )
    target_infimum = np.min(target_Omega_trchi, axis=0)
    target_supremum = np.max(target_Omega_trchi, axis=0)

    inverse_g = tangent_inverse(source_grid, g)
    tr_chib = tensor_trace(Omega_chib, inverse_g) / Omega
    incoming_supremum = np.max(tr_chib, axis=0)
    trapped = (target_supremum <= 0.0) & (incoming_supremum < 0.0)
    candidate = np.unravel_index(
        int(np.argmin(target_supremum)), target_supremum.shape
    )
    trapped_indices = np.argwhere(trapped)

    def section_record(index: tuple[int, int]) -> dict[str, object]:
        record: dict[str, object] = {
            "u_index": index[0],
            "v_index": index[1],
            "u": float(u[index[0]]),
            "v": float(v[index[1]]),
            "supremum_Omega_trchi": float(target_supremum[index]),
            "infimum_Omega_trchi": float(target_infimum[index]),
            "sampled_supremum_tr_chib": float(incoming_supremum[index]),
        }
        if terminal_update is not None:
            record["terminal_local_update"] = float(terminal_update[index])
        if terminal_residual is not None:
            record["terminal_combined_residual"] = float(
                terminal_residual[index]
            )
        return record

    first = None
    if len(trapped_indices):
        first_index = min(
            (tuple(map(int, value)) for value in trapped_indices),
            key=lambda value: (value[1], value[0]),
        )
        first = section_record(first_index)

    halo = scalar_config.solver.derivative_halo_u
    interior_trapped = trapped.copy()
    if halo:
        interior_trapped[:halo] = False
        interior_trapped[-halo:] = False
    interior_indices = np.argwhere(interior_trapped)
    first_interior = None
    if len(interior_indices):
        first_interior_index = min(
            (tuple(map(int, value)) for value in interior_indices),
            key=lambda value: (value[1], value[0]),
        )
        first_interior = section_record(first_interior_index)

    summary: dict[str, object] = {
        "schema": "nee-trapped-section-audit-1",
        "state": str(state_path.resolve()),
        "scalar_config": str(config_path.resolve()),
        "criterion": (
            "sup_S Omega_trchi<=0 and sup_S tr(chib)<0; "
            "Omega_trchi=Omega tr(chi) has the same sign as tr(chi)"
        ),
        "source_sphere_count": source_grid.count,
        "independent_sphere_count": target_count,
        "independent_scalar_basis_condition": float(target_condition),
        "minimum_lapse": float(np.min(Omega)),
        "sampled_maximum_incoming_expansion": float(np.max(tr_chib)),
        "trapped_section_count": int(np.count_nonzero(trapped)),
        "first_trapped_section": first,
        "coordinate_endpoint_halo_u": halo,
        "interior_trapped_section_count": int(
            np.count_nonzero(interior_trapped)
        ),
        "first_interior_trapped_section": first_interior,
        "most_focused_section": {
            **section_record((int(candidate[0]), int(candidate[1]))),
            "sampled_supremum_tr_chi": float(
                np.max((Omega_trchi / Omega)[:, candidate[0], candidate[1]])
            ),
        },
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "trapped-surface-audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    plotted = (-u[:, None]) * target_supremum
    figure, axes = plt.subplots(
        1, 2, figsize=(12.0, 5.0), constrained_layout=True
    )
    scale = max(
        float(np.max(np.abs(plotted))),
        np.finfo(float).tiny,
    )
    image = None
    for axis in axes:
        image = axis.pcolormesh(
            v,
            u,
            plotted,
            shading="auto",
            cmap="coolwarm",
            vmin=-scale,
            vmax=scale,
        )
        if float(np.min(target_supremum)) <= 0.0 <= float(
            np.max(target_supremum)
        ):
            axis.contour(
                v, u, target_supremum, levels=[0.0], colors="black"
            )
        if np.any(trapped):
            axis.contourf(
                v,
                u,
                trapped.astype(float),
                levels=[0.5, 1.5],
                colors=["none"],
                hatches=["////"],
            )
        axis.set_xlabel(r"$v$")
        axis.set_ylabel(r"$u$")
    axes[0].set_title("Full continuation domain")
    zoom_bottom = min(-0.06, float(u[-1]))
    axes[1].set_ylim(zoom_bottom, float(u[-1]))
    axes[1].set_title("Near-singularity sign crossing")
    figure.suptitle(
        r"Independent angular $(-u)\sup_{S_{u,v}}"
        r"(\Omega\,\mathrm{tr}\chi)$"
    )
    assert image is not None
    figure.colorbar(image, ax=axes, label=r"$(-u)\sup_S Omega_trchi$")
    figure.savefig(output_directory / "trapped-surface-region.png", dpi=220)
    figure.savefig(output_directory / "trapped-surface-region.pdf")
    plt.close(figure)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--scalar_config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=1000)
    args = parser.parse_args()
    result = run(
        args.state,
        args.scalar_config,
        args.output_directory,
        args.target_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
