"""Generate/load initial data and run the configurable ESE Picard iteration."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from .scalar_config import ExperimentConfig
from .scalar_coordinates import mesh_from_config
from .scalar_iteration import (
    ESEState,
    initial_state,
    picard_step,
    update_map,
    update_norm,
)
from .scalar_residual import components, l2_maps, safe_summary
from .scalar_initial_data import (
    InitialDataBundle,
    build_angular,
    construct_initial_data,
    write_summary,
)


Array = np.ndarray
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]


def numerical_fingerprint() -> str:
    digest = hashlib.sha256()
    paths = [
        HERE / name
        for name in (
            "scalar_config.py",
            "scalar_coordinates.py",
            "scalar_initial_data.py",
            "scalar_iteration.py",
            "scalar_residual.py",
            "scalar_run.py",
        )
    ]
    paths.extend(
        HERE / name
        for name in (
            "spherical_harmonics.py",
            "lgl.py",
            "vacuum_iteration.py",
            "vacuum_residual.py",
            "sphere.py",
            "ricci_residual.py",
            "coordinate_quadrature.py",
            "vacuum_state.py",
            "coordinate_differentiation.py",
        )
    )
    for path in paths:
        digest.update(str(path.relative_to(PROJECT)).encode())
        digest.update(path.read_bytes())
    return f"sha256:{digest.hexdigest()}"


def save_state(
    path: Path,
    state: ESEState,
    update_maps: list[Array],
    residual_maps: dict[str, list[Array]],
    u: Array,
    v: Array,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: getattr(state, name)
        for name in (
            "metric",
            "omega",
            "zeta_up",
            "shift",
            "q",
            "shear",
            "weighted_chib",
            "weighted_omega",
            "weighted_omegab",
            "phi",
            "scalar_p",
            "incoming_scalar",
        )
    }
    arrays["u"] = u
    arrays["v"] = v
    arrays["update_maps"] = np.stack(update_maps)
    for name, history in residual_maps.items():
        arrays[f"residual_maps__{name}"] = np.stack(history)
    np.savez_compressed(path, **arrays)


def run(
    scalar_config: ExperimentConfig,
    output_directory: Path,
    initial_data_path: Path,
    regenerate_initial_data: bool,
) -> dict[str, object]:
    if regenerate_initial_data or not initial_data_path.exists():
        bundle, grid, angular, mesh = construct_initial_data(scalar_config)
        bundle.save(initial_data_path)
        write_summary(
            bundle,
            initial_data_path.with_name(
                f"{initial_data_path.stem}-summary.json"
            ),
        )
    else:
        bundle = InitialDataBundle.load(initial_data_path)
        grid, angular = build_angular(scalar_config)
        mesh = mesh_from_config(scalar_config.scalar_coordinates)
    state = initial_state(bundle, angular)
    iteration_summaries: list[dict[str, object]] = []
    update_maps: list[Array] = []
    residual_maps: dict[str, list[Array]] = {}
    for iteration in range(1, scalar_config.solver.picard_iterations + 1):
        previous = state
        state, context = picard_step(
            grid,
            angular,
            mesh,
            previous,
            bundle,
            metric_substeps=scalar_config.solver.metric_substeps,
        )
        change = update_norm(state, previous)
        change_map = update_map(state, previous)
        residual_values = components(
            grid, state, previous, context, mesh
        )
        maps = l2_maps(grid, state, mesh, residual_values)
        summary = safe_summary(
            maps,
            scalar_config.solver.derivative_halo_u,
            scalar_config.solver.derivative_halo_v,
            mesh,
        )
        update_maps.append(change_map)
        for name, value in maps.items():
            residual_maps.setdefault(name, []).append(value)
        iteration_summary = {
            "iteration": iteration,
            "update_norm": change,
            "update_map_maximum": float(np.max(change_map)),
            "residual": summary,
            "minimum_q": float(np.min(state.q)),
            "maximum_q": float(np.max(state.q)),
            "minimum_omega": float(np.min(state.omega)),
            "minimum_metric_eigenvalue": float(
                np.min(
                    np.linalg.eigvalsh(
                        np.einsum(
                            "nia,n...ij,njb->n...ab",
                            grid.frames,
                            state.metric,
                            grid.frames,
                        )
                    )
                )
            ),
        }
        iteration_summaries.append(iteration_summary)
        print(
            f"iteration {iteration}: update={change:.6e}, "
            f"combined median={summary['combined']['median']:.6e}, "
            f"max={summary['combined']['maximum']:.6e}",
            flush=True,
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    save_state(
        output_directory / "final-state.npz",
        state,
        update_maps,
        residual_maps,
        mesh.u,
        mesh.v,
    )
    result: dict[str, object] = {
        "schema": "nee-scalar-field-run-1",
        "scalar_config": scalar_config.to_dict(),
        "numerical_fingerprint": numerical_fingerprint(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "scalar_initial_data": {
            "path": str(initial_data_path),
            "schema": bundle.metadata.get("schema"),
            "minimum_incoming_scalar_radicand": bundle.metadata.get(
                "minimum_incoming_scalar_radicand"
            ),
            "corner_mismatches": bundle.metadata.get("corner_mismatches"),
        },
        "angular": angular.diagnostics(),
        "scalar_coordinates": mesh.diagnostics(),
        "iterations": iteration_summaries,
        "converged_update": bool(
            len(iteration_summaries) >= 2
            and iteration_summaries[-1]["update_norm"]
            < iteration_summaries[-2]["update_norm"]
        ),
        "limitations": [
            "This run uses RK4 stage solves on common LGL nodes; it has not "
            "yet been upgraded to the EVE tau-SDC production solver.",
            "The literal sin(theta) coefficients are not smooth at the poles "
            "and therefore do not admit spectral angular convergence.",
        ],
    }
    (output_directory / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar_config", default="scalar_config/default.json")
    parser.add_argument("--output-directory", default="results/default")
    parser.add_argument(
        "--initial-data", default="initial-data/default.npz"
    )
    parser.add_argument("--regenerate-initial-data", action="store_true")
    parser.add_argument("--iterations", type=int)
    args = parser.parse_args()
    scalar_config = ExperimentConfig.load(args.scalar_config)
    if args.iterations is not None:
        value = scalar_config.to_dict()
        value["solver"]["picard_iterations"] = args.iterations
        scalar_config = ExperimentConfig.from_dict(value)
    run(
        scalar_config,
        Path(args.output_directory),
        Path(args.scalar_initial_data),
        args.regenerate_initial_data,
    )


if __name__ == "__main__":
    main()
