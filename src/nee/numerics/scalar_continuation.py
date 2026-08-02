"""Continue a converged ESE state by appending whole tau-LGL elements."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .scalar_config import ExperimentConfig
from .scalar_coordinates import mesh_from_config
from .scalar_iteration import (
    ESEState,
    impose_characteristic_faces,
    initial_state,
    picard_step,
    update_map,
    update_norm,
    validate_state,
)
from .scalar_residual import components, l2_maps, safe_summary
from .scalar_initial_data import construct_initial_data, write_summary
from .scalar_run import numerical_fingerprint, save_state


STATE_FIELDS = (
    "g",
    "Omega",
    "zeta",
    "b",
    "Omega_trchi",
    "Omega_chih",
    "Omega_chib",
    "Omega_omega",
    "Omega_omegab",
    "phi",
    "Omega_e4phi",
    "Omega_e3phi",
)


def load_state(path: Path) -> tuple[ESEState, np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.array(archive[name], copy=True) for name in STATE_FIELDS}
        return (
            ESEState(**values),
            np.array(archive["u"], copy=True),
            np.array(archive["v"], copy=True),
        )


def prolong_appended_tau_elements(
    source: ESEState,
    source_u: np.ndarray,
    source_v: np.ndarray,
    seed: ESEState,
    target_u: np.ndarray,
    target_v: np.ndarray,
    bundle: object,
) -> tuple[ESEState, dict[str, object]]:
    """Copy the common prefix exactly and use baseline increments on the tail."""

    old_count = len(source_u)
    if old_count >= len(target_u):
        raise ValueError("target mesh must append at least one u node")
    if not np.array_equal(source_v, target_v):
        raise ValueError("u continuation requires the same v nodes")
    if not np.allclose(
        source_u, target_u[:old_count], rtol=2.0e-14, atol=2.0e-15
    ):
        raise ValueError("target tau mesh does not preserve the source prefix")

    values: dict[str, np.ndarray] = {}
    prefix_mismatch: dict[str, float] = {}
    for name in STATE_FIELDS:
        old = np.asarray(getattr(source, name))
        baseline = np.asarray(getattr(seed, name))
        target = np.array(baseline, copy=True)
        target[:, :old_count] = old
        correction = old[:, -1:] - baseline[:, old_count - 1 : old_count]
        target[:, old_count:] += correction
        values[name] = target
        prefix_mismatch[name] = float(
            np.max(np.abs(target[:, :old_count] - old))
        )

    prolonged = ESEState(**values)
    impose_characteristic_faces(prolonged, bundle)
    validate_state(None, prolonged)
    return prolonged, {
        "method": (
            "exact common tau prefix plus the fresh characteristic seed's "
            "tail increment"
        ),
        "source_u_count": old_count,
        "target_u_count": len(target_u),
        "added_u_count": len(target_u) - old_count,
        "source_u_right": float(source_u[-1]),
        "target_u_right": float(target_u[-1]),
        "prefix_maximum_mismatch_by_field": prefix_mismatch,
        "prefix_bitwise_preserved_before_face_imposition": bool(
            max(prefix_mismatch.values()) == 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar_config", required=True)
    parser.add_argument("--source-state", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--initial-data", required=True)
    args = parser.parse_args()

    scalar_config = ExperimentConfig.load(args.scalar_config)
    output = Path(args.output_directory)
    initial_data_path = Path(args.scalar_initial_data)
    bundle, grid, angular, mesh = construct_initial_data(scalar_config)
    initial_data_path.parent.mkdir(parents=True, exist_ok=True)
    bundle.save(initial_data_path)
    write_summary(
        bundle,
        initial_data_path.with_name(f"{initial_data_path.stem}-summary.json"),
    )

    source, source_u, source_v = load_state(Path(args.source_state))
    seed = initial_state(bundle, angular)
    state, prolongation = prolong_appended_tau_elements(
        source, source_u, source_v, seed, mesh.u, mesh.v, bundle
    )

    iteration_summaries: list[dict[str, object]] = []
    update_maps: list[np.ndarray] = []
    residual_maps: dict[str, list[np.ndarray]] = {}
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
        residual_values = components(grid, state, previous, context, mesh)
        maps = l2_maps(grid, state, mesh, residual_values)
        residual = safe_summary(
            maps,
            scalar_config.solver.derivative_halo_u,
            scalar_config.solver.derivative_halo_v,
            mesh,
        )
        update_maps.append(change_map)
        for name, value in maps.items():
            residual_maps.setdefault(name, []).append(value)
        iteration_summaries.append(
            {
                "iteration": iteration,
                "update_norm": change,
                "update_map_maximum": float(np.max(change_map)),
                "residual": residual,
                "minimum_Omega_trchi": float(np.min(state.Omega_trchi)),
                "maximum_Omega_trchi": float(np.max(state.Omega_trchi)),
                "minimum_omega": float(np.min(state.Omega)),
            }
        )
        print(
            f"iteration {iteration}: update={change:.6e}, "
            f"combined median={residual['combined']['median']:.6e}, "
            f"max={residual['combined']['maximum']:.6e}",
            flush=True,
        )

    output.mkdir(parents=True, exist_ok=True)
    save_state(
        output / "final-state.npz",
        state,
        update_maps,
        residual_maps,
        mesh.u,
        mesh.v,
    )
    summary = {
        "schema": "nee-scalar-u-continuation-1",
        "scalar_config": scalar_config.to_dict(),
        "source_state": str(Path(args.source_state).resolve()),
        "numerical_fingerprint": numerical_fingerprint(),
        "prolongation": prolongation,
        "iterations": iteration_summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
