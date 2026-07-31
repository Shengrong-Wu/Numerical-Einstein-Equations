"""Nonspherical Einstein--scalar characteristic data.

The free data use globally smooth real harmonics and are completed by the
Einstein--scalar characteristic constraints.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np



from nee.numerics import scalar_initial_data as idata  # noqa: E402
from nee.numerics import scalar_run as ese_run  # noqa: E402
from nee.numerics.scalar_config import (  # noqa: E402
    AngularConfig,
    CoordinateConfig,
    ExperimentConfig,
    InitialDataConfig,
    SolverConfig,
)
from nee.numerics.sphere import (  # noqa: E402
    connection_difference,
    lie_covariant_tensor,
    scalar_gradient,
    tangent_inverse,
    tensor_norm_sq,
    tensor_trace,
    tensor_tracefree,
    tracefree_symmetric_gradient,
)


Array = np.ndarray


def _rotation_z(angle: float) -> Array:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotated_harmonic_fields(
    points: Array, rotation: Array
) -> tuple[Array, Array, Array]:
    """Return normalized Y20, grad(Y21R), and 0.1 grad(Y22R)."""

    local = points @ rotation
    x, y, z = local.T
    y20 = 3.0 * z**2 - 1.0
    y20 /= np.max(np.abs(y20))

    ambient_21_local = np.column_stack([z, np.zeros_like(z), x])
    ambient_22_local = np.column_stack([2.0 * x, -2.0 * y, np.zeros_like(z)])
    ambient_21 = ambient_21_local @ rotation.T
    ambient_22 = ambient_22_local @ rotation.T
    grad_21 = ambient_21 - np.einsum(
        "ni,ni->n", ambient_21, points
    )[:, None] * points
    grad_22 = ambient_22 - np.einsum(
        "ni,ni->n", ambient_22, points
    )[:, None] * points
    grad_21 /= np.max(np.linalg.norm(grad_21, axis=1))
    grad_22 *= 0.1 / np.max(np.linalg.norm(grad_22, axis=1))
    return y20, grad_21, grad_22


def _plan_free_data(rotation: Array) -> Callable[..., dict[str, Array]]:
    def generator(
        grid: Any, mesh: Any, config: ExperimentConfig
    ) -> dict[str, Array]:
        data = config.scalar_initial_data
        u = mesh.u
        radius = -u
        y20, vector_b, _ = _rotated_harmonic_fields(grid.points, rotation)
        metric = (
            radius[None, :, None, None] ** 2
            * grid.projector[:, None, :, :]
        )
        omega = np.sqrt(
            radius[None, :] ** data.lapse_radial_power
            * (
                1.0
                + data.lapse_angular_amplitude * y20[:, None]
            )
        )
        shift = np.broadcast_to(
            data.shift_amplitude * vector_b[:, None, :],
            (grid.count, len(u), 3),
        ).copy()

        metric_u = mesh.differentiate_u(metric, axis=1)
        weighted_chib = 0.5 * (
            metric_u + lie_covariant_tensor(grid, shift, metric)
        )
        log_omega = np.log(omega)
        d3_log_omega = mesh.differentiate_u(log_omega, axis=1) + np.einsum(
            "nui,nui->nu", shift, scalar_gradient(grid, log_omega)
        )
        weighted_omegab = -0.5 * d3_log_omega
        inverse = tangent_inverse(grid, metric)
        weighted_trace = tensor_trace(weighted_chib, inverse)
        weighted_hat = tensor_tracefree(weighted_chib, metric, inverse)
        d3_trace = mesh.differentiate_u(weighted_trace, axis=1) + np.einsum(
            "nui,nui->nu",
            shift,
            scalar_gradient(grid, weighted_trace),
        )
        radicand = (
            -d3_trace
            - 0.5 * weighted_trace**2
            - tensor_norm_sq(weighted_hat, inverse)
            - 4.0 * weighted_omegab * weighted_trace
        )
        return {
            "metric": metric,
            "omega": omega,
            "shift": shift,
            "weighted_chib_exact": weighted_chib,
            "weighted_omegab_exact": weighted_omegab,
            "weighted_hatchib_exact": weighted_hat,
            "incoming_scalar_exact": np.sqrt(np.maximum(radicand, 0.0)),
        }

    return generator


def _plan_reference_shear(rotation: Array) -> Callable[..., Array]:
    def generator(grid: Any, amplitude: float, profile: str) -> Array:
        del profile
        _, _, vector_chi = _rotated_harmonic_fields(grid.points, rotation)
        vector_chi = (amplitude / 0.1) * vector_chi
        metric = grid.projector
        inverse = tangent_inverse(grid, metric)
        difference, _ = connection_difference(grid, metric, inverse)
        # On the round unit sphere lowering an ambient tangent vector with the
        # projector leaves its components unchanged.
        return tracefree_symmetric_gradient(
            grid, vector_chi, metric, difference, inverse
        )

    return generator


def _scaled_construct(
    original: Callable[..., Any],
    lambda_phi: float,
    lambda_chi: float,
) -> Callable[..., Any]:
    def construct(config: ExperimentConfig) -> Any:
        bundle, grid, angular, mesh = original(config)
        zero_shear = abs(lambda_chi) <= 1.0e-15
        if abs(lambda_phi - 1.0) <= 1.0e-15 and not zero_shear:
            bundle.metadata["official_scalar_increment_multiplier"] = lambda_phi
            return bundle, grid, angular, mesh

        incoming = bundle.incoming
        outgoing = bundle.outgoing
        data = config.scalar_initial_data
        v = mesh.v
        corner_p = data.corner_outgoing_scalar
        scalar_p = (
            corner_p
            + lambda_phi * v[None, :] ** data.outgoing_scalar_power
        )
        scalar_p = np.broadcast_to(
            scalar_p, (grid.count, len(v))
        ).copy()
        phi = (
            corner_p * v[None, :]
            + lambda_phi
            * v[None, :] ** (1.0 + data.outgoing_scalar_power)
            / (1.0 + data.outgoing_scalar_power)
        )
        phi = np.broadcast_to(phi, scalar_p.shape).copy()
        reference_shear = (
            np.zeros_like(outgoing["reference_shear"])
            if zero_shear
            else outgoing["reference_shear"]
        )
        metric, expansion, shear = idata._solve_outgoing_metric(
            grid,
            angular,
            mesh,
            incoming["metric"][:, 0],
            incoming["weighted_expansion"][:, 0],
            outgoing["omega"],
            outgoing["weighted_omega"],
            scalar_p,
            reference_shear,
            data.boundary_substeps,
        )
        outgoing.update(
            {
                "metric": metric,
                "weighted_expansion": expansion,
                "q": expansion / outgoing["omega"] ** 2,
                "shear": shear,
                "scalar_p": scalar_p,
                "phi": phi,
                "reference_shear": reference_shear,
            }
        )
        bundle.metadata["official_scalar_increment_multiplier"] = lambda_phi
        bundle.metadata["minimum_outgoing_expansion"] = float(
            np.min(expansion)
        )
        return bundle, grid, angular, mesh

    return construct


def _config(
    *,
    name: str,
    cap: float,
    lambda_omega: float,
    lambda_b: float,
    lambda_chi: float,
    retained: int = 3,
    work: int = 6,
    points: int = 100,
    tau_elements: int = 1,
    s_elements: int = 1,
    tau_degree: int = 4,
    s_degree: int = 6,
    metric_substeps: int = 1,
    derivative_halo: int = 1,
    iterations: int = 3,
) -> ExperimentConfig:
    result = ExperimentConfig(
        name=name,
        angular=AngularConfig(
            point_count=points,
            neighbor_count=min(24, points - 1),
            retained_degree=retained,
            work_degree=work,
        ),
        scalar_coordinates=CoordinateConfig(
            u_left=-1.0,
            u_right=-0.5,
            v_max=cap,
            tau_elements=tau_elements,
            tau_degree=tau_degree,
            s_elements=s_elements,
            s_degree=s_degree,
            fractional_power=0.1,
        ),
        scalar_initial_data=InitialDataConfig(
            lapse_radial_power=0.25,
            lapse_angular_amplitude=0.01 * lambda_omega,
            shift_amplitude=0.05 * lambda_b,
            corner_outgoing_expansion=1.6,
            corner_outgoing_scalar=8.0 * math.sqrt(2.0) / 5.0,
            outgoing_scalar_power=0.1,
            # The constraint generator rejects an exactly zero shear before
            # returning its otherwise valid incoming constraint solution.
            # A vanishing official case is generated through a harmless tiny seed
            # and then reconstructed with exact zero in _scaled_construct.
            shear_vector_amplitude=0.1 * (
                lambda_chi if lambda_chi != 0.0 else 1.0e-10
            ),
            shear_profile_power=0.1,
            shear_profile="quadrupole",
            boundary_substeps=8,
        ),
        solver=SolverConfig(
            picard_iterations=iterations,
            metric_substeps=metric_substeps,
            derivative_halo_u=derivative_halo,
            derivative_halo_v=derivative_halo,
        ),
    )
    result.validate()
    return result


def _run_case(
    *,
    output: Path,
    name: str,
    cap: float,
    multipliers: tuple[float, float, float, float],
    rotation_angle: float = 0.0,
    retained: int = 3,
    work: int = 6,
    points: int = 100,
    tau_elements: int = 1,
    s_elements: int = 1,
    tau_degree: int = 4,
    s_degree: int = 6,
    metric_substeps: int = 1,
    derivative_halo: int = 1,
    iterations: int = 3,
) -> dict[str, Any]:
    case_output = output / "cases" / name
    initial_path = output / "boundary-data" / f"{name}.npz"
    if (case_output / "summary.json").exists():
        return json.loads(
            (case_output / "summary.json").read_text(encoding="utf-8")
        )
    lo, lb, lc, lp = multipliers
    config = _config(
        name=name,
        cap=cap,
        lambda_omega=lo,
        lambda_b=lb,
        lambda_chi=lc,
        retained=retained,
        work=work,
        points=points,
        tau_elements=tau_elements,
        s_elements=s_elements,
        tau_degree=tau_degree,
        s_degree=s_degree,
        metric_substeps=metric_substeps,
        derivative_halo=derivative_halo,
        iterations=iterations,
    )
    rotation = _rotation_z(rotation_angle)
    original_free = idata.analytic_incoming_free_data
    original_shear = idata._draft_reference_shear
    original_construct = idata.construct_initial_data
    original_run_construct = ese_run.construct_initial_data
    idata.analytic_incoming_free_data = _plan_free_data(rotation)
    idata._draft_reference_shear = _plan_reference_shear(rotation)
    scaled = _scaled_construct(original_construct, lp, lc)
    idata.construct_initial_data = scaled
    ese_run.construct_initial_data = scaled
    try:
        result = ese_run.run(
            config,
            case_output,
            initial_path,
            regenerate_initial_data=True,
        )
    finally:
        idata.analytic_incoming_free_data = original_free
        idata._draft_reference_shear = original_shear
        idata.construct_initial_data = original_construct
        ese_run.construct_initial_data = original_run_construct
    result["official_multipliers"] = {
        "lambda_omega": lo,
        "lambda_b": lb,
        "lambda_chi": lc,
        "lambda_phi": lp,
    }
    result["official_rotation_angle"] = rotation_angle
    (case_output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _failed(output: Path, name: str, error: Exception) -> dict[str, Any]:
    case_output = output / "cases" / name
    case_output.mkdir(parents=True, exist_ok=True)
    result = {
        "case_id": name,
        "terminal_status": "failed",
        "failure_classification": (
            "constraint radicand, positivity, Picard contraction, or resolution"
        ),
        "error": repr(error),
    }
    (case_output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def run_experiment(output: Path, quick: bool) -> dict[str, Any]:
    if not quick:
        raise NotImplementedError("declare a production stress budget explicitly")
    cases: list[tuple[str, float, tuple[float, float, float, float], dict[str, Any]]] = []

    # Baseline continuation caps.
    for cap in (0.01, 0.02, 0.04, 0.10):
        cases.append(
            (
                f"baseline-cap-{cap:.2f}",
                cap,
                (1.0, 1.0, 1.0, 1.0),
                {},
            )
        )

    # One-at-a-time and joint stress matrix at the representative v=0.04 cap.
    for axis in range(4):
        for value in (0.0, 0.5, 1.0, 2.0):
            multipliers = [1.0, 1.0, 1.0, 1.0]
            multipliers[axis] = value
            cases.append(
                (
                    f"stress-axis-{axis}-value-{value:.1f}",
                    0.04,
                    tuple(multipliers),
                    {},
                )
            )
    for value in (0.0, 0.5, 1.0, 2.0):
        cases.append(
            (
                f"stress-joint-{value:.1f}",
                0.04,
                (value, value, value, value),
                {},
            )
        )

    # Rotation covariance control and two further angular/coordinate levels.
    cases.extend(
        [
            (
                "rotated-baseline",
                0.04,
                (1.0, 1.0, 1.0, 1.0),
                {"rotation_angle": 0.731},
            ),
            (
                "coordinate-level-1",
                0.04,
                (1.0, 1.0, 1.0, 1.0),
                {"tau_elements": 2, "s_elements": 2, "iterations": 4},
            ),
            (
                "coordinate-level-2",
                0.04,
                (1.0, 1.0, 1.0, 1.0),
                {"tau_elements": 4, "s_elements": 4, "iterations": 4},
            ),
            (
                "angular-level-1",
                0.04,
                (1.0, 1.0, 1.0, 1.0),
                {"retained": 4, "work": 8, "points": 140, "iterations": 4},
            ),
            (
                "angular-level-2",
                0.04,
                (1.0, 1.0, 1.0, 1.0),
                {"retained": 5, "work": 10, "points": 180, "iterations": 4},
            ),
        ]
    )

    # Remove exact duplicate baseline stress cases while retaining labels in
    # the aggregate via references to the canonical run.
    summaries: list[dict[str, Any]] = []
    seen: dict[tuple[Any, ...], dict[str, Any]] = {}
    for name, cap, multipliers, options in cases:
        key = (
            cap,
            multipliers,
            tuple(sorted(options.items())),
        )
        if key in seen:
            summaries.append(
                {
                    "case_id": name,
                    "terminal_status": "alias",
                    "canonical_case": seen[key].get("config", {}).get(
                        "name", "baseline-cap-0.04"
                    ),
                }
            )
            continue
        try:
            summary = _run_case(
                output=output,
                name=name,
                cap=cap,
                multipliers=multipliers,
                **options,
            )
        except Exception as error:
            summary = _failed(output, name, error)
        seen[key] = summary
        summaries.append(summary)

    aggregate = {
        "schema": "nee-official-experiment-06-aggregate-v1",
        "experiment": 6,
        "runs": summaries,
        "harmonics": {
            "Y_Omega": "normalized real Y_20",
            "V_b": "normalized grad(real Y_21)",
            "V_chi": "0.1 normalized grad(real Y_22)",
            "corner_power": 0.1,
            "corner_regularization": "none",
        },
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
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"immutable run directory exists: {args.output}")
    args.output.mkdir(parents=True)
    run_experiment(args.output, args.quick)


if __name__ == "__main__":
    main()
