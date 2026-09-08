"""MOTS reconstruction and apparent-horizon tracing for Experiment 8."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, least_squares

from nee.io.state_artifact import load_state
from nee.numerics.scalar_config import ExperimentConfig
from nee.numerics.scalar_coordinates import mesh_from_config
from nee.numerics.scalar_initial_data import build_angular
from nee.numerics.sphere import spherical_harmonic_collocation
from nee.state.fields import PrimitiveFields

from .mots_geometry import NullConeGeometry


Array = np.ndarray


@dataclass
class Patch:
    name: str
    root: Path
    cap: float
    u_right: float
    state: object
    u: Array
    grid: object
    fields: PrimitiveFields
    mesh: object


def load_patch(root: Path, name: str) -> Patch:
    directory = root / name
    report = json.loads((directory / "summary.json").read_text())
    config = ExperimentConfig.from_dict(report["config"])
    public, u, _, _ = load_state(directory / "final-state.npz")
    grid, _ = build_angular(config)
    return Patch(
        name=name,
        root=directory,
        cap=float(report["curved_region"]["v_cap"]),
        u_right=float(report["curved_region"]["u_right"]),
        state=public,
        u=np.asarray(u),
        grid=grid,
        fields=PrimitiveFields(
            public.g, public.log_Omega, public.b, public.phi
        ),
        mesh=mesh_from_config(config.scalar_coordinates),
    )


def field_norms(values: Array) -> dict[str, float]:
    return {
        "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
        "rms": float(np.sqrt(np.mean(values * values))),
        "linf": float(np.max(np.abs(values))),
    }


def acceptance_errors(report: dict, tolerance: float = 1.0e-5) -> list[str]:
    """Checks for accepting a numerical MOTS, rather than a solver return."""
    errors = []
    surface = report['surface']
    if not report['nonlinear_success']:
        errors.append('nonlinear solve did not succeed')
    if not report['sign_check']['bracketed']:
        errors.append('displaced sections do not bracket the outgoing expansion')
    for key in ('h_min', 'h_max', 'area'):
        if not np.isfinite(surface[key]):
            errors.append(f'nonfinite {key}')
    residual = surface['theta_out']['linf']
    if not np.isfinite(residual) or residual > tolerance:
        errors.append('outgoing expansion exceeds acceptance tolerance')
    incoming = surface['theta_in']['maximum']
    if not np.isfinite(incoming) or incoming >= 0:
        errors.append('incoming expansion is not strictly negative')
    if surface.get('distance_to_patch_boundary', surface['distance_to_patch_right']) <= 0:
        errors.append('surface is outside the patch interior')
    return errors


def solve_mots(
    patch: Patch, v_value: float, degree: int
) -> tuple[dict, dict[str, Array]]:
    """Solve the outermost graph MOTS on one incoming null cone."""

    cone = NullConeGeometry.create_at_v(
        patch.grid, patch.fields, patch.mesh, float(v_value)
    )
    _, basis, condition = spherical_harmonic_collocation(
        patch.grid.points, degree
    )
    inverse = np.linalg.pinv(basis, rcond=1.0e-13)

    def mean_expansion(value: float) -> float:
        return float(
            np.mean(
                cone.expansion(
                    np.full(patch.grid.count, value)
                ).theta_out
            )
        )

    scan_u = np.linspace(float(patch.u[2]), float(patch.u[-1]), 241)
    scan_theta = np.asarray([mean_expansion(value) for value in scan_u])
    brackets = []
    for left, right, f_left, f_right in zip(
        scan_u[:-1], scan_u[1:], scan_theta[:-1], scan_theta[1:], strict=True
    ):
        if f_left * f_right <= 0.0:
            brackets.append((float(left), float(right)))
    if not brackets:
        raise RuntimeError("no constant-section expansion sign bracket")
    bracket = max(brackets, key=lambda pair: pair[1])
    root = float(brentq(mean_expansion, *bracket, xtol=2.0e-13))
    coefficients0 = inverse @ np.full(patch.grid.count, root)
    lower = float(patch.u[0])
    upper = float(patch.u[-1])

    def residual(coefficients: Array) -> Array:
        h = basis @ coefficients
        if h.min() < lower or h.max() > upper:
            violation = np.maximum(lower - h, 0.0) + np.maximum(
                h - upper, 0.0
            )
            return inverse @ (20.0 + 1.0e4 * violation)
        return inverse @ cone.expansion(h).theta_out

    fit = least_squares(
        residual,
        coefficients0,
        method="lm",
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=1000,
    )
    coefficients = fit.x
    h = basis @ coefficients
    graph = cone.expansion(h)
    distance = min(float(h.min() - lower), float(upper - h.max()))
    offset = min(0.002, max(2.0e-5, 0.25 * distance))
    inward = cone.expansion(h - offset).theta_out
    outward = cone.expansion(h + offset).theta_out
    area = float(4.0 * math.pi * np.mean(graph.area_density))
    null_out = np.einsum(
        "ni,nij,nj->n",
        graph.outgoing_normal,
        graph.spacetime_metric,
        graph.outgoing_normal,
    )
    null_in = np.einsum(
        "ni,nij,nj->n",
        graph.incoming_normal,
        graph.spacetime_metric,
        graph.incoming_normal,
    )
    cross = np.einsum(
        "ni,nij,nj->n",
        graph.outgoing_normal,
        graph.spacetime_metric,
        graph.incoming_normal,
    )
    report = {
        "v": float(v_value),
        "patch": patch.name,
        "patch_cap": patch.cap,
        "patch_u_right": patch.u_right,
        "constant_sign_brackets": [list(pair) for pair in brackets],
        "outermost_constant_root": root,
        "basis_condition": float(condition),
        "nonlinear_success": bool(fit.success),
        "nonlinear_evaluations": int(fit.nfev),
        "surface": {
            "h_min": float(h.min()),
            "h_mean": float(h.mean()),
            "h_max": float(h.max()),
            "h_peak_to_peak": float(np.ptp(h)),
            "distance_to_patch_right": float(upper - h.max()),
            "distance_to_patch_boundary": float(distance),
            "area": area,
            "areal_radius": float(math.sqrt(area / (4.0 * math.pi))),
            "theta_out": field_norms(graph.theta_out),
            "theta_in": field_norms(graph.theta_in),
        },
        "normalization_checks": {
            "g_l_l_linf": float(np.max(np.abs(null_out))),
            "g_n_n_linf": float(np.max(np.abs(null_in))),
            "g_l_n_plus_2_linf": float(np.max(np.abs(cross + 2.0))),
        },
        "sign_check": {
            "offset": offset,
            "inward": field_norms(inward),
            "outward": field_norms(outward),
            "bracketed": bool(
                np.min(inward) > 0.0 and np.max(outward) < 0.0
            ),
        },
    }
    arrays = {
        "points": np.asarray(patch.grid.points),
        "coefficients": coefficients,
        "h": h,
        "theta_out": graph.theta_out,
        "theta_in": graph.theta_in,
        "area_density": graph.area_density,
    }
    return report, arrays


def save_surface(output: Path, report: dict, arrays: dict[str, Array]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(output / "surface.npz", **arrays)


def trace_horizon(
    patches: Iterable[Patch],
    output: Path,
    *,
    degree: int,
    requested_v: Array,
    tolerance: float = 1.0e-5,
    diagnostic_only: bool = False,
) -> dict:
    """Trace the outermost MOTS through the supplied overlapping patches."""

    output.mkdir(parents=True, exist_ok=False)
    ordered_patches = sorted(patches, key=lambda item: item.cap)
    sections = []
    failures = []
    for v_value in np.asarray(requested_v)[::-1]:
        candidates = [
            patch
            for patch in ordered_patches
            if v_value <= patch.cap * (1.0 + 2.0e-13)
        ]
        candidates.sort(key=lambda item: item.u_right, reverse=True)
        errors = []
        for patch in candidates:
            try:
                report, arrays = solve_mots(patch, float(v_value), degree)
            except Exception as error:
                errors.append({"patch": patch.name, "reason": str(error)})
                continue
            reasons = acceptance_errors(report, tolerance)
            report['accepted'] = not reasons
            report['acceptance_tolerance'] = tolerance
            report['rejection_reasons'] = reasons
            if reasons and not diagnostic_only:
                errors.append({'patch': patch.name, 'reasons': reasons, 'report': report})
                continue
            save_surface(output / f"v-{v_value:.10f}", report, arrays)
            sections.append(report)
            print(
                f"MOTS degree={degree} v={v_value:.10f} "
                f"patch={patch.name} h={report['surface']['h_mean']:.10f} "
                f"theta={report['surface']['theta_out']['linf']:.3e}",
                flush=True,
            )
            break
        else:
            failures.append({"v": float(v_value), "attempts": errors})
            print(f"MOTS degree={degree} v={v_value:.10f} UNSOLVED", flush=True)

    sections.sort(key=lambda item: item["v"])
    v = np.asarray([item["v"] for item in sections])
    h = np.asarray([item["surface"]["h_mean"] for item in sections])
    area = np.asarray([item["surface"]["area"] for item in sections])
    theta = np.asarray(
        [item["surface"]["theta_out"]["linf"] for item in sections]
    )
    report = {
        "schema": "nee-exp08-apparent-horizon-v1",
        "degree": degree,
        "requested_v_count": int(len(requested_v)),
        "diagnostic_only": diagnostic_only,
        "acceptance_tolerance": tolerance,
        "solved_v_count": int(len(sections)),
        "failed_v_count": int(len(failures)),
        "sections": sections,
        "failures": failures,
    }
    (output / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    np.savez_compressed(
        output / "apparent-horizon.npz",
        v=v,
        h_mean=h,
        area=area,
        theta_out_linf=theta,
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    axes[0].plot(v, h, "o-")
    axes[0].set(xlabel="v", ylabel="mean MOTS u", title="Horizon location")
    axes[1].plot(v, np.sqrt(area / (4.0 * np.pi)), "o-")
    axes[1].set(xlabel="v", ylabel="areal radius", title="MOTS areal radius")
    axes[2].semilogy(v, theta, "o-")
    axes[2].set(
        xlabel="v",
        ylabel=r"$||\theta_+||_\infty$",
        title="Expansion residual",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "apparent-horizon.png", dpi=180)
    plt.close(fig)
    return report
