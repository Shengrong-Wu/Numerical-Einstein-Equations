"""Experiment 7 nonspherical Einstein--scalar orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nee.initial_data.nonspherical_scalar import (
    _config,
    _plan_free_data,
    _plan_reference_shear,
    _rotation_z,
    _scaled_construct,
)
from nee.numerics import scalar_initial_data as idata
from nee.numerics import scalar_run as ese_run


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
    construct_wrapper=_scaled_construct,
    run_hooks: dict[str, Any] | None = None,
    public_config: Any | None = None,
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
        public_config=public_config,
    )
    rotation = _rotation_z(rotation_angle)
    def construct(numerical_config):
        return idata.construct_initial_data(
            numerical_config,
            free_data_generator=_plan_free_data(rotation),
            shear_generator=_plan_reference_shear(rotation),
        )

    scaled = construct_wrapper(construct, lp, lc)
    result = ese_run.run(
        config, case_output, initial_path, regenerate_initial_data=True,
        construct_initial_data_fn=scaled, **(run_hooks or {}),
    )
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
