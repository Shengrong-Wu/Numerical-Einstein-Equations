"""Run manifest creation and atomic JSON output."""

from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .provenance import runtime_provenance


def peak_memory_kib() -> int:
    """Return peak resident memory in KiB on supported Unix platforms.

    ``ru_maxrss`` is reported in bytes by macOS and in KiB by Linux.  The
    manifest schema uses KiB on both platforms.
    """

    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value // 1024 if sys.platform == "darwin" else value


def array_schema(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in sorted(arrays.items())
    }


def build_manifest(
    *,
    root: str | Path,
    experiment: str,
    started_at: float,
    config: dict[str, Any],
    inputs: dict[str, str],
    outputs: dict[str, str],
    arrays: dict[str, np.ndarray],
) -> dict[str, Any]:
    return {
        "schema": "nee-run-manifest-1",
        "experiment": experiment,
        "status": "complete",
        "wall_time_seconds": time.time() - started_at,
        "peak_memory_kib": peak_memory_kib(),
        "software": runtime_provenance(root),
        "resolved_dimensions": {
            "u_nodes": config["coordinates"]["u_node_count"],
            "v_nodes": config["coordinates"]["v_node_count"],
            "scalar_retained_basis": config["angular"]["scalar_retained_dimension"],
            "scalar_work_basis": config["angular"]["scalar_work_dimension"],
        },
        "input_hashes": inputs,
        "output_hashes": outputs,
        "array_schemas": array_schema(arrays),
    }


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(target)
