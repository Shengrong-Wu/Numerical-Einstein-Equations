"""Deterministic serialization for immutable characteristic data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from nee.state.boundary import BoundaryData


SCHEMA = "nee-characteristic-boundary-1"


def save_boundary_data(
    path: str | Path,
    boundary: BoundaryData,
    *,
    u: np.ndarray,
    v: np.ndarray,
) -> None:
    arrays: dict[str, Any] = {
        "schema": np.asarray(SCHEMA),
        "content_hash": np.asarray(boundary.content_hash),
        "metadata_json": np.asarray(json.dumps(dict(boundary.metadata), sort_keys=True, separators=(",", ":"))),
        "u": np.asarray(u, dtype=np.float64),
        "v": np.asarray(v, dtype=np.float64),
    }
    arrays.update({f"outgoing__{name}": np.asarray(value) for name, value in boundary.outgoing.items()})
    arrays.update({f"incoming__{name}": np.asarray(value) for name, value in boundary.incoming.items()})
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **arrays)


def load_boundary_data(path: str | Path) -> tuple[BoundaryData, np.ndarray, np.ndarray]:
    with np.load(Path(path)) as artifact:
        schema = str(artifact["schema"])
        if schema != SCHEMA:
            raise ValueError(f"unsupported boundary schema {schema!r}")
        outgoing = {
            name.removeprefix("outgoing__"): artifact[name].copy()
            for name in artifact.files
            if name.startswith("outgoing__")
        }
        incoming = {
            name.removeprefix("incoming__"): artifact[name].copy()
            for name in artifact.files
            if name.startswith("incoming__")
        }
        metadata = json.loads(str(artifact["metadata_json"]))
        expected = str(artifact["content_hash"])
        u = artifact["u"].copy()
        v = artifact["v"].copy()
    boundary = BoundaryData.create(outgoing, incoming, metadata=metadata)
    if boundary.content_hash != expected:
        raise RuntimeError("boundary artifact failed its content-hash check")
    return boundary, u, v

