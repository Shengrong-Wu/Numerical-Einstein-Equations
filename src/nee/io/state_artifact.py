"""Final-state serialization and content hashing."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from nee.state.iterate import PicardState


def file_hash(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def save_state(
    path: str | Path,
    state: PicardState,
    *,
    u: np.ndarray,
    v: np.ndarray,
    extra: dict[str, Any] | None = None,
) -> str:
    target = Path(path)
    state.save(target, u=u, v=v, extra=extra)
    return file_hash(target)


def load_state(path: str | Path) -> tuple[PicardState, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    return PicardState.load(Path(path))

