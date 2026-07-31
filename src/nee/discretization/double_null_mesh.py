"""Complete numerical context supplied to a Picard sweep."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Discretization:
    grid: Any
    u: np.ndarray
    v: np.ndarray
    coordinates: Any | None = None
    angular: Any | None = None
    metric_substeps: int = 2
    options: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        u = np.asarray(self.u, dtype=np.float64)
        v = np.asarray(self.v, dtype=np.float64)
        if u.ndim != 1 or v.ndim != 1 or len(u) < 2 or len(v) < 2:
            raise ValueError("double-null coordinate arrays must be one-dimensional with at least two nodes")
        if np.any(np.diff(u) <= 0.0) or np.any(np.diff(v) <= 0.0):
            raise ValueError("double-null coordinates must be strictly increasing")
        object.__setattr__(self, "u", u)
        object.__setattr__(self, "v", v)

