"""Vacuum source provider, Ric(g)=0."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nee.state.iterate import PicardState


@dataclass(frozen=True)
class VacuumEquations:
    name: str = "vacuum"
    has_scalar: bool = False

    def ricci_source(self, state: PicardState) -> dict[str, np.ndarray]:
        shape = state.sphere_metric.shape[:-2]
        return {"33": np.zeros(shape), "44": np.zeros(shape), "34": np.zeros(shape)}

