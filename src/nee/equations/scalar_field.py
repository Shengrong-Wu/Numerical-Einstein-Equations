"""Massless scalar source provider, Ric=dphi tensor dphi."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from nee.state.iterate import PicardState


@dataclass(frozen=True)
class ScalarFieldEquations:
    name: str = "scalar_field"
    has_scalar: bool = True

    def ricci_source(self, state: PicardState) -> dict[str, np.ndarray]:
        if state.Omega_e3phi is None or state.Omega_e4phi is None:
            raise ValueError("scalar equations require both weighted null scalar derivatives")
        return {
            "33": state.Omega_e3phi**2,
            "44": state.Omega_e4phi**2,
            "34": state.Omega_e3phi * state.Omega_e4phi,
        }

