"""Equation-system interface shared by the geometric transport hierarchy."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from nee.state.iterate import PicardState


class EquationSystem(Protocol):
    name: str
    has_scalar: bool

    def ricci_source(self, state: PicardState) -> dict[str, np.ndarray]: ...

