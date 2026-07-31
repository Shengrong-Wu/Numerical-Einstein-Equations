"""Stable public solver API."""

from __future__ import annotations

from typing import Any

from .equations.base import EquationSystem
from .solver.result import SweepResult
from .state.boundary import BoundaryData
from .state.iterate import PicardState


def picard_sweep(
    state: PicardState,
    boundary: BoundaryData,
    discretization: Any,
    equations: EquationSystem,
) -> SweepResult:
    """Return ``U[i+1]`` without mutating ``U[i]`` or the boundary data."""

    from .solver.sweep import picard_sweep as implementation

    return implementation(state, boundary, discretization, equations)

