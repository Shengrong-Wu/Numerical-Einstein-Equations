"""Return type and diagnostics for a complete Picard sweep."""

from __future__ import annotations

from dataclasses import dataclass

from nee.state.iterate import PicardState


@dataclass(frozen=True)
class SweepResult:
    state: PicardState
    weighted_update: float
    maximum_update_map: float
    trace_errors: dict[str, float]
    diagnostics: dict[str, object]

