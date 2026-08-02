"""Configuration objects for the ESE numerical experiment.

The numerical code consumes these dataclasses rather than embedding one
particular grid, angular band, or characteristic datum in the solvers.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AngularConfig:
    point_count: int = 350
    neighbor_count: int = 40
    retained_degree: int = 8
    work_degree: int = 16


@dataclass(frozen=True)
class CoordinateConfig:
    u_left: float = -1.0
    u_right: float = -0.5
    v_max: float = 0.1
    tau_elements: int = 2
    tau_degree: int = 8
    s_elements: int = 2
    s_degree: int = 11
    fractional_power: float = 0.1


@dataclass(frozen=True)
class InitialDataConfig:
    lapse_radial_power: float = 0.25
    lapse_angular_amplitude: float = 0.01
    shift_amplitude: float = 0.05
    corner_outgoing_expansion: float = 1.6
    corner_outgoing_scalar: float = 8.0 * 2.0**0.5 / 5.0
    incoming_scalar_branch: str = "positive"
    outgoing_scalar_power: float = 0.1
    outgoing_scalar_amplitude: float = 1.0
    outgoing_profile_scale: float = 0.0
    shear_vector_amplitude: float = 0.1
    shear_profile_power: float = 0.1
    shear_profile: str = "quadrupole"
    boundary_substeps: int = 8
    corner_zeta: float = 0.0
    corner_weighted_omega: float = 0.0


@dataclass(frozen=True)
class SolverConfig:
    picard_iterations: int = 6
    metric_substeps: int = 2
    derivative_halo_u: int = 2
    derivative_halo_v: int = 2


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "default"
    angular: AngularConfig = field(default_factory=AngularConfig)
    scalar_coordinates: CoordinateConfig = field(default_factory=CoordinateConfig)
    scalar_initial_data: InitialDataConfig = field(default_factory=InitialDataConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)

    def validate(self) -> None:
        a = self.angular
        c = self.scalar_coordinates
        d = self.scalar_initial_data
        s = self.solver
        if a.retained_degree < 1 or a.work_degree < a.retained_degree:
            raise ValueError("require 1 <= retained_degree <= work_degree")
        if (a.work_degree + 2) ** 2 >= a.point_count:
            raise ValueError(
                "point_count must overdetermine the degree-(work_degree+1) "
                "differentiation space"
            )
        if a.neighbor_count >= a.point_count:
            raise ValueError("neighbor_count must be smaller than point_count")
        if not (-1.0 <= c.u_left < c.u_right < 0.0):
            raise ValueError("require -1 <= u_left < u_right < 0")
        if c.v_max <= 0.0:
            raise ValueError("v_max must be positive")
        if c.tau_elements < 1 or c.s_elements < 1:
            raise ValueError("each coordinate needs at least one element")
        if c.tau_degree < 2 or c.s_degree < 2:
            raise ValueError("LGL degrees must be at least two")
        if not 0.0 < c.fractional_power <= 1.0:
            raise ValueError("fractional_power must lie in (0,1]")
        if abs(c.fractional_power - d.outgoing_scalar_power) > 1.0e-14:
            raise ValueError(
                "the regular coordinate power must match the scalar datum"
            )
        if abs(d.shear_profile_power - d.outgoing_scalar_power) > 1.0e-14:
            raise ValueError(
                "the current regular coordinate assumes equal scalar and "
                "Omega_chih fractional powers"
            )
        if d.shear_profile not in {"quadrupole", "draft-conformal-killing"}:
            raise ValueError(f"unknown Omega_chih profile {d.shear_profile!r}")
        if d.incoming_scalar_branch not in {"positive", "negative"}:
            raise ValueError(
                "incoming_scalar_branch must be 'positive' or 'negative'"
            )
        if not math.isfinite(d.outgoing_scalar_amplitude):
            raise ValueError("outgoing_scalar_amplitude must be finite")
        if not d.outgoing_profile_scale >= 0.0:
            raise ValueError("outgoing_profile_scale must be nonnegative")
        if d.boundary_substeps < 1:
            raise ValueError("boundary_substeps must be positive")
        if s.picard_iterations < 1 or s.metric_substeps < 1:
            raise ValueError("iteration and substep counts must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        result = cls(
            name=str(value.get("name", "default")),
            angular=AngularConfig(**value.get("angular", {})),
            scalar_coordinates=CoordinateConfig(**value.get("scalar_coordinates", {})),
            scalar_initial_data=InitialDataConfig(**value.get("scalar_initial_data", {})),
            solver=SolverConfig(**value.get("solver", {})),
        )
        result.validate()
        return result

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with Path(path).open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


def write_default(path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(ExperimentConfig().to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
