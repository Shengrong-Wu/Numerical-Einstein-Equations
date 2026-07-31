"""Validated experiment configuration and deterministic TOML serialization."""

from __future__ import annotations

import math
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ExperimentSection:
    identifier: str
    title: str
    equation_system: str


@dataclass(frozen=True)
class CoordinateSection:
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    u_map: str
    v_map: str
    u_breakpoints: tuple[float, ...]
    v_breakpoints: tuple[float, ...]
    u_degrees: tuple[int, ...]
    v_degrees: tuple[int, ...]

    def validate(self) -> None:
        for name, values, degrees, lower, upper in (
            ("u", self.u_breakpoints, self.u_degrees, self.u_min, self.u_max),
            ("v", self.v_breakpoints, self.v_degrees, self.v_min, self.v_max),
        ):
            if len(values) != len(degrees) + 1:
                raise ValueError(f"{name} breakpoints must have one more entry than degrees")
            if any(right <= left for left, right in zip(values, values[1:])):
                raise ValueError(f"{name} breakpoints must be strictly increasing")
            if not math.isclose(values[0], lower, rel_tol=0.0, abs_tol=1.0e-13):
                raise ValueError(f"{name} first breakpoint does not match the domain")
            if not math.isclose(values[-1], upper, rel_tol=0.0, abs_tol=1.0e-13):
                raise ValueError(f"{name} last breakpoint does not match the domain")
            if any(degree < 2 for degree in degrees):
                raise ValueError(f"{name} LGL degrees must be at least two")

    @staticmethod
    def node_count(degrees: Sequence[int]) -> int:
        return 1 + sum(degrees)


@dataclass(frozen=True)
class AngularSection:
    construction: str
    point_count: int
    neighbor_count: int
    seed: int
    retained_degree: int
    work_degree: int
    differentiation_degree: int

    def validate(self) -> None:
        if not (0 <= self.retained_degree <= self.work_degree <= self.differentiation_degree):
            raise ValueError("require retained <= work <= differentiation angular degree")
        if self.point_count <= (self.work_degree + 1) ** 2:
            raise ValueError("sphere point count must overdetermine the work scalar basis")
        if not (3 <= self.neighbor_count < self.point_count):
            raise ValueError("invalid sphere neighbor count")


@dataclass(frozen=True)
class SolverSection:
    maximum_sweeps: int
    tolerance: float
    relaxation: float
    slab_direction: str
    rk_substeps: int
    sdc_sweeps: int
    metric_substeps: int
    boundary_substeps: int
    stopping_norm: str

    def validate(self) -> None:
        if self.maximum_sweeps < 1 or self.tolerance <= 0.0:
            raise ValueError("solver needs positive sweep limit and tolerance")
        if not 0.0 < self.relaxation <= 1.0:
            raise ValueError("relaxation must lie in (0,1]")
        if self.slab_direction not in {"u", "v", "none"}:
            raise ValueError("slab_direction must be u, v, or none")
        if min(self.rk_substeps, self.sdc_sweeps, self.metric_substeps, self.boundary_substeps) < 1:
            raise ValueError("all configured substep counts must be positive")


@dataclass(frozen=True)
class ProjectionSection:
    nonlinear_products: str
    retained_projection: str
    filter: str
    filter_strength: float


@dataclass(frozen=True)
class AuditSection:
    coordinate_overgrid_increment: int
    angular_overgrid_degree: int
    sphere_point_count: int
    derivative_halo_u: int
    derivative_halo_v: int
    endpoint_mask_u: int
    endpoint_mask_v: int

    def validate(self) -> None:
        values = asdict(self)
        if any(value < 0 for name, value in values.items() if name != "sphere_point_count"):
            raise ValueError("audit degrees and halos cannot be negative")
        if self.sphere_point_count < 4:
            raise ValueError("audit needs at least four sphere points")


@dataclass(frozen=True)
class OutputSection:
    checkpoints: bool
    checkpoint_every: int
    save_residual_maps: bool
    save_figures: bool


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment: ExperimentSection
    physics: Mapping[str, Any]
    initial_data: Mapping[str, Any]
    coordinates: CoordinateSection
    angular: AngularSection
    solver: SolverSection
    projection: ProjectionSection
    audit: AuditSection
    output: OutputSection
    source_path: Path | None = field(default=None, compare=False, repr=False)

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported config schema {self.schema_version}")
        if not self.experiment.identifier.startswith("exp"):
            raise ValueError("experiment identifier must start with exp")
        if self.experiment.equation_system not in {"vacuum", "scalar_field"}:
            raise ValueError("equation_system must be vacuum or scalar_field")
        self.coordinates.validate()
        self.angular.validate()
        self.solver.validate()
        self.audit.validate()
        if self.output.checkpoints and self.output.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive when checkpoints are enabled")

    def resolved(self) -> dict[str, Any]:
        result = asdict(self)
        result.pop("source_path", None)
        coordinates = result["coordinates"]
        coordinates["u_node_count"] = CoordinateSection.node_count(self.coordinates.u_degrees)
        coordinates["v_node_count"] = CoordinateSection.node_count(self.coordinates.v_degrees)
        result["angular"]["scalar_retained_dimension"] = (self.angular.retained_degree + 1) ** 2
        result["angular"]["scalar_work_dimension"] = (self.angular.work_degree + 1) ** 2
        return result


def _section(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"config section [{name}] is required")
    return dict(value)


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    with source.open("rb") as stream:
        data = tomllib.load(stream)
    expected = {
        "schema_version",
        "experiment",
        "physics",
        "initial_data",
        "coordinates",
        "angular",
        "solver",
        "projection",
        "audit",
        "output",
    }
    unknown = set(data).difference(expected)
    missing = expected.difference(data)
    if unknown or missing:
        raise ValueError(f"invalid top-level config keys; missing={sorted(missing)}, unknown={sorted(unknown)}")
    coordinate_data = _section(data, "coordinates")
    for name in ("u_breakpoints", "v_breakpoints"):
        coordinate_data[name] = tuple(float(value) for value in coordinate_data[name])
    for name in ("u_degrees", "v_degrees"):
        coordinate_data[name] = tuple(int(value) for value in coordinate_data[name])
    config = ExperimentConfig(
        schema_version=int(data["schema_version"]),
        experiment=ExperimentSection(**_section(data, "experiment")),
        physics=_section(data, "physics"),
        initial_data=_section(data, "initial_data"),
        coordinates=CoordinateSection(**coordinate_data),
        angular=AngularSection(**_section(data, "angular")),
        solver=SolverSection(**_section(data, "solver")),
        projection=ProjectionSection(**_section(data, "projection")),
        audit=AuditSection(**_section(data, "audit")),
        output=OutputSection(**_section(data, "output")),
        source_path=source,
    )
    config.validate()
    return config


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("resolved TOML cannot contain nonfinite values")
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML value {type(value).__name__}")


def dump_resolved_config(config: ExperimentConfig, path: str | Path) -> None:
    data = config.resolved()
    lines = [f"schema_version = {_toml_value(data.pop('schema_version'))}", ""]
    for section, values in data.items():
        lines.append(f"[{section}]")
        for name, value in values.items():
            lines.append(f"{name} = {_toml_value(value)}")
        lines.append("")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")

