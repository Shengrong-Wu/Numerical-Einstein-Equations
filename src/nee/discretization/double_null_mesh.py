"""Complete numerical context supplied to a Picard sweep."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nee.numerics.lgl import CompositeLGLMesh


Array = np.ndarray


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


@dataclass(frozen=True)
class DoubleSqrtLGLMesh:
    """Composite LGL mesh in ``t=sqrt(2(u+1))`` and ``s=sqrt(2v)``."""

    t: CompositeLGLMesh
    s: CompositeLGLMesh
    u: Array
    v: Array

    @classmethod
    def create(
        cls,
        t_breakpoints: Array,
        t_degrees: int | Array,
        s_breakpoints: Array,
        s_degrees: int | Array,
    ) -> "DoubleSqrtLGLMesh":
        t = CompositeLGLMesh.create(t_breakpoints, t_degrees)
        s = CompositeLGLMesh.create(s_breakpoints, s_degrees)
        return cls(
            t=t,
            s=s,
            u=-1.0 + 0.5 * t.nodes**2,
            v=0.5 * s.nodes**2,
        )

    @staticmethod
    def _divide_by_corner_coordinate(
        derivative: Array, coordinate: Array, axis: int
    ) -> Array:
        moved = np.moveaxis(derivative, axis, 0)
        result = np.empty_like(moved)
        if coordinate[0] == 0.0:
            shape = (len(coordinate) - 1,) + (1,) * (moved.ndim - 1)
            result[1:] = moved[1:] / coordinate[1:].reshape(shape)
            # The data are only assumed smooth in the square-root coordinate.
            # The physical derivative may diverge at the corner.  Copying the
            # first positive trace keeps array diagnostics finite; all
            # reported protected maxima exclude that endpoint.
            result[0] = result[1]
        else:
            shape = (len(coordinate),) + (1,) * (moved.ndim - 1)
            result[:] = moved / coordinate.reshape(shape)
        return np.moveaxis(result, 0, axis)

    def differentiate_u(self, values: Array, axis: int = 1) -> Array:
        return self._divide_by_corner_coordinate(
            self.t.differentiate(values, axis=axis),
            self.t.nodes,
            axis,
        )

    def differentiate_v(self, values: Array, axis: int = 2) -> Array:
        return self._divide_by_corner_coordinate(
            self.s.differentiate(values, axis=axis),
            self.s.nodes,
            axis,
        )

    def integrate_v(self, source: Array, axis: int = 2) -> Array:
        shape = [1] * source.ndim
        shape[axis] = len(self.s.nodes)
        return self.s.integrate(
            self.s.nodes.reshape(shape) * source,
            axis=axis,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "coordinate_system": "t=sqrt(2(u+1)), s=sqrt(2v)",
            "t_breakpoints": [
                self.t.segments[0].left,
                *[segment.right for segment in self.t.segments],
            ],
            "t_degrees": [segment.degree for segment in self.t.segments],
            "s_breakpoints": [
                self.s.segments[0].left,
                *[segment.right for segment in self.s.segments],
            ],
            "s_degrees": [segment.degree for segment in self.s.segments],
            "u_count": len(self.u),
            "v_count": len(self.v),
        }
