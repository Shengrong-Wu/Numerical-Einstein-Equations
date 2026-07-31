"""Common LGL elements in tau=-log(-u) and a configurable power of v."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lgl import CompositeLGLMesh


Array = np.ndarray


@dataclass(frozen=True)
class CharacteristicPowerMesh:
    """Tensor-product characteristic mesh.

    ``s=(v/v_max)^delta`` makes the prescribed ``v^delta`` scalar and shear
    profiles linear at the singular corner.  Differentiation and integration
    use the same composite LGL polynomial in each coordinate.
    """

    tau: CompositeLGLMesh
    s: CompositeLGLMesh
    v1: float
    delta: float
    u: Array
    v: Array

    @classmethod
    def create(
        cls,
        tau_breakpoints: Array,
        tau_degrees: int | Array,
        s_breakpoints: Array,
        s_degrees: int | Array,
        v1: float,
        delta: float,
    ) -> "CharacteristicPowerMesh":
        if not np.isfinite(v1) or v1 <= 0.0:
            raise ValueError("v1 must be positive")
        if not np.isfinite(delta) or not 0.0 < delta <= 1.0:
            raise ValueError("delta must lie in (0,1]")
        tau = CompositeLGLMesh.create(tau_breakpoints, tau_degrees)
        s = CompositeLGLMesh.create(s_breakpoints, s_degrees)
        return cls(
            tau=tau,
            s=s,
            v1=float(v1),
            delta=float(delta),
            u=-np.exp(-tau.nodes),
            v=float(v1) * s.nodes ** (1.0 / float(delta)),
        )

    def differentiate_u(self, values: Array, axis: int = 1) -> Array:
        derivative_tau = self.tau.differentiate(
            values, axis=axis, interface_rule="average"
        )
        shape = [1] * values.ndim
        shape[axis] = len(self.u)
        return derivative_tau / (-self.u).reshape(shape)

    def _dv_ds(self) -> Array:
        exponent = 1.0 / self.delta
        result = (
            self.v1
            * exponent
            * self.s.nodes ** np.maximum(exponent - 1.0, 0.0)
        )
        if exponent == 1.0:
            result[:] = self.v1
        return result

    def integrate_v(self, source: Array, axis: int = 2) -> Array:
        shape = [1] * source.ndim
        shape[axis] = len(self.s.nodes)
        return self.s.integrate(
            self._dv_ds().reshape(shape) * source, axis=axis
        )

    def differentiate_v(self, values: Array, axis: int = 2) -> Array:
        derivative_s = self.s.differentiate(
            values, axis=axis, interface_rule="average"
        )
        moved = np.moveaxis(derivative_s, axis, 0)
        result = np.empty_like(moved)
        jacobian = self._dv_ds()
        positive = jacobian > 64.0 * np.finfo(float).tiny
        result[positive] = moved[positive] / jacobian[positive].reshape(
            (int(np.count_nonzero(positive)),)
            + (1,) * (moved.ndim - 1)
        )
        first_positive = int(np.flatnonzero(positive)[0])
        result[~positive] = result[first_positive]
        return np.moveaxis(result, 0, axis)

    def diagnostics(self) -> dict[str, object]:
        return {
            "coordinate_system": (
                f"tau=-log(-u), s=(v/v_max)^{self.delta:.12g}"
            ),
            "tau_breakpoints": [
                self.tau.segments[0].left,
                *[segment.right for segment in self.tau.segments],
            ],
            "tau_degrees": [segment.degree for segment in self.tau.segments],
            "s_breakpoints": [
                self.s.segments[0].left,
                *[segment.right for segment in self.s.segments],
            ],
            "s_degrees": [segment.degree for segment in self.s.segments],
            "u_count": len(self.u),
            "v_count": len(self.v),
            "v_max": self.v1,
            "fractional_power": self.delta,
        }


def uniform_breakpoints(left: float, right: float, count: int) -> Array:
    if count < 1:
        raise ValueError("element count must be positive")
    return np.linspace(float(left), float(right), count + 1)


def mesh_from_config(scalar_config: object) -> CharacteristicPowerMesh:
    tau_left = -np.log(-float(scalar_config.u_left))
    tau_right = -np.log(-float(scalar_config.u_right))
    return CharacteristicPowerMesh.create(
        uniform_breakpoints(tau_left, tau_right, int(scalar_config.tau_elements)),
        int(scalar_config.tau_degree),
        uniform_breakpoints(0.0, 1.0, int(scalar_config.s_elements)),
        int(scalar_config.s_degree),
        float(scalar_config.v_max),
        float(scalar_config.fractional_power),
    )
