"""Lightweight typed views of sampled angular fields."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class ScalarField:
    values: Array

    def __post_init__(self) -> None:
        value = np.asarray(self.values, dtype=np.float64)
        if value.ndim < 1:
            raise ValueError("a sampled scalar field needs an angular axis")
        object.__setattr__(self, "values", value)


@dataclass(frozen=True)
class TangentVectorField:
    values: Array

    def __post_init__(self) -> None:
        value = np.asarray(self.values, dtype=np.float64)
        if value.ndim < 2 or value.shape[-1] != 3:
            raise ValueError("a tangent vector field must end in three ambient components")
        object.__setattr__(self, "values", value)


@dataclass(frozen=True)
class SymmetricTensorField:
    values: Array

    def __post_init__(self) -> None:
        value = np.asarray(self.values, dtype=np.float64)
        if value.ndim < 3 or value.shape[-2:] != (3, 3):
            raise ValueError("a symmetric tensor field must end in a 3-by-3 ambient tensor")
        if not np.allclose(value, np.swapaxes(value, -1, -2), rtol=0.0, atol=1.0e-12):
            raise ValueError("tensor samples are not symmetric")
        object.__setattr__(self, "values", value)


@dataclass(frozen=True)
class PrimitiveFields:
    """Four-g primitive fields accepted by independent audits."""

    g: Array
    log_Omega: Array
    b: Array
    phi: Array | None = None


def tangent_inverse(g: Array) -> Array:
    symmetric = 0.5 * (g + np.swapaxes(g, -1, -2))
    return np.linalg.pinv(symmetric, rcond=1.0e-13, hermitian=True)


def tensor_trace(tensor: Array, inverse_g: Array) -> Array:
    return np.einsum("n...ij,n...ij->n...", tensor, inverse_g)


def tracefree(tensor: Array, g: Array, inverse_g: Array | None = None) -> Array:
    inverse_g = tangent_inverse(g) if inverse_g is None else inverse_g
    return tensor - 0.5 * tensor_trace(tensor, inverse_g)[..., None, None] * g
