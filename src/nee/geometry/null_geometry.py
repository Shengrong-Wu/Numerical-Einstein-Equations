"""Null-frame differential operations shared by data and residual modules."""

from __future__ import annotations

import numpy as np

from nee.geometry.sphere import PointSphereGrid


Array = np.ndarray


def one_form_lie_derivative(
    grid: PointSphereGrid, vector: Array, form: Array
) -> Array:
    """Return the intrinsic Lie derivative of a sphere one-form."""

    derivative_form = grid.reference_derivative(form, tensor_rank=1)
    derivative_vector = grid.reference_derivative(vector, tensor_rank=1)
    return np.einsum(
        "n...k,n...ki->n...i", vector, derivative_form
    ) + np.einsum("n...k,n...ik->n...i", form, derivative_vector)
