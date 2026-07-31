"""Temporary smooth datum ``hat(chi)^0(v)=v X`` for a controlled test.

The fixed tensor is the normalized trace-free round Hessian of ``z^2``.
It is smooth on the whole sphere, symmetric, round-metric trace-free, and
has continuum maximum norm one.  The production experiments continue to use
``smooth_pulse``; this module is selected only by an explicit CLI
flag.
"""

from __future__ import annotations

import math

import numpy as np

from .sphere import PointSphereGrid, tangent_inverse, tensor_norm_sq
from .smooth_pulse import (
    BoundaryDegeneracy,
    minimum_tangent_eigenvalue,
    solve_boundary_zeta,
    transferred_shear,
)


Array = np.ndarray


def fixed_tracefree_tensor(sphere: PointSphereGrid) -> Array:
    """Return X=(Hess(z^2))^TF/sqrt(2), whose max round norm is one."""

    z = sphere.points[:, 2]
    axis = np.array([0.0, 0.0, 1.0])
    dz = axis[None, :] - z[:, None] * sphere.points
    one_minus_z_sq = 1.0 - z**2
    tensor = (
        2.0 * np.einsum("ni,nj->nij", dz, dz)
        - one_minus_z_sq[:, None, None] * sphere.projector
    ) / math.sqrt(2.0)
    trace = np.einsum("nij,nij->n", sphere.projector, tensor)
    if float(np.max(np.abs(trace))) > 2.0e-14:
        raise AssertionError("the fixed tensor is not round-trace-free")
    return tensor


def solve_linear_fixed_boundary(
    sphere: PointSphereGrid,
    v: Array,
    minimum_eigenvalue: float = 1.0e-7,
) -> dict[str, Array | float]:
    """Solve the outgoing constraints with reference shear ``v X``."""

    if len(v) < 9 or abs(float(v[0])) > 1.0e-15 or np.any(np.diff(v) <= 0.0):
        raise ValueError("v must be a strictly increasing grid beginning at zero")
    tensor = fixed_tracefree_tensor(sphere)
    count = len(v)
    metric = np.zeros((sphere.count, count, 3, 3), dtype=float)
    expansion = np.zeros((sphere.count, count), dtype=float)
    shear = np.zeros_like(metric)
    reference = np.zeros_like(metric)
    metric[:, 0] = sphere.projector
    expansion[:, 0] = 2.0

    def reference_shear(value: float) -> Array:
        return value * tensor

    def rhs(
        value: float, current_metric: Array, current_expansion: Array
    ) -> tuple[Array, Array]:
        current_shear = transferred_shear(
            sphere, reference_shear(value), current_metric
        )
        inverse = tangent_inverse(sphere, current_metric)
        norm_sq = tensor_norm_sq(current_shear, inverse)
        return (
            current_expansion[:, None, None] * current_metric
            + 2.0 * current_shear,
            -0.5 * current_expansion**2 - norm_sq,
        )

    for j in range(count - 1):
        step = float(v[j + 1] - v[j])
        value = float(v[j])
        current_metric = metric[:, j]
        current_expansion = expansion[:, j]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                k1_metric, k1_expansion = rhs(
                    value, current_metric, current_expansion
                )
                k2_metric, k2_expansion = rhs(
                    value + 0.5 * step,
                    current_metric + 0.5 * step * k1_metric,
                    current_expansion + 0.5 * step * k1_expansion,
                )
                k3_metric, k3_expansion = rhs(
                    value + 0.5 * step,
                    current_metric + 0.5 * step * k2_metric,
                    current_expansion + 0.5 * step * k2_expansion,
                )
                k4_metric, k4_expansion = rhs(
                    value + step,
                    current_metric + step * k3_metric,
                    current_expansion + step * k3_expansion,
                )
                next_metric = current_metric + step * (
                    k1_metric + 2.0 * k2_metric + 2.0 * k3_metric + k4_metric
                ) / 6.0
                next_expansion = current_expansion + step * (
                    k1_expansion
                    + 2.0 * k2_expansion
                    + 2.0 * k3_expansion
                    + k4_expansion
                ) / 6.0
            except (FloatingPointError, np.linalg.LinAlgError) as error:
                raise BoundaryDegeneracy(
                    float(v[j]), float(v[j + 1]), -math.inf
                ) from error
        eigenvalue = minimum_tangent_eigenvalue(sphere, next_metric)
        if eigenvalue <= minimum_eigenvalue:
            raise BoundaryDegeneracy(float(v[j]), float(v[j + 1]), eigenvalue)
        metric[:, j + 1] = next_metric
        expansion[:, j + 1] = next_expansion

    inverse = tangent_inverse(sphere, metric)
    for j, value in enumerate(v):
        reference[:, j] = reference_shear(float(value))
        shear[:, j] = transferred_shear(
            sphere, reference[:, j], metric[:, j]
        )
    reference_norm = np.sqrt(
        np.maximum(np.einsum("nvij,nvij->nv", reference, reference), 0.0)
    )
    physical_norm = np.sqrt(np.maximum(tensor_norm_sq(shear, inverse), 0.0))
    trace = np.einsum("nvij,nvji->nv", inverse, shear)
    reference_trace = np.einsum("nvij,nij->nv", reference, sphere.projector)
    tensor_norm = np.sqrt(
        np.maximum(np.einsum("nij,nij->n", tensor, tensor), 0.0)
    )
    zeta, shift, zeta_source, zeta_iterations, zeta_update = solve_boundary_zeta(
        sphere, metric, expansion, shear, v
    )
    return {
        "metric": metric,
        "inverse": inverse,
        "expansion": expansion,
        "shear": shear,
        "reference_shear": reference,
        "zeta_up": zeta,
        "shift": shift,
        "zeta_source": zeta_source,
        "minimum_eigenvalue": minimum_tangent_eigenvalue(sphere, metric),
        "minimum_reference_l1": float(
            np.min(np.trapezoid(reference_norm, v, axis=1))
        ),
        "minimum_physical_l1": float(
            np.min(np.trapezoid(physical_norm, v, axis=1))
        ),
        "maximum_reference_trace": float(np.max(np.abs(reference_trace))),
        "maximum_physical_trace": float(np.max(np.abs(trace))),
        "minimum_expansion": float(np.min(expansion)),
        "maximum_final_expansion": float(np.max(expansion[:, -1])),
        "fixed_tensor_sampled_maximum_norm": float(np.max(tensor_norm)),
        "fixed_tensor_continuum_maximum_norm": 1.0,
        "zeta_iterations": zeta_iterations,
        "zeta_final_update": zeta_update,
    }
