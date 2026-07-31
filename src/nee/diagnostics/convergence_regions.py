"""Reliability masks for spectral convergence-region diagnostics."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def composite_axis_mask(
    node_count: int,
    *,
    element_degree: int,
    interface_halo: int,
) -> Array:
    """Mask unreliable boundary elements and element-interface stencils.

    The first coordinate element touches the square-root corner, where a
    physical derivative contains division by the root coordinate.  The last
    element touches the terminal face and has no two-sided continuation.
    Both elements are excluded.  Around every remaining shared LGL endpoint,
    ``interface_halo`` nodes on each side are excluded because the fresh
    derivative diagnostic is assembled element by element.
    """

    if element_degree < 1:
        raise ValueError("element degree must be positive")
    if interface_halo < 0:
        raise ValueError("interface halo cannot be negative")
    if (node_count - 1) % element_degree != 0:
        raise ValueError(
            "node count is inconsistent with a uniform composite LGL mesh"
        )
    element_count = (node_count - 1) // element_degree
    if element_count < 3:
        raise ValueError(
            "a reliability audit requires at least three coordinate elements"
        )

    mask = np.ones(node_count, dtype=bool)
    mask[: element_degree + 1] = False
    mask[-(element_degree + 1) :] = False
    for interface in range(
        element_degree,
        element_count * element_degree,
        element_degree,
    ):
        left = max(0, interface - interface_halo)
        right = min(node_count, interface + interface_halo + 1)
        mask[left:right] = False
    if not np.any(mask):
        raise ValueError("reliability mask removed every coordinate node")
    return mask


def composite_uv_mask(
    u_count: int,
    v_count: int,
    *,
    element_degree: int,
    interface_halo: int,
) -> Array:
    """Return the tensor-product coordinate reliability mask."""

    u_mask = composite_axis_mask(
        u_count,
        element_degree=element_degree,
        interface_halo=interface_halo,
    )
    v_mask = composite_axis_mask(
        v_count,
        element_degree=element_degree,
        interface_halo=interface_halo,
    )
    return u_mask[:, None] & v_mask[None, :]
