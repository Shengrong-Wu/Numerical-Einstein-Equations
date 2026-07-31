"""Trapped-section sign diagnostics with explicit endpoint halos."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def trapped_sections(
    outgoing_expansion: Array,
    incoming_expansion: Array,
    *,
    u_endpoint_halo: int = 3,
) -> tuple[Array, Array]:
    """Return full and u-halo-masked trapped-section indicators."""

    outgoing = np.asarray(outgoing_expansion, dtype=float)
    incoming = np.asarray(incoming_expansion, dtype=float)
    if outgoing.shape != incoming.shape or outgoing.ndim != 2:
        raise ValueError("expansion suprema must be equally shaped (u,v) maps")
    trapped = (outgoing < 0.0) & (incoming < 0.0)
    interior = trapped.copy()
    if u_endpoint_halo < 0:
        raise ValueError("u_endpoint_halo must be nonnegative")
    if u_endpoint_halo:
        if 2 * u_endpoint_halo >= interior.shape[0]:
            raise ValueError("u endpoint halo removes the whole coordinate grid")
        interior[:u_endpoint_halo] = False
        interior[-u_endpoint_halo:] = False
    return trapped, interior
