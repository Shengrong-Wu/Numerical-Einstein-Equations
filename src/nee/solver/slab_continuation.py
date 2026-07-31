"""Reusable characteristic-slab continuation helpers."""

from __future__ import annotations

from dataclasses import fields
from typing import Iterable

import numpy as np

from nee.state.boundary import BoundaryData
from nee.state.iterate import PicardState as WeightedState


Array = np.ndarray


def terminal_incoming_data(state: WeightedState) -> dict[str, Array]:
    """Use one slab's terminal section as the next slab's fixed data."""

    return {
        "metric": state.metric[:, :, -1].copy(),
        "weighted_tr_chib": state.a_in[:, :, -1].copy(),
        "weighted_hatchib": state.sigma_in[:, :, -1].copy(),
        "weighted_chib": state.x_in[:, :, -1].copy(),
        "weighted_tr_chi": state.a_out[:, :, -1].copy(),
        "weighted_hatchi": state.sigma_out[:, :, -1].copy(),
        "weighted_chi": state.x_out[:, :, -1].copy(),
        "omega": state.omega[:, :, -1].copy(),
        "weighted_omega": state.w_out[:, :, -1].copy(),
        "weighted_omegab": state.w_in[:, :, -1].copy(),
        "zeta_up": state.zeta_up[:, :, -1].copy(),
        "shift": state.shift[:, :, -1].copy(),
    }


def outgoing_segment(
    outgoing: dict[str, Array],
    global_v: Array,
    local_v: Array,
    *,
    tolerance: float = 5.0e-14,
) -> dict[str, Array]:
    """Extract one slab's outgoing-face nodes from a global artifact."""

    indices = []
    for value in np.asarray(local_v):
        index = int(np.argmin(np.abs(np.asarray(global_v) - value)))
        error = abs(float(global_v[index] - value))
        if error > tolerance:
            raise ValueError(
                "local slab node is absent from the global boundary mesh: "
                f"v={value:.17g}, nearest error={error:.3e}"
            )
        indices.append(index)
    return {
        name: np.take(np.asarray(value), indices, axis=1).copy()
        for name, value in outgoing.items()
    }


def slab_boundary(
    outgoing: dict[str, Array],
    incoming: dict[str, Array],
) -> BoundaryData:
    """Create and corner-check one slab's immutable data."""

    for name in ("metric", "zeta_up", "shift"):
        mismatch = float(
            np.max(
                np.abs(
                    np.asarray(outgoing[name])[:, 0]
                    - np.asarray(incoming[name])[:, 0]
                )
            )
        )
        if mismatch > 5.0e-11:
            raise ValueError(
                f"slab corner mismatch in {name}: {mismatch:.6e}"
            )
    return BoundaryData.create(outgoing, incoming)


def concatenate_states(states: Iterable[WeightedState]) -> WeightedState:
    """Join settled slabs, retaining a shared interface only once."""

    materialized = list(states)
    if not materialized:
        raise ValueError("at least one settled slab is required")
    arrays: dict[str, Array | None] = {}
    for item in fields(WeightedState):
        pieces = [getattr(state, item.name) for state in materialized]
        if pieces[0] is None:
            if any(piece is not None for piece in pieces):
                raise ValueError(
                    f"inconsistent optional state field {item.name!r}"
                )
            arrays[item.name] = None
            continue
        arrays[item.name] = np.concatenate(
            [
                np.asarray(piece) if index == 0 else np.asarray(piece)[:, :, 1:]
                for index, piece in enumerate(pieces)
            ],
            axis=2,
        )
    return WeightedState(**arrays)
