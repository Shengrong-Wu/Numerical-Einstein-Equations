"""Complete immutable-input Picard sweep."""

from __future__ import annotations

from typing import Any

import numpy as np

from nee.discretization.double_null_mesh import Discretization
from nee.equations.base import EquationSystem
from nee.state.boundary import BoundaryData
from nee.state.iterate import PicardState
from nee.state.validation import validate_state

from .backend import scalar_step, vacuum_step, weighted_update_map, weighted_update_norm
from .result import SweepResult


def _trace_errors(state: PicardState, boundary: BoundaryData) -> dict[str, float]:
    aliases = {
        "metric": "sphere_metric",
        "sphere_metric": "sphere_metric",
        "shift": "shift",
        "omega": "lapse",
        "lapse": "lapse",
        "zeta_up": "torsion",
        "torsion": "torsion",
        "weighted_chib": "incoming_null_form",
        "incoming_null_form": "incoming_null_form",
        "weighted_omega": "outgoing_weighted_omega",
        "weighted_omegab": "incoming_weighted_omega",
        "phi": "scalar",
        "scalar_p": "scalar_e4",
        "incoming_scalar": "scalar_e3",
    }
    errors: dict[str, float] = {}
    for side, values, state_index, boundary_index in (
        ("outgoing", boundary.outgoing, (slice(None), 0), (slice(None),)),
        ("incoming", boundary.incoming, (slice(None), slice(None), 0), (slice(None), slice(None))),
    ):
        for name, expected in values.items():
            field = aliases.get(name)
            if field is None:
                continue
            actual = getattr(state, field)
            if actual is None:
                continue
            selected = np.asarray(actual)[state_index]
            target = np.asarray(expected)[boundary_index]
            if selected.shape == target.shape:
                errors[f"{side}.{name}"] = float(np.max(np.abs(selected - target)))
    return errors


def picard_sweep(
    state: PicardState,
    boundary: BoundaryData,
    discretization: Discretization,
    equations: EquationSystem,
) -> SweepResult:
    """Return the complete next iterate while preserving both inputs."""

    state_before = {name: value.tobytes() for name, value in state.arrays().items()}
    boundary_hash = boundary.content_hash
    outgoing = {name: np.asarray(value) for name, value in boundary.outgoing.items()}
    incoming = {name: np.asarray(value) for name, value in boundary.incoming.items()}
    options: dict[str, Any] = dict(discretization.options)
    options.setdefault("metric_substeps", discretization.metric_substeps)
    if equations.has_scalar:
        if discretization.angular is None or discretization.coordinates is None:
            raise ValueError("scalar sweeps require angular and coordinate discretizations")
        candidate, context = scalar_step(
            discretization.grid,
            discretization.angular,
            discretization.coordinates,
            state,
            outgoing,
            incoming,
            **options,
        )
    else:
        options.setdefault("angular", discretization.angular)
        options.setdefault("scalar_coordinates", discretization.coordinates)
        candidate, context = vacuum_step(
            discretization.grid,
            state,
            outgoing,
            incoming,
            discretization.u,
            discretization.v,
            **options,
        )
    validate_state(candidate, discretization.grid.frames)
    boundary.verify_unchanged()
    if boundary.content_hash != boundary_hash:
        raise RuntimeError("boundary content hash changed during the sweep")
    for name, value in state.arrays().items():
        if value.tobytes() != state_before[name]:
            raise RuntimeError(f"input state field {name} was mutated")
    update_map = weighted_update_map(candidate, state)
    return SweepResult(
        state=candidate,
        weighted_update=weighted_update_norm(candidate, state),
        maximum_update_map=float(np.max(update_map)),
        trace_errors=_trace_errors(candidate, boundary),
        diagnostics=dict(context),
    )

