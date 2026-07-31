"""Boundary-preserving Picard I/O for characteristic vacuum slabs.

The shared equation kernel returns a raw full-state candidate.  This module
owns the experiment-specific seed, fixed characteristic traces, relaxation,
and trace validation.  In particular, angular projection is never allowed to
replace prescribed nodal data on either characteristic face.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from nee.solver.backend import vacuum_picard_step
from nee.state.boundary import BoundaryData
from nee.state.iterate import PicardState as WeightedState
from nee.state.fields import (
    tangent_inverse,
    tensor_trace,
    tracefree,
)


Array = np.ndarray


def _extend_in_v(value: Array, v_count: int) -> Array:
    """Return the constant-v extension of an incoming-face array."""

    value = np.asarray(value)
    return np.broadcast_to(
        value[:, :, None],
        value.shape[:2] + (v_count,) + value.shape[2:],
    ).copy()


def initial_iterate(
    grid: Any,
    v: Array,
    boundary: BoundaryData,
) -> WeightedState:
    """Construct exactly the prescribed zeroth Picard iterate.

    Every field except the shift is extended constantly from ``v=0``.  The
    shift is obtained by integrating

        partial_v b^(0) = -4 (Omega^(0))^2 zeta^(0)

    with all right-hand-side fields held at their incoming-face values.
    """

    incoming = boundary.incoming
    v = np.asarray(v)
    v_count = len(v)
    metric = _extend_in_v(incoming["metric"], v_count)
    omega = _extend_in_v(incoming["omega"], v_count)
    zeta_up = _extend_in_v(incoming["zeta_up"], v_count)
    x_in = _extend_in_v(incoming["weighted_chib"], v_count)

    weighted_tr_chi = _extend_in_v(
        incoming["weighted_tr_chi"], v_count
    )
    if "weighted_chi" in incoming:
        x_out = _extend_in_v(incoming["weighted_chi"], v_count)
    else:
        sigma_out_face = np.asarray(
            incoming.get(
                "weighted_hatchi",
                np.zeros_like(incoming["weighted_chib"]),
            )
        )
        sigma_out = _extend_in_v(sigma_out_face, v_count)
        x_out = (
            0.5 * weighted_tr_chi[..., None, None] * metric + sigma_out
        )

    shift_face = np.asarray(incoming["shift"])
    shift = _extend_in_v(shift_face, v_count)
    shift -= (
        4.0
        * omega[..., None] ** 2
        * zeta_up
        * (v - v[0]).reshape((1, 1, v_count, 1))
    )

    scalar_shape = metric.shape[:-2]
    w_out = _extend_in_v(
        np.asarray(
            incoming.get(
                "weighted_omega",
                np.zeros(metric.shape[:2], dtype=metric.dtype),
            )
        ),
        v_count,
    )
    w_in = _extend_in_v(incoming["weighted_omegab"], v_count)
    state = WeightedState(
        sphere_metric=metric,
        shift=shift,
        log_lapse=np.log(omega),
        outgoing_null_form=x_out,
        incoming_null_form=x_in,
        torsion=zeta_up,
        outgoing_weighted_omega=np.broadcast_to(w_out, scalar_shape).copy(),
        incoming_weighted_omega=np.broadcast_to(w_in, scalar_shape).copy(),
    )
    state.validate(grid.frames)
    assert_incoming_traces(state, boundary)
    return state


def impose_characteristic_traces(
    state: WeightedState,
    boundary: BoundaryData,
) -> WeightedState:
    """Restore all prescribed face values on a copied state."""

    result = state.copy()
    outgoing = boundary.outgoing
    incoming = boundary.incoming

    # H_{-1}: array index u=0.
    result.metric[:, 0] = outgoing["metric"]
    result.log_omega[:, 0] = np.log(outgoing["omega"])
    result.shift[:, 0] = outgoing["shift"]
    result.zeta_up[:, 0] = outgoing["zeta_up"]
    result.w_out[:, 0] = outgoing["weighted_omega"]
    result.x_out[:, 0] = (
        outgoing["shear"]
        + 0.5
        * outgoing["expansion"][..., None, None]
        * outgoing["metric"]
    )

    # Hbar_0: array index v=0.  Apply this second so the single shared corner
    # is represented by the incoming copy; corner validation guarantees that
    # the prescribed fields agree there.
    result.metric[:, :, 0] = incoming["metric"]
    result.log_omega[:, :, 0] = np.log(incoming["omega"])
    result.shift[:, :, 0] = incoming["shift"]
    result.zeta_up[:, :, 0] = incoming["zeta_up"]
    result.w_in[:, :, 0] = incoming["weighted_omegab"]
    result.x_in[:, :, 0] = incoming["weighted_chib"]

    if "weighted_chi" in incoming:
        # A restarted slab carries the complete terminal value of the
        # preceding slab.
        result.x_out[:, :, 0] = incoming["weighted_chi"]
    else:
        # On the original v=0 face the data prescribe A_+ but no independent
        # transverse shear.  Preserve the candidate shear and restore A_+.
        incoming_metric = result.metric[:, :, 0]
        incoming_inverse = tangent_inverse(incoming_metric)
        sigma_out = tracefree(
            result.x_out[:, :, 0],
            incoming_metric,
            incoming_inverse,
        )
        result.x_out[:, :, 0] = (
            sigma_out
            + 0.5
            * incoming["weighted_tr_chi"][..., None, None]
            * incoming_metric
        )
    if "weighted_omega" in incoming:
        result.w_out[:, :, 0] = incoming["weighted_omega"]
    return result


def incoming_trace_errors(
    state: WeightedState,
    boundary: BoundaryData,
) -> dict[str, float]:
    """Return maximum component errors on the fixed ``v=0`` face."""

    incoming = boundary.incoming
    errors = {
        "metric": float(
            np.max(np.abs(state.metric[:, :, 0] - incoming["metric"]))
        ),
        "omega": float(
            np.max(np.abs(state.omega[:, :, 0] - incoming["omega"]))
        ),
        "shift": float(
            np.max(np.abs(state.shift[:, :, 0] - incoming["shift"]))
        ),
        "zeta_up": float(
            np.max(np.abs(state.zeta_up[:, :, 0] - incoming["zeta_up"]))
        ),
        "x_in": float(
            np.max(
                np.abs(
                    state.x_in[:, :, 0] - incoming["weighted_chib"]
                )
            )
        ),
        "w_in": float(
            np.max(
                np.abs(
                    state.w_in[:, :, 0] - incoming["weighted_omegab"]
                )
            )
        ),
        "a_out": float(
            np.max(
                np.abs(
                    tensor_trace(
                        state.x_out[:, :, 0],
                        tangent_inverse(state.metric[:, :, 0]),
                    )
                    - incoming["weighted_tr_chi"]
                )
            )
        ),
    }
    if "weighted_chi" in incoming:
        errors["x_out"] = float(
            np.max(
                np.abs(
                    state.x_out[:, :, 0] - incoming["weighted_chi"]
                )
            )
        )
    if "weighted_omega" in incoming:
        errors["w_out"] = float(
            np.max(
                np.abs(
                    state.w_out[:, :, 0]
                    - incoming["weighted_omega"]
                )
            )
        )
    return errors


def outgoing_trace_errors(
    state: WeightedState,
    boundary: BoundaryData,
) -> dict[str, float]:
    """Return maximum component errors on the fixed ``u=-1`` face."""

    outgoing = boundary.outgoing
    expected_x_out = (
        outgoing["shear"]
        + 0.5
        * outgoing["expansion"][..., None, None]
        * outgoing["metric"]
    )
    return {
        "metric": float(
            np.max(np.abs(state.metric[:, 0] - outgoing["metric"]))
        ),
        "omega": float(
            np.max(np.abs(state.omega[:, 0] - outgoing["omega"]))
        ),
        "shift": float(
            np.max(np.abs(state.shift[:, 0] - outgoing["shift"]))
        ),
        "zeta_up": float(
            np.max(np.abs(state.zeta_up[:, 0] - outgoing["zeta_up"]))
        ),
        "x_out": float(
            np.max(np.abs(state.x_out[:, 0] - expected_x_out))
        ),
        "w_out": float(
            np.max(
                np.abs(
                    state.w_out[:, 0] - outgoing["weighted_omega"]
                )
            )
        ),
    }


def assert_incoming_traces(
    state: WeightedState,
    boundary: BoundaryData,
    *,
    tolerance: float = 5.0e-12,
) -> None:
    errors = incoming_trace_errors(state, boundary)
    maximum = max(errors.values())
    if maximum > tolerance:
        raise RuntimeError(
            "incoming characteristic trace changed: "
            f"maximum={maximum:.6e}, fields={errors}"
        )


def assert_characteristic_traces(
    state: WeightedState,
    boundary: BoundaryData,
    *,
    tolerance: float = 5.0e-12,
) -> None:
    errors = {
        "incoming": incoming_trace_errors(state, boundary),
        "outgoing": outgoing_trace_errors(state, boundary),
    }
    maximum = max(
        value
        for face_errors in errors.values()
        for value in face_errors.values()
    )
    if maximum > tolerance:
        raise RuntimeError(
            "characteristic trace changed: "
            f"maximum={maximum:.6e}, fields={errors}"
        )


def raw_picard_candidate(
    grid: Any,
    mesh: Any,
    angular: Any,
    current: WeightedState,
    boundary: BoundaryData,
    *,
    metric_substeps: int,
    transport_cfl: float = 0.45,
) -> tuple[WeightedState, dict[str, Any]]:
    """Apply one equation sweep and restore its prescribed traces."""

    # The declared U^(0) is the incoming-face extension and therefore does
    # not also contain the nontrivial H_{-1} data.  The equation kernel reads
    # its current-state values on H_{-1} while constructing the u-marches.
    # Supply those immutable data on a private working copy: the public input
    # state remains unchanged.
    working = current.copy()
    candidate, context = vacuum_picard_step(
        grid,
        working,
        dict(boundary.outgoing),
        mesh.u,
        mesh.v,
        metric_substeps=metric_substeps,
        angular=angular,
        coordinates=mesh,
        metric_parameterization="cholesky",
        metric_integrator="rk4",
        u_integrator="rk4",
        incoming=dict(boundary.incoming),
        impose_previous_outgoing_boundary=True,
        transport_cfl=transport_cfl,
    )
    pre_restore = {
        "incoming": incoming_trace_errors(candidate, boundary),
        "outgoing": outgoing_trace_errors(candidate, boundary),
    }
    candidate = impose_characteristic_traces(candidate, boundary)
    candidate.validate(grid.frames)
    assert_characteristic_traces(candidate, boundary)
    context = dict(context)
    context["trace_errors_before_restore"] = pre_restore
    context["trace_errors_after_restore"] = {
        "incoming": incoming_trace_errors(candidate, boundary),
        "outgoing": outgoing_trace_errors(candidate, boundary),
    }
    return candidate, context


def relaxed_next_iterate(
    previous: WeightedState,
    candidate: WeightedState,
    boundary: BoundaryData,
    relaxation: float,
) -> WeightedState:
    """Relax the full state and then reimpose the immutable face data."""

    if not 0.0 < relaxation <= 1.0:
        raise ValueError("Picard relaxation must lie in (0,1]")

    def blend(name: str) -> Array:
        old = np.asarray(getattr(previous, name))
        new = np.asarray(getattr(candidate, name))
        return old + relaxation * (new - old)

    relaxed = WeightedState(
        sphere_metric=blend("metric"),
        shift=blend("shift"),
        log_lapse=blend("log_omega"),
        outgoing_null_form=blend("x_out"),
        incoming_null_form=blend("x_in"),
        torsion=blend("zeta_up"),
        outgoing_weighted_omega=blend("w_out"),
        incoming_weighted_omega=blend("w_in"),
    )
    relaxed = impose_characteristic_traces(relaxed, boundary)
    assert_characteristic_traces(relaxed, boundary)
    return relaxed
