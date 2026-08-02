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
from nee.state.iterate import PicardState
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
) -> PicardState:
    """Construct exactly the prescribed zeroth Picard iterate.

    Every field except the b is extended constantly from ``v=0``.  The
    b is obtained by integrating

        partial_v b^(0) = -4 (Omega^(0))^2 zeta^(0)

    with all right-hand-side fields held at their incoming-face values.
    """

    incoming = boundary.incoming
    v = np.asarray(v)
    v_count = len(v)
    g = _extend_in_v(incoming["g"], v_count)
    Omega = _extend_in_v(incoming["Omega"], v_count)
    zeta = _extend_in_v(incoming["zeta"], v_count)
    Omega_chib = _extend_in_v(incoming["Omega_chib"], v_count)

    Omega_trchi = _extend_in_v(
        incoming["Omega_trchi"], v_count
    )
    if "Omega_chi" in incoming:
        Omega_chi = _extend_in_v(incoming["Omega_chi"], v_count)
    else:
        sigma_out_face = np.asarray(
            incoming.get(
                "Omega_chih",
                np.zeros_like(incoming["Omega_chib"]),
            )
        )
        Omega_chih = _extend_in_v(sigma_out_face, v_count)
        Omega_chi = (
            0.5 * Omega_trchi[..., None, None] * g + Omega_chih
        )

    shift_face = np.asarray(incoming["b"])
    b = _extend_in_v(shift_face, v_count)
    b -= (
        4.0
        * Omega[..., None] ** 2
        * zeta
        * (v - v[0]).reshape((1, 1, v_count, 1))
    )

    scalar_shape = g.shape[:-2]
    Omega_omega = _extend_in_v(
        np.asarray(
            incoming.get(
                "Omega_omega",
                np.zeros(g.shape[:2], dtype=g.dtype),
            )
        ),
        v_count,
    )
    Omega_omegab = _extend_in_v(incoming["Omega_omegab"], v_count)
    state = PicardState(
        g=g,
        b=b,
        log_Omega=np.log(Omega),
        Omega_chi=Omega_chi,
        Omega_chib=Omega_chib,
        zeta=zeta,
        Omega_omega=np.broadcast_to(Omega_omega, scalar_shape).copy(),
        Omega_omegab=np.broadcast_to(Omega_omegab, scalar_shape).copy(),
    )
    state.validate(grid.frames)
    assert_incoming_traces(state, boundary)
    return state


def impose_characteristic_traces(
    state: PicardState,
    boundary: BoundaryData,
) -> PicardState:
    """Restore all prescribed face values on a copied state."""

    result = state.copy()
    outgoing = boundary.outgoing
    incoming = boundary.incoming

    # H_{-1}: array index u=0.
    result.g[:, 0] = outgoing["g"]
    result.log_Omega[:, 0] = np.log(outgoing["Omega"])
    result.b[:, 0] = outgoing["b"]
    result.zeta[:, 0] = outgoing["zeta"]
    result.Omega_omega[:, 0] = outgoing["Omega_omega"]
    result.Omega_chi[:, 0] = (
        outgoing["Omega_chih"]
        + 0.5
        * outgoing["Omega_trchi"][..., None, None]
        * outgoing["g"]
    )

    # Hbar_0: array index v=0.  Apply this second so the single shared corner
    # is represented by the incoming copy; corner validation guarantees that
    # the prescribed fields agree there.
    result.g[:, :, 0] = incoming["g"]
    result.log_Omega[:, :, 0] = np.log(incoming["Omega"])
    result.b[:, :, 0] = incoming["b"]
    result.zeta[:, :, 0] = incoming["zeta"]
    result.Omega_omegab[:, :, 0] = incoming["Omega_omegab"]
    result.Omega_chib[:, :, 0] = incoming["Omega_chib"]

    if "Omega_chi" in incoming:
        # A restarted slab carries the complete terminal value of the
        # preceding slab.
        result.Omega_chi[:, :, 0] = incoming["Omega_chi"]
    else:
        # On the original v=0 face the data prescribe A_+ but no independent
        # transverse Omega_chih.  Preserve the candidate Omega_chih and restore A_+.
        incoming_metric = result.g[:, :, 0]
        incoming_inverse = tangent_inverse(incoming_metric)
        Omega_chih = tracefree(
            result.Omega_chi[:, :, 0],
            incoming_metric,
            incoming_inverse,
        )
        result.Omega_chi[:, :, 0] = (
            Omega_chih
            + 0.5
            * incoming["Omega_trchi"][..., None, None]
            * incoming_metric
        )
    if "Omega_omega" in incoming:
        result.Omega_omega[:, :, 0] = incoming["Omega_omega"]
    return result


def incoming_trace_errors(
    state: PicardState,
    boundary: BoundaryData,
) -> dict[str, float]:
    """Return maximum component errors on the fixed ``v=0`` face."""

    incoming = boundary.incoming
    errors = {
        "g": float(
            np.max(np.abs(state.g[:, :, 0] - incoming["g"]))
        ),
        "Omega": float(
            np.max(np.abs(state.Omega[:, :, 0] - incoming["Omega"]))
        ),
        "b": float(
            np.max(np.abs(state.b[:, :, 0] - incoming["b"]))
        ),
        "zeta": float(
            np.max(np.abs(state.zeta[:, :, 0] - incoming["zeta"]))
        ),
        "Omega_chib": float(
            np.max(
                np.abs(
                    state.Omega_chib[:, :, 0] - incoming["Omega_chib"]
                )
            )
        ),
        "Omega_omegab": float(
            np.max(
                np.abs(
                    state.Omega_omegab[:, :, 0] - incoming["Omega_omegab"]
                )
            )
        ),
        "Omega_trchi": float(
            np.max(
                np.abs(
                    tensor_trace(
                        state.Omega_chi[:, :, 0],
                        tangent_inverse(state.g[:, :, 0]),
                    )
                    - incoming["Omega_trchi"]
                )
            )
        ),
    }
    if "Omega_chi" in incoming:
        errors["Omega_chi"] = float(
            np.max(
                np.abs(
                    state.Omega_chi[:, :, 0] - incoming["Omega_chi"]
                )
            )
        )
    if "Omega_omega" in incoming:
        errors["Omega_omega"] = float(
            np.max(
                np.abs(
                    state.Omega_omega[:, :, 0]
                    - incoming["Omega_omega"]
                )
            )
        )
    return errors


def outgoing_trace_errors(
    state: PicardState,
    boundary: BoundaryData,
) -> dict[str, float]:
    """Return maximum component errors on the fixed ``u=-1`` face."""

    outgoing = boundary.outgoing
    expected_x_out = (
        outgoing["Omega_chih"]
        + 0.5
        * outgoing["Omega_trchi"][..., None, None]
        * outgoing["g"]
    )
    return {
        "g": float(
            np.max(np.abs(state.g[:, 0] - outgoing["g"]))
        ),
        "Omega": float(
            np.max(np.abs(state.Omega[:, 0] - outgoing["Omega"]))
        ),
        "b": float(
            np.max(np.abs(state.b[:, 0] - outgoing["b"]))
        ),
        "zeta": float(
            np.max(np.abs(state.zeta[:, 0] - outgoing["zeta"]))
        ),
        "Omega_chi": float(
            np.max(np.abs(state.Omega_chi[:, 0] - expected_x_out))
        ),
        "Omega_omega": float(
            np.max(
                np.abs(
                    state.Omega_omega[:, 0] - outgoing["Omega_omega"]
                )
            )
        ),
    }


def assert_incoming_traces(
    state: PicardState,
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
    state: PicardState,
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
    current: PicardState,
    boundary: BoundaryData,
    *,
    metric_substeps: int,
    transport_cfl: float = 0.45,
) -> tuple[PicardState, dict[str, Any]]:
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
    previous: PicardState,
    candidate: PicardState,
    boundary: BoundaryData,
    relaxation: float,
) -> PicardState:
    """Relax the full state and then reimpose the immutable face data."""

    if not 0.0 < relaxation <= 1.0:
        raise ValueError("Picard relaxation must lie in (0,1]")

    def blend(name: str) -> Array:
        old = np.asarray(getattr(previous, name))
        new = np.asarray(getattr(candidate, name))
        return old + relaxation * (new - old)

    relaxed = PicardState(
        g=blend("g"),
        b=blend("b"),
        log_Omega=blend("log_Omega"),
        Omega_chi=blend("Omega_chi"),
        Omega_chib=blend("Omega_chib"),
        zeta=blend("zeta"),
        Omega_omega=blend("Omega_omega"),
        Omega_omegab=blend("Omega_omegab"),
    )
    relaxed = impose_characteristic_traces(relaxed, boundary)
    assert_characteristic_traces(relaxed, boundary)
    return relaxed
