"""State, positivity, and characteristic-trace validation."""

from __future__ import annotations

import numpy as np

from .fields import tensor_trace
from .iterate import PicardState


def validate_state(state: PicardState, frames: np.ndarray | None = None) -> dict[str, float]:
    arrays = state.arrays()
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise FloatingPointError("state contains nonfinite values")
    if float(np.min(state.Omega)) <= 0.0:
        raise FloatingPointError("Omega is nonpositive")
    symmetry = max(
        float(np.max(np.abs(value - np.swapaxes(value, -1, -2))))
        for value in (
            state.g,
            state.Omega_chi,
            state.Omega_chib,
        )
    )
    if symmetry > 1.0e-9:
        raise ValueError(f"symmetric-tensor drift is {symmetry:.3e}")
    result = {
        "minimum_lapse": float(np.min(state.Omega)),
        "maximum_outgoing_shear_trace": float(
            np.max(np.abs(tensor_trace(state.Omega_chih, state.inverse_g)))
        ),
        "maximum_incoming_shear_trace": float(
            np.max(np.abs(tensor_trace(state.Omega_chibh, state.inverse_g)))
        ),
    }
    if frames is not None:
        local = np.einsum("nia,n...ij,njb->n...ab", frames, state.g, frames)
        minimum = float(np.min(np.linalg.eigvalsh(local)))
        if minimum <= 0.0:
            raise FloatingPointError(f"sphere g left the positive cone: {minimum:.12g}")
        result["minimum_metric_eigenvalue"] = minimum
    return result

