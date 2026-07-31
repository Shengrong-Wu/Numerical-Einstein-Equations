"""Intrinsic sphere-tensor algebra used throughout the solver."""

from nee.geometry.sphere import (
    projected_spin2,
    tangent_inverse,
    tensor_norm_sq,
    tensor_trace,
    tensor_tracefree,
    tracefree_square,
)

__all__ = [
    "projected_spin2",
    "tangent_inverse",
    "tensor_norm_sq",
    "tensor_trace",
    "tensor_tracefree",
    "tracefree_square",
]
