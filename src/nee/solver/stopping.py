"""Stopping predicates for Picard sweeps and slabs."""

import math


def has_converged(update: float, tolerance: float) -> bool:
    """Return whether a finite nonnegative update meets the tolerance."""

    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if not math.isfinite(update) or update < 0.0:
        return False
    return update <= tolerance
