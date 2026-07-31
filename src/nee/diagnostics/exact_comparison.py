"""Fieldwise exact-solution comparison helpers."""

from __future__ import annotations

import numpy as np


Array = np.ndarray


def field_error(actual: Array, expected: Array) -> dict[str, float]:
    """Return maximum, RMS, and relative-RMS errors."""

    difference = np.asarray(actual) - np.asarray(expected)
    absolute_rms = float(np.sqrt(np.mean(difference**2)))
    exact_rms = float(np.sqrt(np.mean(np.asarray(expected) ** 2)))
    return {
        "absolute_maximum": float(np.max(np.abs(difference))),
        "absolute_rms": absolute_rms,
        "relative_rms": absolute_rms / max(exact_rms, 1.0e-14),
    }
