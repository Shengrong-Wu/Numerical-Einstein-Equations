"""Analytic scalar-pulse data and exact-prefix continuation rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np


Array = np.ndarray
Field = TypeVar("Field", bound=np.ndarray)


@dataclass(frozen=True)
class TrappedSectionParameters:
    """Physical and coordinate parameters for Experiment 8."""

    kappa: float = 0.25
    delta: float = 0.1
    pulse_scale: float = 0.01
    scalar_amplitude: float = -2.0
    shear_amplitude: float = 0.1
    v_max: float = 0.04
    base_tau_elements: int = 12

    def validate(self) -> None:
        if not 0.0 < self.kappa < 1.0:
            raise ValueError("kappa must lie in (0,1)")
        if not 0.0 < self.delta <= 1.0:
            raise ValueError("delta must lie in (0,1]")
        if self.pulse_scale <= 0.0 or self.v_max <= 0.0:
            raise ValueError("pulse_scale and v_max must be positive")
        if not np.isfinite(self.scalar_amplitude):
            raise ValueError("scalar_amplitude must be finite")
        if self.shear_amplitude < 0.0:
            raise ValueError("shear_amplitude must be nonnegative")
        if self.base_tau_elements < 1:
            raise ValueError("base_tau_elements must be positive")

    @property
    def base_tau_right(self) -> float:
        return float(np.log(50.0))

    @property
    def tau_element_width(self) -> float:
        return self.base_tau_right / self.base_tau_elements

    @property
    def continued_tau_right(self) -> float:
        return self.base_tau_right + self.tau_element_width

    @property
    def continued_u_right(self) -> float:
        return -float(np.exp(-self.continued_tau_right))


def pulse_profile(v: Array, parameters: TrappedSectionParameters) -> Array:
    """Return h(v)=(v/(v+rho))**delta with its exact corner value."""

    parameters.validate()
    values = np.asarray(v, dtype=float)
    if np.any(values < 0.0):
        raise ValueError("v must be nonnegative")
    profile = (values / (values + parameters.pulse_scale)) ** parameters.delta
    return np.where(values == 0.0, 0.0, profile)


def incoming_spherical_data(
    u: Array, parameters: TrappedSectionParameters
) -> dict[str, Array]:
    """Evaluate the analytic data on the incoming characteristic face."""

    parameters.validate()
    values = np.asarray(u, dtype=float)
    if np.any(values >= 0.0):
        raise ValueError("u must be negative")
    radius = -values
    kappa = parameters.kappa
    omega_sq = radius**kappa
    outgoing_trace = 2.0 / (1.0 + kappa) * radius ** (kappa - 1.0)
    return {
        "radius": radius,
        "omega_squared": omega_sq,
        "phi": np.sqrt(2.0 * kappa) * np.log(radius),
        "incoming_weighted_trace": -2.0 / radius,
        "incoming_weighted_lapse": kappa / (4.0 * radius),
        "Omega_e3phi": -np.sqrt(2.0 * kappa) / radius,
        "outgoing_weighted_trace": outgoing_trace,
        "outgoing_inverse_lapse_trace": outgoing_trace / omega_sq,
        "outgoing_scalar": (
            -np.sqrt(2.0 / kappa)
            / (1.0 + kappa)
            * radius ** (kappa - 1.0)
        ),
        "outgoing_weighted_lapse": (
            radius ** (kappa - 1.0) - 1.0
        )
        / (2.0 * (1.0 + kappa)),
    }


def outgoing_scalar_data(
    v: Array, parameters: TrappedSectionParameters
) -> Array:
    """Evaluate the prescribed outgoing scalar derivative."""

    corner = -np.sqrt(2.0 / parameters.kappa) / (
        1.0 + parameters.kappa
    )
    return corner + parameters.scalar_amplitude * pulse_profile(v, parameters)


def quadrupole_coordinate_components(
    theta: Array, parameters: TrappedSectionParameters
) -> tuple[Array, Array, Array]:
    """Return covariant (X_theta_theta, X_theta_phi, X_phi_phi)."""

    parameters.validate()
    sine = np.sin(np.asarray(theta, dtype=float))
    amplitude = parameters.shear_amplitude
    return (
        -amplitude * sine**2,
        np.zeros_like(sine),
        amplitude * sine**4,
    )


def prolong_exact_prefix(
    source: Field,
    seed: Field,
    source_u_count: int,
    *,
    u_axis: int = 1,
) -> Field:
    """Copy a complete prefix and add its endpoint correction to the tail."""

    old = np.asarray(source)
    fresh = np.asarray(seed)
    if old.ndim != fresh.ndim:
        raise ValueError("source and seed ranks differ")
    axis = int(u_axis) % fresh.ndim
    if source_u_count < 1 or source_u_count >= fresh.shape[axis]:
        raise ValueError("source_u_count must define a proper nonempty prefix")
    if old.shape[axis] != source_u_count:
        raise ValueError("source does not have the declared u count")
    for index, (old_size, new_size) in enumerate(
        zip(old.shape, fresh.shape, strict=True)
    ):
        if index != axis and old_size != new_size:
            raise ValueError("source and seed transverse shapes differ")

    result = np.array(fresh, copy=True)
    prefix = [slice(None)] * result.ndim
    prefix[axis] = slice(0, source_u_count)
    result[tuple(prefix)] = old

    old_end = [slice(None)] * old.ndim
    old_end[axis] = slice(source_u_count - 1, source_u_count)
    seed_end = [slice(None)] * fresh.ndim
    seed_end[axis] = slice(source_u_count - 1, source_u_count)
    tail = [slice(None)] * result.ndim
    tail[axis] = slice(source_u_count, None)
    result[tuple(tail)] += old[tuple(old_end)] - fresh[tuple(seed_end)]
    return result
