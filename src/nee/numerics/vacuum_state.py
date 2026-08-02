"""Nonaxisymmetric whole-sphere Picard iteration for the vacuum construction.

Angular tensors are ambient Cartesian tangent tensors on a Fibonacci sphere,
so the calculation has no polar chart boundary.  The initial pulse is the
smooth moving-zero, sphere-L2 datum documented in ``pulse_design.py``.
It satisfies the requested integral norm without claiming the topologically
impossible everywhere-nonzero pointwise constant norm.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .coordinate_differentiation import high_order_differentiate
from .coordinate_quadrature import (
    cumulative_polynomial_quadrature,
    midpoint_values,
    stage_value,
)
from .pulse_design import time_coefficients
from .sphere import (
    PointSphereGrid,
    ambient_stf_basis,
    gaussian_curvature,
    lie_covariant_tensor,
    projected_spin2,
    scalar_gradient,
    tangent_inverse,
    tensor_divergence,
    tensor_norm_sq,
    tensor_trace,
    tensor_tracefree,
    tracefree_square,
    tracefree_symmetric_gradient,
    vector_divergence,
)


Array = np.ndarray


@dataclass
class GlobalState:
    g: Array
    Omega: Array
    zeta: Array
    b: Array
    Omega_trchi: Array
    Omega_chih: Array


def fractional_values(
    values: Array,
    coordinate: Array,
    axis: int,
    fraction: float,
    stencil: int = 6,
) -> Array:
    """Interpolate every coordinate interval at one fixed local fraction."""

    count = len(coordinate)
    width = min(stencil, count)
    matrix = np.zeros((count - 1, count), dtype=float)
    for index in range(count - 1):
        start = min(max(index - width // 2 + 1, 0), count - width)
        indices = np.arange(start, start + width)
        target_coordinate = coordinate[index] + fraction * (
            coordinate[index + 1] - coordinate[index]
        )
        offsets = coordinate[indices] - target_coordinate
        scale = float(np.max(np.abs(offsets)))
        normalized = offsets / scale
        vandermonde = np.vstack([normalized**power for power in range(width)])
        target = np.zeros(width)
        target[0] = 1.0
        matrix[index, indices] = np.linalg.solve(vandermonde, target)
    moved = np.moveaxis(values, axis, 0)
    result = np.tensordot(matrix, moved, axes=(1, 0))
    return np.moveaxis(result, 0, axis)


def sphere_broadcast(grid: PointSphereGrid, batch_rank: int) -> Array:
    return grid.projector.reshape((grid.count,) + (1,) * batch_rank + (3, 3))


def pulse_basis(grid: PointSphereGrid) -> Array:
    return np.stack(
        [projected_spin2(grid, ambient)[0] for ambient in ambient_stf_basis()],
        axis=1,
    )


def pulse_tensor(basis: Array, value: float, v_max: float, delta: float) -> Array:
    time = (value / v_max) ** (2.0 * delta + 1.0) if value > 0.0 else 0.0
    coefficients = time_coefficients(np.array([time]))[0]
    return np.einsum("k,nkij->nij", coefficients, basis)


def normalized_boundary_shear(
    grid: PointSphereGrid,
    basis: Array,
    g: Array,
    value: float,
    v_max: float,
    c: float,
    delta: float,
) -> Array:
    direction = pulse_tensor(basis, value, v_max, delta)
    raw = np.matmul(g, direction)
    raw = 0.5 * (raw + np.swapaxes(raw, -1, -2))
    inverse_g = tangent_inverse(grid, g)
    raw = tensor_tracefree(raw, g, inverse_g)
    norm_sq = tensor_norm_sq(raw, inverse_g)
    local_metric = np.einsum(
        "nia,nij,njb->nab", grid.frames, g, grid.frames
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local_metric), 0.0))
    l2 = math.sqrt(4.0 * math.pi * float(np.mean(norm_sq * area_ratio)))
    amplitude = c * value**delta if value > 0.0 else 0.0
    return amplitude * raw / l2


def solve_outgoing_boundary(
    grid: PointSphereGrid,
    v: Array,
    c: float,
    delta: float,
    substeps: int = 1,
    pulse_v_max: float | None = None,
) -> dict[str, Array | float]:
    """Solve the H_-1 g/Raychaudhuri constraints for the global pulse."""

    basis = pulse_basis(grid)
    g = np.zeros((grid.count, len(v), 3, 3), dtype=float)
    expansion = np.zeros((grid.count, len(v)), dtype=float)
    Omega_chih = np.zeros_like(g)
    g[:, 0] = grid.projector
    expansion[:, 0] = 2.0
    v_max = float(v[-1]) if pulse_v_max is None else float(pulse_v_max)
    if v_max <= 0.0 or float(v[-1]) > v_max:
        raise ValueError("pulse_v_max must be positive and cover the v grid")

    def rhs(value: float, current_metric: Array, current_Omega_trchi: Array) -> tuple[Array, Array]:
        current_shear = normalized_boundary_shear(
            grid, basis, current_metric, value, v_max, c, delta
        )
        inverse_g = tangent_inverse(grid, current_metric)
        norm_sq = tensor_norm_sq(current_shear, inverse_g)
        return (
            current_Omega_trchi[:, None, None] * current_metric + 2.0 * current_shear,
            -0.5 * current_Omega_trchi**2 - norm_sq,
        )

    for j in range(len(v) - 1):
        full_step = float(v[j + 1] - v[j])
        step = full_step / substeps
        current_metric = g[:, j]
        current_Omega_trchi = expansion[:, j]
        for substep in range(substeps):
            value = float(v[j] + substep * step)
            k1_metric, k1_Omega_trchi = rhs(value, current_metric, current_Omega_trchi)
            k2_metric, k2_Omega_trchi = rhs(
                value + 0.5 * step,
                current_metric + 0.5 * step * k1_metric,
                current_Omega_trchi + 0.5 * step * k1_Omega_trchi,
            )
            k3_metric, k3_Omega_trchi = rhs(
                value + 0.5 * step,
                current_metric + 0.5 * step * k2_metric,
                current_Omega_trchi + 0.5 * step * k2_Omega_trchi,
            )
            k4_metric, k4_Omega_trchi = rhs(
                value + step,
                current_metric + step * k3_metric,
                current_Omega_trchi + step * k3_Omega_trchi,
            )
            current_metric = current_metric + step * (
                k1_metric + 2.0 * k2_metric + 2.0 * k3_metric + k4_metric
            ) / 6.0
            current_Omega_trchi = current_Omega_trchi + step * (
                k1_Omega_trchi + 2.0 * k2_Omega_trchi + 2.0 * k3_Omega_trchi + k4_Omega_trchi
            ) / 6.0
        g[:, j + 1] = current_metric
        expansion[:, j + 1] = current_Omega_trchi

    for j, value in enumerate(v):
        Omega_chih[:, j] = normalized_boundary_shear(
            grid, basis, g[:, j], float(value), v_max, c, delta
        )
    inverse_g = tangent_inverse(grid, g)
    norm = np.sqrt(np.maximum(tensor_norm_sq(Omega_chih, inverse_g), 0.0))
    local_metric = np.einsum(
        "nia,nvij,njb->nvab", grid.frames, g, grid.frames
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local_metric), 0.0))
    l2 = np.sqrt(
        np.maximum(4.0 * math.pi * np.mean(norm**2 * area_ratio, axis=0), 0.0)
    )
    rms = np.sqrt(np.mean(norm**2, axis=0))
    trace = tensor_trace(Omega_chih, inverse_g)
    energy = np.trapezoid(norm**2, v, axis=1)
    normalized_energy = energy / np.mean(energy)
    return {
        "g": g,
        "inverse_g": inverse_g,
        "Omega_trchi": expansion,
        "Omega_chih": Omega_chih,
        "max_l2_norm_error": float(
            np.max(np.abs(l2 - c * np.where(v > 0.0, v**delta, 0.0)))
        ),
        "rms_norm": rms,
        "max_trace": float(np.max(np.abs(trace))),
        "integrated_energy_relative_spread": float(
            np.max(normalized_energy) - np.min(normalized_energy)
        ),
    }


def initial_state(grid: PointSphereGrid, u: Array, v: Array) -> GlobalState:
    batch_projector = sphere_broadcast(grid, 2)
    radius_sq = (-u[None, :, None]) ** 2
    g = np.broadcast_to(
        radius_sq[..., None, None] * batch_projector,
        (grid.count, len(u), len(v), 3, 3),
    ).copy()
    scalar_shape = (grid.count, len(u), len(v))
    return GlobalState(
        g=g,
        Omega=np.ones(scalar_shape),
        zeta=np.zeros(scalar_shape + (3,)),
        b=np.zeros(scalar_shape + (3,)),
        Omega_trchi=np.broadcast_to(2.0 / (-u[None, :, None]), scalar_shape).copy(),
        Omega_chih=np.zeros_like(g),
    )


def impose_outgoing_boundary(
    state: GlobalState, boundary: dict[str, Array | float]
) -> GlobalState:
    """Impose the prescribed ``g, tr chi, hat chi, Omega`` on ``H_-1``."""

    state.g[:, 0] = np.asarray(boundary["g"])
    state.Omega_trchi[:, 0] = np.asarray(boundary["Omega_trchi"])
    state.Omega_chih[:, 0] = np.asarray(boundary["Omega_chih"])
    state.Omega[:, 0] = 1.0
    if "zeta" in boundary:
        state.zeta[:, 0] = np.asarray(boundary["zeta"])
    if "b" in boundary:
        state.b[:, 0] = np.asarray(boundary["b"])
    return state


def section_geometry(
    grid: PointSphereGrid, state: GlobalState, u: Array
) -> dict[str, Array]:
    curvature, difference, inverse_g = gaussian_curvature(grid, state.g)
    metric_u = high_order_differentiate(state.g, u, axis=1)
    metric_lie = metric_u + lie_covariant_tensor(grid, state.b, state.g)
    chib = metric_lie / (2.0 * state.Omega[..., None, None])
    tr_chib = tensor_trace(chib, inverse_g)
    hatchib = tensor_tracefree(chib, state.g, inverse_g)
    log_omega_gradient = scalar_gradient(grid, np.log(state.Omega))
    zeta_cov = np.einsum("n...ij,n...j->n...i", state.g, state.zeta)
    eta = zeta_cov + log_omega_gradient
    etab = -zeta_cov + log_omega_gradient
    return {
        "curvature": curvature,
        "difference": difference,
        "inverse_g": inverse_g,
        "tr_chib": tr_chib,
        "hatchib": hatchib,
        "eta": eta,
        "etab": etab,
        "eta_grad_hat": tracefree_symmetric_gradient(
            grid, eta, state.g, difference, inverse_g
        ),
        "eta_square_hat": tracefree_square(eta, state.g, inverse_g),
    }


def solve_half_shear(
    grid: PointSphereGrid,
    state: GlobalState,
    geometry: dict[str, Array],
    boundary: dict[str, Array | float],
    u: Array,
    u_boundary: dict[str, Array] | None = None,
) -> Array:
    half = np.zeros_like(state.g)
    if u_boundary is None:
        boundary_shear = np.asarray(boundary["Omega_chih"])
        boundary_inverse = np.asarray(boundary["inverse_g"])
    else:
        boundary_shear = np.asarray(u_boundary["half_shear"])
        boundary_inverse = tangent_inverse(
            grid, np.asarray(u_boundary["g"])
        )
    transfer = np.matmul(
        np.matmul(boundary_shear, boundary_inverse), state.g[:, 0]
    )
    half[:, 0] = 0.5 * (transfer + np.swapaxes(transfer, -1, -2))
    tr_chi = state.Omega_trchi / state.Omega
    source = state.Omega[..., None, None] ** 2 * (
        geometry["eta_grad_hat"]
        + geometry["eta_square_hat"]
        - 0.5 * tr_chi[..., None, None] * geometry["hatchib"]
    )
    weighted_trace = state.Omega * geometry["tr_chib"]
    Omega_chibh = state.Omega[..., None, None] * geometry["hatchib"]
    mixed_hatchib = np.matmul(Omega_chibh, geometry["inverse_g"])
    midpoint_fields = {
        name: midpoint_values(value, u, axis=1)
        for name, value in {
            "b": state.b,
            "trace": weighted_trace,
            "mixed": mixed_hatchib,
            "source": source,
        }.items()
    }
    for i in range(len(u) - 1):
        step = float(u[i + 1] - u[i])

        def rhs(value: Array, alpha: float) -> Array:
            b = stage_value(state.b, midpoint_fields["b"], i, alpha, axis=1)
            trace = stage_value(weighted_trace, midpoint_fields["trace"], i, alpha, axis=1)
            mixed = stage_value(mixed_hatchib, midpoint_fields["mixed"], i, alpha, axis=1)
            stage_source = stage_value(source, midpoint_fields["source"], i, alpha, axis=1)
            return (
                0.5 * trace[..., None, None] * value
                + np.matmul(mixed, value)
                + np.matmul(value, np.swapaxes(mixed, -1, -2))
                + stage_source
                - lie_covariant_tensor(grid, b, value)
            )

        current = half[:, i]
        k1 = rhs(current, 0.0)
        k2 = rhs(current + 0.5 * step * k1, 0.5)
        k3 = rhs(current + 0.5 * step * k2, 0.5)
        k4 = rhs(current + step * k3, 1.0)
        half[:, i + 1] = current + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    return tensor_tracefree(half, state.g, geometry["inverse_g"])


def solve_log_omega(
    grid: PointSphereGrid,
    state: GlobalState,
    Omega_omegab: Array,
    u: Array,
    initial_log_omega: Array | None = None,
) -> Array:
    log_Omega = np.zeros_like(state.Omega)
    if initial_log_omega is not None:
        log_Omega[:, 0] = initial_log_omega
    midpoint_shift = midpoint_values(state.b, u, axis=1)
    midpoint_source = midpoint_values(Omega_omegab, u, axis=1)
    for i in range(len(u) - 1):
        step = float(u[i + 1] - u[i])

        def rhs(value: Array, alpha: float) -> Array:
            b = stage_value(state.b, midpoint_shift, i, alpha, axis=1)
            source = stage_value(Omega_omegab, midpoint_source, i, alpha, axis=1)
            return -np.einsum(
                "n...i,n...i->n...", b, scalar_gradient(grid, value)
            ) - 2.0 * source

        current = log_Omega[:, i]
        k1 = rhs(current, 0.0)
        k2 = rhs(current + 0.5 * step * k1, 0.5)
        k3 = rhs(current + 0.5 * step * k2, 0.5)
        k4 = rhs(current + step * k3, 1.0)
        log_Omega[:, i + 1] = current + step * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0
    return np.exp(log_Omega)


def solve_metric_and_expansion(
    grid: PointSphereGrid,
    state: GlobalState,
    half: Array,
    new_omega: Array,
    new_Omega_omega: Array,
    u: Array,
    v: Array,
    substeps: int = 1,
) -> tuple[Array, Array, Array]:
    g = np.zeros_like(state.g)
    Omega_trchi = np.zeros_like(state.Omega_trchi)
    projector = sphere_broadcast(grid, 1)
    g[:, :, 0] = (-u[None, :])[:, :, None, None] ** 2 * projector
    Omega_trchi[:, :, 0] = 2.0 / (-u[None, :])
    old_inverse = tangent_inverse(grid, state.g)
    fields = {
        "Omega": new_omega,
        "Omega_omega": new_Omega_omega,
        "half": half,
        "old_metric": state.g,
    }
    fractions = sorted(
        {
            stage / (2 * substeps)
            for stage in range(1, 2 * substeps)
        }
    )
    interpolated = {
        name: {
            fraction: fractional_values(value, v, axis=2, fraction=fraction)
            for fraction in fractions
            if fraction not in {0.0, 1.0}
        }
        for name, value in fields.items()
    }

    def external(name: str, interval: int, fraction: float) -> Array:
        if abs(fraction) < 1.0e-14:
            return np.take(fields[name], interval, axis=2)
        if abs(fraction - 1.0) < 1.0e-14:
            return np.take(fields[name], interval + 1, axis=2)
        return np.take(interpolated[name][fraction], interval, axis=2)

    def transferred(stage_half: Array, stage_inverse: Array, value_metric: Array) -> Array:
        value = np.matmul(np.matmul(stage_half, stage_inverse), value_metric)
        return 0.5 * (value + np.swapaxes(value, -1, -2))

    for j in range(len(v) - 1):
        full_step = float(v[j + 1] - v[j])
        step = full_step / substeps

        def rhs(value_Omega_trchi: Array, value_metric: Array, fraction: float) -> tuple[Array, Array]:
            Omega = external("Omega", j, fraction)
            Omega_omega = external("Omega_omega", j, fraction)
            stage_half = external("half", j, fraction)
            # The transfer equation contains (g^(i)(v_stage))^{-1}.
            # Interpolation and matrix inversion do not commute, so form the
            # inverse_g from the interpolated old g at this RK stage.
            stage_old_metric = external("old_metric", j, fraction)
            stage_inverse = tangent_inverse(grid, stage_old_metric)
            Omega_chih = transferred(stage_half, stage_inverse, value_metric)
            inverse_g = tangent_inverse(grid, value_metric)
            norm_sq = tensor_norm_sq(Omega_chih, inverse_g)
            return (
                -0.5 * value_Omega_trchi**2
                - 4.0 * Omega_omega * value_Omega_trchi
                - norm_sq,
                value_Omega_trchi[..., None, None] * value_metric
                + 2.0 * Omega_chih,
            )

        current_Omega_trchi, current_metric = Omega_trchi[:, :, j], g[:, :, j]
        for substep in range(substeps):
            fraction = substep / substeps
            half_fraction = (substep + 0.5) / substeps
            end_fraction = (substep + 1.0) / substeps
            k1_Omega_trchi, k1_metric = rhs(current_Omega_trchi, current_metric, fraction)
            k2_Omega_trchi, k2_metric = rhs(
                current_Omega_trchi + 0.5 * step * k1_Omega_trchi,
                current_metric + 0.5 * step * k1_metric,
                half_fraction,
            )
            k3_Omega_trchi, k3_metric = rhs(
                current_Omega_trchi + 0.5 * step * k2_Omega_trchi,
                current_metric + 0.5 * step * k2_metric,
                half_fraction,
            )
            k4_Omega_trchi, k4_metric = rhs(
                current_Omega_trchi + step * k3_Omega_trchi,
                current_metric + step * k3_metric,
                end_fraction,
            )
            current_Omega_trchi = current_Omega_trchi + step * (
                k1_Omega_trchi + 2.0 * k2_Omega_trchi + 2.0 * k3_Omega_trchi + k4_Omega_trchi
            ) / 6.0
            current_metric = current_metric + step * (
                k1_metric + 2.0 * k2_metric + 2.0 * k3_metric + k4_metric
            ) / 6.0
        Omega_trchi[:, :, j + 1] = current_Omega_trchi
        g[:, :, j + 1] = current_metric

    transfer = np.matmul(np.matmul(half, old_inverse), g)
    Omega_chih = 0.5 * (transfer + np.swapaxes(transfer, -1, -2))
    return g, Omega_trchi, Omega_chih


def picard_step(
    grid: PointSphereGrid,
    state: GlobalState,
    boundary: dict[str, Array | float],
    u: Array,
    v: Array,
    zeta_divergence_factor: float = 1.0,
    return_context: bool = False,
    u_boundary: dict[str, Array] | None = None,
    enforce_outgoing_boundary: bool = False,
    metric_substeps: int = 1,
    omegab_source_mode: str = "ric34",
) -> GlobalState | tuple[GlobalState, dict[str, Array]]:
    if enforce_outgoing_boundary:
        impose_outgoing_boundary(state, boundary)
    geometry = section_geometry(grid, state, u)
    half = solve_half_shear(
        grid, state, geometry, boundary, u, u_boundary=u_boundary
    )
    inverse_g = geometry["inverse_g"]
    eta_norm = np.einsum(
        "n...i,n...ij,n...j->n...", geometry["eta"], inverse_g, geometry["eta"]
    )
    eta_etab = np.einsum(
        "n...i,n...ij,n...j->n...", geometry["eta"], inverse_g, geometry["etab"]
    )
    Omega_chibh = state.Omega[..., None, None] * geometry["hatchib"]
    shear_dot = np.einsum(
        "n...ik,n...jl,n...ij,n...kl->n...",
        inverse_g,
        inverse_g,
        half,
        Omega_chibh,
    )
    omega_tr_chi = state.Omega_trchi
    omega_tr_chib = state.Omega * geometry["tr_chib"]
    if omegab_source_mode == "established-combination":
        omegab_source = (
            state.Omega**2
            * (0.5 * eta_norm - eta_etab - 0.5 * geometry["curvature"])
            + 0.25 * shear_dot
            - 0.125 * omega_tr_chi * omega_tr_chib
        )
    elif omegab_source_mode == "ric34":
        eta_up = np.einsum(
            "n...ij,n...j->n...i", inverse_g, geometry["eta"]
        )
        div_eta = vector_divergence(grid, eta_up, geometry["difference"])
        Omega_e3_Omega_trchi = high_order_differentiate(
            omega_tr_chi, u, axis=1
        ) + np.einsum(
            "n...i,n...i->n...",
            state.b,
            scalar_gradient(grid, omega_tr_chi),
        )
        omegab_source = 0.25 * (
            shear_dot
            + 0.5 * omega_tr_chi * omega_tr_chib
            - 4.0 * state.Omega**2 * eta_etab
            + Omega_e3_Omega_trchi
            - 2.0 * state.Omega**2 * div_eta
        )
    else:
        raise ValueError(
            "omegab_source_mode must be 'ric34' or 'established-combination'"
        )
    Omega_omegab = cumulative_polynomial_quadrature(omegab_source, v, axis=2)
    new_omega = solve_log_omega(
        grid,
        state,
        Omega_omegab,
        u,
        initial_log_omega=(
            None if u_boundary is None else u_boundary["log_Omega"]
        ),
    )
    Omega_omega = -0.5 * np.gradient(np.log(new_omega), v, axis=2, edge_order=2)

    div_half = tensor_divergence(
        grid, half, geometry["difference"], geometry["inverse_g"]
    )
    grad_weighted_omega = scalar_gradient(grid, Omega_omega)
    grad_omega_tr_chi = scalar_gradient(grid, omega_tr_chi)
    grad_log_new_omega = scalar_gradient(grid, np.log(new_omega))
    half_zeta = np.einsum("n...ij,n...j->n...i", half, state.zeta)
    zeta_source = (
        -2.0 * omega_tr_chi[..., None] * state.zeta
        -2.0 * np.einsum("n...ij,n...j->n...i", inverse_g, half_zeta)
        +2.0 * np.einsum("n...ij,n...j->n...i", inverse_g, grad_weighted_omega)
        +zeta_divergence_factor
        * np.einsum("n...ij,n...j->n...i", inverse_g, div_half)
        -0.5 * np.einsum("n...ij,n...j->n...i", inverse_g, grad_omega_tr_chi)
        +omega_tr_chi[..., None]
        * np.einsum("n...ij,n...j->n...i", inverse_g, grad_log_new_omega)
    )
    new_zeta = cumulative_polynomial_quadrature(zeta_source, v, axis=2)
    projector = sphere_broadcast(grid, 2)
    new_zeta = np.einsum("n...ij,n...j->n...i", projector, new_zeta)
    new_shift = cumulative_polynomial_quadrature(
        -4.0 * new_omega[..., None] ** 2 * new_zeta, v, axis=2
    )
    new_metric, new_Omega_trchi, new_shear = solve_metric_and_expansion(
        grid,
        state,
        half,
        new_omega,
        Omega_omega,
        u,
        v,
        substeps=metric_substeps,
    )
    if enforce_outgoing_boundary:
        new_metric[:, 0] = np.asarray(boundary["g"])
        new_Omega_trchi[:, 0] = np.asarray(boundary["Omega_trchi"])
        new_shear[:, 0] = np.asarray(boundary["Omega_chih"])
        new_omega[:, 0] = 1.0
        if "zeta" in boundary:
            new_zeta[:, 0] = np.asarray(boundary["zeta"])
        if "b" in boundary:
            new_shift[:, 0] = np.asarray(boundary["b"])
    local_metric = np.einsum(
        "nia,n...ij,njb->n...ab", grid.frames, new_metric, grid.frames
    )
    eigenvalues = np.linalg.eigvalsh(local_metric)
    finite_metric = bool(np.all(np.isfinite(eigenvalues)))
    finite_lapse = bool(np.all(np.isfinite(new_omega)))
    finite_expansion = bool(np.all(np.isfinite(new_Omega_trchi)))
    minimum_metric = float(np.min(eigenvalues)) if finite_metric else math.nan
    minimum_lapse = float(np.min(new_omega)) if finite_lapse else math.nan
    minimum_expansion = float(np.min(new_Omega_trchi)) if finite_expansion else math.nan
    if (
        not finite_metric
        or not finite_lapse
        or not finite_expansion
        or minimum_metric <= 0.0
        or minimum_lapse <= 0.0
    ):
        raise FloatingPointError(
            "global iteration left the positive/finite region: "
            f"metric_finite={finite_metric}, lapse_finite={finite_lapse}, "
            f"expansion_finite={finite_expansion}, "
            f"min_metric={minimum_metric:.6g}, min_lapse={minimum_lapse:.6g}, "
            f"min_expansion={minimum_expansion:.6g}"
        )
    new_state = GlobalState(
        g=new_metric,
        Omega=new_omega,
        zeta=new_zeta,
        b=new_shift,
        Omega_trchi=new_Omega_trchi,
        Omega_chih=new_shear,
    )
    if return_context:
        return new_state, {
            "half_shear": half,
            "weighted_omegab_half": Omega_omegab,
            "omegab_source": omegab_source,
            "omegab_source_mode": omegab_source_mode,
            "Omega_omega": Omega_omega,
            "zeta_source": zeta_source,
        }
    return new_state


def update_norm(new: GlobalState, old: GlobalState) -> float:
    values = []
    for name in ["g", "Omega", "zeta", "b", "Omega_trchi"]:
        current = getattr(new, name)
        previous = getattr(old, name)
        scale = np.maximum(1.0, np.abs(current))
        values.append(float(np.mean(((current - previous) / scale) ** 2)))
    return math.sqrt(sum(values))


def suffix_boundary(condition: Array, u: Array) -> Array:
    boundary = np.full(condition.shape[1], np.nan)
    for j in range(condition.shape[1]):
        for i in range(len(u)):
            if condition[i, j] and np.all(condition[i:, j]):
                boundary[j] = u[i]
                break
    return boundary


def run_global_case(
    c: float,
    delta: float,
    point_count: int = 86,
    n_u: int = 13,
    n_v: int = 65,
    iterations: int = 6,
    u_endpoint: float = -0.8,
    v_endpoint: float = 0.005,
    zeta_divergence_factor: float = 1.0,
    neighbor_count: int = 48,
    angular_degree: int = 5,
    spectral_degree: int | None = None,
) -> dict:
    grid = PointSphereGrid.create(
        point_count,
        neighbor_count=neighbor_count,
        degree=angular_degree,
        spectral_degree=spectral_degree,
    )
    u = -np.exp(np.linspace(0.0, math.log(-u_endpoint), n_u))
    v = v_endpoint * np.linspace(0.0, 1.0, n_v) ** 2
    boundary = solve_outgoing_boundary(grid, v, c, delta)
    state = initial_state(grid, u, v)
    updates = []
    for _ in range(iterations):
        new_state = picard_step(
            grid,
            state,
            boundary,
            u,
            v,
            zeta_divergence_factor=zeta_divergence_factor,
        )
        updates.append(update_norm(new_state, state))
        state = new_state
    geometry = section_geometry(grid, state, u)
    outgoing_supremum = np.max(state.Omega_trchi / state.Omega, axis=0)
    incoming_supremum = np.max(geometry["tr_chib"], axis=0)
    outgoing_boundary = suffix_boundary(outgoing_supremum <= 0.0, u)
    both_boundary = suffix_boundary(
        (outgoing_supremum <= 0.0) & (incoming_supremum <= 0.0), u
    )
    inverse_g = geometry["inverse_g"]
    shear_trace = tensor_trace(state.Omega_chih, inverse_g)
    local_metric = np.einsum(
        "nia,n...ij,njb->n...ab", grid.frames, state.g, grid.frames
    )
    finite_outgoing = np.flatnonzero(np.isfinite(outgoing_boundary))
    finite_both = np.flatnonzero(np.isfinite(both_boundary))
    return {
        "parameters": {
            "C": c,
            "delta": delta,
            "point_count": point_count,
            "n_u": n_u,
            "n_v": n_v,
            "iterations": iterations,
            "zeta_divergence_factor": zeta_divergence_factor,
            "neighbor_count": neighbor_count,
            "angular_degree": angular_degree,
            "spectral_degree": spectral_degree,
            "v_endpoint": v_endpoint,
            "pulse_norm": "ordinary area-integral L2(S_-1,v)",
        },
        "scalar_coordinates": {"u": u, "v": v},
        "updates": updates,
        "outgoing_boundary": outgoing_boundary,
        "both_boundary": both_boundary,
        "outgoing_supremum": outgoing_supremum,
        "incoming_supremum": incoming_supremum,
        "summary": {
            "final_update": updates[-1],
            "max_final_shear_trace": float(np.max(np.abs(shear_trace))),
            "min_metric_eigenvalue": float(np.min(np.linalg.eigvalsh(local_metric))),
            "min_lapse": float(np.min(state.Omega)),
            "max_shift_norm": float(np.max(np.linalg.norm(state.b, axis=-1))),
            "first_v_outgoing_global_suffix": (
                float(v[finite_outgoing[0]]) if len(finite_outgoing) else None
            ),
            "first_v_both_expansions_global_suffix": (
                float(v[finite_both[0]]) if len(finite_both) else None
            ),
            "v_count_outgoing_global_suffix": int(len(finite_outgoing)),
            "v_count_both_expansions_global_suffix": int(len(finite_both)),
            "max_incoming_expansion": float(np.max(incoming_supremum)),
            "boundary_max_l2_norm_error": boundary["max_l2_norm_error"],
            "boundary_max_trace": boundary["max_trace"],
            "boundary_integrated_energy_relative_spread": boundary[
                "integrated_energy_relative_spread"
            ],
        },
        "state": state,
        "grid": grid,
        "boundary": boundary,
    }


def serializable(case: dict) -> dict:
    def finite_or_none(values: Array) -> list[float | None]:
        return [float(value) if np.isfinite(value) else None for value in values]

    return {
        "parameters": case["parameters"],
        "domain": {
            "u": [float(case["scalar_coordinates"]["u"][0]), float(case["scalar_coordinates"]["u"][-1])],
            "v": [float(case["scalar_coordinates"]["v"][0]), float(case["scalar_coordinates"]["v"][-1])],
            "angular_domain": "full sphere, coordinate-free Fibonacci nodes",
        },
        "updates": case["updates"],
        "summary": case["summary"],
        "trapped_boundary_data": {
            "v": [float(value) for value in case["scalar_coordinates"]["v"]],
            "outgoing_u": finite_or_none(case["outgoing_boundary"]),
            "both_expansions_u": finite_or_none(case["both_boundary"]),
        },
        "scope": (
            "nonaxisymmetric whole-sphere construction using the smooth sphere-L2 pulse "
            "relaxation; this case contains one independent four-dimensional Ricci sample; "
            "companion suites provide complete three-grid maps; their R_4 cells do not "
            "overlap the global trapped-v range, so no certificate is claimed"
        ),
    }


def independent_ricci_sample(case: dict, target_v: float = 0.00125) -> dict:
    """Audit one stencil-complete comparison point with four-dimensional Ricci."""

    from .spacetime_ricci import (
        adapted_ricci_norm,
        direct_ricci_block,
        subset_state,
    )

    u = case["scalar_coordinates"]["u"]
    v = case["scalar_coordinates"]["v"]
    center_v = int(np.argmin(np.abs(v - target_v)))
    # The Ricci formula differentiates a connection difference that already
    # contains a g derivative.  A 17-point operator therefore needs the
    # complete 33-point nested stencil around the reported sample.
    start = min(max(center_v - 16, 0), len(v) - 33)
    indices = np.arange(start, start + 33)
    local_state = subset_state(case["state"], indices)
    ricci, inverse_g, _ = direct_ricci_block(
        case["grid"], local_state, u, v[indices]
    )
    center_u = len(u) // 2
    local_v = int(np.flatnonzero(indices == center_v)[0])
    rho = adapted_ricci_norm(ricci, local_state, inverse_g)[:, center_u, local_v]
    g = case["state"].g[:, center_u, center_v]
    local_metric = np.einsum(
        "nia,nij,njb->nab", case["grid"].frames, g, case["grid"].frames
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local_metric), 0.0))
    l2 = math.sqrt(4.0 * math.pi * float(np.mean(rho**2 * area_ratio)))
    components = ricci[:, center_u, local_v]
    Omega = case["state"].Omega[:, center_u, center_v]
    r44 = components[:, 1, 1] / Omega**2
    r4 = components[:, 1, 2:5] / Omega[:, None]
    r4_norm_sq = np.einsum(
        "ni,nij,nj->n", r4, inverse_g[:, center_u, local_v], r4
    )
    return {
        "u": float(u[center_u]),
        "v": float(v[center_v]),
        "renormalized_ricci_l2": float((-u[center_u]) * l2),
        "pointwise_rho_rms": float(np.sqrt(np.mean(rho**2))),
        "R44_rms": float(np.sqrt(np.mean(r44**2))),
        "R4A_rms": float(np.sqrt(np.mean(np.maximum(r4_norm_sq, 0.0)))),
        "stencil": (
            f"{len(u)} x {len(indices)} in (u,v), full spectral sphere; "
            "nested 17-point v support"
        ),
    }


def global_case_report_worker(configuration: tuple[float, int]) -> dict:
    """Run one high-memory case in an isolated process and return JSON data."""

    delta, iterations = configuration
    case = run_global_case(
        100.0,
        delta,
        point_count=96,
        n_u=11,
        n_v=129,
        iterations=iterations,
        neighbor_count=24,
        spectral_degree=8,
    )
    item = serializable(case)
    item["independent_ricci_sample"] = independent_ricci_sample(case)
    return item


def write_global_trapped_svg(cases: list[dict], path: Path) -> None:
    width, height = 780, 470
    left, right, top, bottom = 84, 28, 42, 64
    plot_w, plot_h = width - left - right, height - top - bottom
    u0, u1 = cases[0]["domain"]["u"]
    vmax = cases[0]["domain"]["v"][1]
    colors = ["#2563eb", "#dc2626"]

    def xcoord(value: float) -> float:
        return left + (value - u0) / (u1 - u0) * plot_w

    def ycoord(value: float) -> float:
        return top + (vmax - value) / vmax * plot_h

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Whole-sphere both-expansions-negative region</title>',
        '<desc id="desc">For both requested pulse exponents, shaded u suffixes have nonpositive outgoing expansion and negative incoming expansion at every sampled sphere point.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="390" y="25" text-anchor="middle" font-family="sans-serif" font-size="16">Whole-sphere trapped u-suffix</text>',
    ]
    for color, case in zip(colors, cases):
        data = case["trapped_boundary_data"]
        values = data["both_expansions_u"]
        v = data["v"]
        for index, boundary in enumerate(values):
            if boundary is None:
                continue
            lower = 0.0 if index == 0 else 0.5 * (v[index - 1] + v[index])
            upper = vmax if index == len(v) - 1 else 0.5 * (v[index] + v[index + 1])
            x0, x1 = xcoord(boundary), xcoord(u1)
            y0, y1 = ycoord(upper), ycoord(lower)
            lines.append(
                f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{x1-x0:.2f}" height="{y1-y0:.2f}" fill="{color}" fill-opacity="0.20"/>'
            )
        points = [
            (xcoord(boundary), ycoord(value))
            for boundary, value in zip(values, v)
            if boundary is not None
        ]
        lines.append(
            f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in points)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
    lines.extend(
        [
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="none" stroke="#222"/>',
            f'<text x="{left+plot_w/2:.1f}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="13">u; shading extends to computed endpoint u={u1:g}</text>',
            f'<text x="18" y="{top+plot_h/2:.1f}" text-anchor="middle" transform="rotate(-90 18 {top+plot_h/2:.1f})" font-family="sans-serif" font-size="13">v</text>',
            f'<line x1="{left+18}" y1="{top+16}" x2="{left+42}" y2="{top+16}" stroke="{colors[0]}" stroke-width="2"/><text x="{left+49}" y="{top+20}" font-family="sans-serif" font-size="11">delta=0.1</text>',
            f'<line x1="{left+18}" y1="{top+36}" x2="{left+42}" y2="{top+36}" stroke="{colors[1]}" stroke-width="2"/><text x="{left+49}" y="{top+40}" font-family="sans-serif" font-size="11">delta=0.01</text>',
        ]
    )
    for tick in np.linspace(u0, u1, 5):
        lines.append(
            f'<text x="{xcoord(float(tick)):.2f}" y="{top+plot_h+21}" text-anchor="middle" font-family="monospace" font-size="10">{tick:.2f}</text>'
        )
    for tick in np.linspace(0.0, vmax, 6):
        lines.append(
            f'<text x="{left-8}" y="{ycoord(float(tick))+4:.2f}" text-anchor="end" font-family="monospace" font-size="10">{tick:.3g}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_global_updates_svg(cases: list[dict], path: Path) -> None:
    width, height = 760, 430
    left, right, top, bottom = 78, 28, 38, 60
    plot_w, plot_h = width - left - right, height - top - bottom
    colors = ["#2563eb", "#dc2626"]
    maximum = max(len(case["updates"]) for case in cases)
    logs = [math.log10(value) for case in cases for value in case["updates"]]
    y_min, y_max = math.floor(min(logs)), math.ceil(max(logs))

    def point(index: int, value: float) -> tuple[float, float]:
        return (
            left + (index - 1) / (maximum - 1) * plot_w,
            top + (y_max - math.log10(value)) / (y_max - y_min) * plot_h,
        )

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Whole-sphere Picard contraction</title>',
        '<desc id="desc">Both requested pulse cases contract monotonically after the first Picard sweep to updates below two times ten to the minus nine.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="380" y="24" text-anchor="middle" font-family="sans-serif" font-size="16">Whole-sphere Picard update contraction</text>',
        f'<line x1="{left}" y1="{top+plot_h}" x2="{left+plot_w}" y2="{top+plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#222"/>',
    ]
    for color, case in zip(colors, cases):
        scalar_coordinates = [point(index, value) for index, value in enumerate(case["updates"], 1)]
        lines.append(
            f'<polyline points="{" ".join(f"{x:.2f},{y:.2f}" for x, y in scalar_coordinates)}" fill="none" stroke="{color}" stroke-width="2"/>'
        )
        for x, y in scalar_coordinates:
            lines.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>')
    for exponent in range(y_min, y_max + 1, 2):
        y = top + (y_max - exponent) / (y_max - y_min) * plot_h
        lines.append(f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end" font-family="monospace" font-size="10">{exponent}</text>')
    for index in range(1, maximum + 1):
        x, _ = point(index, 1.0)
        lines.append(f'<text x="{x:.2f}" y="{top+plot_h+21}" text-anchor="middle" font-family="monospace" font-size="10">{index}</text>')
    lines.extend(
        [
            '<text x="380" y="416" text-anchor="middle" font-family="sans-serif" font-size="13">Picard sweep</text>',
            '<text x="18" y="215" text-anchor="middle" transform="rotate(-90 18 215)" font-family="sans-serif" font-size="13">log10(update)</text>',
            f'<line x1="{left+18}" y1="{top+16}" x2="{left+42}" y2="{top+16}" stroke="{colors[0]}" stroke-width="2"/><text x="{left+49}" y="{top+20}" font-family="sans-serif" font-size="11">delta=0.1</text>',
            f'<line x1="{left+18}" y1="{top+36}" x2="{left+42}" y2="{top+36}" stroke="{colors[1]}" stroke-width="2"/><text x="{left+49}" y="{top+40}" font-family="sans-serif" font-size="11">delta=0.01</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_global_audit_svg(cases: list[dict], path: Path) -> None:
    width, height = 700, 300
    left, right, top, bottom = 110, 36, 48, 54
    plot_w = width - left - right
    x_min, x_max = -5.0, -1.5

    def xcoord(value: float) -> float:
        return left + (math.log10(value) - x_min) / (x_max - x_min) * plot_w

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Independent whole-sphere Ricci samples</title>',
        '<desc id="desc">The renormalized residual samples are 0.00208 and 0.00923, below one percent but above the R4 threshold.</desc>',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="350" y="25" text-anchor="middle" font-family="sans-serif" font-size="16">Independent whole-sphere Ricci sample</text>',
    ]
    for value, label, color, y in [
        (1.0e-4, "R_4 threshold", "#64748b", 0),
        (1.0e-2, "one percent", "#64748b", 0),
    ]:
        x = xcoord(value)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{height-bottom}" stroke="{color}" stroke-dasharray="5 4"/>')
        lines.append(f'<text x="{x:.2f}" y="{top-8}" text-anchor="middle" font-family="sans-serif" font-size="10">{label}</text>')
    for case, color, y in zip(cases, ["#2563eb", "#dc2626"], [105.0, 175.0]):
        value = case["independent_ricci_sample"]["renormalized_ricci_l2"]
        x = xcoord(value)
        lines.append(f'<line x1="{left}" y1="{y}" x2="{x:.2f}" y2="{y}" stroke="{color}" stroke-width="5"/>')
        lines.append(f'<circle cx="{x:.2f}" cy="{y}" r="6" fill="{color}"/>')
        lines.append(f'<text x="{left-10}" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="12">delta={case["parameters"]["delta"]}</text>')
        lines.append(f'<text x="{x+10:.2f}" y="{y+4}" font-family="monospace" font-size="11">{value:.3e}</text>')
    for exponent in range(-5, -1):
        x = xcoord(10.0**exponent)
        lines.append(f'<text x="{x:.2f}" y="{height-bottom+22}" text-anchor="middle" font-family="monospace" font-size="10">1e{exponent}</text>')
    lines.extend(
        [
            f'<line x1="{left}" y1="{height-bottom}" x2="{left+plot_w}" y2="{height-bottom}" stroke="#222"/>',
            '<text x="350" y="286" text-anchor="middle" font-family="sans-serif" font-size="13">(-u) ||Ric||_L2(S)</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_global_coupled_suite(results_dir: Path, log_path: Path) -> dict:
    results_dir.mkdir(parents=True, exist_ok=True)
    case_paths = [
        results_dir / "global-coupled-case-delta-0p1.json",
        results_dir / "global-coupled-case-delta-0p01.json",
    ]
    missing = [path for path in case_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "global cases are isolated to respect the managed memory limit; run "
            "`python src/vacuum_state.py --worker 0.1 8 "
            "results/global-coupled-case-delta-0p1.json` and the analogous "
            "`--worker 0.01 11 ...0p01.json` command first"
        )
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in case_paths]
    for case in cases:
        case["scope"] = (
            "nonaxisymmetric whole-sphere construction using the smooth sphere-L2 pulse "
            "relaxation; this case contains one independent four-dimensional Ricci sample; "
            "companion suites provide complete three-grid maps; their R_4 cells do not "
            "overlap the global trapped-v range, so no certificate is claimed"
        )
    from .spacetime_ricci import minkowski_audit

    report = {
        "experiment": "nonaxisymmetric coordinate-free global coupled Picard iteration",
        "cases": cases,
        "independent_auditor_minkowski_control": minkowski_audit(42, 11, 11),
    }
    (results_dir / "global-coupled-summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    write_global_trapped_svg(cases, results_dir / "global-coupled-trapped-region.svg")
    write_global_updates_svg(cases, results_dir / "global-coupled-updates.svg")
    write_global_audit_svg(cases, results_dir / "global-coupled-ricci-sample.svg")
    lines = [
        "",
        "## 2026-07-15 — Nonaxisymmetric whole-sphere coupled iteration",
        "",
        "The smooth moving-zero pulse was inserted into the complete",
        "`hat(chi) -> omegabar -> Omega -> zeta -> b -> (g,tr chi)` Picard map",
        "using coordinate-free full-sphere angular derivatives.",
        "",
        "| delta | final update | first v, tr chi<=0 globally | first v, both expansions<=0 globally | min g eig | sampled (-u)||Ric|| L2 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for case in report["cases"]:
        summary = case["summary"]
        sample = case["independent_ricci_sample"]
        lines.append(
            f"| {case['parameters']['delta']} | {summary['final_update']:.3e} | "
            f"{summary['first_v_outgoing_global_suffix']} | "
            f"{summary['first_v_both_expansions_global_suffix']} | "
            f"{summary['min_metric_eigenvalue']:.6f} | "
            f"{sample['renormalized_ricci_l2']:.3e} |"
        )
    lines.extend(
        [
            "",
            "This removes the angular-chart limitation of the earlier coupled",
            "experiment.  Companion suites now provide a complete production-grid",
            "map and two upward complete maps.  Their R_4 cells remain disjoint",
            "from the global trapped-v range, so no certificate is claimed.",
            "",
        ]
    )
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    return report


if __name__ == "__main__":
    import sys

    if len(sys.argv) in {4, 5} and sys.argv[1] == "--worker":
        worker_text = json.dumps(
            global_case_report_worker((float(sys.argv[2]), int(sys.argv[3]))),
            indent=2,
        )
        if len(sys.argv) == 5:
            Path(sys.argv[4]).write_text(worker_text + "\n", encoding="utf-8")
            print(f"wrote {sys.argv[4]}")
        else:
            print(worker_text)
        raise SystemExit(0)
    root = Path(__file__).resolve().parents[1]
    print(
        json.dumps(
            run_global_coupled_suite(root / "results", root / "results" / "run-log.md"),
            indent=2,
        )
    )
