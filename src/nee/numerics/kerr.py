"""Genuinely nonspherical Kerr benchmark in a local double-null chart.

This implements the quasi-spherical optical construction of Pretorius--Israel
as presented by Franzen--Girão (arXiv:2008.13513), specialized to vacuum Kerr.
The angular coordinate is transported so that the pulled-back metric has the
numerical one-shift form.  The chart is built on a finite exterior patch and
is verified by pulling the Boyer--Lindquist metric back independently.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import fields
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent

from .coordinate_differentiation import high_order_differentiate  # noqa: E402
from .vacuum_exact import (  # noqa: E402
    characteristic_data,
    face_compatible_seed,
    relative_field_error,
)
from .vacuum_iteration import (  # noqa: E402
    FirstOrderState,
    picard_step,
    update_norm,
)
from .sphere import (  # noqa: E402
    PointSphereGrid,
    lie_covariant_tensor,
    scalar_gradient,
    tangent_inverse,
    tensor_trace,
    tensor_tracefree,
)


Array = np.ndarray


def _gauss_integral(
    function: object, left: float, right: float, order: int = 24
) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(order)
    points = 0.5 * (right - left) * nodes + 0.5 * (right + left)
    return float(
        0.5 * (right - left) * np.sum(weights * function(points))
    )


def _branch_theta_at_radius(
    radius: float,
    theta_star: float,
    *,
    mass: float,
    rotation: float,
    reference_radius: float,
) -> float:
    """Solve the separated eikonal relation F=0 on the upper hemisphere."""

    if theta_star <= 1.0e-13 or abs(theta_star - 0.5 * math.pi) <= 1.0e-13:
        return theta_star

    def radial_integrand(value: Array) -> Array:
        delta = value**2 - 2.0 * mass * value + rotation**2
        q_sq = (
            (value**2 + rotation**2) ** 2
            - rotation**2 * math.sin(theta_star) ** 2 * delta
        )
        return 1.0 / np.sqrt(q_sq)

    radial = _gauss_integral(
        radial_integrand, radius, reference_radius, order=24
    )

    def angular_integral(drop: float) -> float:
        if drop <= 0.0:
            return 0.0
        endpoint = math.sqrt(drop)

        def transformed(y: Array) -> Array:
            angle = theta_star - y**2
            denominator = abs(rotation) * np.sqrt(
                np.maximum(
                    math.sin(theta_star) ** 2 - np.sin(angle) ** 2,
                    1.0e-300,
                )
            )
            return 2.0 * y / denominator

        return _gauss_integral(transformed, 0.0, endpoint, order=24)

    low, high = 0.0, theta_star
    if angular_integral(high) < radial:
        raise RuntimeError("the local quasi-spherical branch reached the axis")
    for _ in range(55):
        middle = 0.5 * (low + high)
        if angular_integral(middle) < radial:
            low = middle
        else:
            high = middle
    return theta_star - 0.5 * (low + high)


def _constant_optical_seed(
    upper_theta_star: Array,
    *,
    width: float,
    mass: float,
    rotation: float,
    reference_radius: float,
) -> tuple[Array, Array, Array]:
    """Locate the surface ``r_star=-width`` below the reference sphere."""

    angles = np.asarray(upper_theta_star)
    radius = np.full_like(angles, reference_radius)
    theta = angles.copy()
    offset = np.zeros_like(angles)
    radius, _, _ = _rk4_step(
        radius,
        theta,
        offset,
        -width,
        angles,
        mass,
        rotation,
    )
    nodes, weights = np.polynomial.legendre.leggauss(20)
    sine_sq = np.sin(angles) ** 2
    for _ in range(4):
        points = (
            0.5 * (reference_radius - radius)[:, None] * nodes[None]
            + 0.5 * (reference_radius + radius)[:, None]
        )
        delta = points**2 - 2.0 * mass * points + rotation**2
        q_sq = (
            (points**2 + rotation**2) ** 2
            - rotation**2 * sine_sq[:, None] * delta
        )
        q_value = np.sqrt(q_sq)
        radial_drop = reference_radius - points
        q_reference_sq = (
            (reference_radius**2 + rotation**2) ** 2
            - rotation**2
            * sine_sq
            * (
                reference_radius**2
                - 2.0 * mass * reference_radius
                + rotation**2
            )
        )
        theta_approx = angles[:, None] - (
            rotation**2
            * np.maximum(np.sin(2.0 * angles), 0.0)[:, None]
            * radial_drop**2
            / np.maximum(4.0 * q_reference_sq[:, None], 1.0e-30)
        )
        upsilon = (
            (points**2 + rotation**2) ** 2
            - rotation**2 * np.sin(theta_approx) ** 2 * delta
        )
        integrand = upsilon / (delta * q_value)
        distance = (
            0.5
            * (reference_radius - radius)
            * np.sum(weights[None] * integrand, axis=1)
        )
        delta_radius = radius**2 - 2.0 * mass * radius + rotation**2
        q_radius = np.sqrt(
            (radius**2 + rotation**2) ** 2
            - rotation**2 * sine_sq * delta_radius
        )
        theta_radius = angles - (
            rotation**2
            * np.maximum(np.sin(2.0 * angles), 0.0)
            * (reference_radius - radius) ** 2
            / np.maximum(4.0 * q_reference_sq, 1.0e-30)
        )
        upsilon_radius = (
            (radius**2 + rotation**2) ** 2
            - rotation**2 * np.sin(theta_radius) ** 2 * delta_radius
        )
        integrand_radius = upsilon_radius / (delta_radius * q_radius)
        radius += (distance - width) / integrand_radius

    theta = np.array(
        [
            _branch_theta_at_radius(
                float(value_radius),
                float(value_theta),
                mass=mass,
                rotation=rotation,
                reference_radius=reference_radius,
            )
            for value_radius, value_theta in zip(radius, angles, strict=True)
        ]
    )
    delta_reference = (
        reference_radius**2
        - 2.0 * mass * reference_radius
        + rotation**2
    )
    q_reference_sq = (
        (reference_radius**2 + rotation**2) ** 2
        - rotation**2 * sine_sq * delta_reference
    )
    leading_drop = (
        rotation**2
        * np.maximum(np.sin(2.0 * angles), 0.0)
        * (reference_radius - radius) ** 2
        / np.maximum(4.0 * q_reference_sq, 1.0e-30)
    )
    # For very small widths the quadrature root may round back to the
    # nonphysical constant-theta branch.  The positive quadratic term selects
    # the unique quasi-spherical branch leaving the square-root endpoint.
    theta = np.minimum(theta, angles - leading_drop)
    points = (
        0.5 * (reference_radius - radius)[:, None] * nodes[None]
        + 0.5 * (reference_radius + radius)[:, None]
    )
    delta = points**2 - 2.0 * mass * points + rotation**2
    q_value = np.sqrt(
        (points**2 + rotation**2) ** 2
        - rotation**2 * sine_sq[:, None] * delta
    )
    offset_integrand = (
        2.0 * mass * rotation * points / (delta * q_value)
    )
    offset = (
        0.5
        * (reference_radius - radius)
        * np.sum(weights[None] * offset_integrand, axis=1)
    )
    return radius, theta, offset


def _kerr_functions(
    radius: Array,
    theta: Array,
    theta_star: Array,
    mass: float,
    rotation: float,
) -> tuple[Array, Array, Array, Array, Array]:
    delta = radius**2 - 2.0 * mass * radius + rotation**2
    q_sq = (
        (radius**2 + rotation**2) ** 2
        - rotation**2 * np.sin(theta_star) ** 2 * delta
    )
    p_sq = rotation**2 * np.maximum(
        np.sin(theta_star) ** 2 - np.sin(theta) ** 2, 0.0
    )
    upsilon = (
        (radius**2 + rotation**2) ** 2
        - rotation**2 * np.sin(theta) ** 2 * delta
    )
    return delta, np.sqrt(q_sq), np.sqrt(p_sq), upsilon, q_sq


def _map_rhs(
    radius: Array,
    upper_theta: Array,
    azimuth_offset: Array,
    upper_theta_star: Array,
    mass: float,
    rotation: float,
) -> tuple[Array, Array, Array]:
    del azimuth_offset
    delta, q_value, p_value, upsilon, _ = _kerr_functions(
        radius, upper_theta, upper_theta_star, mass, rotation
    )
    radius_s = delta * q_value / upsilon
    theta_s = delta * p_value / upsilon
    angular_velocity = (
        rotation
        * ((radius**2 + rotation**2) - delta)
        / upsilon
    )
    return radius_s, theta_s, -angular_velocity


def _rk4_step(
    radius: Array,
    theta: Array,
    offset: Array,
    step: float,
    theta_star: Array,
    mass: float,
    rotation: float,
) -> tuple[Array, Array, Array]:
    k1 = _map_rhs(radius, theta, offset, theta_star, mass, rotation)
    k2 = _map_rhs(
        radius + 0.5 * step * k1[0],
        theta + 0.5 * step * k1[1],
        offset + 0.5 * step * k1[2],
        theta_star,
        mass,
        rotation,
    )
    k3 = _map_rhs(
        radius + 0.5 * step * k2[0],
        theta + 0.5 * step * k2[1],
        offset + 0.5 * step * k2[2],
        theta_star,
        mass,
        rotation,
    )
    k4 = _map_rhs(
        radius + step * k3[0],
        theta + step * k3[1],
        offset + step * k3[2],
        theta_star,
        mass,
        rotation,
    )
    return tuple(
        value
        + step * (one + 2.0 * two + 2.0 * three + four) / 6.0
        for value, one, two, three, four in zip(
            (radius, theta, offset), k1, k2, k3, k4, strict=True
        )
    )


def optical_map(
    s_values: Array,
    theta_star: Array,
    *,
    mass: float,
    rotation: float,
    reference_radius: float,
    maximum_step: float = 2.0e-3,
) -> tuple[Array, Array, Array]:
    """Return ``(r,theta,h)`` at every ``(theta_star,s)``.

    The reference two-sphere is the largest requested ``s`` and has
    ``r=reference_radius``, ``theta=theta_star``, and ``h=0``.  The
    nontrivial branch leaving the square-root endpoint is initialized with
    its quadratic expansion and then integrated backwards.
    """

    values_s = np.asarray(s_values, dtype=float)
    angles = np.asarray(theta_star, dtype=float)
    if np.any(np.diff(values_s) <= 0.0):
        raise ValueError("s_values must be strictly increasing")
    if not (mass > 0.0 and 0.0 < abs(rotation) < mass):
        raise ValueError("require a subextremal rotating Kerr metric")
    outer_horizon = mass + math.sqrt(mass**2 - rotation**2)
    if reference_radius <= outer_horizon:
        raise ValueError("reference radius must lie outside the outer horizon")

    reflected = angles > 0.5 * math.pi
    upper_star = np.where(reflected, math.pi - angles, angles)
    s_reference = float(values_s[-1])
    # Non-equatorial characteristics leave the reference slice quadratically
    # in s.  Shrinking this seed with the null-grid spacing can round the
    # near-polar angular displacement to zero and select the spurious
    # constant-theta branch.  Keep branch selection grid-independent.
    seed_width = 1.0e-3
    seed_step = -seed_width
    radius_seed, theta_seed, offset_seed = _constant_optical_seed(
        upper_star,
        width=seed_width,
        mass=mass,
        rotation=rotation,
        reference_radius=reference_radius,
    )
    radius, theta, offset = radius_seed, theta_seed, offset_seed
    current_s = s_reference - seed_width

    radius_result = np.empty((len(angles), len(values_s)))
    theta_result = np.empty_like(radius_result)
    offset_result = np.empty_like(radius_result)
    radius_result[:, -1] = reference_radius
    theta_result[:, -1] = angles
    offset_result[:, -1] = 0.0

    for target_index in range(len(values_s) - 2, -1, -1):
        target = float(values_s[target_index])
        distance = target - current_s
        steps = max(1, int(math.ceil(abs(distance) / maximum_step)))
        step = distance / steps
        for _ in range(steps):
            radius, theta, offset = _rk4_step(
                radius,
                theta,
                offset,
                step,
                upper_star,
                mass,
                rotation,
            )
            theta = np.minimum(np.maximum(theta, 0.0), upper_star)
        current_s = target
        physical_theta = np.where(reflected, math.pi - theta, theta)
        radius_result[:, target_index] = radius
        theta_result[:, target_index] = physical_theta
        offset_result[:, target_index] = offset
    return radius_result, theta_result, offset_result


def _mapping_with_angular_derivatives(
    grid: PointSphereGrid,
    s_values: Array,
    *,
    mass: float,
    rotation: float,
    reference_radius: float,
    angular_step: float = 2.0e-5,
) -> dict[str, Array]:
    theta_star = np.arccos(np.clip(grid.points[:, 2], -1.0, 1.0))
    stacked = np.concatenate(
        (theta_star, theta_star + angular_step, theta_star - angular_step)
    )
    if np.any(stacked <= 0.0) or np.any(stacked >= math.pi):
        raise ValueError("angular finite-difference samples crossed a pole")
    radius_all, theta_all, offset_all = optical_map(
        s_values,
        stacked,
        mass=mass,
        rotation=rotation,
        reference_radius=reference_radius,
    )
    count = grid.count
    radius = radius_all[:count]
    theta = theta_all[:count]
    offset = offset_all[:count]
    return {
        "radius": radius,
        "theta": theta,
        "offset": offset,
        "radius_theta": (
            radius_all[count : 2 * count] - radius_all[2 * count :]
        )
        / (2.0 * angular_step),
        "theta_theta": (
            theta_all[count : 2 * count] - theta_all[2 * count :]
        )
        / (2.0 * angular_step),
        "offset_theta": (
            offset_all[count : 2 * count] - offset_all[2 * count :]
        )
        / (2.0 * angular_step),
    }


def boyer_lindquist_metric(
    radius: Array, theta: Array, mass: float, rotation: float
) -> Array:
    rho_sq = radius**2 + rotation**2 * np.cos(theta) ** 2
    delta = radius**2 - 2.0 * mass * radius + rotation**2
    sin_sq = np.sin(theta) ** 2
    result = np.zeros((*radius.shape, 4, 4))
    result[..., 0, 0] = -(1.0 - 2.0 * mass * radius / rho_sq)
    result[..., 1, 1] = rho_sq / delta
    result[..., 2, 2] = rho_sq
    result[..., 0, 3] = result[..., 3, 0] = (
        -2.0 * mass * rotation * radius * sin_sq / rho_sq
    )
    result[..., 3, 3] = (
        sin_sq
        * (
            (radius**2 + rotation**2) ** 2
            - rotation**2 * delta * sin_sq
        )
        / rho_sq
    )
    return result


def kerr_double_null_geometry(
    grid: PointSphereGrid,
    u: Array,
    v: Array,
    *,
    mass: float,
    rotation: float,
    reference_radius: float,
) -> tuple[dict[str, Array], dict[str, float]]:
    uu, vv = np.meshgrid(np.asarray(u), np.asarray(v), indexing="ij")
    # Relative to the paper's scalar_coordinates, use
    # u_code=v_paper and v_code=-u_paper.  Hence
    # t=u+v and r_star=u-v; this gives the solver's negative du dv sign
    # while keeping the single angular shift on du.
    s_grid = uu - vv
    # Tensor-product decimal nodes can produce nominally identical optical
    # scalar_coordinates separated only by binary roundoff.  Treating those as a
    # genuine first interval collapses the non-Lipschitz branch initializer.
    unique_s, inverse_s = np.unique(
        np.round(s_grid, decimals=14), return_inverse=True
    )
    mapping = _mapping_with_angular_derivatives(
        grid,
        unique_s,
        mass=mass,
        rotation=rotation,
        reference_radius=reference_radius,
    )

    def expand(name: str) -> Array:
        value = mapping[name][:, inverse_s]
        return value.reshape((grid.count, len(u), len(v)))

    radius = expand("radius")
    theta = expand("theta")
    offset = expand("offset")
    radius_theta = expand("radius_theta")
    theta_theta = expand("theta_theta")
    offset_theta = expand("offset_theta")
    theta_star = np.arccos(np.clip(grid.points[:, 2], -1.0, 1.0))
    theta_star_full = np.broadcast_to(
        theta_star[:, None, None], radius.shape
    )
    radius_s, theta_s_upper, offset_s = _map_rhs(
        radius,
        np.where(
            theta_star_full > 0.5 * math.pi, math.pi - theta, theta
        ),
        offset,
        np.where(
            theta_star_full > 0.5 * math.pi,
            math.pi - theta_star_full,
            theta_star_full,
        ),
        mass,
        rotation,
    )
    theta_s = np.where(
        theta_star_full > 0.5 * math.pi, -theta_s_upper, theta_s_upper
    )

    jacobian = np.zeros((*radius.shape, 4, 4))
    jacobian[..., 0, :] = np.stack(
        (np.ones_like(radius), radius_s, theta_s, offset_s), axis=-1
    )
    jacobian[..., 1, :] = np.stack(
        (
            np.ones_like(radius),
            -radius_s,
            -theta_s,
            -offset_s,
        ),
        axis=-1,
    )
    jacobian[..., 2, :] = np.stack(
        (
            np.zeros_like(radius),
            radius_theta,
            theta_theta,
            offset_theta,
        ),
        axis=-1,
    )
    jacobian[..., 3, 3] = 1.0
    bl_metric = boyer_lindquist_metric(radius, theta, mass, rotation)
    pulled = np.einsum(
        "p...ai,p...ij,p...bj->p...ab",
        jacobian,
        bl_metric,
        jacobian,
    )

    gamma_coordinate = pulled[..., 2:, 2:]
    inverse_gamma_coordinate = np.linalg.inv(gamma_coordinate)
    shift_coordinate = -np.einsum(
        "n...AB,n...B->n...A",
        inverse_gamma_coordinate,
        pulled[..., 0, 2:],
    )
    omega_sq = -0.5 * pulled[..., 0, 1]

    phi = np.arctan2(grid.points[:, 1], grid.points[:, 0])
    sin_star = np.sin(theta_star)
    cos_star = np.cos(theta_star)
    e_theta = np.column_stack(
        (cos_star * np.cos(phi), cos_star * np.sin(phi), -sin_star)
    )
    e_phi = np.column_stack(
        (-np.sin(phi), np.cos(phi), np.zeros_like(phi))
    )
    orthonormal_gamma = gamma_coordinate.copy()
    orthonormal_gamma[..., 0, 1] /= sin_star[:, None, None]
    orthonormal_gamma[..., 1, 0] /= sin_star[:, None, None]
    orthonormal_gamma[..., 1, 1] /= sin_star[:, None, None] ** 2
    angular_frame = np.stack((e_theta, e_phi), axis=-1)
    metric = np.einsum(
        "niA,n...AB,njB->n...ij",
        angular_frame,
        orthonormal_gamma,
        angular_frame,
    )
    shift_local = shift_coordinate.copy()
    shift_local[..., 1] *= sin_star[:, None, None]
    shift = np.einsum("niA,n...A->n...i", angular_frame, shift_local)

    gamma_shift_sq = np.einsum(
        "n...A,n...AB,n...B->n...",
        shift_coordinate,
        gamma_coordinate,
        shift_coordinate,
    )
    scale = np.maximum(np.max(np.abs(pulled), axis=(-1, -2)), 1.0)
    diagnostics = {
        "g_vv_relative_max": float(
            np.max(np.abs(pulled[..., 1, 1]) / scale)
        ),
        "g_vA_relative_max": float(
            np.max(np.abs(pulled[..., 1, 2:]) / scale[..., None])
        ),
        "g_uu_closure_relative_max": float(
            np.max(
                np.abs(pulled[..., 0, 0] - gamma_shift_sq) / scale
            )
        ),
        "minimum_omega_squared": float(np.min(omega_sq)),
        "minimum_gamma_eigenvalue": float(
            np.min(np.linalg.eigvalsh(orthonormal_gamma))
        ),
        "minimum_radius": float(np.min(radius)),
        "maximum_radius": float(np.max(radius)),
        "outer_horizon": float(
            mass + math.sqrt(mass**2 - rotation**2)
        ),
    }
    return {
        "metric": metric,
        "omega": np.sqrt(omega_sq),
        "shift": shift,
        "radius": radius,
        "theta": theta,
        "offset": offset,
        "pulled_metric": pulled,
    }, diagnostics


def kerr_exact_state(
    grid: PointSphereGrid,
    u: Array,
    v: Array,
    *,
    mass: float,
    rotation: float,
    reference_radius: float,
) -> tuple[FirstOrderState, dict[str, float]]:
    geometry, diagnostics = kerr_double_null_geometry(
        grid,
        u,
        v,
        mass=mass,
        rotation=rotation,
        reference_radius=reference_radius,
    )
    metric = geometry["metric"]
    omega = geometry["omega"]
    shift = geometry["shift"]
    metric_v = high_order_differentiate(metric, v, axis=2)
    weighted_chi = 0.5 * metric_v
    inverse = tangent_inverse(grid, metric)
    weighted_tr_chi = tensor_trace(weighted_chi, inverse)
    shear = tensor_tracefree(weighted_chi, metric, inverse)
    q = weighted_tr_chi / omega**2

    metric_u = high_order_differentiate(metric, u, axis=1)
    weighted_chib = 0.5 * (
        metric_u + lie_covariant_tensor(grid, shift, metric)
    )
    log_omega = np.log(omega)
    weighted_omega = -0.5 * high_order_differentiate(
        log_omega, v, axis=2
    )
    grad_log_omega = scalar_gradient(grid, log_omega)
    weighted_omegab = -0.5 * (
        high_order_differentiate(log_omega, u, axis=1)
        + np.einsum("n...i,n...i->n...", shift, grad_log_omega)
    )
    shift_v = high_order_differentiate(shift, v, axis=2)
    zeta_up = -shift_v / (4.0 * omega[..., None] ** 2)
    return (
        FirstOrderState(
            metric=metric,
            omega=omega,
            zeta_up=zeta_up,
            shift=shift,
            q=q,
            shear=shear,
            weighted_chib=weighted_chib,
            weighted_omega=weighted_omega,
            weighted_omegab=weighted_omegab,
        ),
        diagnostics,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    grid = PointSphereGrid.create(
        args.points,
        neighbor_count=min(args.neighbors, args.points - 1),
        degree=args.angular_degree,
        spectral_degree=args.spectral_degree,
    )
    u = np.linspace(args.u_min, args.u_max, args.u_count)
    v = np.linspace(args.v_min, args.v_max, args.v_count)
    exact, geometry_diagnostics = kerr_exact_state(
        grid,
        u,
        v,
        mass=args.mass,
        rotation=args.rotation,
        reference_radius=args.reference_radius,
    )
    print(json.dumps({"geometry": geometry_diagnostics}, indent=2), flush=True)
    if (
        geometry_diagnostics["g_vv_relative_max"] > args.gauge_tolerance
        or geometry_diagnostics["g_vA_relative_max"] > args.gauge_tolerance
        or geometry_diagnostics["g_uu_closure_relative_max"]
        > args.gauge_tolerance
        or geometry_diagnostics["minimum_omega_squared"] <= 0.0
        or geometry_diagnostics["minimum_gamma_eigenvalue"] <= 0.0
    ):
        raise RuntimeError("the Kerr pullback failed the double-null gauge gate")
    if args.geometry_only:
        return {"geometry": geometry_diagnostics}

    outgoing, incoming = characteristic_data(grid, exact)
    state = face_compatible_seed(exact, u)
    records: list[dict[str, object]] = []
    for sweep in range(1, args.iterations + 1):
        started = time.perf_counter()
        new_state, _ = picard_step(
            grid,
            state,
            outgoing,
            u,
            v,
            metric_substeps=args.metric_substeps,
            incoming=incoming,
        )
        record = {
            "sweep": sweep,
            "seconds": time.perf_counter() - started,
            "picard_update": update_norm(new_state, state),
            "metric_error": relative_field_error(new_state, exact, "metric"),
            "lapse_error": relative_field_error(new_state, exact, "omega"),
            "shift_error": relative_field_error(new_state, exact, "shift"),
            "shear_error": relative_field_error(new_state, exact, "shear"),
        }
        records.append(record)
        print(json.dumps(record), flush=True)
        state = new_state

    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "state-and-exact.npz",
        u=u,
        v=v,
        **{
            item.name: np.asarray(getattr(state, item.name))
            for item in fields(FirstOrderState)
        },
        **{
            f"exact_{item.name}": np.asarray(getattr(exact, item.name))
            for item in fields(FirstOrderState)
        },
    )
    np.savez_compressed(
        args.output / "characteristic-data.npz",
        u=u,
        v=v,
        **{f"outgoing_{name}": value for name, value in outgoing.items()},
        **{f"incoming_{name}": value for name, value in incoming.items()},
    )
    summary = {
        "metric": "Kerr",
        "mass": args.mass,
        "rotation": args.rotation,
        "reference_radius": args.reference_radius,
        "domain": {
            "u": [args.u_min, args.u_max],
            "v": [args.v_min, args.v_max],
        },
        "resolution": {
            "sphere_points": args.points,
            "spectral_degree": args.spectral_degree,
            "u_count": args.u_count,
            "v_count": args.v_count,
            "metric_substeps": args.metric_substeps,
        },
        "geometry_diagnostics": geometry_diagnostics,
        "records": records,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mass", type=float, default=1.0)
    parser.add_argument("--rotation", type=float, default=0.1)
    parser.add_argument("--reference-radius", type=float, default=4.0)
    parser.add_argument("--points", type=int, default=50)
    parser.add_argument("--neighbors", type=int, default=28)
    parser.add_argument("--angular-degree", type=int, default=3)
    parser.add_argument("--spectral-degree", type=int, default=5)
    parser.add_argument("--u-count", type=int, default=17)
    parser.add_argument("--v-count", type=int, default=17)
    parser.add_argument("--u-min", type=float, default=-1.0)
    parser.add_argument("--u-max", type=float, default=0.5)
    parser.add_argument("--v-min", type=float, default=0.0)
    parser.add_argument("--v-max", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--metric-substeps", type=int, default=2)
    parser.add_argument("--gauge-tolerance", type=float, default=2.0e-6)
    parser.add_argument("--geometry-only", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE.parent / "results" / "exact-vacuum-benchmarks" / "kerr-low",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
