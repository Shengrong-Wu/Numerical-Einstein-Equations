"""Spherical Picard--Chebyshev solver on the Experiment-3 curved domain."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import mpmath
import numpy as np
from scipy.integrate import cumulative_simpson, cumulative_trapezoid, simpson


Array = np.ndarray


def chebyshev_lobatto(
    count: int, left: float, right: float, dtype: Any = np.longdouble
) -> tuple[Array, Array]:
    if count < 5:
        raise ValueError("Chebyshev grid requires at least five nodes")
    index = np.arange(count, dtype=dtype)
    nodes = np.cos(np.asarray(math.pi, dtype=dtype) * index / (count - 1))[::-1]
    nodes = np.asarray(left, dtype=dtype) + 0.5 * (nodes + 1) * np.asarray(
        right - left, dtype=dtype
    )
    weights = np.ones(count, dtype=dtype)
    weights[0] = weights[-1] = 0.5
    weights *= np.where(np.arange(count) % 2 == 0, 1.0, -1.0)
    # Reversal changes all weights by a common sign only, which cancels.
    difference = nodes[:, None] - nodes[None, :]
    matrix = np.zeros((count, count), dtype=dtype)
    mask = ~np.eye(count, dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        matrix[mask] = (
            weights[None, :] / weights[:, None] / difference
        )[mask]
    matrix[np.diag_indices(count)] = -np.sum(matrix, axis=1)
    return nodes, matrix


def differentiate(matrix: Array, values: Array, axis: int) -> Array:
    moved = np.moveaxis(values, axis, 0)
    result = np.tensordot(matrix, moved, axes=(1, 0))
    return np.moveaxis(result, 0, axis)


def _barycentric_weights(nodes: Array) -> Array:
    count = len(nodes)
    result = np.ones(count, dtype=np.longdouble)
    for i in range(count):
        result[i] = 1.0 / np.prod(nodes[i] - np.delete(nodes, i))
    result /= np.max(np.abs(result))
    return result


def _basis_values(nodes: Array, points: Array) -> Array:
    nodes = np.asarray(nodes, dtype=np.longdouble)
    points = np.asarray(points, dtype=np.longdouble)
    weights = _barycentric_weights(nodes)
    difference = points[:, None] - nodes[None, :]
    result = np.empty((len(points), len(nodes)), dtype=np.longdouble)
    for row in range(len(points)):
        exact = np.flatnonzero(np.abs(difference[row]) < 8 * np.finfo(np.longdouble).eps)
        if len(exact):
            result[row] = 0.0
            result[row, exact[0]] = 1.0
        else:
            raw = weights / difference[row]
            result[row] = raw / np.sum(raw)
    return result


def integration_weights(nodes: Array, left: float, right: float) -> Array:
    if right == left:
        return np.zeros(len(nodes), dtype=np.longdouble)
    order = max(8, len(nodes) + 2)
    points, weights = np.polynomial.legendre.leggauss(order)
    points = np.asarray(points, dtype=np.longdouble)
    weights = np.asarray(weights, dtype=np.longdouble)
    mapped = 0.5 * (right - left) * points + 0.5 * (right + left)
    basis = _basis_values(nodes, mapped)
    return 0.5 * (right - left) * np.einsum(
        "Omega_trchi,qj->j", weights, basis
    )


def rectangle_integrator(u: Array, xi: Array, v0: Array) -> Array:
    """Return W with integral[i,j] = W[i,j,k,l] rhs[k,l]."""

    nu, nx = len(u), len(xi)
    result = np.zeros((nu, nx, nu, nx), dtype=np.longdouble)
    xi_rows: dict[tuple[int, int, int], Array] = {}
    for i in range(1, nu):
        u_weights = integration_weights(u[: i + 1], u[0], u[i])
        for j in range(1, nx):
            target_v = xi[j] * v0[i]
            for k in range(i + 1):
                upper = target_v / v0[k]
                if upper > 1.0 + 1.0e-12:
                    raise ValueError("curved-domain rectangle escaped the domain")
                upper = min(np.longdouble(1.0), upper)
                key = (i, j, k)
                xi_rows[key] = integration_weights(
                    xi, xi[0], upper
                )
                result[i, j, k] = (
                    u_weights[k] * v0[k] * xi_rows[key]
                )
    return result


def _integral_samples(values: Array, coordinate: Array) -> np.longdouble:
    if len(coordinate) < 2:
        return np.longdouble(0.0)
    if len(coordinate) == 2:
        return np.longdouble(np.trapezoid(values, x=coordinate))
    return np.longdouble(simpson(values, x=coordinate))


def _cumulative_samples(values: Array, coordinate: Array) -> Array:
    if len(coordinate) < 3:
        return np.asarray(
            cumulative_trapezoid(values, x=coordinate, initial=0.0),
            dtype=np.longdouble,
        )
    return np.asarray(
        cumulative_simpson(values, x=coordinate, initial=0.0),
        dtype=np.longdouble,
    )


def volterra_step(
    radius_rhs: Array,
    lapse_rhs: Array,
    exact: dict[str, Array],
    u: Array,
    xi: Array,
    mass: float,
) -> tuple[Array, Array]:
    """Integrate the two Goursat equations along physical null rectangles."""

    nu, nx = radius_rhs.shape
    v0 = exact["v0"]
    radius_next = np.empty_like(radius_rhs)
    log_next = np.empty_like(lapse_rhs)
    radius_v = np.empty_like(radius_rhs)
    ell_v = np.empty_like(lapse_rhs)
    xi_weights = _barycentric_weights(xi)

    def interpolate(row: Array, location: np.longdouble) -> np.longdouble:
        difference = location - xi
        exact_node = np.flatnonzero(
            np.abs(difference) < 8 * np.finfo(np.longdouble).eps
        )
        if len(exact_node):
            return np.longdouble(row[exact_node[0]])
        raw = xi_weights / difference
        return np.longdouble(np.sum(raw * row) / np.sum(raw))

    incoming_radius = exact["radius"][:, 0]
    incoming_log = exact["log_Omega"][:, 0]
    for i in range(nu):
        physical_v = xi * v0[i]
        outgoing_radius = kruskal_radius(
            physical_v + np.longdouble(0.2), mass, True
        )
        coefficient = (
            4
            * np.longdouble(mass) ** 2
            / outgoing_radius
            * np.exp(-outgoing_radius / (2 * np.longdouble(mass)))
        )
        boundary_radius_v = -coefficient
        boundary_ell_v = (
            -0.5
            * (
                1 / outgoing_radius
                + 1 / (2 * np.longdouble(mass))
            )
            * boundary_radius_v
        )
        for j, target_v in enumerate(physical_v):
            radial_samples = np.empty(i + 1, dtype=np.longdouble)
            lapse_samples = np.empty(i + 1, dtype=np.longdouble)
            for k in range(i + 1):
                location = target_v / v0[k]
                radial_samples[k] = interpolate(radius_rhs[k], location)
                lapse_samples[k] = interpolate(lapse_rhs[k], location)
            radius_v[i, j] = boundary_radius_v[j] + _integral_samples(
                radial_samples, u[: i + 1]
            )
            ell_v[i, j] = boundary_ell_v[j] + _integral_samples(
                lapse_samples, u[: i + 1]
            )
        radius_next[i] = incoming_radius[i] + _cumulative_samples(
            radius_v[i], physical_v
        )
        log_next[i] = incoming_log[i] + _cumulative_samples(
            ell_v[i], physical_v
        )
    return radius_next, log_next


def kruskal_radius(product: Array, mass: float, high_precision: bool) -> Array:
    values = np.asarray(product)
    if not high_precision:
        from scipy.special import lambertw

        return np.asarray(
            2.0 * mass * (1.0 + np.real(lambertw(-values / math.e))),
            dtype=np.longdouble,
        )
    mpmath.mp.dps = 40
    flat = [
        2
        * mpmath.mpf(str(mass))
        * (
            1
            + mpmath.lambertw(
                -mpmath.mpf(str(float(item))) / mpmath.e, 0
            )
        )
        for item in values.ravel()
    ]
    return np.asarray([str(item) for item in flat], dtype=np.longdouble).reshape(
        values.shape
    )


def exact_fields(
    u: Array,
    xi: Array,
    epsilon: float,
    mass: float,
    *,
    high_precision: bool,
) -> dict[str, Array]:
    epsilon_ld = np.longdouble(epsilon)
    x_s = (1 - epsilon_ld) * np.exp(epsilon_ld)
    v0 = x_s / (u + 2) - np.longdouble(0.2)
    v0_prime = -x_s / (u + 2) ** 2
    v = xi[None, :] * v0[:, None]
    U = u[:, None] + 2
    V = v + np.longdouble(0.2)
    radius = kruskal_radius(U * V, mass, high_precision)
    y = radius / (2 * np.longdouble(mass))
    omega_sq = (
        8
        * np.longdouble(mass) ** 3
        / radius
        * np.exp(-y)
    )
    return {
        "v0": v0,
        "v0_prime": v0_prime,
        "v": v,
        "radius": radius,
        "log_Omega": 0.5 * np.log(omega_sq),
        "omega_squared": omega_sq,
    }


def physical_derivatives(
    values: Array,
    d_u: Array,
    d_xi: Array,
    xi: Array,
    v0: Array,
    v0_prime: Array,
) -> tuple[Array, Array]:
    at_xi = differentiate(d_u, values, axis=0)
    in_xi = differentiate(d_xi, values, axis=1)
    at_v = at_xi - (
        xi[None, :] * v0_prime[:, None] / v0[:, None]
    ) * in_xi
    in_v = in_xi / v0[:, None]
    return at_v, in_v


def rhs(
    radius: Array,
    log_Omega: Array,
    d_u: Array,
    d_xi: Array,
    xi: Array,
    v0: Array,
    v0_prime: Array,
) -> tuple[Array, Array]:
    radius_u, radius_v = physical_derivatives(
        radius, d_u, d_xi, xi, v0, v0_prime
    )
    common = radius_u * radius_v + np.exp(2 * log_Omega)
    return -common / radius, common / radius**2


def face_blend(
    exact: dict[str, Array], u: Array, xi: Array, mass: float
) -> tuple[Array, Array]:
    physical_v = exact["v"]
    incoming_radius = exact["radius"][:, :1]
    incoming_log = exact["log_Omega"][:, :1]
    outgoing_radius = kruskal_radius(
        physical_v + np.longdouble(0.2), mass, True
    )
    outgoing_log = 0.5 * np.log(
        8
        * np.longdouble(mass) ** 3
        / outgoing_radius
        * np.exp(-outgoing_radius / (2 * np.longdouble(mass)))
    )
    radius = incoming_radius + outgoing_radius - outgoing_radius[0, 0]
    log_Omega = incoming_log + outgoing_log - outgoing_log[0, 0]
    return radius, log_Omega


def warped_product_residual(
    radius: Array,
    log_Omega: Array,
    u: Array,
    xi: Array,
    d_u: Array,
    d_xi: Array,
    v0: Array,
    v0_prime: Array,
    *,
    halo: int,
) -> dict[str, Any]:
    """Independent 80-bit Ricci audit derived from the reconstructed g."""

    ru, rv = physical_derivatives(
        radius, d_u, d_xi, xi, v0, v0_prime
    )
    ellu, ellv = physical_derivatives(
        log_Omega, d_u, d_xi, xi, v0, v0_prime
    )
    ruu, _ = physical_derivatives(ru, d_u, d_xi, xi, v0, v0_prime)
    _, rvv = physical_derivatives(rv, d_u, d_xi, xi, v0, v0_prime)
    ruv_a, _ = physical_derivatives(rv, d_u, d_xi, xi, v0, v0_prime)
    _, ruv_b = physical_derivatives(ru, d_u, d_xi, xi, v0, v0_prime)
    ruv = 0.5 * (ruv_a + ruv_b)
    elluv_a, _ = physical_derivatives(ellv, d_u, d_xi, xi, v0, v0_prime)
    _, elluv_b = physical_derivatives(ellu, d_u, d_xi, xi, v0, v0_prime)
    elluv = 0.5 * (elluv_a + elluv_b)
    omega_sq = np.exp(2 * log_Omega)

    ric_uu = -2 * (ruu - 2 * ellu * ru) / radius
    ric_vv = -2 * (rvv - 2 * ellv * rv) / radius
    ric_uv = -2 * elluv - 2 * ruv / radius
    angular_coefficient = 1 + (ru * rv + radius * ruv) / omega_sq
    r33 = ric_uu / omega_sq
    r44 = ric_vv / omega_sq
    r34 = ric_uv / omega_sq
    angular_norm_sq = 2 * (angular_coefficient / radius**2) ** 2
    pointwise = np.sqrt(
        np.maximum(
            r33**2 + r44**2 + 2 * r34**2 + angular_norm_sq,
            np.longdouble(0),
        )
    )
    section_l2 = np.sqrt(4 * np.longdouble(math.pi) * radius**2) * pointwise
    scaled = (-u[:, None]) * section_l2
    safe = np.zeros(scaled.shape, dtype=bool)
    safe[halo : len(u) - halo, halo : len(xi) - halo] = True
    if not np.any(safe):
        safe[:] = True
    return {
        "precision": {
            "dtype": str(radius.dtype),
            "mantissa_bits": int(np.finfo(radius.dtype).nmant),
        },
        "raw_maximum": float(np.max(scaled)),
        "masked_maximum": float(np.max(scaled[safe])),
        "component_raw_maxima": {
            "Ric33": float(np.max(np.abs(r33))),
            "Ric44": float(np.max(np.abs(r44))),
            "Ric34": float(np.max(np.abs(r34))),
            "RicAB_coefficient": float(np.max(np.abs(angular_coefficient))),
        },
        "pointwise": np.asarray(pointwise, dtype=float),
        "section_l2": np.asarray(section_l2, dtype=float),
    }


def weighted_spherical_residual(radius, log_Omega, x_out, x_in, omega, omegab,
                                u, xi, d_u, d_xi, v0, v0_prime, *, halo):
    """First derivatives of the stored weighted forms in spherical symmetry."""
    A, Ab = 2*x_out/radius**2, 2*x_in/radius**2
    Au, Av = physical_derivatives(A, d_u, d_xi, xi, v0, v0_prime)
    Abu, _ = physical_derivatives(Ab, d_u, d_xi, xi, v0, v0_prime)
    _, Wbv = physical_derivatives(omegab, d_u, d_xi, xi, v0, v0_prime)
    omega2 = np.exp(2*log_Omega)
    r33 = -(Abu + 4*omegab*Ab + Ab**2/2)/omega2
    r44 = -(Av + 4*omega*A + A**2/2)/omega2
    r34 = (4*Wbv - A*Ab/2 - Au)/omega2
    trace = (Au + A*Ab + 2*omega2/radius**2)/omega2
    pointwise = np.sqrt(r33**2 + r44**2 + 2*r34**2 + trace**2/2)
    section = np.sqrt(4*np.longdouble(math.pi))*radius*pointwise
    scaled = -u[:, None]*section
    safe = np.zeros(scaled.shape, bool)
    safe[halo:-halo, halo:-halo] = True
    if not np.any(safe):
        raise ValueError('spherical residual mask is empty')
    return {'method': 'first derivatives of weighted spherical connections',
        'raw_maximum': float(np.max(scaled)), 'masked_maximum': float(np.max(scaled[safe])),
        'component_raw_maxima': {'Ric33': float(np.max(np.abs(r33))),
            'Ric44': float(np.max(np.abs(r44))), 'Ric34': float(np.max(np.abs(r34))),
            'RicAB_coefficient': float(np.max(np.abs(trace/2)))},
        'pointwise': np.asarray(pointwise, float), 'section_l2': np.asarray(section, float)}


def warped_product_residual_mpmath(
    radius: Array,
    log_Omega: Array,
    epsilon: float,
    mass: float,
    *,
    halo: int,
    decimal_digits: int = 35,
) -> dict[str, Any]:
    """Repeat the direct warped-product Ricci audit above 80-bit precision."""

    mpmath.mp.dps = decimal_digits
    mp = mpmath.mp
    nu, nx = radius.shape

    def grid_and_derivative(count: int, left: Any, right: Any):
        raw = [mp.cos(mp.pi * j / (count - 1)) for j in range(count)][::-1]
        nodes = [left + (x + 1) * (right - left) / 2 for x in raw]
        weights = [
            (mp.mpf("0.5") if j in (0, count - 1) else mp.mpf(1))
            * (-1 if j % 2 else 1)
            for j in range(count)
        ]
        derivative = mpmath.matrix(count, count)
        for i in range(count):
            for j in range(count):
                if i != j:
                    derivative[i, j] = (
                        weights[j] / weights[i] / (nodes[i] - nodes[j])
                    )
            derivative[i, i] = -sum(
                derivative[i, j] for j in range(count) if j != i
            )
        return nodes, derivative

    u_nodes, du = grid_and_derivative(nu, mp.mpf(-1), mp.mpf("-0.5"))
    xi_nodes, dxi = grid_and_derivative(nx, mp.mpf(0), mp.mpf(1))
    r = mpmath.matrix(
        [[mp.mpf(str(float(radius[i, j]))) for j in range(nx)] for i in range(nu)]
    )
    ell = mpmath.matrix(
        [
            [mp.mpf(str(float(log_Omega[i, j]))) for j in range(nx)]
            for i in range(nu)
        ]
    )
    eps = mp.mpf(str(epsilon))
    mass_mp = mp.mpf(str(mass))
    xs = (1 - eps) * mp.exp(eps)
    v0 = [xs / (u + 2) - mp.mpf("0.2") for u in u_nodes]
    v0p = [-xs / (u + 2) ** 2 for u in u_nodes]

    def transpose(matrix):
        return matrix.T

    def physical(values):
        at_xi = du * values
        in_xi = values * transpose(dxi)
        at_v = mpmath.matrix(nu, nx)
        in_v = mpmath.matrix(nu, nx)
        for i in range(nu):
            for j in range(nx):
                at_v[i, j] = (
                    at_xi[i, j]
                    - xi_nodes[j] * v0p[i] / v0[i] * in_xi[i, j]
                )
                in_v[i, j] = in_xi[i, j] / v0[i]
        return at_v, in_v

    ru, rv = physical(r)
    ellu, ellv = physical(ell)
    ruu, _ = physical(ru)
    _, rvv = physical(rv)
    ruv_a, _ = physical(rv)
    _, ruv_b = physical(ru)
    elluv_a, _ = physical(ellv)
    _, elluv_b = physical(ellu)

    raw_max = mp.mpf(0)
    masked_max = mp.mpf(0)
    component = {
        "Ric33": mp.mpf(0),
        "Ric44": mp.mpf(0),
        "Ric34": mp.mpf(0),
        "RicAB_coefficient": mp.mpf(0),
    }
    for i in range(nu):
        for j in range(nx):
            ruv = (ruv_a[i, j] + ruv_b[i, j]) / 2
            elluv = (elluv_a[i, j] + elluv_b[i, j]) / 2
            omega_sq = mp.exp(2 * ell[i, j])
            ric_uu = -2 * (ruu[i, j] - 2 * ellu[i, j] * ru[i, j]) / r[i, j]
            ric_vv = -2 * (rvv[i, j] - 2 * ellv[i, j] * rv[i, j]) / r[i, j]
            ric_uv = -2 * elluv - 2 * ruv / r[i, j]
            angular = 1 + (
                ru[i, j] * rv[i, j] + r[i, j] * ruv
            ) / omega_sq
            r33 = ric_uu / omega_sq
            r44 = ric_vv / omega_sq
            r34 = ric_uv / omega_sq
            norm = mp.sqrt(
                r33**2
                + r44**2
                + 2 * r34**2
                + 2 * (angular / r[i, j] ** 2) ** 2
            )
            scaled = (-u_nodes[i]) * mp.sqrt(
                4 * mp.pi * r[i, j] ** 2
            ) * norm
            raw_max = max(raw_max, abs(scaled))
            if halo <= i < nu - halo and halo <= j < nx - halo:
                masked_max = max(masked_max, abs(scaled))
            component["Ric33"] = max(component["Ric33"], abs(r33))
            component["Ric44"] = max(component["Ric44"], abs(r44))
            component["Ric34"] = max(component["Ric34"], abs(r34))
            component["RicAB_coefficient"] = max(
                component["RicAB_coefficient"], abs(angular)
            )
    if masked_max == 0:
        masked_max = raw_max
    return {
        "precision": {
            "backend": "mpmath",
            "decimal_digits": decimal_digits,
            "minimum_binary_bits": int(decimal_digits * math.log2(10)),
        },
        "raw_maximum": float(raw_max),
        "masked_maximum": float(masked_max),
        "component_raw_maxima": {
            name: float(value) for name, value in component.items()
        },
    }


def warped_product_rectangular_residual_mpmath(
    radius: Array,
    log_Omega: Array,
    u: Array,
    v: Array,
    *,
    halo: int,
    decimal_digits: int = 35,
) -> dict[str, Any]:
    """Arbitrary-precision direct Ricci audit on a rectangular spherical grid."""

    mpmath.mp.dps = decimal_digits
    mp = mpmath.mp
    nu, nv = radius.shape

    def derivative_matrix(nodes_array: Array, stencil: int = 9):
        nodes = [mp.mpf(str(float(item))) for item in nodes_array]
        count = len(nodes)
        width = min(stencil, count)
        if width < 3:
            raise ValueError("the local derivative stencil needs at least three nodes")
        if width % 2 == 0 and width < count:
            width -= 1
        matrix = mpmath.matrix(count, count)
        for i in range(count):
            start = max(0, min(i - width // 2, count - width))
            indices = list(range(start, start + width))
            offsets = [nodes[j] - nodes[i] for j in indices]
            vandermonde = mpmath.matrix(
                [[offset**degree for offset in offsets] for degree in range(width)]
            )
            target = mpmath.matrix(
                [mp.mpf(1) if degree == 1 else mp.mpf(0) for degree in range(width)]
            )
            local_weights = mpmath.lu_solve(vandermonde, target)
            for local, j in enumerate(indices):
                matrix[i, j] = local_weights[local]
        return nodes, matrix, width

    u_nodes, du, u_stencil = derivative_matrix(u)
    v_nodes, dv, v_stencil = derivative_matrix(v)
    r = mpmath.matrix(
        [[mp.mpf(str(float(radius[i, j]))) for j in range(nv)] for i in range(nu)]
    )
    ell = mpmath.matrix(
        [
            [mp.mpf(str(float(log_Omega[i, j]))) for j in range(nv)]
            for i in range(nu)
        ]
    )
    ru = du * r
    rv = r * dv.T
    ellu = du * ell
    ellv = ell * dv.T
    ruu = du * ru
    rvv = rv * dv.T
    ruv = (du * rv + ru * dv.T) / 2
    elluv = (du * ellv + ellu * dv.T) / 2
    raw_max = mp.mpf(0)
    masked_max = mp.mpf(0)
    component = {
        "Ric33": mp.mpf(0),
        "Ric44": mp.mpf(0),
        "Ric34": mp.mpf(0),
        "RicAB_coefficient": mp.mpf(0),
    }
    for i in range(nu):
        for j in range(nv):
            omega_sq = mp.exp(2 * ell[i, j])
            ric_uu = -2 * (ruu[i, j] - 2 * ellu[i, j] * ru[i, j]) / r[i, j]
            ric_vv = -2 * (rvv[i, j] - 2 * ellv[i, j] * rv[i, j]) / r[i, j]
            ric_uv = -2 * elluv[i, j] - 2 * ruv[i, j] / r[i, j]
            angular = 1 + (
                ru[i, j] * rv[i, j] + r[i, j] * ruv[i, j]
            ) / omega_sq
            r33 = ric_uu / omega_sq
            r44 = ric_vv / omega_sq
            r34 = ric_uv / omega_sq
            norm = mp.sqrt(
                r33**2
                + r44**2
                + 2 * r34**2
                + 2 * (angular / r[i, j] ** 2) ** 2
            )
            scaled = (-u_nodes[i]) * mp.sqrt(
                4 * mp.pi * r[i, j] ** 2
            ) * norm
            raw_max = max(raw_max, abs(scaled))
            if halo <= i < nu - halo and halo <= j < nv - halo:
                masked_max = max(masked_max, abs(scaled))
            component["Ric33"] = max(component["Ric33"], abs(r33))
            component["Ric44"] = max(component["Ric44"], abs(r44))
            component["Ric34"] = max(component["Ric34"], abs(r34))
            component["RicAB_coefficient"] = max(
                component["RicAB_coefficient"], abs(angular)
            )
    if masked_max == 0:
        masked_max = raw_max
    return {
        "precision": {
            "backend": "mpmath",
            "decimal_digits": decimal_digits,
            "minimum_binary_bits": int(decimal_digits * math.log2(10)),
        },
        "differentiation": {
            "method": "local arbitrary-precision finite-difference weights",
            "u_stencil": u_stencil,
            "v_stencil": v_stencil,
        },
        "raw_maximum": float(raw_max),
        "masked_maximum": float(masked_max),
        "component_raw_maxima": {
            name: float(value) for name, value in component.items()
        },
    }


def spectral_tail(values: Array) -> float:
    coefficients = np.polynomial.chebyshev.chebfit(
        np.linspace(-1.0, 1.0, values.shape[1]),
        np.asarray(values, dtype=float).T,
        deg=values.shape[1] - 1,
    )
    scale = max(float(np.max(np.abs(coefficients))), 1.0e-30)
    return float(np.max(np.abs(coefficients[-3:])) / scale)


@dataclass(frozen=True)
class CurvedSolution:
    u: Array
    xi: Array
    v: Array
    v0: Array
    radius: Array
    log_Omega: Array
    x_out_scalar: Array
    x_in_scalar: Array
    Omega_omega: Array
    Omega_omegab: Array
    records: list[dict[str, float | None]]
    diagnostics: dict[str, Any]


def solve(
    epsilon: float,
    u_count: int,
    xi_count: int,
    *,
    mass: float = 1.0,
    iterations: int = 12,
    tolerance: float = 1.0e-12,
) -> CurvedSolution:
    dtype = np.longdouble
    u, d_u = chebyshev_lobatto(u_count, -1.0, -0.5, dtype)
    xi, d_xi = chebyshev_lobatto(xi_count, 0.0, 1.0, dtype)
    high_precision = epsilon <= 2.0**-4
    exact = exact_fields(
        u, xi, epsilon, mass, high_precision=high_precision
    )
    radius, log_Omega = face_blend(exact, u, xi, mass)
    records: list[dict[str, float | None]] = []
    previous_update = math.inf
    positivity_stabilized = False
    for iteration in range(1, iterations + 1):
        radius_rhs, lapse_rhs = rhs(
            radius,
            log_Omega,
            d_u,
            d_xi,
            xi,
            exact["v0"],
            exact["v0_prime"],
        )
        raw_radius_next, raw_log_next = volterra_step(
            radius_rhs, lapse_rhs, exact, u, xi, mass
        )
        relaxation = 1.0
        while True:
            radius_next = (
                (1.0 - relaxation) * radius
                + relaxation * raw_radius_next
            )
            log_next = (
                (1.0 - relaxation) * log_Omega
                + relaxation * raw_log_next
            )
            scale_r = np.maximum(
                np.abs(radius_next), np.longdouble(1.0)
            )
            update = float(
                np.sqrt(
                    np.mean(((radius_next - radius) / scale_r) ** 2)
                    + np.mean((log_next - log_Omega) ** 2)
                )
            )
            finite_positive = bool(
                np.all(np.isfinite(radius_next))
                and np.all(np.isfinite(log_next))
                and np.min(radius_next) > 0.0
            )
            nonincreasing = (
                not positivity_stabilized
                or not math.isfinite(previous_update)
                or update <= previous_update
            )
            if finite_positive and nonincreasing:
                break
            positivity_stabilized = True
            relaxation *= 0.5
            if relaxation < 2.0**-30:
                raise FloatingPointError(
                    "curved Picard positivity backtracking exhausted at "
                    f"epsilon={epsilon}, iteration={iteration}"
                )
        contraction = (
            update / previous_update
            if math.isfinite(previous_update)
            else None
        )
        records.append(
            {
                "iteration": float(iteration),
                "update": update,
                "contraction": contraction,
                "relaxation": relaxation,
            }
        )
        radius, log_Omega = radius_next, log_next
        previous_update = update
        if update <= tolerance:
            break

    ru, rv = physical_derivatives(
        radius,
        d_u,
        d_xi,
        xi,
        exact["v0"],
        exact["v0_prime"],
    )
    ellu, ellv = physical_derivatives(
        log_Omega,
        d_u,
        d_xi,
        xi,
        exact["v0"],
        exact["v0_prime"],
    )
    exact_radius = exact["radius"]
    exact_log = exact["log_Omega"]
    relative_radius = np.abs(radius - exact_radius) / np.maximum(
        np.abs(exact_radius), np.longdouble(1.0e-30)
    )
    residual = weighted_spherical_residual(radius, log_Omega, radius*rv, radius*ru,
        -0.5*ellv, -0.5*ellu, u, xi, d_u, d_xi, exact['v0'], exact['v0_prime'],
        halo=max(2, min(u_count, xi_count)//8))
    high_precision_residual = None
    diagnostics = {
        "epsilon": epsilon,
        "minimum_radius": float(np.min(radius)),
        "exact_minimum_radius": float(np.min(exact_radius)),
        "future_boundary_radius_error": float(
            np.max(np.abs(radius[:, -1] - 2 * mass * epsilon))
        ),
        "radius_relative_raw_maximum": float(np.max(relative_radius)),
        "radius_relative_rms": float(np.sqrt(np.mean(relative_radius**2))),
        "log_omega_absolute_raw_maximum": float(
            np.max(np.abs(log_Omega - exact_log))
        ),
        "maximum_kretschmann": float(
            np.max(48 * np.longdouble(mass) ** 2 / radius**6)
        ),
        "declared_maximum_kretschmann": float(
            3 / (4 * np.longdouble(mass) ** 4 * np.longdouble(epsilon) ** 6)
        ),
        "spectral_tail_radius": spectral_tail(radius),
        "working_precision": {
            "dtype": str(dtype),
            "mantissa_bits": int(np.finfo(dtype).nmant),
            "mpmath_reference_decimal_digits": 40 if high_precision else None,
        },
        "independent_warped_product_residual": {
            key: value
            for key, value in residual.items()
            if key not in {"pointwise", "section_l2"}
        },
        "independent_warped_product_residual_high_precision": (
            high_precision_residual
        ),
        "degrees_of_freedom": int(2 * u_count * xi_count),
    }
    return CurvedSolution(
        u=u,
        xi=xi,
        v=exact["v"],
        v0=exact["v0"],
        radius=radius,
        log_Omega=log_Omega,
        x_out_scalar=radius * rv,
        x_in_scalar=radius * ru,
        Omega_omega=-0.5 * ellv,
        Omega_omegab=-0.5 * ellu,
        records=records,
        diagnostics=diagnostics,
    )


def overgrid_audit(
    solution: CurvedSolution,
    epsilon: float,
    *,
    mass: float = 1.0,
    extra_nodes: int = 4,
) -> dict[str, Any]:
    """Audit a curved-domain solution on absent Chebyshev nodes."""

    u, d_u = chebyshev_lobatto(
        len(solution.u) + extra_nodes, -1.0, -0.5
    )
    xi, d_xi = chebyshev_lobatto(
        len(solution.xi) + extra_nodes, 0.0, 1.0
    )
    basis_u = _basis_values(solution.u, u)
    basis_xi = _basis_values(solution.xi, xi)

    def transfer(values: Array) -> Array:
        return basis_u @ values @ basis_xi.T

    radius = transfer(solution.radius)
    log_Omega = transfer(solution.log_Omega)
    exact = exact_fields(
        u,
        xi,
        epsilon,
        mass,
        high_precision=epsilon <= 2.0**-4,
    )
    residual = weighted_spherical_residual(radius, log_Omega,
        transfer(solution.x_out_scalar), transfer(solution.x_in_scalar),
        transfer(solution.Omega_omega), transfer(solution.Omega_omegab),
        u, xi, d_u, d_xi, exact['v0'], exact['v0_prime'],
        halo=max(2, min(len(u), len(xi))//8))
    high_precision = None
    relative_radius = np.abs(radius - exact["radius"]) / np.maximum(
        np.abs(exact["radius"]), np.longdouble(1.0e-30)
    )
    return {
        "grid_relation": (
            "tensor-product barycentric evaluation on Chebyshev nodes "
            "absent from the construction grid"
        ),
        "u_count": len(u),
        "xi_count": len(xi),
        "radius_relative_raw_maximum": float(np.max(relative_radius)),
        "log_omega_absolute_raw_maximum": float(
            np.max(np.abs(log_Omega - exact["log_Omega"]))
        ),
        "warped_product_residual": {
            key: value
            for key, value in residual.items()
            if key not in {"pointwise", "section_l2"}
        },
        "warped_product_residual_high_precision": high_precision,
    }
