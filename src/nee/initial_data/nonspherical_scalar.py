"""Nonspherical Einstein--scalar characteristic data.

The free data use globally smooth real harmonics and are completed by the
Einstein--scalar characteristic constraints.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Callable

import numpy as np



from nee.numerics import scalar_initial_data as idata  # noqa: E402
from nee.numerics.scalar_config import (  # noqa: E402
    AngularConfig,
    CoordinateConfig,
    ExperimentConfig,
    InitialDataConfig,
    SolverConfig,
)
from nee.numerics.sphere import (  # noqa: E402
    connection_difference,
    lie_covariant_tensor,
    scalar_gradient,
    tangent_inverse,
    tensor_norm_sq,
    tensor_trace,
    tensor_tracefree,
    tracefree_symmetric_gradient,
)


Array = np.ndarray


def _rotation_z(angle: float) -> Array:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rotated_harmonic_fields(
    points: Array, rotation: Array
) -> tuple[Array, Array, Array]:
    """Return normalized Y20, grad(Y21R), and 0.1 grad(Y22R)."""

    local = points @ rotation
    x, y, z = local.T
    y20 = 3.0 * z**2 - 1.0
    y20 /= 2.0

    ambient_21_local = np.column_stack([z, np.zeros_like(z), x])
    ambient_22_local = np.column_stack([2.0 * x, -2.0 * y, np.zeros_like(z)])
    ambient_21 = ambient_21_local @ rotation.T
    ambient_22 = ambient_22_local @ rotation.T
    grad_21 = ambient_21 - np.einsum(
        "ni,ni->n", ambient_21, points
    )[:, None] * points
    grad_22 = ambient_22 - np.einsum(
        "ni,ni->n", ambient_22, points
    )[:, None] * points
    # The analytic suprema are 1 for grad(xz) and 2 for grad(x^2-y^2).
    grad_22 *= 0.05
    return y20, grad_21, grad_22


def _plan_free_data(rotation: Array) -> Callable[..., dict[str, Array]]:
    def generator(
        grid: Any, mesh: Any, config: ExperimentConfig
    ) -> dict[str, Array]:
        data = config.scalar_initial_data
        u = mesh.u
        radius = -u
        y20, vector_b, _ = _rotated_harmonic_fields(grid.points, rotation)
        g = (
            radius[None, :, None, None] ** 2
            * grid.projector[:, None, :, :]
        )
        Omega = np.sqrt(
            radius[None, :] ** data.lapse_radial_power
            * (
                1.0
                + data.lapse_angular_amplitude * y20[:, None]
            )
        )
        b = np.broadcast_to(
            data.shift_amplitude * vector_b[:, None, :],
            (grid.count, len(u), 3),
        ).copy()

        metric_u = mesh.differentiate_u(g, axis=1)
        Omega_chib = 0.5 * (
            metric_u + lie_covariant_tensor(grid, b, g)
        )
        log_Omega = np.log(Omega)
        Omega_e3_log_Omega = mesh.differentiate_u(log_Omega, axis=1) + np.einsum(
            "nui,nui->nu", b, scalar_gradient(grid, log_Omega)
        )
        Omega_omegab = -0.5 * Omega_e3_log_Omega
        inverse_g = tangent_inverse(grid, g)
        weighted_trace = tensor_trace(Omega_chib, inverse_g)
        weighted_hat = tensor_tracefree(Omega_chib, g, inverse_g)
        Omega_e3_Omega_trchib = mesh.differentiate_u(weighted_trace, axis=1) + np.einsum(
            "nui,nui->nu",
            b,
            scalar_gradient(grid, weighted_trace),
        )
        radicand = (
            -Omega_e3_Omega_trchib
            - 0.5 * weighted_trace**2
            - tensor_norm_sq(weighted_hat, inverse_g)
            - 4.0 * Omega_omegab * weighted_trace
        )
        return {
            "g": g,
            "Omega": Omega,
            "b": b,
            "weighted_chib_exact": Omega_chib,
            "weighted_omegab_exact": Omega_omegab,
            "weighted_hatchib_exact": weighted_hat,
            "incoming_scalar_exact": np.sqrt(np.maximum(radicand, 0.0)),
        }

    return generator


def _plan_reference_shear(rotation: Array) -> Callable[..., Array]:
    def generator(grid: Any, amplitude: float, profile: str) -> Array:
        del profile
        _, _, vector_chi = _rotated_harmonic_fields(grid.points, rotation)
        vector_chi = (amplitude / 0.1) * vector_chi
        g = grid.projector
        inverse_g = tangent_inverse(grid, g)
        difference, _ = connection_difference(grid, g, inverse_g)
        # On the round unit sphere lowering an ambient tangent vector with the
        # projector leaves its components unchanged.
        return tracefree_symmetric_gradient(
            grid, vector_chi, g, difference, inverse_g
        )

    return generator


def _scaled_construct(
    original: Callable[..., Any],
    lambda_phi: float,
    lambda_chi: float,
) -> Callable[..., Any]:
    def construct(config: ExperimentConfig) -> Any:
        bundle, grid, angular, mesh = original(config)
        zero_shear = abs(lambda_chi) <= 1.0e-15
        if abs(lambda_phi - 1.0) <= 1.0e-15 and not zero_shear:
            bundle.metadata["official_scalar_increment_multiplier"] = lambda_phi
            return bundle, grid, angular, mesh

        incoming = bundle.incoming
        outgoing = bundle.outgoing
        data = config.scalar_initial_data
        v = mesh.v
        corner_p = data.corner_outgoing_scalar
        Omega_e4phi = (
            corner_p
            + lambda_phi * v[None, :] ** data.outgoing_scalar_power
        )
        Omega_e4phi = np.broadcast_to(
            Omega_e4phi, (grid.count, len(v))
        ).copy()
        phi = (
            corner_p * v[None, :]
            + lambda_phi
            * v[None, :] ** (1.0 + data.outgoing_scalar_power)
            / (1.0 + data.outgoing_scalar_power)
        )
        phi = np.broadcast_to(phi, Omega_e4phi.shape).copy()
        reference_shear = (
            np.zeros_like(outgoing["reference_shear"])
            if zero_shear
            else outgoing["reference_shear"]
        )
        g, expansion, Omega_chih = idata._solve_outgoing_metric(
            grid,
            angular,
            mesh,
            incoming["g"][:, 0],
            incoming["Omega_trchi"][:, 0],
            outgoing["Omega"],
            outgoing["Omega_omega"],
            Omega_e4phi,
            reference_shear,
            data.boundary_substeps,
        )
        outgoing.update(
            {
                "g": g,
                "Omega_trchi": expansion,
                "Omega_chih": Omega_chih,
                "Omega_e4phi": Omega_e4phi,
                "phi": phi,
                "reference_shear": reference_shear,
            }
        )
        bundle.metadata["official_scalar_increment_multiplier"] = lambda_phi
        bundle.metadata["minimum_outgoing_expansion"] = float(
            np.min(expansion)
        )
        return bundle, grid, angular, mesh

    return construct


def _config(
    *,
    name: str,
    cap: float,
    lambda_omega: float,
    lambda_b: float,
    lambda_chi: float,
    retained: int = 3,
    work: int = 6,
    points: int = 100,
    tau_elements: int = 1,
    s_elements: int = 1,
    tau_degree: int = 4,
    s_degree: int = 6,
    metric_substeps: int = 1,
    derivative_halo: int = 1,
    iterations: int = 3,
    public_config=None,
) -> ExperimentConfig:
    result = ExperimentConfig(
        name=name,
        angular=AngularConfig(
            point_count=points,
            neighbor_count=min(24, points - 1),
            retained_degree=retained,
            work_degree=work,
        ),
        scalar_coordinates=CoordinateConfig(
            u_left=-1.0,
            u_right=-0.5,
            v_max=cap,
            tau_elements=tau_elements,
            tau_degree=tau_degree,
            s_elements=s_elements,
            s_degree=s_degree,
            fractional_power=0.1,
        ),
        scalar_initial_data=InitialDataConfig(
            lapse_radial_power=0.25,
            lapse_angular_amplitude=0.01 * lambda_omega,
            shift_amplitude=0.05 * lambda_b,
            corner_outgoing_expansion=1.6,
            corner_outgoing_scalar=8.0 * math.sqrt(2.0) / 5.0,
            outgoing_scalar_power=0.1,
            # The constraint generator rejects an exactly zero Omega_chih before
            # returning its otherwise valid incoming constraint solution.
            # A vanishing official case is generated through a harmless tiny seed
            # and then reconstructed with exact zero in _scaled_construct.
            shear_vector_amplitude=0.1 * (
                lambda_chi if lambda_chi != 0.0 else 1.0e-10
            ),
            shear_profile_power=0.1,
            shear_profile="quadrupole",
            boundary_substeps=8,
        ),
        solver=SolverConfig(
            picard_iterations=iterations,
            metric_substeps=metric_substeps,
            derivative_halo_u=derivative_halo,
            derivative_halo_v=derivative_halo,
        ),
    )
    if public_config is not None:
        data = public_config.initial_data
        result = replace(result,
            angular=replace(result.angular, neighbor_count=min(public_config.angular.neighbor_count, points-1)),
            scalar_coordinates=replace(result.scalar_coordinates,
                u_left=public_config.coordinates.u_min, u_right=public_config.coordinates.u_max,
                fractional_power=float(data['scalar_power'])),
            scalar_initial_data=replace(result.scalar_initial_data,
                lapse_radial_power=float(data['lapse_radial_power']),
                lapse_angular_amplitude=float(data['lapse_angular_amplitude'])*lambda_omega,
                shift_amplitude=float(data['shift_amplitude'])*lambda_b,
                shear_vector_amplitude=float(data['shear_amplitude'])*(lambda_chi if lambda_chi else 1e-10),
                outgoing_scalar_power=float(data['scalar_power']),
                shear_profile_power=float(data['scalar_power']),
                boundary_substeps=public_config.solver.boundary_substeps))
    result.validate()
    return result
