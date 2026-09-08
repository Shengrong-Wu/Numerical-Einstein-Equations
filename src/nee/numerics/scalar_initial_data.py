"""Characteristic initial-data construction for ESE numerical.

The free g, Omega, and b on ``v=0`` are sampled from the draft.
Connection quantities are then derived from their defining equations on the
same angular Galerkin/LGL discretization used by the iteration.  The remaining
fields are obtained by solving the characteristic constraints.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .scalar_config import ExperimentConfig
from .scalar_coordinates import CharacteristicPowerMesh, mesh_from_config
from .spherical_harmonics import AngularGalerkin
from .coordinate_quadrature import midpoint_values, stage_value
from .ricci_residual import one_form_lie_derivative
from .sphere import (
    PointSphereGrid,
    connection_difference,
    lie_covariant_tensor,
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
Face = dict[str, Array]


@dataclass
class InitialDataBundle:
    incoming: Face
    outgoing: Face
    raw: Face
    metadata: dict[str, object]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, Array] = {}
        for face_name, face in (
            ("incoming", self.incoming),
            ("outgoing", self.outgoing),
            ("raw", self.raw),
        ):
            arrays.update(
                {f"{face_name}__{name}": np.asarray(value) for name, value in face.items()}
            )
        arrays["metadata_json"] = np.asarray(
            json.dumps(self.metadata, sort_keys=True)
        )
        np.savez_compressed(target, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "InitialDataBundle":
        with np.load(Path(path), allow_pickle=False) as archive:
            faces: dict[str, Face] = {"incoming": {}, "outgoing": {}, "raw": {}}
            for name in archive.files:
                if "__" not in name:
                    continue
                face_name, field_name = name.split("__", 1)
                if face_name in faces:
                    faces[face_name][field_name] = archive[name].copy()
            metadata = json.loads(str(archive["metadata_json"]))
        return cls(
            incoming=faces["incoming"],
            outgoing=faces["outgoing"],
            raw=faces["raw"],
            metadata=metadata,
        )


def build_angular(
    scalar_config: ExperimentConfig,
) -> tuple[PointSphereGrid, AngularGalerkin]:
    angular_config = scalar_config.angular
    grid = PointSphereGrid.create(
        angular_config.point_count,
        neighbor_count=angular_config.neighbor_count,
        spectral_degree=angular_config.work_degree + 1,
    )
    angular = AngularGalerkin(
        grid,
        angular_config.retained_degree,
        angular_config.work_degree,
    )
    return grid, angular


def spherical_frame(grid: PointSphereGrid) -> tuple[Array, Array, Array]:
    """Return sin(theta), e_theta, and e_phi on the pole-free Fibonacci grid."""

    points = grid.points
    sine = np.sqrt(np.maximum(points[:, 0] ** 2 + points[:, 1] ** 2, 0.0))
    if float(np.min(sine)) <= 0.0:
        raise ValueError("the analytic spherical frame is undefined at a pole")
    e_theta = np.column_stack(
        [
            points[:, 2] * points[:, 0] / sine,
            points[:, 2] * points[:, 1] / sine,
            -sine,
        ]
    )
    e_phi = np.column_stack(
        [-points[:, 1] / sine, points[:, 0] / sine, np.zeros(grid.count)]
    )
    return sine, e_theta, e_phi


def _outer(first: Array, second: Array) -> Array:
    return np.einsum("ni,nj->nij", first, second)


def analytic_incoming_free_data(
    grid: PointSphereGrid,
    mesh: CharacteristicPowerMesh,
    scalar_config: ExperimentConfig,
) -> Face:
    """Sample the draft's explicit ``v=0`` g, Omega, and b."""

    data = scalar_config.scalar_initial_data
    u = mesh.u
    radius = -u
    sine, e_theta, e_phi = spherical_frame(grid)
    z = grid.points[:, 2]
    projector = grid.projector

    g = (
        radius[None, :, None, None] ** 2 * projector[:, None, :, :]
    )
    lapse_factor = 1.0 + data.lapse_angular_amplitude * sine
    Omega = np.sqrt(
        radius[None, :] ** data.lapse_radial_power * lapse_factor[:, None]
    )
    # Coordinate vector b=beta sin(theta) partial_phi.  Since
    # partial_phi=sin(theta)e_phi as an ambient vector, b=beta sin^2(theta)e_phi.
    shift_one_section = (
        data.shift_amplitude
        * sine[:, None, None] ** 2
        * e_phi[:, None, :]
    )
    b = np.broadcast_to(
        shift_one_section, (grid.count, len(u), 3)
    ).copy()

    # Exact coordinate formulas retained as a certificate for the numerical
    # definitions below.
    cross = 0.5 * (
        _outer(e_theta, e_phi) + _outer(e_phi, e_theta)
    )
    cross_coefficient = (
        radius[None, :] ** 2
        * data.shift_amplitude
        * sine[:, None]
        * z[:, None]
    )
    weighted_chib_exact = (
        -radius[None, :, None, None] * projector[:, None, :, :]
        + cross_coefficient[..., None, None] * cross[:, None, :, :]
    )
    weighted_omegab_exact = np.broadcast_to(
        data.lapse_radial_power / (4.0 * radius[None, :]),
        Omega.shape,
    ).copy()
    hatchib_exact = weighted_chib_exact + (
        radius[None, :, None, None] * projector[:, None, :, :]
    )
    hatchib_norm_sq = 0.5 * (
        data.shift_amplitude**2
        * sine[:, None] ** 2
        * z[:, None] ** 2
    )
    incoming_scalar_exact = np.sqrt(
        np.maximum(
            2.0 * data.lapse_radial_power / radius[None, :] ** 2
            - hatchib_norm_sq,
            0.0,
        )
    )
    return {
        "g": g,
        "Omega": Omega,
        "b": b,
        "weighted_chib_exact": weighted_chib_exact,
        "weighted_omegab_exact": weighted_omegab_exact,
        "weighted_hatchib_exact": hatchib_exact,
        "incoming_scalar_exact": incoming_scalar_exact,
    }


def _project_scalar(angular: AngularGalerkin, value: Array) -> Array:
    return angular.project_scalar(value)


def _project_vector(angular: AngularGalerkin, value: Array) -> Array:
    return angular.project_vector(value)


def _project_sym2(angular: AngularGalerkin, value: Array) -> Array:
    return angular.project_sym2(0.5 * (value + np.swapaxes(value, -1, -2)))


def _rk4_u(
    initial: Array,
    u: Array,
    rhs: Callable[[Array, int, float], Array],
) -> Array:
    result = np.zeros((initial.shape[0], len(u), *initial.shape[1:]), dtype=float)
    result[:, 0] = initial
    for index in range(len(u) - 1):
        step = float(u[index + 1] - u[index])
        current = result[:, index]
        k1 = rhs(current, index, 0.0)
        k2 = rhs(current + 0.5 * step * k1, index, 0.5)
        k3 = rhs(current + 0.5 * step * k2, index, 0.5)
        k4 = rhs(current + step * k3, index, 1.0)
        result[:, index + 1] = current + step * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0
    return result


def _stage(field: Array, midpoint: Array, index: int, alpha: float) -> Array:
    return stage_value(field, midpoint, index, alpha, axis=1)


def _solve_scalar_Omega_e3(
    grid: PointSphereGrid,
    angular: AngularGalerkin,
    u: Array,
    b: Array,
    source: Array,
    initial: Array,
    damping: Array | None = None,
) -> Array:
    midpoint_shift = midpoint_values(b, u, axis=1)
    midpoint_source = midpoint_values(source, u, axis=1)
    midpoint_damping = (
        None if damping is None else midpoint_values(damping, u, axis=1)
    )

    def rhs(value: Array, index: int, alpha: float) -> Array:
        stage_shift = _stage(b, midpoint_shift, index, alpha)
        complete = _stage(source, midpoint_source, index, alpha)
        complete = complete - np.einsum(
            "ni,ni->n", stage_shift, scalar_gradient(grid, value)
        )
        if damping is not None and midpoint_damping is not None:
            complete = complete - _stage(
                damping, midpoint_damping, index, alpha
            ) * value
        return _project_scalar(angular, complete)

    return _rk4_u(initial, u, rhs)


def _solve_zeta_covector(
    grid: PointSphereGrid,
    angular: AngularGalerkin,
    u: Array,
    b: Array,
    Omega_trchib: Array,
    source: Array,
    corner: float,
) -> Array:
    midpoint_shift = midpoint_values(b, u, axis=1)
    midpoint_trace = midpoint_values(Omega_trchib, u, axis=1)
    midpoint_source = midpoint_values(source, u, axis=1)

    def rhs(value: Array, index: int, alpha: float) -> Array:
        stage_shift = _stage(b, midpoint_shift, index, alpha)
        complete = _stage(source, midpoint_source, index, alpha)
        complete = complete - one_form_lie_derivative(
            grid, stage_shift, value
        )
        complete = complete - _stage(
            Omega_trchib, midpoint_trace, index, alpha
        )[..., None] * value
        return _project_vector(angular, complete)

    initial = np.full((grid.count, 3), float(corner))
    initial = np.einsum("nij,nj->ni", grid.projector, initial)
    return _rk4_u(initial, u, rhs)


def _solve_incoming_shear(
    grid: PointSphereGrid,
    angular: AngularGalerkin,
    mesh: CharacteristicPowerMesh,
    g: Array,
    inverse_g: Array,
    b: Array,
    Omega_trchib: Array,
    Omega_chibh: Array,
    source: Array,
) -> Array:
    u = mesh.u
    mixed_hatchib = np.matmul(Omega_chibh, inverse_g)
    metric_u = mesh.differentiate_u(g, axis=1)
    midpoint_fields = {
        name: midpoint_values(value, u, axis=1)
        for name, value in {
            "b": b,
            "trace": Omega_trchib,
            "mixed": mixed_hatchib,
            "source": source,
            "g": g,
            "metric_u": metric_u,
        }.items()
    }

    def rhs(value: Array, index: int, alpha: float) -> Array:
        stage_shift = _stage(
            b, midpoint_fields["b"], index, alpha
        )
        stage_trace = _stage(
            Omega_trchib, midpoint_fields["trace"], index, alpha
        )
        stage_mixed = _stage(
            mixed_hatchib, midpoint_fields["mixed"], index, alpha
        )
        stage_source = _stage(source, midpoint_fields["source"], index, alpha)
        stage_metric = _stage(g, midpoint_fields["g"], index, alpha)
        stage_metric_u = _stage(
            metric_u, midpoint_fields["metric_u"], index, alpha
        )
        stage_inverse = tangent_inverse(grid, stage_metric)
        stage_inverse_u = -np.matmul(
            np.matmul(stage_inverse, stage_metric_u), stage_inverse
        )
        stage_tensor = angular.project_g_tracefree(value, stage_inverse)
        complete = (
            0.5 * stage_trace[..., None, None] * stage_tensor
            + np.matmul(stage_mixed, stage_tensor)
            + np.matmul(stage_tensor, np.swapaxes(stage_mixed, -1, -2))
            + stage_source
            - lie_covariant_tensor(grid, stage_shift, stage_tensor)
        )
        return angular.project_g_tracefree_derivative(
            stage_tensor,
            complete,
            stage_inverse,
            stage_inverse_u,
        )

    initial = np.zeros((grid.count, 3, 3))
    result = _rk4_u(initial, u, rhs)
    for index in range(len(u)):
        result[:, index] = angular.project_g_tracefree(
            result[:, index], inverse_g[:, index]
        )
    return result


def _draft_reference_shear(
    grid: PointSphereGrid,
    amplitude: float,
    profile: str,
) -> Array:
    """Return nabla-hat-tensor V on the unit round corner sphere.

    The literal draft field is returned as exact zero because
    ``sin(theta) partial_theta`` is conformal Killing.  The quadrupole is the
    minimal axisymmetric replacement used by the experiment.
    """

    sine, e_theta, e_phi = spherical_frame(grid)
    if profile == "draft-conformal-killing":
        return np.zeros((grid.count, 3, 3))
    if profile != "quadrupole":
        raise ValueError(f"unknown Omega_chih profile {profile!r}")
    return amplitude * sine[:, None, None] ** 2 * (
        _outer(e_phi, e_phi) - _outer(e_theta, e_theta)
    )


def _solve_outgoing_metric(
    grid: PointSphereGrid,
    angular: AngularGalerkin,
    mesh: CharacteristicPowerMesh,
    corner_metric: Array,
    corner_weighted_expansion: Array,
    Omega: Array,
    Omega_omega: Array,
    Omega_e4phi: Array,
    reference_shear: Array,
    substeps: int,
) -> tuple[Array, Array, Array]:
    v = mesh.v
    g = np.zeros((grid.count, len(v), 3, 3))
    Omega_trchi = np.zeros((grid.count, len(v)))
    Omega_chih = np.zeros_like(g)
    g[:, 0] = corner_metric
    Omega_trchi[:, 0] = corner_weighted_expansion
    corner_inverse = tangent_inverse(grid, corner_metric)

    fields = {
        "reference": reference_shear,
        "Omega": Omega,
        "Omega_omega": Omega_omega,
        "Omega_e4phi": Omega_e4phi,
    }

    def interpolate(field: Array, index: int, fraction: float) -> Array:
        if fraction <= 1.0e-14:
            return field[:, index]
        if fraction >= 1.0 - 1.0e-14:
            return field[:, index + 1]
        return (1.0 - fraction) * field[:, index] + fraction * field[:, index + 1]

    for index in range(len(v) - 1):
        full_step = float(v[index + 1] - v[index])
        step = full_step / substeps
        value_metric = g[:, index]
        value_expansion = Omega_trchi[:, index]

        def rhs(
            stage_expansion: Array,
            stage_metric: Array,
            fraction: float,
        ) -> tuple[Array, Array, Array]:
            stage_reference = interpolate(
                fields["reference"], index, fraction
            )
            raw = np.matmul(
                np.matmul(stage_reference, corner_inverse), stage_metric
            )
            raw = 0.5 * (raw + np.swapaxes(raw, -1, -2))
            stage_inverse = tangent_inverse(grid, stage_metric)
            stage_shear = tensor_tracefree(raw, stage_metric, stage_inverse)
            stage_omega = interpolate(fields["Omega"], index, fraction)
            stage_w = interpolate(fields["Omega_omega"], index, fraction)
            stage_p = interpolate(fields["Omega_e4phi"], index, fraction)
            shear_norm = tensor_norm_sq(stage_shear, stage_inverse)
            expansion_rhs = (
                -0.5 * stage_expansion**2
                - 4.0 * stage_w * stage_expansion
                - stage_p**2
                - shear_norm
            )
            metric_rhs = (
                stage_expansion[..., None, None] * stage_metric
                + 2.0 * stage_shear
            )
            return (
                _project_scalar(angular, expansion_rhs),
                _project_sym2(angular, metric_rhs),
                stage_shear,
            )

        for substep in range(substeps):
            start = substep / substeps
            middle = (substep + 0.5) / substeps
            end = (substep + 1.0) / substeps
            k1_e, k1_g, _ = rhs(value_expansion, value_metric, start)
            k2_e, k2_g, _ = rhs(
                value_expansion + 0.5 * step * k1_e,
                value_metric + 0.5 * step * k1_g,
                middle,
            )
            k3_e, k3_g, _ = rhs(
                value_expansion + 0.5 * step * k2_e,
                value_metric + 0.5 * step * k2_g,
                middle,
            )
            k4_e, k4_g, _ = rhs(
                value_expansion + step * k3_e,
                value_metric + step * k3_g,
                end,
            )
            value_expansion = value_expansion + step * (
                k1_e + 2.0 * k2_e + 2.0 * k3_e + k4_e
            ) / 6.0
            value_metric = value_metric + step * (
                k1_g + 2.0 * k2_g + 2.0 * k3_g + k4_g
            ) / 6.0
        g[:, index + 1] = value_metric
        Omega_trchi[:, index + 1] = value_expansion
    transfer = np.matmul(
        np.matmul(reference_shear, corner_inverse[:, None]), g
    )
    transfer = 0.5 * (transfer + np.swapaxes(transfer, -1, -2))
    inverse_g = tangent_inverse(grid, g)
    Omega_chih = tensor_tracefree(transfer, g, inverse_g)
    for index in range(len(v)):
        Omega_chih[:, index] = angular.project_g_tracefree(
            Omega_chih[:, index], inverse_g[:, index]
        )
    return g, Omega_trchi, Omega_chih


def construct_initial_data(
    scalar_config: ExperimentConfig,
    *,
    free_data_generator=analytic_incoming_free_data,
    shear_generator=_draft_reference_shear,
) -> tuple[
    InitialDataBundle,
    PointSphereGrid,
    AngularGalerkin,
    CharacteristicPowerMesh,
]:
    scalar_config.validate()
    grid, angular = build_angular(scalar_config)
    mesh = mesh_from_config(scalar_config.scalar_coordinates)
    raw = free_data_generator(grid, mesh, scalar_config)
    data = scalar_config.scalar_initial_data
    u = mesh.u
    radius = -u

    g = _project_sym2(angular, raw["g"])
    log_Omega = _project_scalar(angular, np.log(raw["Omega"]))
    Omega = np.exp(log_Omega)
    b = _project_vector(angular, raw["b"])
    metric_u = mesh.differentiate_u(g, axis=1)
    Omega_chib = 0.5 * (
        metric_u + lie_covariant_tensor(grid, b, g)
    )
    Omega_chib = _project_sym2(angular, Omega_chib)
    inverse_g = tangent_inverse(grid, g)
    Omega_trchib = tensor_trace(Omega_chib, inverse_g)
    Omega_chibh = tensor_tracefree(Omega_chib, g, inverse_g)
    Omega_e3_log_Omega = mesh.differentiate_u(log_Omega, axis=1) + np.einsum(
        "nui,nui->nu", b, scalar_gradient(grid, log_Omega)
    )
    Omega_omegab = _project_scalar(angular, -0.5 * Omega_e3_log_Omega)

    Omega_e3_Omega_trchib = mesh.differentiate_u(
        Omega_trchib, axis=1
    ) + np.einsum(
        "nui,nui->nu",
        b,
        scalar_gradient(grid, Omega_trchib),
    )
    incoming_radicand = (
        -Omega_e3_Omega_trchib
        - 0.5 * Omega_trchib**2
        - tensor_norm_sq(Omega_chibh, inverse_g)
        - 4.0 * Omega_omegab * Omega_trchib
    )
    minimum_radicand = float(np.min(incoming_radicand))
    if minimum_radicand <= 0.0:
        raise FloatingPointError(
            "the incoming scalar constraint has nonpositive radicand: "
            f"{minimum_radicand:.12g}"
        )
    incoming_scalar_sign = (
        -1.0 if data.incoming_scalar_branch == "negative" else 1.0
    )
    Omega_e3phi = _project_scalar(
        angular, incoming_scalar_sign * np.sqrt(incoming_radicand)
    )

    phi = _solve_scalar_Omega_e3(
        grid,
        angular,
        u,
        b,
        Omega_e3phi,
        np.zeros(grid.count),
    )
    difference, _ = connection_difference(grid, g, inverse_g)
    nabla_phi = scalar_gradient(grid, phi)
    grad_log_omega = scalar_gradient(grid, log_Omega)
    div_hatchib = tensor_divergence(
        grid, Omega_chibh, difference, inverse_g
    )
    zeta_source = (
        -2.0 * scalar_gradient(grid, Omega_omegab)
        - div_hatchib
        + 0.5 * scalar_gradient(grid, Omega_trchib)
        - Omega_trchib[..., None] * grad_log_omega
        + Omega_e3phi[..., None] * nabla_phi
    )
    zeta_cov = _solve_zeta_covector(
        grid,
        angular,
        u,
        b,
        Omega_trchib,
        zeta_source,
        data.corner_zeta,
    )
    zeta = _project_vector(
        angular,
        np.einsum("nuij,nuj->nui", inverse_g, zeta_cov),
    )
    eta = zeta_cov + grad_log_omega
    etab = -zeta_cov + grad_log_omega
    eta_up = np.einsum("nuij,nuj->nui", inverse_g, eta)
    div_eta = vector_divergence(grid, eta_up, difference)
    eta_norm = np.einsum("nui,nuij,nuj->nu", eta, inverse_g, eta)
    grad_phi_norm = np.einsum(
        "nui,nuij,nuj->nu", nabla_phi, inverse_g, nabla_phi
    )
    curvature = np.broadcast_to(
        1.0 / radius[None, :] ** 2, Omega.shape
    ).copy()
    expansion_source = Omega**2 * (
        2.0 * div_eta
        + 2.0 * eta_norm
        - 2.0 * curvature
        + grad_phi_norm
    )
    Omega_trchi = _solve_scalar_Omega_e3(
        grid,
        angular,
        u,
        b,
        expansion_source,
        np.full(grid.count, data.corner_outgoing_expansion),
        damping=Omega_trchib,
    )

    eta_grad_hat = tracefree_symmetric_gradient(
        grid, eta, g, difference, inverse_g
    )
    eta_square_hat = tracefree_square(eta, g, inverse_g)
    phi_square_hat = tracefree_square(nabla_phi, g, inverse_g)
    incoming_shear_source = Omega[..., None, None] ** 2 * (
        eta_grad_hat
        + eta_square_hat
        + 0.5 * phi_square_hat
    ) - 0.5 * Omega_trchi[..., None, None] * Omega_chibh
    Omega_chih = _solve_incoming_shear(
        grid,
        angular,
        mesh,
        g,
        inverse_g,
        b,
        Omega_trchib,
        Omega_chibh,
        incoming_shear_source,
    )

    eta_grad_phi = np.einsum(
        "nui,nuij,nuj->nu", eta, inverse_g, nabla_phi
    )
    grad_phi_up = np.einsum("nuij,nuj->nui", inverse_g, nabla_phi)
    lap_phi = vector_divergence(grid, grad_phi_up, difference)
    scalar_p_source = (
        Omega**2 * lap_phi
        - 0.5 * Omega_trchi * Omega_e3phi
        + 2.0 * Omega**2 * eta_grad_phi
    )
    Omega_e4phi = _solve_scalar_Omega_e3(
        grid,
        angular,
        u,
        b,
        scalar_p_source,
        np.full(grid.count, data.corner_outgoing_scalar),
        damping=0.5 * Omega_trchib,
    )

    shear_cross = np.einsum(
        "nuik,nujl,nuij,nukl->nu",
        inverse_g,
        inverse_g,
        Omega_chih,
        Omega_chibh,
    )
    etab_norm = np.einsum("nui,nuij,nuj->nu", etab, inverse_g, etab)
    eta_etab = np.einsum("nui,nuij,nuj->nu", eta, inverse_g, etab)
    weighted_omega_source = (
        0.25 * Omega_e3phi * Omega_e4phi
        + 0.25 * Omega**2 * grad_phi_norm
        - 0.5 * Omega**2 * curvature
        + 0.25 * shear_cross
        - 0.125 * Omega_trchi * Omega_trchib
        + 0.5 * Omega**2 * etab_norm
        - Omega**2 * eta_etab
    )
    Omega_omega = _solve_scalar_Omega_e3(
        grid,
        angular,
        u,
        b,
        weighted_omega_source,
        np.full(grid.count, data.corner_weighted_omega),
    )

    incoming = {
        "u": u.copy(),
        "g": g,
        "Omega": Omega,
        "b": b,
        "Omega_chib": Omega_chib,
        "Omega_omegab": Omega_omegab,
        "Omega_trchib": Omega_trchib,
        "Omega_chibh": Omega_chibh,
        "Omega_e3phi": Omega_e3phi,
        "phi": phi,
        "zeta": zeta,
        "Omega_trchi": Omega_trchi,
        "Omega_chih": Omega_chih,
        "Omega_e4phi": Omega_e4phi,
        "Omega_omega": Omega_omega,
    }

    v = mesh.v
    omega_h = np.broadcast_to(Omega[:, :1], (grid.count, len(v))).copy()
    weighted_omega_h = np.zeros_like(omega_h)
    if data.outgoing_profile_scale > 0.0:
        outgoing_profile = (
            v / (v + data.outgoing_profile_scale)
        ) ** data.outgoing_scalar_power
        outgoing_profile[0] = 0.0
    else:
        outgoing_profile = v**data.outgoing_scalar_power
    scalar_p_h = (
        data.corner_outgoing_scalar
        + data.outgoing_scalar_amplitude * outgoing_profile[None, :]
    )
    scalar_p_h = np.broadcast_to(scalar_p_h, omega_h.shape).copy()
    phi_h = mesh.integrate_v(
        scalar_p_h,
        axis=1,
    )
    reference_shape = shear_generator(
        grid,
        data.shear_vector_amplitude,
        data.shear_profile,
    )
    if float(np.max(np.abs(reference_shape))) <= 1.0e-14:
        raise ValueError(
            "the draft-conformal-killing Omega_chih profile vanishes identically; "
            "select the quadrupole profile"
        )
    reference_shape = angular.project_g_tracefree(
        reference_shape,
        inverse_g[:, 0],
    )
    reference_shear = (
        outgoing_profile[None, :, None, None]
        * reference_shape[:, None, :, :]
    )
    boundary_metric, boundary_expansion, boundary_shear = (
        _solve_outgoing_metric(
            grid,
            angular,
            mesh,
            g[:, 0],
            Omega_trchi[:, 0],
            omega_h,
            weighted_omega_h,
            scalar_p_h,
            reference_shear,
            data.boundary_substeps,
        )
    )
    (
        refined_boundary_metric,
        refined_boundary_expansion,
        refined_boundary_shear,
    ) = _solve_outgoing_metric(
        grid,
        angular,
        mesh,
        g[:, 0],
        Omega_trchi[:, 0],
        omega_h,
        weighted_omega_h,
        scalar_p_h,
        reference_shear,
        2 * data.boundary_substeps,
    )
    outgoing = {
        "v": v.copy(),
        "g": boundary_metric,
        "Omega": omega_h,
        "Omega_omega": weighted_omega_h,
        "Omega_trchi": boundary_expansion,
        "Omega_chih": boundary_shear,
        "reference_shear": reference_shear,
        "Omega_e4phi": scalar_p_h,
        "phi": phi_h,
    }

    exact_chib_error = float(
        np.max(np.abs(Omega_chib - raw["weighted_chib_exact"]))
    )
    exact_omegab_error = float(
        np.max(np.abs(Omega_omegab - raw["weighted_omegab_exact"]))
    )
    corner_mismatches = {
        name: float(np.max(np.abs(outgoing[out_name][:, 0] - incoming[in_name][:, 0])))
        for name, out_name, in_name in (
            ("g", "g", "g"),
            ("Omega", "Omega", "Omega"),
            ("Omega_trchi", "Omega_trchi", "Omega_trchi"),
            ("Omega_chih", "Omega_chih", "Omega_chih"),
            ("Omega_e4phi", "Omega_e4phi", "Omega_e4phi"),
            ("phi", "phi", "phi"),
            ("Omega_omega", "Omega_omega", "Omega_omega"),
        )
    }
    metadata: dict[str, object] = {
        "schema": "nee-scalar-initial-data-1",
        "scalar_config": scalar_config.to_dict(),
        "angular": angular.diagnostics(),
        "scalar_coordinates": mesh.diagnostics(),
        "derivation": {
            "Omega_chib": "0.5*(partial_u + Lie_b)g",
            "Omega_omegab": "-0.5*(partial_u+b.grad)log(Omega)",
            "Omega_e3phi": (
                f"{data.incoming_scalar_branch} root of incoming "
                "Raychaudhuri"
            ),
            "remaining_incoming_fields": "characteristic constraint ODEs",
            "outgoing_metric": "coupled g/Omega_chih/Raychaudhuri RK4",
        },
        "corrections_to_draft": {
            "outgoing_lapse": (
                "Omega(-1,v)=sqrt(1+0.01 sin(theta)) for corner compatibility"
            ),
            "shear_vector": (
                "0.1 sin(theta)cos(theta) partial_theta; the literal draft "
                "field is conformal Killing and has zero trace-free gradient"
            ),
            "scalar_tracefree_factor": (
                "0.5*nabla(phi) hat-tensor nabla(phi)"
            ),
        },
        "minimum_incoming_scalar_radicand": minimum_radicand,
        "incoming_scalar_branch": data.incoming_scalar_branch,
        "outgoing_profile": {
            "formula": (
                "(v/(v+scale))^power"
                if data.outgoing_profile_scale > 0.0
                else "v^power"
            ),
            "power": data.outgoing_scalar_power,
            "scale": data.outgoing_profile_scale,
            "scalar_amplitude": data.outgoing_scalar_amplitude,
        },
        "analytic_connection_sample_errors": {
            "weighted_chib_max_abs": exact_chib_error,
            "weighted_omegab_max_abs": exact_omegab_error,
        },
        "corner_mismatches": corner_mismatches,
        "minimum_outgoing_expansion": float(np.min(boundary_expansion)),
        "outgoing_rk_refinement": {
            "base_substeps": data.boundary_substeps,
            "reference_substeps": 2 * data.boundary_substeps,
            "metric_max_abs_difference": float(
                np.max(np.abs(boundary_metric - refined_boundary_metric))
            ),
            "weighted_expansion_max_abs_difference": float(
                np.max(
                    np.abs(
                        boundary_expansion - refined_boundary_expansion
                    )
                )
            ),
            "shear_max_abs_difference": float(
                np.max(np.abs(boundary_shear - refined_boundary_shear))
            ),
        },
        "maximum_outgoing_shear_norm": float(
            np.max(
                np.sqrt(
                    np.maximum(
                        tensor_norm_sq(
                            boundary_shear,
                            tangent_inverse(grid, boundary_metric),
                        ),
                        0.0,
                    )
                )
            )
        ),
    }
    return (
        InitialDataBundle(
            incoming=incoming,
            outgoing=outgoing,
            raw=raw,
            metadata=metadata,
        ),
        grid,
        angular,
        mesh,
    )


def write_summary(bundle: InitialDataBundle, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(bundle.metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
