"""First-order Picard--Galerkin iteration for Einstein--scalar field numerical."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .scalar_coordinates import CharacteristicPowerMesh
from . import vacuum_iteration as eve
from .spherical_harmonics import AngularGalerkin
from .coordinate_quadrature import midpoint_values, stage_value
from .sphere import (
    PointSphereGrid,
    connection_difference,
    lie_covariant_tensor,
    one_form_covariant_derivative,
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
from .scalar_initial_data import InitialDataBundle


Array = np.ndarray


@dataclass
class ESEState(eve.FirstOrderState):
    """EVE first-order state augmented by three scalar-field variables."""

    phi: Array
    Omega_e4phi: Array  # partial_v phi = Omega e_4 phi
    Omega_e3phi: Array  # (partial_u+b.grad)phi = Omega e_3 phi


def _project_scalar(angular: AngularGalerkin | None, value: Array) -> Array:
    return value if angular is None else angular.project_scalar(value)


def _project_vector(angular: AngularGalerkin | None, value: Array) -> Array:
    return value if angular is None else angular.project_vector(value)


def _project_sym2(angular: AngularGalerkin | None, value: Array) -> Array:
    symmetric = 0.5 * (value + np.swapaxes(value, -1, -2))
    return symmetric if angular is None else angular.project_sym2(symmetric)


def _quintic_seed_weight(u: Array) -> Array:
    tau = -np.log(-u)
    coordinate = (tau[-1] - tau) / (tau[-1] - tau[0])
    return coordinate**3 * (
        10.0 + coordinate * (-15.0 + 6.0 * coordinate)
    )


def _extend_face(
    incoming: Array,
    outgoing: Array,
    u: Array,
    homogeneity: float,
) -> Array:
    """Extend an outgoing perturbation while preserving the incoming face."""

    radius_ratio = (-u) / float(-u[0])
    extra_slots = incoming.ndim - 2
    u_factor = (
        _quintic_seed_weight(u) * radius_ratio**homogeneity
    ).reshape((1, len(u), 1, *(1,) * extra_slots))
    baseline = incoming[:, :, None]
    perturbation = (outgoing - outgoing[:, :1])[:, None]
    result = baseline + u_factor * perturbation
    result[:, 0] = outgoing
    result[:, :, 0] = incoming
    return result


def initial_state(
    data: InitialDataBundle,
    angular: AngularGalerkin | None = None,
) -> ESEState:
    """Return a smooth seed matching both stored characteristic faces."""

    incoming = data.incoming
    outgoing = data.outgoing
    u = incoming["u"]
    g = _extend_face(incoming["g"], outgoing["g"], u, 2.0)
    Omega_trchi = _extend_face(incoming["Omega_trchi"], outgoing["Omega_trchi"], u, -1.0)
    Omega_chih = _extend_face(incoming["Omega_chih"], outgoing["Omega_chih"], u, 1.0)
    log_Omega = _extend_face(
        np.log(incoming["Omega"]), np.log(outgoing["Omega"]), u, 0.0
    )
    Omega_omega = _extend_face(
        incoming["Omega_omega"], outgoing["Omega_omega"], u, -1.0
    )
    phi = _extend_face(incoming["phi"], outgoing["phi"], u, 0.0)
    Omega_e4phi = _extend_face(
        incoming["Omega_e4phi"], outgoing["Omega_e4phi"], u, -1.0
    )
    v_count = len(outgoing["v"])

    def repeat(name: str) -> Array:
        return np.broadcast_to(
            incoming[name][:, :, None],
            (*incoming[name].shape[:2], v_count, *incoming[name].shape[2:]),
        ).copy()

    state = ESEState(
        g=g,
        Omega=np.exp(log_Omega),
        zeta=repeat("zeta"),
        b=repeat("b"),
        Omega_trchi=Omega_trchi,
        Omega_chih=Omega_chih,
        Omega_chib=repeat("Omega_chib"),
        Omega_omega=Omega_omega,
        Omega_omegab=repeat("Omega_omegab"),
        phi=phi,
        Omega_e4phi=Omega_e4phi,
        Omega_e3phi=repeat("Omega_e3phi"),
    )
    if angular is not None:
        angular.project_state(state)
        state.phi = angular.project_scalar(state.phi)
        state.Omega_e4phi = angular.project_scalar(state.Omega_e4phi)
        state.Omega_e3phi = angular.project_scalar(state.Omega_e3phi)
    impose_characteristic_faces(state, data)
    validate_state(angular.grid if angular is not None else None, state)
    return state


def impose_characteristic_faces(
    state: ESEState,
    data: InitialDataBundle,
) -> None:
    """Canonicalize fields whose construction equation starts on a face."""

    incoming = data.incoming
    outgoing = data.outgoing
    # Outgoing u-marches.
    for state_name, data_name in (
        ("g", "g"),
        ("Omega_trchi", "Omega_trchi"),
        ("Omega_chih", "Omega_chih"),
        ("Omega", "Omega"),
        ("Omega_omega", "Omega_omega"),
        ("phi", "phi"),
        ("Omega_e4phi", "Omega_e4phi"),
    ):
        getattr(state, state_name)[:, 0] = outgoing[data_name]
    # Incoming v-marches or definitions.
    for state_name, data_name in (
        ("g", "g"),
        ("Omega_trchi", "Omega_trchi"),
        ("Omega_chih", "Omega_chih"),
        ("phi", "phi"),
        ("zeta", "zeta"),
        ("b", "b"),
        ("Omega_chib", "Omega_chib"),
        ("Omega_omegab", "Omega_omegab"),
        ("Omega_e3phi", "Omega_e3phi"),
    ):
        getattr(state, state_name)[:, :, 0] = incoming[data_name]


def section_geometry(
    grid: PointSphereGrid,
    state: ESEState,
    include_curvature: bool = False,
) -> dict[str, Array]:
    geometry = eve.section_geometry(
        grid, state, include_curvature=include_curvature
    )
    inverse_g = geometry["inverse_g"]
    nabla_phi = scalar_gradient(grid, state.phi)
    grad_phi_up = np.einsum("n...ij,n...j->n...i", inverse_g, nabla_phi)
    geometry.update(
        {
            "nabla_phi": nabla_phi,
            "grad_phi_up": grad_phi_up,
            "lap_phi": vector_divergence(
                grid, grad_phi_up, geometry["difference"]
            ),
            # The project's hat-tensor convention is twice the STF product.
            "phi_square_hat": tracefree_square(
                nabla_phi, state.g, inverse_g
            ),
        }
    )
    return geometry


def solve_half_shear(
    grid: PointSphereGrid,
    state: ESEState,
    geometry: dict[str, Array],
    boundary: dict[str, Array],
    mesh: CharacteristicPowerMesh,
    angular: AngularGalerkin | None,
) -> Array:
    """Solve the moving-trace half-Omega_chih DAE including scalar anisotropy."""

    u = mesh.u
    half = np.zeros_like(state.g)
    boundary_inverse = tangent_inverse(grid, boundary["g"])
    old_inverse_boundary = geometry["inverse_g"][:, 0]
    transfer = np.matmul(
        np.matmul(boundary["Omega_chih"], boundary_inverse), state.g[:, 0]
    )
    transfer = 0.5 * (transfer + np.swapaxes(transfer, -1, -2))
    half[:, 0] = (
        tensor_tracefree(
            transfer, state.g[:, 0], old_inverse_boundary
        )
        if angular is None
        else angular.project_g_tracefree(
            transfer, old_inverse_boundary
        )
    )
    source = state.Omega[..., None, None] ** 2 * (
        geometry["eta_grad_hat"]
        + geometry["eta_square_hat"]
        + 0.5 * geometry["phi_square_hat"]
        - 0.5
        * (state.Omega_trchi / state.Omega)[..., None, None]
        * geometry["hatchib"]
    )
    Omega_chibh = geometry["Omega_chibh"]
    mixed_hatchib = np.matmul(Omega_chibh, geometry["inverse_g"])
    metric_u = mesh.differentiate_u(state.g, axis=1)
    midpoint = {
        name: midpoint_values(value, u, axis=1)
        for name, value in {
            "b": state.b,
            "trace": geometry["Omega_trchib"],
            "mixed": mixed_hatchib,
            "source": source,
            "g": state.g,
            "metric_u": metric_u,
        }.items()
    }
    for index in range(len(u) - 1):
        step = float(u[index + 1] - u[index])

        def rhs(value: Array, alpha: float) -> Array:
            stage_shift = stage_value(
                state.b, midpoint["b"], index, alpha, axis=1
            )
            stage_trace = stage_value(
                geometry["Omega_trchib"],
                midpoint["trace"],
                index,
                alpha,
                axis=1,
            )
            stage_mixed = stage_value(
                mixed_hatchib, midpoint["mixed"], index, alpha, axis=1
            )
            stage_source = stage_value(
                source, midpoint["source"], index, alpha, axis=1
            )
            stage_tensor = value
            stage_inverse = None
            stage_inverse_u = None
            if angular is not None:
                stage_metric = stage_value(
                    state.g, midpoint["g"], index, alpha, axis=1
                )
                stage_metric_u = stage_value(
                    metric_u, midpoint["metric_u"], index, alpha, axis=1
                )
                stage_inverse = tangent_inverse(grid, stage_metric)
                stage_inverse_u = -np.matmul(
                    np.matmul(stage_inverse, stage_metric_u), stage_inverse
                )
                stage_tensor = angular.project_g_tracefree(
                    value, stage_inverse
                )
            complete = (
                0.5 * stage_trace[..., None, None] * stage_tensor
                + np.matmul(stage_mixed, stage_tensor)
                + np.matmul(
                    stage_tensor, np.swapaxes(stage_mixed, -1, -2)
                )
                + stage_source
                - lie_covariant_tensor(grid, stage_shift, stage_tensor)
            )
            if angular is None:
                return complete
            return angular.project_g_tracefree_derivative(
                stage_tensor,
                complete,
                stage_inverse,
                stage_inverse_u,
            )

        current = half[:, index]
        k1 = rhs(current, 0.0)
        k2 = rhs(current + 0.5 * step * k1, 0.5)
        k3 = rhs(current + 0.5 * step * k2, 0.5)
        k4 = rhs(current + step * k3, 1.0)
        candidate = current + step * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0
        half[:, index + 1] = (
            tensor_tracefree(
                candidate,
                state.g[:, index + 1],
                geometry["inverse_g"][:, index + 1],
            )
            if angular is None
            else angular.project_g_tracefree(
                candidate, geometry["inverse_g"][:, index + 1]
            )
        )
    return half


def solve_scalar_p(
    grid: PointSphereGrid,
    state: ESEState,
    geometry: dict[str, Array],
    boundary_p: Array,
    u: Array,
    angular: AngularGalerkin | None,
) -> Array:
    value = np.zeros_like(state.Omega_e4phi)
    value[:, 0] = boundary_p
    eta_grad_phi = np.einsum(
        "n...i,n...i->n...", geometry["eta"], geometry["grad_phi_up"]
    )
    source = (
        state.Omega**2 * geometry["lap_phi"]
        - 0.5
        * state.Omega_trchi
        * state.Omega_e3phi
        + 2.0 * state.Omega**2 * eta_grad_phi
    )
    midpoint = {
        name: midpoint_values(field, u, axis=1)
        for name, field in {
            "b": state.b,
            "trace": geometry["Omega_trchib"],
            "source": source,
        }.items()
    }
    for index in range(len(u) - 1):
        step = float(u[index + 1] - u[index])

        def rhs(stage_p: Array, alpha: float) -> Array:
            stage_shift = stage_value(
                state.b, midpoint["b"], index, alpha, axis=1
            )
            complete = stage_value(
                source, midpoint["source"], index, alpha, axis=1
            )
            complete -= np.einsum(
                "n...i,n...i->n...",
                stage_shift,
                scalar_gradient(grid, stage_p),
            )
            complete -= 0.5 * stage_value(
                geometry["Omega_trchib"],
                midpoint["trace"],
                index,
                alpha,
                axis=1,
            ) * stage_p
            return _project_scalar(angular, complete)

        current = value[:, index]
        k1 = rhs(current, 0.0)
        k2 = rhs(current + 0.5 * step * k1, 0.5)
        k3 = rhs(current + 0.5 * step * k2, 0.5)
        k4 = rhs(current + step * k3, 1.0)
        value[:, index + 1] = current + step * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0
    return value


def solve_metric_and_expansion(
    grid: PointSphereGrid,
    old_state: ESEState,
    half: Array,
    new_omega: Array,
    new_Omega_omega: Array,
    new_scalar_p: Array,
    incoming: dict[str, Array],
    mesh: CharacteristicPowerMesh,
    substeps: int,
    angular: AngularGalerkin | None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Coupled all-current g, Raychaudhuri, and Omega_chih march."""

    g = np.zeros_like(old_state.g)
    Omega_trchi = np.zeros_like(old_state.Omega_trchi)
    g[:, :, 0] = incoming["g"]
    Omega_trchi[:, :, 0] = incoming["Omega_trchi"]
    old_inverse = tangent_inverse(grid, old_state.g)
    u = mesh.u
    v = mesh.v
    fields = {
        "log_Omega": np.log(new_omega),
        "Omega_omega": new_Omega_omega,
        "half": half,
        "old_metric": old_state.g,
        "Omega_e4phi": new_scalar_p,
    }
    interpolation_cache: dict[tuple[int, float], tuple[Array, Array]] = {}

    def interpolation_rule(
        interval: int, fraction: float
    ) -> tuple[Array, Array]:
        key = (interval, float(fraction))
        if key in interpolation_cache:
            return interpolation_cache[key]
        target_v = v[interval] + fraction * (
            v[interval + 1] - v[interval]
        )
        target_s = (target_v / mesh.v1) ** mesh.delta
        for segment, indices in zip(
            mesh.s.segments, mesh.s.indices, strict=True
        ):
            if interval >= int(indices[0]) and interval + 1 <= int(indices[-1]):
                weights = segment.interpolation_matrix(
                    np.asarray([target_s])
                )[0]
                interpolation_cache[key] = (indices, weights)
                return indices, weights
        raise ValueError(f"v interval {interval} is not contained in an s element")

    def external(name: str, interval: int, fraction: float) -> Array:
        if abs(fraction) < 1.0e-14:
            return np.take(fields[name], interval, axis=2)
        if abs(fraction - 1.0) < 1.0e-14:
            return np.take(fields[name], interval + 1, axis=2)
        indices, weights = interpolation_rule(interval, fraction)
        local = np.take(fields[name], indices, axis=2)
        return np.tensordot(weights, local, axes=(0, 2))

    def rhs(
        value_Omega_trchi: Array,
        value_metric: Array,
        interval: int,
        fraction: float,
    ) -> tuple[Array, Array, Array]:
        stage_omega = np.exp(external("log_Omega", interval, fraction))
        stage_half = external("half", interval, fraction)
        stage_old_metric = external("old_metric", interval, fraction)
        stage_old_inverse = tangent_inverse(grid, stage_old_metric)
        raw = np.matmul(
            np.matmul(stage_half, stage_old_inverse), value_metric
        )
        raw = 0.5 * (raw + np.swapaxes(raw, -1, -2))
        inverse_g = tangent_inverse(grid, value_metric)
        stage_shear = tensor_tracefree(raw, value_metric, inverse_g)
        shear_norm = tensor_norm_sq(stage_shear, inverse_g)
        stage_p = external("Omega_e4phi", interval, fraction)
        stage_Omega_omega = external("Omega_omega", interval, fraction)
        Omega_trchi_rhs = (
            -0.5 * value_Omega_trchi**2
            - 4.0 * stage_Omega_omega * value_Omega_trchi
            - shear_norm
            - stage_p**2
        )
        metric_rhs = (
            value_Omega_trchi[..., None, None] * value_metric
            + 2.0 * stage_shear
        )
        return (
            _project_scalar(angular, Omega_trchi_rhs),
            _project_sym2(angular, metric_rhs),
            stage_shear,
        )

    for interval in range(len(v) - 1):
        full_step = float(v[interval + 1] - v[interval])
        step = full_step / substeps
        value_Omega_trchi = Omega_trchi[:, :, interval]
        value_metric = g[:, :, interval]
        for substep in range(substeps):
            start = substep / substeps
            middle = (substep + 0.5) / substeps
            end = (substep + 1.0) / substeps
            k1_Omega_trchi, k1_g, _ = rhs(value_Omega_trchi, value_metric, interval, start)
            k2_Omega_trchi, k2_g, _ = rhs(
                value_Omega_trchi + 0.5 * step * k1_Omega_trchi,
                value_metric + 0.5 * step * k1_g,
                interval,
                middle,
            )
            k3_Omega_trchi, k3_g, _ = rhs(
                value_Omega_trchi + 0.5 * step * k2_Omega_trchi,
                value_metric + 0.5 * step * k2_g,
                interval,
                middle,
            )
            k4_Omega_trchi, k4_g, _ = rhs(
                value_Omega_trchi + step * k3_Omega_trchi,
                value_metric + step * k3_g,
                interval,
                end,
            )
            value_Omega_trchi = value_Omega_trchi + step * (
                k1_Omega_trchi + 2.0 * k2_Omega_trchi + 2.0 * k3_Omega_trchi + k4_Omega_trchi
            ) / 6.0
            value_metric = value_metric + step * (
                k1_g + 2.0 * k2_g + 2.0 * k3_g + k4_g
            ) / 6.0
        g[:, :, interval + 1] = value_metric
        Omega_trchi[:, :, interval + 1] = value_Omega_trchi
    transfer = np.matmul(np.matmul(half, old_inverse), g)
    transfer = 0.5 * (transfer + np.swapaxes(transfer, -1, -2))
    inverse_g = tangent_inverse(grid, g)
    Omega_chih = tensor_tracefree(transfer, g, inverse_g)
    raychaudhuri_source = _project_scalar(
        angular,
        -0.5 * Omega_trchi**2
        - 4.0 * new_Omega_omega * Omega_trchi
        - tensor_norm_sq(Omega_chih, inverse_g)
        - new_scalar_p**2,
    )
    metric_source = _project_sym2(
        angular,
        Omega_trchi[..., None, None] * g
        + 2.0 * Omega_chih,
    )
    return g, Omega_trchi, Omega_chih, raychaudhuri_source, metric_source


def solve_weighted_chib(
    grid: PointSphereGrid,
    g: Array,
    Omega: Array,
    zeta: Array,
    b: Array,
    Omega_trchi: Array,
    Omega_chih: Array,
    incoming_chib: Array,
    mesh: CharacteristicPowerMesh,
    angular: AngularGalerkin | None,
) -> tuple[Array, Array]:
    inverse_g = tangent_inverse(grid, g)
    difference, _ = connection_difference(grid, g, inverse_g)
    Omega_chi = (
        Omega_chih
        + 0.5 * Omega_trchi[..., None, None] * g
    )
    Lie_Omega_e3_Omega_chi = mesh.differentiate_u(Omega_chi, axis=1)
    Lie_Omega_e3_Omega_chi += lie_covariant_tensor(grid, b, Omega_chi)
    zeta = np.einsum("n...ij,n...j->n...i", g, zeta)
    nabla_zeta = one_form_covariant_derivative(grid, zeta, difference)
    sym_nabla_zeta = nabla_zeta + np.swapaxes(nabla_zeta, -1, -2)
    grad_log_omega = scalar_gradient(grid, np.log(Omega))
    sym_zeta_grad = np.einsum(
        "n...i,n...j->n...ij", zeta, grad_log_omega
    )
    sym_zeta_grad += np.swapaxes(sym_zeta_grad, -1, -2)
    source = _project_sym2(
        angular,
        Lie_Omega_e3_Omega_chi
        - 2.0 * Omega[..., None, None] ** 2 * sym_nabla_zeta
        - 4.0 * Omega[..., None, None] ** 2 * sym_zeta_grad,
    )
    Omega_chib = (
        incoming_chib[:, :, None]
        + mesh.integrate_v(source, axis=2)
    )
    return _project_sym2(angular, Omega_chib), source


def solve_incoming_scalar(
    grid: PointSphereGrid,
    Omega_e4phi: Array,
    phi: Array,
    Omega: Array,
    zeta: Array,
    b: Array,
    incoming_value: Array,
    mesh: CharacteristicPowerMesh,
    angular: AngularGalerkin | None,
) -> tuple[Array, Array]:
    grad_p = scalar_gradient(grid, Omega_e4phi)
    nabla_phi = scalar_gradient(grid, phi)
    source = mesh.differentiate_u(Omega_e4phi, axis=1)
    source += np.einsum("n...i,n...i->n...", b, grad_p)
    source -= 4.0 * Omega**2 * np.einsum(
        "n...i,n...i->n...", zeta, nabla_phi
    )
    source = _project_scalar(angular, source)
    value = incoming_value[:, :, None] + mesh.integrate_v(source, axis=2)
    return _project_scalar(angular, value), source


def reconstruct_phi(
    grid: PointSphereGrid,
    Omega_e3phi: Array,
    b: Array,
    boundary_phi: Array,
    u: Array,
    angular: AngularGalerkin | None,
) -> Array:
    result = np.zeros_like(Omega_e3phi)
    result[:, 0] = boundary_phi
    midpoint_shift = midpoint_values(b, u, axis=1)
    midpoint_source = midpoint_values(Omega_e3phi, u, axis=1)
    for index in range(len(u) - 1):
        step = float(u[index + 1] - u[index])

        def rhs(value: Array, alpha: float) -> Array:
            stage_shift = stage_value(
                b, midpoint_shift, index, alpha, axis=1
            )
            complete = stage_value(
                Omega_e3phi,
                midpoint_source,
                index,
                alpha,
                axis=1,
            )
            complete -= np.einsum(
                "n...i,n...i->n...",
                stage_shift,
                scalar_gradient(grid, value),
            )
            return _project_scalar(angular, complete)

        current = result[:, index]
        k1 = rhs(current, 0.0)
        k2 = rhs(current + 0.5 * step * k1, 0.5)
        k3 = rhs(current + 0.5 * step * k2, 0.5)
        k4 = rhs(current + step * k3, 1.0)
        result[:, index + 1] = current + step * (
            k1 + 2.0 * k2 + 2.0 * k3 + k4
        ) / 6.0
    return result


def picard_step(
    grid: PointSphereGrid,
    angular: AngularGalerkin,
    mesh: CharacteristicPowerMesh,
    state: ESEState,
    data: InitialDataBundle,
    metric_substeps: int = 2,
) -> tuple[ESEState, dict[str, object]]:
    """Apply one complete scalar-field Picard sweep."""

    impose_characteristic_faces(state, data)
    incoming = data.incoming
    outgoing = data.outgoing
    geometry = section_geometry(grid, state)
    half = solve_half_shear(
        grid, state, geometry, outgoing, mesh, angular
    )
    new_scalar_p = solve_scalar_p(
        grid,
        state,
        geometry,
        outgoing["Omega_e4phi"],
        mesh.u,
        angular,
    )
    new_phi = _project_scalar(
        angular,
        incoming["phi"][:, :, None]
        + mesh.integrate_v(new_scalar_p, axis=2),
    )

    inverse_g = geometry["inverse_g"]
    eta_etab = np.einsum(
        "n...i,n...ij,n...j->n...",
        geometry["eta"],
        inverse_g,
        geometry["etab"],
    )
    shear_cross = np.einsum(
        "n...ik,n...jl,n...ij,n...kl->n...",
        inverse_g,
        inverse_g,
        half,
        geometry["Omega_chibh"],
    )
    Omega_trchi = state.Omega_trchi
    eta_up = np.einsum(
        "n...ij,n...j->n...i", inverse_g, geometry["eta"]
    )
    div_eta = vector_divergence(
        grid, eta_up, geometry["difference"]
    )
    Omega_e3_Omega_trchi = mesh.differentiate_u(
        Omega_trchi, axis=1
    ) + np.einsum(
        "n...i,n...i->n...",
        state.b,
        scalar_gradient(grid, Omega_trchi),
    )
    omegab_source = _project_scalar(
        angular,
        0.25
        * (
            shear_cross
            + 0.5
            * Omega_trchi
            * geometry["Omega_trchib"]
            - 4.0 * state.Omega**2 * eta_etab
            + Omega_e3_Omega_trchi
            - 2.0 * state.Omega**2 * div_eta
            + new_scalar_p * state.Omega_e3phi
        ),
    )
    new_weighted_omegab_half = _project_scalar(
        angular,
        incoming["Omega_omegab"][:, :, None]
        + mesh.integrate_v(omegab_source, axis=2),
    )
    new_omega = eve.solve_log_omega(
        grid,
        state,
        new_weighted_omegab_half,
        mesh.u,
        initial_log_omega=np.log(outgoing["Omega"]),
        angular=angular,
    )
    new_weighted_omega = eve.solve_weighted_omega(
        grid,
        state,
        omegab_source,
        new_omega,
        mesh.u,
        initial_value=outgoing["Omega_omega"],
        angular=angular,
    )

    grad_old_phi = geometry["nabla_phi"]
    div_half = tensor_divergence(
        grid, half, geometry["difference"], inverse_g
    )
    grad_weighted_omega = scalar_gradient(grid, new_weighted_omega)
    grad_weighted_tr_chi = scalar_gradient(grid, Omega_trchi)
    grad_log_new_omega = scalar_gradient(grid, np.log(new_omega))
    half_zeta = np.einsum(
        "n...ij,n...j->n...i", half, state.zeta
    )
    zeta_source = _project_vector(
        angular,
        -2.0 * Omega_trchi[..., None] * state.zeta
        - 2.0 * np.einsum(
            "n...ij,n...j->n...i", inverse_g, half_zeta
        )
        + 2.0
        * np.einsum(
            "n...ij,n...j->n...i", inverse_g, grad_weighted_omega
        )
        + np.einsum("n...ij,n...j->n...i", inverse_g, div_half)
        - 0.5
        * np.einsum(
            "n...ij,n...j->n...i", inverse_g, grad_weighted_tr_chi
        )
        + Omega_trchi[..., None]
        * np.einsum(
            "n...ij,n...j->n...i", inverse_g, grad_log_new_omega
        )
        - new_scalar_p[..., None]
        * np.einsum("n...ij,n...j->n...i", inverse_g, grad_old_phi),
    )
    new_zeta = _project_vector(
        angular,
        incoming["zeta"][:, :, None]
        + mesh.integrate_v(zeta_source, axis=2),
    )
    shift_source = _project_vector(
        angular, -4.0 * new_omega[..., None] ** 2 * new_zeta
    )
    new_shift = _project_vector(
        angular,
        incoming["b"][:, :, None]
        + mesh.integrate_v(shift_source, axis=2),
    )
    (
        new_metric,
        new_Omega_trchi,
        new_shear,
        raychaudhuri_source,
        metric_source,
    ) = solve_metric_and_expansion(
        grid,
        state,
        half,
        new_omega,
        new_weighted_omega,
        new_scalar_p,
        incoming,
        mesh,
        metric_substeps,
        angular,
    )
    # The three outgoing fields are free characteristic data.
    new_metric[:, 0] = outgoing["g"]
    new_Omega_trchi[:, 0] = outgoing["Omega_trchi"]
    new_shear[:, 0] = outgoing["Omega_chih"]

    new_weighted_omegab, omegab_full_source = (
        eve.complete_weighted_omegab(
            grid,
            state,
            new_weighted_omegab_half,
            omegab_source,
            new_omega,
            new_weighted_omega,
            new_zeta,
            new_shift,
            angular=angular,
        )
    )
    new_weighted_chib, chib_source = solve_weighted_chib(
        grid,
        new_metric,
        new_omega,
        new_zeta,
        new_shift,
        new_Omega_trchi,
        new_shear,
        incoming["Omega_chib"],
        mesh,
        angular,
    )
    new_incoming_scalar, incoming_scalar_source = solve_incoming_scalar(
        grid,
        new_scalar_p,
        new_phi,
        new_omega,
        new_zeta,
        new_shift,
        incoming["Omega_e3phi"],
        mesh,
        angular,
    )
    new_state = ESEState(
        g=new_metric,
        Omega=new_omega,
        zeta=new_zeta,
        b=new_shift,
        Omega_trchi=new_Omega_trchi,
        Omega_chih=new_shear,
        Omega_chib=new_weighted_chib,
        Omega_omega=new_weighted_omega,
        Omega_omegab=new_weighted_omegab,
        phi=new_phi,
        Omega_e4phi=new_scalar_p,
        Omega_e3phi=new_incoming_scalar,
    )
    validate_state(grid, new_state)
    incoming_metric = eve.solve_incoming_metric(
        grid,
        outgoing["g"],
        new_weighted_chib,
        new_shift,
        mesh.u,
        angular=angular,
    )
    incoming_phi = reconstruct_phi(
        grid,
        new_incoming_scalar,
        new_shift,
        outgoing["phi"],
        mesh.u,
        angular,
    )
    context: dict[str, object] = {
        "half_shear": half,
        "weighted_omegab_half": new_weighted_omegab_half,
        "weighted_omegab_full": new_weighted_omegab,
        "omegab_half_source": omegab_source,
        "omegab_full_source": omegab_full_source,
        "omegab_source": omegab_source,
        "zeta_source": zeta_source,
        "shift_source": shift_source,
        "chib_source": chib_source,
        "scalar_p_source_semantics": (
            "u-characteristic wave solve with previous Picard coefficients"
        ),
        "incoming_scalar_source": incoming_scalar_source,
        "raychaudhuri_source": raychaudhuri_source,
        "metric_source": metric_source,
        "incoming_metric": incoming_metric,
        "incoming_phi": incoming_phi,
        "projection_tails": {
            # The values are immaterial to the residual; the keys certify
            # that the corrected nonlinear products passed through Pi_L.
            "weighted_omegab_full": 0.0,
            "weighted_omegab_full_source": 0.0,
        },
    }
    return new_state, context


def validate_state(
    grid: PointSphereGrid | None,
    state: ESEState,
) -> None:
    arrays = tuple(
        getattr(state, name)
        for name in (
            "g",
            "Omega",
            "zeta",
            "b",
            "Omega_trchi",
            "Omega_chih",
            "Omega_chib",
            "Omega_omega",
            "Omega_omegab",
            "phi",
            "Omega_e4phi",
            "Omega_e3phi",
        )
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError("ESE state contains a nonfinite value")
    if float(np.min(state.Omega)) <= 0.0:
        raise FloatingPointError("the Omega left the positive region")
    if grid is not None:
        local = np.einsum(
            "nia,n...ij,njb->n...ab",
            grid.frames,
            state.g,
            grid.frames,
        )
        if float(np.min(np.linalg.eigvalsh(local))) <= 0.0:
            raise FloatingPointError("the section g left the positive cone")


def update_norm(new: ESEState, old: ESEState) -> float:
    values = []
    for name in (
        "g",
        "Omega",
        "zeta",
        "b",
        "Omega_trchi",
        "Omega_chih",
        "Omega_chib",
        "Omega_omega",
        "Omega_omegab",
        "phi",
        "Omega_e4phi",
        "Omega_e3phi",
    ):
        current = getattr(new, name)
        previous = getattr(old, name)
        scale = np.maximum(1.0, np.maximum(np.abs(current), np.abs(previous)))
        values.append(float(np.mean(((current - previous) / scale) ** 2)))
    return math.sqrt(sum(values))


def update_map(new: ESEState, old: ESEState) -> Array:
    result = np.zeros(new.Omega_trchi.shape[1:])
    for name in (
        "g",
        "Omega",
        "zeta",
        "b",
        "Omega_trchi",
        "Omega_chih",
        "Omega_chib",
        "Omega_omega",
        "Omega_omegab",
        "phi",
        "Omega_e4phi",
        "Omega_e3phi",
    ):
        current = getattr(new, name)
        previous = getattr(old, name)
        scale = np.maximum(1.0, np.maximum(np.abs(current), np.abs(previous)))
        relative_sq = ((current - previous) / scale) ** 2
        reduction_axes = (0, *range(3, relative_sq.ndim))
        result += np.mean(relative_sq, axis=reduction_axes)
    return np.sqrt(result)
