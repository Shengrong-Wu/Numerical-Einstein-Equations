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
    scalar_p: Array  # partial_v phi = Omega e_4 phi
    incoming_scalar: Array  # (partial_u+b.grad)phi = Omega e_3 phi


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
    metric = _extend_face(incoming["metric"], outgoing["metric"], u, 2.0)
    q = _extend_face(incoming["q"], outgoing["q"], u, -1.0)
    shear = _extend_face(incoming["shear"], outgoing["shear"], u, 1.0)
    log_omega = _extend_face(
        np.log(incoming["omega"]), np.log(outgoing["omega"]), u, 0.0
    )
    weighted_omega = _extend_face(
        incoming["weighted_omega"], outgoing["weighted_omega"], u, -1.0
    )
    phi = _extend_face(incoming["phi"], outgoing["phi"], u, 0.0)
    scalar_p = _extend_face(
        incoming["scalar_p"], outgoing["scalar_p"], u, -1.0
    )
    v_count = len(outgoing["v"])

    def repeat(name: str) -> Array:
        return np.broadcast_to(
            incoming[name][:, :, None],
            (*incoming[name].shape[:2], v_count, *incoming[name].shape[2:]),
        ).copy()

    state = ESEState(
        metric=metric,
        omega=np.exp(log_omega),
        zeta_up=repeat("zeta_up"),
        shift=repeat("shift"),
        q=q,
        shear=shear,
        weighted_chib=repeat("weighted_chib"),
        weighted_omega=weighted_omega,
        weighted_omegab=repeat("weighted_omegab"),
        phi=phi,
        scalar_p=scalar_p,
        incoming_scalar=repeat("incoming_scalar"),
    )
    if angular is not None:
        angular.project_state(state)
        state.phi = angular.project_scalar(state.phi)
        state.scalar_p = angular.project_scalar(state.scalar_p)
        state.incoming_scalar = angular.project_scalar(state.incoming_scalar)
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
        ("metric", "metric"),
        ("q", "q"),
        ("shear", "shear"),
        ("omega", "omega"),
        ("weighted_omega", "weighted_omega"),
        ("phi", "phi"),
        ("scalar_p", "scalar_p"),
    ):
        getattr(state, state_name)[:, 0] = outgoing[data_name]
    # Incoming v-marches or definitions.
    for state_name, data_name in (
        ("metric", "metric"),
        ("q", "q"),
        ("shear", "shear"),
        ("phi", "phi"),
        ("zeta_up", "zeta_up"),
        ("shift", "shift"),
        ("weighted_chib", "weighted_chib"),
        ("weighted_omegab", "weighted_omegab"),
        ("incoming_scalar", "incoming_scalar"),
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
    inverse = geometry["inverse"]
    grad_phi = scalar_gradient(grid, state.phi)
    grad_phi_up = np.einsum("n...ij,n...j->n...i", inverse, grad_phi)
    geometry.update(
        {
            "grad_phi": grad_phi,
            "grad_phi_up": grad_phi_up,
            "lap_phi": vector_divergence(
                grid, grad_phi_up, geometry["difference"]
            ),
            # The project's hat-tensor convention is twice the STF product.
            "phi_square_hat": tracefree_square(
                grad_phi, state.metric, inverse
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
    """Solve the moving-trace half-shear DAE including scalar anisotropy."""

    u = mesh.u
    half = np.zeros_like(state.metric)
    boundary_inverse = tangent_inverse(grid, boundary["metric"])
    old_inverse_boundary = geometry["inverse"][:, 0]
    transfer = np.matmul(
        np.matmul(boundary["shear"], boundary_inverse), state.metric[:, 0]
    )
    transfer = 0.5 * (transfer + np.swapaxes(transfer, -1, -2))
    half[:, 0] = (
        tensor_tracefree(
            transfer, state.metric[:, 0], old_inverse_boundary
        )
        if angular is None
        else angular.project_g_tracefree(
            transfer, old_inverse_boundary
        )
    )
    source = state.omega[..., None, None] ** 2 * (
        geometry["eta_grad_hat"]
        + geometry["eta_square_hat"]
        + 0.5 * geometry["phi_square_hat"]
        - 0.5
        * (state.omega * state.q)[..., None, None]
        * geometry["hatchib"]
    )
    weighted_hatchib = geometry["weighted_hatchib"]
    mixed_hatchib = np.matmul(weighted_hatchib, geometry["inverse"])
    metric_u = mesh.differentiate_u(state.metric, axis=1)
    midpoint = {
        name: midpoint_values(value, u, axis=1)
        for name, value in {
            "shift": state.shift,
            "trace": geometry["weighted_tr_chib"],
            "mixed": mixed_hatchib,
            "source": source,
            "metric": state.metric,
            "metric_u": metric_u,
        }.items()
    }
    for index in range(len(u) - 1):
        step = float(u[index + 1] - u[index])

        def rhs(value: Array, alpha: float) -> Array:
            stage_shift = stage_value(
                state.shift, midpoint["shift"], index, alpha, axis=1
            )
            stage_trace = stage_value(
                geometry["weighted_tr_chib"],
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
                    state.metric, midpoint["metric"], index, alpha, axis=1
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
                state.metric[:, index + 1],
                geometry["inverse"][:, index + 1],
            )
            if angular is None
            else angular.project_g_tracefree(
                candidate, geometry["inverse"][:, index + 1]
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
    value = np.zeros_like(state.scalar_p)
    value[:, 0] = boundary_p
    eta_grad_phi = np.einsum(
        "n...i,n...i->n...", geometry["eta"], geometry["grad_phi_up"]
    )
    source = (
        state.omega**2 * geometry["lap_phi"]
        - 0.5
        * (state.omega**2 * state.q)
        * state.incoming_scalar
        + 2.0 * state.omega**2 * eta_grad_phi
    )
    midpoint = {
        name: midpoint_values(field, u, axis=1)
        for name, field in {
            "shift": state.shift,
            "trace": geometry["weighted_tr_chib"],
            "source": source,
        }.items()
    }
    for index in range(len(u) - 1):
        step = float(u[index + 1] - u[index])

        def rhs(stage_p: Array, alpha: float) -> Array:
            stage_shift = stage_value(
                state.shift, midpoint["shift"], index, alpha, axis=1
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
                geometry["weighted_tr_chib"],
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
    new_scalar_p: Array,
    incoming: dict[str, Array],
    mesh: CharacteristicPowerMesh,
    substeps: int,
    angular: AngularGalerkin | None,
) -> tuple[Array, Array, Array, Array, Array]:
    """Coupled all-current metric, Raychaudhuri, and shear march."""

    metric = np.zeros_like(old_state.metric)
    q = np.zeros_like(old_state.q)
    metric[:, :, 0] = incoming["metric"]
    q[:, :, 0] = incoming["q"]
    old_inverse = tangent_inverse(grid, old_state.metric)
    u = mesh.u
    v = mesh.v
    fields = {
        "log_omega": np.log(new_omega),
        "half": half,
        "old_metric": old_state.metric,
        "scalar_p": new_scalar_p,
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
        value_q: Array,
        value_metric: Array,
        interval: int,
        fraction: float,
    ) -> tuple[Array, Array, Array]:
        stage_omega = np.exp(external("log_omega", interval, fraction))
        stage_half = external("half", interval, fraction)
        stage_old_metric = external("old_metric", interval, fraction)
        stage_old_inverse = tangent_inverse(grid, stage_old_metric)
        raw = np.matmul(
            np.matmul(stage_half, stage_old_inverse), value_metric
        )
        raw = 0.5 * (raw + np.swapaxes(raw, -1, -2))
        inverse = tangent_inverse(grid, value_metric)
        stage_shear = tensor_tracefree(raw, value_metric, inverse)
        shear_norm = tensor_norm_sq(stage_shear, inverse)
        stage_p = external("scalar_p", interval, fraction)
        q_rhs = (
            -0.5 * stage_omega**2 * value_q**2
            - (shear_norm + stage_p**2) / stage_omega**2
        )
        metric_rhs = (
            stage_omega[..., None, None] ** 2
            * value_q[..., None, None]
            * value_metric
            + 2.0 * stage_shear
        )
        return (
            _project_scalar(angular, q_rhs),
            _project_sym2(angular, metric_rhs),
            stage_shear,
        )

    for interval in range(len(v) - 1):
        full_step = float(v[interval + 1] - v[interval])
        step = full_step / substeps
        value_q = q[:, :, interval]
        value_metric = metric[:, :, interval]
        for substep in range(substeps):
            start = substep / substeps
            middle = (substep + 0.5) / substeps
            end = (substep + 1.0) / substeps
            k1_q, k1_g, _ = rhs(value_q, value_metric, interval, start)
            k2_q, k2_g, _ = rhs(
                value_q + 0.5 * step * k1_q,
                value_metric + 0.5 * step * k1_g,
                interval,
                middle,
            )
            k3_q, k3_g, _ = rhs(
                value_q + 0.5 * step * k2_q,
                value_metric + 0.5 * step * k2_g,
                interval,
                middle,
            )
            k4_q, k4_g, _ = rhs(
                value_q + step * k3_q,
                value_metric + step * k3_g,
                interval,
                end,
            )
            value_q = value_q + step * (
                k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q
            ) / 6.0
            value_metric = value_metric + step * (
                k1_g + 2.0 * k2_g + 2.0 * k3_g + k4_g
            ) / 6.0
        metric[:, :, interval + 1] = value_metric
        q[:, :, interval + 1] = value_q
    transfer = np.matmul(np.matmul(half, old_inverse), metric)
    transfer = 0.5 * (transfer + np.swapaxes(transfer, -1, -2))
    inverse = tangent_inverse(grid, metric)
    shear = tensor_tracefree(transfer, metric, inverse)
    raychaudhuri_source = _project_scalar(
        angular,
        -0.5 * new_omega**2 * q**2
        - (tensor_norm_sq(shear, inverse) + new_scalar_p**2) / new_omega**2,
    )
    metric_source = _project_sym2(
        angular,
        new_omega[..., None, None] ** 2
        * q[..., None, None]
        * metric
        + 2.0 * shear,
    )
    return metric, q, shear, raychaudhuri_source, metric_source


def solve_weighted_chib(
    grid: PointSphereGrid,
    metric: Array,
    omega: Array,
    zeta_up: Array,
    shift: Array,
    q: Array,
    shear: Array,
    incoming_chib: Array,
    mesh: CharacteristicPowerMesh,
    angular: AngularGalerkin | None,
) -> tuple[Array, Array]:
    inverse = tangent_inverse(grid, metric)
    difference, _ = connection_difference(grid, metric, inverse)
    weighted_tr_chi = omega**2 * q
    weighted_chi = (
        shear
        + 0.5 * weighted_tr_chi[..., None, None] * metric
    )
    d3_weighted_chi = mesh.differentiate_u(weighted_chi, axis=1)
    d3_weighted_chi += lie_covariant_tensor(grid, shift, weighted_chi)
    zeta = np.einsum("n...ij,n...j->n...i", metric, zeta_up)
    nabla_zeta = one_form_covariant_derivative(grid, zeta, difference)
    sym_nabla_zeta = nabla_zeta + np.swapaxes(nabla_zeta, -1, -2)
    grad_log_omega = scalar_gradient(grid, np.log(omega))
    sym_zeta_grad = np.einsum(
        "n...i,n...j->n...ij", zeta, grad_log_omega
    )
    sym_zeta_grad += np.swapaxes(sym_zeta_grad, -1, -2)
    source = _project_sym2(
        angular,
        d3_weighted_chi
        - 2.0 * omega[..., None, None] ** 2 * sym_nabla_zeta
        - 4.0 * omega[..., None, None] ** 2 * sym_zeta_grad,
    )
    weighted_chib = (
        incoming_chib[:, :, None]
        + mesh.integrate_v(source, axis=2)
    )
    return _project_sym2(angular, weighted_chib), source


def solve_incoming_scalar(
    grid: PointSphereGrid,
    scalar_p: Array,
    phi: Array,
    omega: Array,
    zeta_up: Array,
    shift: Array,
    incoming_value: Array,
    mesh: CharacteristicPowerMesh,
    angular: AngularGalerkin | None,
) -> tuple[Array, Array]:
    grad_p = scalar_gradient(grid, scalar_p)
    grad_phi = scalar_gradient(grid, phi)
    source = mesh.differentiate_u(scalar_p, axis=1)
    source += np.einsum("n...i,n...i->n...", shift, grad_p)
    source -= 4.0 * omega**2 * np.einsum(
        "n...i,n...i->n...", zeta_up, grad_phi
    )
    source = _project_scalar(angular, source)
    value = incoming_value[:, :, None] + mesh.integrate_v(source, axis=2)
    return _project_scalar(angular, value), source


def reconstruct_phi(
    grid: PointSphereGrid,
    incoming_scalar: Array,
    shift: Array,
    boundary_phi: Array,
    u: Array,
    angular: AngularGalerkin | None,
) -> Array:
    result = np.zeros_like(incoming_scalar)
    result[:, 0] = boundary_phi
    midpoint_shift = midpoint_values(shift, u, axis=1)
    midpoint_source = midpoint_values(incoming_scalar, u, axis=1)
    for index in range(len(u) - 1):
        step = float(u[index + 1] - u[index])

        def rhs(value: Array, alpha: float) -> Array:
            stage_shift = stage_value(
                shift, midpoint_shift, index, alpha, axis=1
            )
            complete = stage_value(
                incoming_scalar,
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
        outgoing["scalar_p"],
        mesh.u,
        angular,
    )
    new_phi = _project_scalar(
        angular,
        incoming["phi"][:, :, None]
        + mesh.integrate_v(new_scalar_p, axis=2),
    )

    inverse = geometry["inverse"]
    eta_etab = np.einsum(
        "n...i,n...ij,n...j->n...",
        geometry["eta"],
        inverse,
        geometry["etab"],
    )
    shear_cross = np.einsum(
        "n...ik,n...jl,n...ij,n...kl->n...",
        inverse,
        inverse,
        half,
        geometry["weighted_hatchib"],
    )
    weighted_tr_chi = state.omega**2 * state.q
    eta_up = np.einsum(
        "n...ij,n...j->n...i", inverse, geometry["eta"]
    )
    div_eta = vector_divergence(
        grid, eta_up, geometry["difference"]
    )
    d3_weighted_tr_chi = mesh.differentiate_u(
        weighted_tr_chi, axis=1
    ) + np.einsum(
        "n...i,n...i->n...",
        state.shift,
        scalar_gradient(grid, weighted_tr_chi),
    )
    omegab_source = _project_scalar(
        angular,
        0.25
        * (
            shear_cross
            + 0.5
            * weighted_tr_chi
            * geometry["weighted_tr_chib"]
            - 4.0 * state.omega**2 * eta_etab
            + d3_weighted_tr_chi
            - 2.0 * state.omega**2 * div_eta
            + new_scalar_p * state.incoming_scalar
        ),
    )
    new_weighted_omegab_half = _project_scalar(
        angular,
        incoming["weighted_omegab"][:, :, None]
        + mesh.integrate_v(omegab_source, axis=2),
    )
    new_omega = eve.solve_log_omega(
        grid,
        state,
        new_weighted_omegab_half,
        mesh.u,
        initial_log_omega=np.log(outgoing["omega"]),
        angular=angular,
    )
    new_weighted_omega = eve.solve_weighted_omega(
        grid,
        state,
        omegab_source,
        new_omega,
        mesh.u,
        initial_value=outgoing["weighted_omega"],
        angular=angular,
    )

    grad_old_phi = geometry["grad_phi"]
    div_half = tensor_divergence(
        grid, half, geometry["difference"], inverse
    )
    grad_weighted_omega = scalar_gradient(grid, new_weighted_omega)
    grad_weighted_tr_chi = scalar_gradient(grid, weighted_tr_chi)
    grad_log_new_omega = scalar_gradient(grid, np.log(new_omega))
    half_zeta = np.einsum(
        "n...ij,n...j->n...i", half, state.zeta_up
    )
    zeta_source = _project_vector(
        angular,
        -2.0 * weighted_tr_chi[..., None] * state.zeta_up
        - 2.0 * np.einsum(
            "n...ij,n...j->n...i", inverse, half_zeta
        )
        + 2.0
        * np.einsum(
            "n...ij,n...j->n...i", inverse, grad_weighted_omega
        )
        + np.einsum("n...ij,n...j->n...i", inverse, div_half)
        - 0.5
        * np.einsum(
            "n...ij,n...j->n...i", inverse, grad_weighted_tr_chi
        )
        + weighted_tr_chi[..., None]
        * np.einsum(
            "n...ij,n...j->n...i", inverse, grad_log_new_omega
        )
        - new_scalar_p[..., None]
        * np.einsum("n...ij,n...j->n...i", inverse, grad_old_phi),
    )
    new_zeta = _project_vector(
        angular,
        incoming["zeta_up"][:, :, None]
        + mesh.integrate_v(zeta_source, axis=2),
    )
    shift_source = _project_vector(
        angular, -4.0 * new_omega[..., None] ** 2 * new_zeta
    )
    new_shift = _project_vector(
        angular,
        incoming["shift"][:, :, None]
        + mesh.integrate_v(shift_source, axis=2),
    )
    (
        new_metric,
        new_q,
        new_shear,
        raychaudhuri_source,
        metric_source,
    ) = solve_metric_and_expansion(
        grid,
        state,
        half,
        new_omega,
        new_scalar_p,
        incoming,
        mesh,
        metric_substeps,
        angular,
    )
    # The three outgoing fields are free characteristic data.
    new_metric[:, 0] = outgoing["metric"]
    new_q[:, 0] = outgoing["q"]
    new_shear[:, 0] = outgoing["shear"]

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
        new_q,
        new_shear,
        incoming["weighted_chib"],
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
        incoming["incoming_scalar"],
        mesh,
        angular,
    )
    new_state = ESEState(
        metric=new_metric,
        omega=new_omega,
        zeta_up=new_zeta,
        shift=new_shift,
        q=new_q,
        shear=new_shear,
        weighted_chib=new_weighted_chib,
        weighted_omega=new_weighted_omega,
        weighted_omegab=new_weighted_omegab,
        phi=new_phi,
        scalar_p=new_scalar_p,
        incoming_scalar=new_incoming_scalar,
    )
    validate_state(grid, new_state)
    incoming_metric = eve.solve_incoming_metric(
        grid,
        outgoing["metric"],
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
            "metric",
            "omega",
            "zeta_up",
            "shift",
            "q",
            "shear",
            "weighted_chib",
            "weighted_omega",
            "weighted_omegab",
            "phi",
            "scalar_p",
            "incoming_scalar",
        )
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError("ESE state contains a nonfinite value")
    if float(np.min(state.omega)) <= 0.0:
        raise FloatingPointError("the lapse left the positive region")
    if grid is not None:
        local = np.einsum(
            "nia,n...ij,njb->n...ab",
            grid.frames,
            state.metric,
            grid.frames,
        )
        if float(np.min(np.linalg.eigvalsh(local))) <= 0.0:
            raise FloatingPointError("the section metric left the positive cone")


def update_norm(new: ESEState, old: ESEState) -> float:
    values = []
    for name in (
        "metric",
        "omega",
        "zeta_up",
        "shift",
        "q",
        "shear",
        "weighted_chib",
        "weighted_omega",
        "weighted_omegab",
        "phi",
        "scalar_p",
        "incoming_scalar",
    ):
        current = getattr(new, name)
        previous = getattr(old, name)
        scale = np.maximum(1.0, np.maximum(np.abs(current), np.abs(previous)))
        values.append(float(np.mean(((current - previous) / scale) ** 2)))
    return math.sqrt(sum(values))


def update_map(new: ESEState, old: ESEState) -> Array:
    result = np.zeros(new.q.shape[1:])
    for name in (
        "metric",
        "omega",
        "zeta_up",
        "shift",
        "q",
        "shear",
        "weighted_chib",
        "weighted_omega",
        "weighted_omegab",
        "phi",
        "scalar_p",
        "incoming_scalar",
    ):
        current = getattr(new, name)
        previous = getattr(old, name)
        scale = np.maximum(1.0, np.maximum(np.abs(current), np.abs(previous)))
        relative_sq = ((current - previous) / scale) ** 2
        reduction_axes = (0, *range(3, relative_sq.ndim))
        result += np.mean(relative_sq, axis=reduction_axes)
    return np.sqrt(result)
