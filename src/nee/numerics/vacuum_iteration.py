"""First-order Ricci-coefficient Picard iteration for the numerical backend.

The incoming null second fundamental form, both weighted null lapse
coefficients, and the outgoing shear are stored iteration variables.  The
incoming form is advanced by its null structure equation instead of being
reconstructed by differentiating the metric in ``u``.

The numerical backend's sphere and quadrature modules are imported as numerical
infrastructure only.  The iteration and diagnostics in this file are new.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


from .coordinate_differentiation import high_order_differentiate  # noqa: E402
from .coordinate_quadrature import (  # noqa: E402
    cumulative_polynomial_quadrature,
    midpoint_values,
    stage_value,
)
from .vacuum_state import (  # noqa: E402
    fractional_values,
    GlobalState,
    sphere_broadcast,
)
from .sphere import (  # noqa: E402
    PointSphereGrid,
    connection_difference,
    gaussian_curvature,
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
from .spherical_harmonics import AngularGalerkin  # noqa: E402
from .lgl import CharacteristicLGLMesh  # noqa: E402
from .sdc import solve_sdc  # noqa: E402


Array = np.ndarray
ProjectionTails = dict[str, float]


class MetricSDCFailure(FloatingPointError):
    """Metric SDC failure carrying every completed element diagnostic."""

    def __init__(self, message: str, diagnostics: list[dict[str, object]]):
        super().__init__(message)
        self.diagnostics = diagnostics


def _differentiate_u(
    values: Array,
    u: Array,
    axis: int,
    scalar_coordinates: CharacteristicLGLMesh | None,
) -> Array:
    if scalar_coordinates is None:
        return high_order_differentiate(values, u, axis=axis)
    if not np.allclose(scalar_coordinates.u, u, rtol=0.0, atol=2.0e-14):
        raise ValueError("the characteristic LGL u grid does not match the state")
    return scalar_coordinates.differentiate_u(values, axis=axis)


def _integrate_v(
    values: Array,
    v: Array,
    axis: int,
    scalar_coordinates: CharacteristicLGLMesh | None,
) -> Array:
    if scalar_coordinates is None:
        return cumulative_polynomial_quadrature(values, v, axis=axis)
    if not np.allclose(scalar_coordinates.v, v, rtol=0.0, atol=2.0e-14):
        raise ValueError("the characteristic LGL v grid does not match the state")
    return scalar_coordinates.integrate_v(values, axis=axis)


def _quadratic_stage_value(
    values: Array,
    midpoints: Array,
    index: int,
    alpha: float,
    axis: int,
) -> Array:
    """Interpolate a known u-field inside one nodal interval.

    The endpoints and the existing high-order LGL midpoint value determine a
    quadratic.  At alpha=0, 1/2, 1 this is exactly the RK stage
    interpolation; the extension to arbitrary alpha is needed only when an
    angular-transport CFL condition subdivides the interval.
    """

    value = float(alpha)
    if value < -1.0e-14 or value > 1.0 + 1.0e-14:
        raise ValueError("RK stage fraction lies outside its u interval")
    left = np.take(values, index, axis=axis)
    middle = np.take(midpoints, index, axis=axis)
    right = np.take(values, index + 1, axis=axis)
    weight_left = 2.0 * (value - 0.5) * (value - 1.0)
    weight_middle = -4.0 * value * (value - 1.0)
    weight_right = 2.0 * value * (value - 0.5)
    return (
        weight_left * left
        + weight_middle * middle
        + weight_right * right
    )


def _transport_substeps(
    grid: PointSphereGrid,
    shift: Array,
    midpoint_shift: Array,
    u: Array,
    index: int,
    cfl: float | None,
) -> int:
    """Return a conservative explicit-RK substep count for b.angular-grad.

    The infinity-norm row sums of the two discrete directional derivative
    matrices bound the semidiscrete advection operator.  This is intentionally
    a stability control, not a change to the continuum or Galerkin equation.
    """

    if cfl is None:
        return 1
    if not math.isfinite(cfl) or cfl <= 0.0:
        raise ValueError("transport_cfl must be positive and finite")
    row_first = np.sum(np.abs(grid.derivative_first), axis=1)
    row_second = np.sum(np.abs(grid.derivative_second), axis=1)
    rate = 0.0
    for alpha in (0.0, 0.5, 1.0):
        stage_shift = _quadratic_stage_value(
            shift, midpoint_shift, index, alpha, axis=1
        )
        component_first = np.einsum(
            "ni,n...i->n...", grid.first, stage_shift
        )
        component_second = np.einsum(
            "ni,n...i->n...", grid.second, stage_shift
        )
        local = (
            np.abs(component_first) * row_first[:, None]
            + np.abs(component_second) * row_second[:, None]
        )
        rate = max(rate, float(np.max(local)))
    count = max(
        1,
        int(
            math.ceil(
                abs(float(u[index + 1] - u[index])) * rate / cfl
            )
        ),
    )
    if count > 512:
        raise FloatingPointError(
            f"angular transport requires {count} u substeps in interval "
            f"{index}; the state is outside the configured explicit CFL cone"
        )
    return count


def _minimum_tangent_eigenvalue(
    grid: PointSphereGrid, metric: Array
) -> float:
    """Return the smallest physical two-dimensional metric eigenvalue."""

    local = np.einsum(
        "nia,n...ij,njb->n...ab", grid.frames, metric, grid.frames
    )
    if not np.all(np.isfinite(local)):
        return -math.inf
    return float(np.min(np.linalg.eigvalsh(local)))


def _require_finite(label: str, **values: Array) -> None:
    """Fail at the first named construction stage, not at sweep exit."""

    for name, value in values.items():
        finite = np.isfinite(value)
        if np.all(finite):
            continue
        first = tuple(int(index) for index in np.argwhere(~finite)[0])
        raise FloatingPointError(
            f"{label}: {name} first became nonfinite at index {first}"
        )


def _record_tail(
    family: object,
    value: Array,
    tails: ProjectionTails | None,
    label: str,
) -> None:
    """Record the largest discarded-to-total coefficient ratio."""

    if tails is None:
        return
    coefficients = family.analyze(value)
    if not np.all(np.isfinite(coefficients)):
        raise FloatingPointError(
            f"angular projection '{label}' produced nonfinite coefficients"
        )
    # Scale before forming coefficient energies.  The ratio is homogeneous,
    # while the unscaled sum of squares can overflow before the state itself
    # becomes nonfinite and thereby hide the actual failing equation.
    scale = float(np.max(np.abs(coefficients)))
    if scale == 0.0:
        ratio = 0.0
    else:
        scaled = np.abs(coefficients) / scale
        total = float(np.sum(scaled**2))
        discarded = float(np.sum(scaled[~family.retained] ** 2))
        ratio = math.sqrt(discarded / max(total, 1.0e-300))
    # A label can be visited at many RK stages.  Retain the worst global
    # coefficient-energy ratio, not a pointwise ratio at a nearly zero field.
    tails[label] = max(tails.get(label, 0.0), ratio)


def _project_scalar(
    angular: AngularGalerkin | None,
    value: Array,
    tails: ProjectionTails | None = None,
    label: str = "scalar",
) -> Array:
    if angular is None:
        return value
    _record_tail(angular.scalar, value, tails, label)
    return angular.project_scalar(value)


def _project_vector(
    angular: AngularGalerkin | None,
    value: Array,
    tails: ProjectionTails | None = None,
    label: str = "vector",
) -> Array:
    if angular is None:
        return value
    _record_tail(angular.vector, value, tails, label)
    return angular.project_vector(value)


def _project_sym2(
    angular: AngularGalerkin | None,
    value: Array,
    tails: ProjectionTails | None = None,
    label: str = "sym2",
) -> Array:
    symmetric = 0.5 * (value + np.swapaxes(value, -1, -2))
    if angular is None:
        return symmetric
    _record_tail(angular.sym2, symmetric, tails, label)
    return angular.project_sym2(symmetric)


def _project_tracefree_sym2(
    angular: AngularGalerkin | None,
    value: Array,
    metric: Array,
    inverse: Array,
    tails: ProjectionTails | None = None,
    label: str = "tracefree_sym2",
) -> Array:
    """Apply the algebraic trace constraint in the declared angular space."""

    symmetric = 0.5 * (value + np.swapaxes(value, -1, -2))
    if angular is None:
        return tensor_tracefree(symmetric, metric, inverse)
    _record_tail(angular.sym2, symmetric, tails, label)
    return angular.project_g_tracefree(symmetric, inverse)


@dataclass
class FirstOrderState(GlobalState):
    """First-order variables; every ``weighted_*`` field includes one Omega.

    ``weighted_omegab`` is the full integer Picard coefficient.  The
    half-step coefficient used to construct ``Omega`` is transient and is
    returned only in the Picard context.
    """

    weighted_chib: Array
    weighted_omega: Array
    weighted_omegab: Array


def initial_state(grid: PointSphereGrid, u: Array, v: Array) -> FirstOrderState:
    projector = sphere_broadcast(grid, 2)
    scalar_shape = (grid.count, len(u), len(v))
    metric = np.broadcast_to(
        (-u[None, :, None])[:, :, :, None, None] ** 2 * projector,
        scalar_shape + (3, 3),
    ).copy()
    # At v=0, Omega chib = (1/2) partial_u[(-u)^2 gamma] = u gamma.
    weighted_chib = np.broadcast_to(
        u[None, :, None, None, None] * projector,
        scalar_shape + (3, 3),
    ).copy()
    return FirstOrderState(
        metric=metric,
        omega=np.ones(scalar_shape),
        zeta_up=np.zeros(scalar_shape + (3,)),
        shift=np.zeros(scalar_shape + (3,)),
        q=np.broadcast_to(2.0 / (-u[None, :, None]), scalar_shape).copy(),
        shear=np.zeros_like(metric),
        weighted_chib=weighted_chib,
        weighted_omega=np.zeros(scalar_shape),
        weighted_omegab=np.zeros(scalar_shape),
    )


def _outgoing_seed_weight(u: Array) -> Array:
    """Return the C2-flat quintic weight from H_left to the far u face.

    The normalized ``tau=-log(-u)`` coordinate is one at the outgoing initial
    hypersurface ``u[0]`` and zero at the opposite face ``u[-1]``.  The
    polynomial

    ``w(x) = 6 x^5 - 15 x^4 + 10 x^3``

    has zero first and second derivatives at both endpoints and is exactly
    degree five in the coordinate used by the u solver.  The complete seed
    also contains explicit homogeneity factors, so its resolution is still
    checked independently between LGL nodes rather than inferred from the
    cutoff degree.
    """

    values = np.asarray(u, dtype=float)
    if (
        values.ndim != 1
        or len(values) < 2
        or not np.all(np.isfinite(values))
        or np.any(np.diff(values) <= 0.0)
        or np.any(values >= 0.0)
    ):
        raise ValueError(
            "u must be a finite, strictly increasing negative grid"
        )
    tau = -np.log(-values)
    width = float(tau[-1] - tau[0])
    coordinate = (tau[-1] - tau) / width
    return coordinate**3 * (
        10.0 + coordinate * (-15.0 + 6.0 * coordinate)
    )


def boundary_compatible_initial_state(
    grid: PointSphereGrid,
    u: Array,
    v: Array,
    boundary: dict[str, Array | float],
) -> FirstOrderState:
    """Return a smooth first Picard seed matching both characteristic faces.

    A raw call to :func:`initial_state` is the exact incoming Minkowski
    iterate.  Overwriting only its first u node with the outgoing data makes
    the degree-p u interpolant see an artificial jump.  This routine instead
    extends every state field accepted by :func:`impose_outgoing_boundary`.

    Let ``rho=-u``, ``R=rho/rho_left`` and let ``w`` be the quintic cutoff
    returned by :func:`_outgoing_seed_weight`.  For a field of homogeneity
    ``p`` the extension is

    ``F_seed = F_flat + w R^p (F_H(v) - F_H(0))``.

    The powers are the flat-cone homogeneities of the stored coordinate
    fields: ``g:2``, ``q:-1``, ``Omega:0``, covariant null second fundamental
    forms ``hat(chi), Omega*chib:1``, ``Omega*omega`` and
    ``Omega*omegab:-1``, contravariant ``zeta:-2``, and ``b:-1``.  In
    particular, the metric is a positive scalar times a convex combination
    of its positive corner and outgoing values.  The seed therefore preserves
    positivity whenever the supplied outgoing metric is positive.

    The outgoing and incoming data must agree at the corner: no smooth
    function can match materially contradictory values there.  A mixed
    roundoff-scale gate accepts differences no larger than ``256 eps`` times
    the field scale, then canonicalizes the accepted corner to the exact
    incoming value.  Thus harmless serialization/interpolation roundoff does
    not block a fresh seed, while both characteristic faces remain bitwise
    compatible.  The cutoff is zero at the far u face, so an arbitrary
    outgoing perturbation is not silently prescribed there.
    """

    values_u = np.asarray(u, dtype=float)
    values_v = np.asarray(v, dtype=float)
    weight = _outgoing_seed_weight(values_u)
    if (
        values_v.ndim != 1
        or len(values_v) < 1
        or not np.all(np.isfinite(values_v))
        or abs(float(values_v[0])) > 1.0e-15
        or (len(values_v) > 1 and np.any(np.diff(values_v) <= 0.0))
    ):
        raise ValueError(
            "v must be a finite, strictly increasing grid beginning at zero"
        )
    radius = -values_u
    if np.any(radius <= 0.0):
        raise ValueError("the boundary-compatible seed requires u < 0")
    radius_ratio = radius / float(radius[0])
    state = initial_state(grid, values_u, values_v)

    # This table is deliberately exhaustive with impose_outgoing_boundary.
    # Optional boundary fields use their exact flat values when absent.
    specifications = (
        ("metric", "metric", 2.0, None),
        ("q", "expansion", -1.0, None),
        ("shear", "shear", 1.0, None),
        ("omega", "omega", 0.0, 1.0),
        ("weighted_omega", "weighted_omega", -1.0, 0.0),
        ("weighted_omegab", "weighted_omegab", -1.0, 0.0),
        ("weighted_chib", "weighted_chib", 1.0, None),
        ("zeta_up", "zeta_up", -2.0, 0.0),
        ("shift", "shift", -1.0, 0.0),
    )
    required = {"metric", "expansion", "shear"}
    for state_name, boundary_name, homogeneity, default in specifications:
        current = getattr(state, state_name)
        boundary_shape = (grid.count, len(values_v), *current.shape[3:])
        if boundary_name not in boundary:
            if boundary_name in required:
                raise ValueError(
                    f"outgoing boundary is missing required field "
                    f"{boundary_name!r}"
                )
            if default is None:
                boundary_values = current[:, 0].copy()
            else:
                boundary_values = np.full(
                    boundary_shape, default, dtype=float
                )
        else:
            supplied = np.asarray(boundary[boundary_name], dtype=float)
            try:
                boundary_values = np.broadcast_to(
                    supplied, boundary_shape
                ).copy()
            except ValueError as error:
                raise ValueError(
                    f"outgoing boundary field {boundary_name!r} has shape "
                    f"{supplied.shape}, expected broadcastable to "
                    f"{boundary_shape}"
                ) from error
        if not np.all(np.isfinite(boundary_values)):
            raise ValueError(
                f"outgoing boundary field {boundary_name!r} is nonfinite"
            )
        flat_corner = current[:, 0, 0]
        mismatch = float(
            np.max(np.abs(boundary_values[:, 0] - flat_corner))
        )
        corner_scale = max(
            1.0,
            float(np.max(np.abs(boundary_values))),
            float(np.max(np.abs(flat_corner))),
        )
        corner_tolerance = 256.0 * np.finfo(float).eps * corner_scale
        if mismatch > corner_tolerance:
            raise ValueError(
                f"outgoing field {boundary_name!r} is incompatible with "
                f"the incoming data at the corner; maximum mismatch="
                f"{mismatch:.6g}, roundoff tolerance={corner_tolerance:.6g}"
            )
        boundary_values[:, 0] = flat_corner

        extra_slots = current.ndim - 3
        u_factor = (
            weight * radius_ratio**homogeneity
        ).reshape((1, len(values_u), 1, *(1,) * extra_slots))
        perturbation = (
            boundary_values - boundary_values[:, :1]
        )[:, None]
        extended = current + u_factor * perturbation
        # Avoid relying on subtract/add roundoff for the exact characteristic
        # trace promised by this constructor.
        extended[:, 0] = boundary_values
        extended[:, :, 0] = current[:, :, 0]
        setattr(state, state_name, extended)
    return state


def impose_outgoing_boundary(
    state: FirstOrderState, boundary: dict[str, Array | float]
) -> FirstOrderState:
    state.metric[:, 0] = np.asarray(boundary["metric"])
    state.q[:, 0] = np.asarray(boundary["expansion"])
    state.shear[:, 0] = np.asarray(boundary["shear"])
    state.omega[:, 0] = np.asarray(boundary.get("omega", 1.0))
    state.weighted_omega[:, 0] = np.asarray(
        boundary.get("weighted_omega", 0.0)
    )
    if "weighted_omegab" in boundary:
        state.weighted_omegab[:, 0] = np.asarray(
            boundary["weighted_omegab"]
        )
    if "weighted_chib" in boundary:
        state.weighted_chib[:, 0] = np.asarray(boundary["weighted_chib"])
    if "zeta_up" in boundary:
        state.zeta_up[:, 0] = np.asarray(boundary["zeta_up"])
    if "shift" in boundary:
        state.shift[:, 0] = np.asarray(boundary["shift"])
    return state


def section_geometry(
    grid: PointSphereGrid,
    state: FirstOrderState,
    include_curvature: bool = False,
) -> dict[str, Array]:
    """Geometry using independent Omega*chib, never a u derivative of g."""

    if include_curvature:
        curvature, difference, inverse = gaussian_curvature(grid, state.metric)
    else:
        difference, inverse = connection_difference(grid, state.metric)
        curvature = np.zeros_like(state.q)
    weighted_tr_chib = tensor_trace(state.weighted_chib, inverse)
    weighted_hatchib = tensor_tracefree(
        state.weighted_chib, state.metric, inverse
    )
    tr_chib = weighted_tr_chib / state.omega
    hatchib = weighted_hatchib / state.omega[..., None, None]
    grad_log_omega = scalar_gradient(grid, np.log(state.omega))
    zeta_cov = np.einsum("n...ij,n...j->n...i", state.metric, state.zeta_up)
    eta = zeta_cov + grad_log_omega
    etab = -zeta_cov + grad_log_omega
    return {
        "curvature": curvature,
        "difference": difference,
        "inverse": inverse,
        "weighted_tr_chib": weighted_tr_chib,
        "weighted_hatchib": weighted_hatchib,
        # Compatibility names used by the reusable half-shear solver.
        "tr_chib": tr_chib,
        "hatchib": hatchib,
        "eta": eta,
        "etab": etab,
        "eta_grad_hat": tracefree_symmetric_gradient(
            grid, eta, state.metric, difference, inverse
        ),
        "eta_square_hat": tracefree_square(eta, state.metric, inverse),
    }


def solve_half_shear(
    grid: PointSphereGrid,
    state: FirstOrderState,
    geometry: dict[str, Array],
    boundary: dict[str, Array | float],
    u: Array,
    angular: AngularGalerkin | None = None,
    projection_tails: ProjectionTails | None = None,
    scalar_coordinates: CharacteristicLGLMesh | None = None,
    transport_cfl: float | None = None,
) -> Array:
    """Advance the half-step outgoing shear with one projected RK RHS.

    The formula is the same null-structure equation used in the numerical backend.  In
    Galerkin mode every *complete* tensor right-hand side is projected after
    all products and Lie-derivative terms have been assembled.  The moving
    constraint ``C(g)c=0`` is enforced as a coefficient-space DAE through
    ``C(g)c_dot + C_dot(g)c=0``.  This is different from naively making the
    derivative trace-free, which would omit the ``C_dot(g)c`` term.
    """

    half = np.zeros_like(state.metric)
    boundary_shear = np.asarray(boundary["shear"])
    boundary_inverse = np.asarray(boundary["inverse"])
    transfer = np.matmul(
        np.matmul(boundary_shear, boundary_inverse), state.metric[:, 0]
    )
    transfer = 0.5 * (transfer + np.swapaxes(transfer, -1, -2))
    half[:, 0] = _project_tracefree_sym2(
        angular,
        transfer,
        state.metric[:, 0],
        geometry["inverse"][:, 0],
        projection_tails,
        "half_shear_boundary",
    )
    tr_chi = state.omega * state.q
    source = state.omega[..., None, None] ** 2 * (
        geometry["eta_grad_hat"]
        + geometry["eta_square_hat"]
        - 0.5 * tr_chi[..., None, None] * geometry["hatchib"]
    )
    weighted_trace = state.omega * geometry["tr_chib"]
    weighted_hatchib = (
        state.omega[..., None, None] * geometry["hatchib"]
    )
    mixed_hatchib = np.matmul(weighted_hatchib, geometry["inverse"])
    metric_u = _differentiate_u(
        state.metric, u, axis=1, scalar_coordinates=scalar_coordinates
    )
    midpoint_fields = {
        name: midpoint_values(value, u, axis=1)
        for name, value in {
            "shift": state.shift,
            "trace": weighted_trace,
            "mixed": mixed_hatchib,
            "source": source,
            "metric": state.metric,
            "metric_u": metric_u,
        }.items()
    }
    for i in range(len(u) - 1):
        step = float(u[i + 1] - u[i])
        substeps = _transport_substeps(
            grid,
            state.shift,
            midpoint_fields["shift"],
            u,
            i,
            transport_cfl,
        )

        def rhs(value: Array, alpha: float) -> Array:
            shift = _quadratic_stage_value(
                state.shift, midpoint_fields["shift"], i, alpha, axis=1
            )
            trace = _quadratic_stage_value(
                weighted_trace,
                midpoint_fields["trace"],
                i,
                alpha,
                axis=1,
            )
            mixed = _quadratic_stage_value(
                mixed_hatchib,
                midpoint_fields["mixed"],
                i,
                alpha,
                axis=1,
            )
            stage_source = _quadratic_stage_value(
                source, midpoint_fields["source"], i, alpha, axis=1
            )
            stage_tensor = value
            stage_metric = None
            stage_inverse = None
            stage_inverse_u = None
            if angular is not None:
                stage_metric = _quadratic_stage_value(
                    state.metric,
                    midpoint_fields["metric"],
                    i,
                    alpha,
                    axis=1,
                )
                stage_metric_u = _quadratic_stage_value(
                    metric_u,
                    midpoint_fields["metric_u"],
                    i,
                    alpha,
                    axis=1,
                )
                stage_inverse = tangent_inverse(grid, stage_metric)
                stage_inverse_u = -np.matmul(
                    np.matmul(stage_inverse, stage_metric_u), stage_inverse
                )
                stage_tensor = angular.project_g_tracefree(
                    value, stage_inverse
                )
            complete = (
                0.5 * trace[..., None, None] * stage_tensor
                + np.matmul(mixed, stage_tensor)
                + np.matmul(stage_tensor, np.swapaxes(mixed, -1, -2))
                + stage_source
                - lie_covariant_tensor(grid, shift, stage_tensor)
            )
            if angular is not None:
                _record_tail(
                    angular.sym2,
                    complete,
                    projection_tails,
                    "half_shear_rhs",
                )
                unconstrained = angular.project_sym2(complete)
                constrained = angular.project_g_tracefree_derivative(
                    stage_tensor,
                    complete,
                    stage_inverse,
                    stage_inverse_u,
                )
                numerator = float(
                    np.linalg.norm(constrained - unconstrained)
                )
                denominator = max(float(np.linalg.norm(unconstrained)), 1.0e-30)
                if projection_tails is not None:
                    projection_tails["half_shear_constraint_correction"] = max(
                        projection_tails.get(
                            "half_shear_constraint_correction", 0.0
                        ),
                        numerator / denominator,
                    )
                return constrained
            return _project_sym2(
                None,
                complete,
                projection_tails,
                "half_shear_rhs",
            )

        candidate = half[:, i]
        local_step = step / substeps
        for substep in range(substeps):
            alpha_left = substep / substeps
            alpha_middle = (substep + 0.5) / substeps
            alpha_right = (substep + 1.0) / substeps
            k1 = rhs(candidate, alpha_left)
            k2 = rhs(
                candidate + 0.5 * local_step * k1, alpha_middle
            )
            k3 = rhs(
                candidate + 0.5 * local_step * k2, alpha_middle
            )
            k4 = rhs(candidate + local_step * k3, alpha_right)
            candidate = candidate + local_step * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            ) / 6.0
        if angular is None:
            half[:, i + 1] = candidate
        else:
            half[:, i + 1] = angular.project_g_tracefree(
                candidate, geometry["inverse"][:, i + 1]
            )
    if angular is None:
        return tensor_tracefree(half, state.metric, geometry["inverse"])
    return half


def solve_log_omega(
    grid: PointSphereGrid,
    state: FirstOrderState,
    weighted_omegab: Array,
    u: Array,
    initial_log_omega: Array | None = None,
    angular: AngularGalerkin | None = None,
    projection_tails: ProjectionTails | None = None,
    transport_cfl: float | None = None,
) -> Array:
    """Solve D3 log(Omega)=-2(Omega omegab) in scalar Galerkin space."""

    log_omega = np.zeros_like(state.omega)
    if initial_log_omega is not None:
        boundary_log_omega = np.broadcast_to(
            np.asarray(initial_log_omega), log_omega[:, 0].shape
        )
        log_omega[:, 0] = _project_scalar(
            angular,
            boundary_log_omega,
            projection_tails,
            "log_omega_boundary",
        )
    midpoint_shift = midpoint_values(state.shift, u, axis=1)
    midpoint_source = midpoint_values(weighted_omegab, u, axis=1)
    for i in range(len(u) - 1):
        step = float(u[i + 1] - u[i])
        substeps = _transport_substeps(
            grid,
            state.shift,
            midpoint_shift,
            u,
            i,
            transport_cfl,
        )

        def rhs(value: Array, alpha: float) -> Array:
            shift = _quadratic_stage_value(
                state.shift, midpoint_shift, i, alpha, axis=1
            )
            source = _quadratic_stage_value(
                weighted_omegab, midpoint_source, i, alpha, axis=1
            )
            complete = -np.einsum(
                "n...i,n...i->n...", shift, scalar_gradient(grid, value)
            ) - 2.0 * source
            return _project_scalar(
                angular,
                complete,
                projection_tails,
                "log_omega_rhs",
            )

        current = log_omega[:, i]
        local_step = step / substeps
        for substep in range(substeps):
            alpha_left = substep / substeps
            alpha_middle = (substep + 0.5) / substeps
            alpha_right = (substep + 1.0) / substeps
            k1 = rhs(current, alpha_left)
            k2 = rhs(current + 0.5 * local_step * k1, alpha_middle)
            k3 = rhs(current + 0.5 * local_step * k2, alpha_middle)
            k4 = rhs(current + local_step * k3, alpha_right)
            current = current + local_step * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            ) / 6.0
        log_omega[:, i + 1] = current
    # Omega is a nonlinear derived field; log(Omega), not Omega, is the
    # retained scalar.  Re-projecting exp(log(Omega)) would change the lapse
    # equation and can destroy positivity.
    return np.exp(log_omega)


def solve_metric_and_expansion(
    grid: PointSphereGrid,
    state: FirstOrderState,
    half: Array,
    new_omega: Array,
    u: Array,
    v: Array,
    substeps: int = 1,
    angular: AngularGalerkin | None = None,
    projection_tails: ProjectionTails | None = None,
    scalar_coordinates: CharacteristicLGLMesh | None = None,
    metric_parameterization: str = "direct",
    initial_metric: Array | None = None,
    initial_q: Array | None = None,
    expansion_excision_threshold: float | None = None,
    excision_diagnostics: dict[str, object] | None = None,
) -> tuple[Array, Array, Array]:
    """Solve the coupled (g, Omega^-1 tr chi, shear) stage system.

    The transferred shear is formed from the current RK metric at every
    stage.  Pointwise trace removal is part of that nonlinear algebra; only
    afterwards is each complete metric/scalar RHS projected to the retained
    angular band.
    """

    if scalar_coordinates is not None and not np.allclose(
        scalar_coordinates.v, v, rtol=0.0, atol=2.0e-14
    ):
        raise ValueError("the characteristic LGL v grid does not match the state")
    if metric_parameterization not in {"direct", "cholesky"}:
        raise ValueError(
            "metric_parameterization must be 'direct' or 'cholesky'"
        )
    if expansion_excision_threshold is None:
        threshold = None
    else:
        threshold = float(expansion_excision_threshold)
        if not math.isfinite(threshold) or threshold >= 0.0:
            raise ValueError(
                "expansion_excision_threshold must be finite and negative"
            )

    metric = np.zeros_like(state.metric)
    q = np.zeros_like(state.q)
    active_u = np.ones(len(u), dtype=bool)
    event_v = np.full(len(u), np.nan)
    event_angular_index = np.full(len(u), -1, dtype=int)
    projector = sphere_broadcast(grid, 1)
    if initial_metric is None:
        metric[:, :, 0] = (-u[None, :])[:, :, None, None] ** 2 * projector
    else:
        metric[:, :, 0] = np.broadcast_to(
            np.asarray(initial_metric), metric[:, :, 0].shape
        )
    if initial_q is None:
        q[:, :, 0] = 2.0 / (-u[None, :])
    else:
        q[:, :, 0] = np.broadcast_to(
            np.asarray(initial_q), q[:, :, 0].shape
        )
    factor = None
    if metric_parameterization == "cholesky":
        local_initial = np.einsum(
            "nia,n...ij,njb->n...ab",
            grid.frames,
            metric[:, :, 0],
            grid.frames,
        )
        factor = np.zeros(
            (*q.shape, 2, 2), dtype=metric.dtype
        )
        factor[:, :, 0] = np.linalg.cholesky(local_initial)
    old_inverse = tangent_inverse(grid, state.metric)
    fields = {
        "log_omega": np.log(new_omega),
        "half": half,
        "old_metric": state.metric,
    }
    fractions = sorted(
        {
            stage / (2 * substeps)
            for stage in range(1, 2 * substeps)
        }
    )
    cell_interpolator = (
        getattr(scalar_coordinates, "fractional_cell_value", None)
        if scalar_coordinates is not None
        else None
    )
    interpolated = (
        {}
        if cell_interpolator is not None
        else {
            name: {
                fraction: fractional_values(
                    value, v, axis=2, fraction=fraction
                )
                for fraction in fractions
                if fraction not in {0.0, 1.0}
            }
            for name, value in fields.items()
        }
    )

    def external(name: str, interval: int, fraction: float) -> Array:
        if abs(fraction) < 1.0e-14:
            return np.take(fields[name], interval, axis=2)
        if abs(fraction - 1.0) < 1.0e-14:
            return np.take(fields[name], interval + 1, axis=2)
        if cell_interpolator is not None:
            return cell_interpolator(
                fields[name],
                axis=2,
                interval=interval,
                fraction=fraction,
            )
        return np.take(interpolated[name][fraction], interval, axis=2)

    def transferred(
        stage_half: Array, stage_inverse: Array, value_metric: Array
    ) -> Array:
        raw = np.matmul(np.matmul(stage_half, stage_inverse), value_metric)
        raw = 0.5 * (raw + np.swapaxes(raw, -1, -2))
        value_inverse = tangent_inverse(grid, value_metric)
        return tensor_tracefree(raw, value_metric, value_inverse)

    def ambient_metric(value_factor: Array) -> Array:
        local_metric = np.matmul(
            value_factor, np.swapaxes(value_factor, -1, -2)
        )
        return np.einsum(
            "nia,n...ab,njb->n...ij",
            grid.frames,
            local_metric,
            grid.frames,
        )

    for j in range(len(v) - 1):
        full_step = float(v[j + 1] - v[j])
        step = full_step / substeps

        def stage_label(fraction: float) -> str:
            value_v = float(v[j] + fraction * full_step)
            return (
                "metric/Raychaudhuri stage "
                f"v={value_v:.12g} in cell [{float(v[j]):.12g}, "
                f"{float(v[j + 1]):.12g}]"
            )

        def record_crossings(
            value_q: Array, fraction: float
        ) -> None:
            """Freeze complete u-slices after the expansion event.

            The user-prescribed excision removes a full sphere once any
            angular point reaches the threshold.  The accepted event is
            recorded only at RK substep endpoints; a smaller substep count
            therefore controls the event-location error without introducing
            interpolation through an already excised region.
            """

            if threshold is None:
                return
            endpoint_omega = np.exp(
                external("log_omega", j, fraction)
            )
            endpoint_expansion = endpoint_omega * value_q
            row_minimum = np.min(endpoint_expansion, axis=0)
            crossed = active_u & (row_minimum < threshold)
            value_v = v[j] + fraction * full_step
            for index_u in np.flatnonzero(crossed):
                event_v[index_u] = value_v
                event_angular_index[index_u] = int(
                    np.argmin(endpoint_expansion[:, index_u])
                )
            active_u[crossed] = False

        def rhs(
            value_q: Array, value_metric: Array, fraction: float
        ) -> tuple[Array, Array]:
            # Interpolate the retained primitive log(Omega), then exponentiate
            # at the stage.  Interpolating exp(log(Omega)) would define a
            # different semidiscrete lapse equation.
            label = stage_label(fraction)
            _require_finite(label, q=value_q, metric=value_metric)
            minimum_eigenvalue = _minimum_tangent_eigenvalue(
                grid, value_metric
            )
            if minimum_eigenvalue <= 0.0:
                raise FloatingPointError(
                    f"{label}: metric left the positive cone; "
                    f"min(g)={minimum_eigenvalue:.12g}, "
                    f"max|q|={float(np.max(np.abs(value_q))):.12g}"
                )
            with np.errstate(over="raise", invalid="raise"):
                omega = np.exp(external("log_omega", j, fraction))
            _require_finite(label, omega=omega)
            if float(np.min(omega)) <= 0.0:
                raise FloatingPointError(
                    f"{label}: lapse left the positive cone; "
                    f"min(Omega)={float(np.min(omega)):.12g}"
                )
            stage_half = external("half", j, fraction)
            stage_old_metric = external("old_metric", j, fraction)
            _require_finite(
                label,
                half_shear=stage_half,
                previous_metric=stage_old_metric,
            )
            old_minimum_eigenvalue = _minimum_tangent_eigenvalue(
                grid, stage_old_metric
            )
            if old_minimum_eigenvalue <= 0.0:
                raise FloatingPointError(
                    f"{label}: interpolated previous metric left the "
                    f"positive cone; min(g_old)={old_minimum_eigenvalue:.12g}"
                )
            stage_inverse = tangent_inverse(grid, stage_old_metric)
            stage_shear = transferred(
                stage_half, stage_inverse, value_metric
            )
            inverse = tangent_inverse(grid, value_metric)
            norm_sq = tensor_norm_sq(stage_shear, inverse)
            try:
                with np.errstate(
                    over="raise", invalid="raise", divide="raise"
                ):
                    q_rhs = (
                        -0.5 * omega**2 * value_q**2
                        - norm_sq / omega**2
                    )
                    metric_rhs = (
                        omega[..., None, None] ** 2
                        * value_q[..., None, None]
                        * value_metric
                        + 2.0 * stage_shear
                    )
            except FloatingPointError as error:
                raise FloatingPointError(
                    f"{label}: coupled metric/Raychaudhuri RHS overflow; "
                    f"min/max Omega=({float(np.min(omega)):.12g},"
                    f"{float(np.max(omega)):.12g}), "
                    f"max|q|={float(np.max(np.abs(value_q))):.12g}, "
                    f"max|g|={float(np.max(np.abs(value_metric))):.12g}, "
                    f"max|hat_chi|^2={float(np.max(norm_sq)):.12g}"
                ) from error
            _require_finite(
                label,
                transferred_shear=stage_shear,
                raychaudhuri_rhs=q_rhs,
                metric_rhs=metric_rhs,
            )
            if threshold is not None and np.any(~active_u):
                # No angular value on an excised S_{u,v} is advanced.  The
                # mask is constant over the complete sphere, so this zeroing
                # also commutes with the angular Galerkin projection.
                q_rhs[:, ~active_u] = 0.0
                metric_rhs[:, ~active_u] = 0.0
            return (
                _project_scalar(
                    angular,
                    q_rhs,
                    projection_tails,
                    "raychaudhuri_rhs",
                ),
                _project_sym2(
                    angular,
                    metric_rhs,
                    projection_tails,
                    "metric_rhs",
                ),
            )

        current_q = q[:, :, j]
        if metric_parameterization == "direct":
            current_metric = metric[:, :, j]
            for substep in range(substeps):
                fraction = substep / substeps
                half_fraction = (substep + 0.5) / substeps
                end_fraction = (substep + 1.0) / substeps
                k1_q, k1_metric = rhs(current_q, current_metric, fraction)
                k2_q, k2_metric = rhs(
                    current_q + 0.5 * step * k1_q,
                    current_metric + 0.5 * step * k1_metric,
                    half_fraction,
                )
                k3_q, k3_metric = rhs(
                    current_q + 0.5 * step * k2_q,
                    current_metric + 0.5 * step * k2_metric,
                    half_fraction,
                )
                k4_q, k4_metric = rhs(
                    current_q + step * k3_q,
                    current_metric + step * k3_metric,
                    end_fraction,
                )
                current_q = current_q + step * (
                    k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q
                ) / 6.0
                current_metric = current_metric + step * (
                    k1_metric
                    + 2.0 * k2_metric
                    + 2.0 * k3_metric
                    + k4_metric
                ) / 6.0
                endpoint_fraction = (substep + 1.0) / substeps
                label = stage_label(endpoint_fraction)
                _require_finite(label, q=current_q, metric=current_metric)
                minimum_eigenvalue = _minimum_tangent_eigenvalue(
                    grid, current_metric
                )
                if minimum_eigenvalue <= 0.0:
                    raise FloatingPointError(
                        f"{label}: accepted metric update left the positive "
                        f"cone; min(g)={minimum_eigenvalue:.12g}, "
                        f"max|q|={float(np.max(np.abs(current_q))):.12g}"
                    )
                record_crossings(
                    current_q, (substep + 1.0) / substeps
                )
        else:
            assert factor is not None
            current_factor = factor[:, :, j]

            def factor_rhs(
                value_q: Array, value_factor: Array, fraction: float
            ) -> tuple[Array, Array]:
                value_metric = ambient_metric(value_factor)
                q_rhs, metric_rhs = rhs(value_q, value_metric, fraction)
                local_rhs = np.einsum(
                    "nia,n...ij,njb->n...ab",
                    grid.frames,
                    metric_rhs,
                    grid.frames,
                )
                try:
                    inverse_transpose = np.swapaxes(
                        np.linalg.inv(value_factor), -1, -2
                    )
                except np.linalg.LinAlgError as error:
                    raise FloatingPointError(
                        f"{stage_label(fraction)}: metric factor became singular"
                    ) from error
                value_factor_rhs = 0.5 * np.matmul(
                    local_rhs, inverse_transpose
                )
                _require_finite(
                    stage_label(fraction),
                    q_rhs=q_rhs,
                    metric_factor_rhs=value_factor_rhs,
                )
                return q_rhs, value_factor_rhs

            for substep in range(substeps):
                fraction = substep / substeps
                half_fraction = (substep + 0.5) / substeps
                end_fraction = (substep + 1.0) / substeps
                k1_q, k1_factor = factor_rhs(
                    current_q, current_factor, fraction
                )
                k2_q, k2_factor = factor_rhs(
                    current_q + 0.5 * step * k1_q,
                    current_factor + 0.5 * step * k1_factor,
                    half_fraction,
                )
                k3_q, k3_factor = factor_rhs(
                    current_q + 0.5 * step * k2_q,
                    current_factor + 0.5 * step * k2_factor,
                    half_fraction,
                )
                k4_q, k4_factor = factor_rhs(
                    current_q + step * k3_q,
                    current_factor + step * k3_factor,
                    end_fraction,
                )
                current_q = current_q + step * (
                    k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q
                ) / 6.0
                current_factor = current_factor + step * (
                    k1_factor
                    + 2.0 * k2_factor
                    + 2.0 * k3_factor
                    + k4_factor
                ) / 6.0
                current_metric = ambient_metric(current_factor)
                label = stage_label((substep + 1.0) / substeps)
                _require_finite(
                    label,
                    q=current_q,
                    metric_factor=current_factor,
                    metric=current_metric,
                )
                minimum_eigenvalue = _minimum_tangent_eigenvalue(
                    grid, current_metric
                )
                if minimum_eigenvalue <= 1.0e-14:
                    raise FloatingPointError(
                        f"{label}: Cholesky metric approached singularity; "
                        f"min(g)={minimum_eigenvalue:.12g}, "
                        f"max|q|={float(np.max(np.abs(current_q))):.12g}"
                    )
                record_crossings(
                    current_q, (substep + 1.0) / substeps
                )
            factor[:, :, j + 1] = current_factor
            current_metric = ambient_metric(current_factor)
        q[:, :, j + 1] = current_q
        metric[:, :, j + 1] = current_metric

    transfer = np.matmul(np.matmul(half, old_inverse), metric)
    transfer = 0.5 * (transfer + np.swapaxes(transfer, -1, -2))
    inverse = tangent_inverse(grid, metric)
    transfer = tensor_tracefree(transfer, metric, inverse)
    # Like Omega=exp(log Omega), the full shear is an algebraically derived
    # nonlinear field.  Keep the exact pointwise transfer/trace constraint on
    # the oversampled work grid; it is not an independent hidden nodal state.
    # Every evolution RHS that consumes it is projected as a complete tensor
    # or scalar expression.
    if angular is not None:
        _record_tail(
            angular.sym2,
            transfer,
            projection_tails,
            "transferred_shear",
        )
    shear = transfer
    if excision_diagnostics is not None:
        excision_diagnostics.clear()
        excision_diagnostics.update(
            {
                "threshold": threshold,
                "event_v": event_v,
                "event_angular_index": event_angular_index,
                "active_u_at_terminal_v": active_u,
            }
        )
    return metric, q, shear


def solve_metric_and_expansion_sdc(
    grid: PointSphereGrid,
    state: FirstOrderState,
    half: Array,
    new_omega: Array,
    u: Array,
    v: Array,
    scalar_coordinates: CharacteristicLGLMesh,
    angular: AngularGalerkin | None = None,
    projection_tails: ProjectionTails | None = None,
    tolerance: float = 1.0e-10,
    overgrid_tolerance: float = np.inf,
    maximum_corrections: int = 12,
) -> tuple[Array, Array, Array, list[dict[str, float | int | bool]]]:
    """Solve the coupled v system by panelwise LGL collocation/SDC.

    The section metric is represented by a free tangent factor ``g=L L^T``.
    This is an identity at the continuum level and keeps every SDC trial in
    the SPD cone.  All known Picard fields are interpolated by the same panel
    polynomial used by the collocation integral, rather than by an unrelated
    local six-point stencil.
    """

    if not np.allclose(scalar_coordinates.u, u, rtol=0.0, atol=2.0e-14) or not np.allclose(
        scalar_coordinates.v, v, rtol=0.0, atol=2.0e-14
    ):
        raise ValueError("the characteristic LGL grid does not match the SDC state")

    metric = np.zeros_like(state.metric)
    q = np.zeros_like(state.q)
    projector = sphere_broadcast(grid, 1)
    metric[:, :, 0] = (-u[None, :])[:, :, None, None] ** 2 * projector
    q[:, :, 0] = 2.0 / (-u[None, :])
    local_initial = np.einsum(
        "nia,n...ij,njb->n...ab",
        grid.frames,
        metric[:, :, 0],
        grid.frames,
    )
    factor = np.zeros((*q.shape, 2, 2), dtype=metric.dtype)
    factor[:, :, 0] = np.linalg.cholesky(local_initial)
    old_inverse = tangent_inverse(grid, state.metric)
    log_omega = np.log(new_omega)
    diagnostics: list[dict[str, float | int | bool]] = []

    def ambient_metric(value_factor: Array) -> Array:
        local_metric = np.matmul(
            value_factor, np.swapaxes(value_factor, -1, -2)
        )
        return np.einsum(
            "nia,n...ab,njb->n...ij",
            grid.frames,
            local_metric,
            grid.frames,
        )

    for element, (segment, index) in enumerate(
        zip(scalar_coordinates.s.segments, scalar_coordinates.s.indices, strict=True)
    ):
        local_log_omega = np.take(log_omega, index, axis=2)
        local_half = np.take(half, index, axis=2)
        local_old_metric = np.take(state.metric, index, axis=2)

        def interpolate(field: Array, value_s: float) -> Array:
            weights = segment.interpolation_matrix(
                np.array([value_s])
            )[0]
            moved = np.moveaxis(field, 2, 0)
            return np.tensordot(weights, moved, axes=(0, 0))

        def unpack(value: Array) -> tuple[Array, Array]:
            value_q = value[..., 0]
            value_factor = value[..., 1:].reshape((*value_q.shape, 2, 2))
            return value_q, value_factor

        def rhs(value_s: float, packed: Array) -> Array:
            value_q, value_factor = unpack(packed)
            label = (
                f"metric/Raychaudhuri SDC element {element + 1} "
                f"at s={value_s:.12g}, v={scalar_coordinates.v1 * value_s**2:.12g}"
            )
            _require_finite(
                label, q=value_q, metric_factor=value_factor
            )
            value_metric = ambient_metric(value_factor)
            _require_finite(label, metric=value_metric)
            minimum_eigenvalue = _minimum_tangent_eigenvalue(
                grid, value_metric
            )
            if minimum_eigenvalue <= 1.0e-14:
                raise FloatingPointError(
                    f"{label}: metric factor approached singularity; "
                    f"min(g)={minimum_eigenvalue:.12g}, "
                    f"max|q|={float(np.max(np.abs(value_q))):.12g}"
                )

            omega = np.exp(interpolate(local_log_omega, value_s))
            stage_half = interpolate(local_half, value_s)
            stage_old_metric = interpolate(local_old_metric, value_s)
            _require_finite(
                label,
                omega=omega,
                half_shear=stage_half,
                previous_metric=stage_old_metric,
            )
            old_minimum = _minimum_tangent_eigenvalue(
                grid, stage_old_metric
            )
            if old_minimum <= 0.0:
                raise FloatingPointError(
                    f"{label}: interpolated previous metric is not SPD; "
                    f"min(g_old)={old_minimum:.12g}"
                )
            stage_inverse = tangent_inverse(grid, stage_old_metric)
            raw_shear = np.matmul(
                np.matmul(stage_half, stage_inverse), value_metric
            )
            raw_shear = 0.5 * (
                raw_shear + np.swapaxes(raw_shear, -1, -2)
            )
            inverse = tangent_inverse(grid, value_metric)
            stage_shear = tensor_tracefree(
                raw_shear, value_metric, inverse
            )
            norm_sq = tensor_norm_sq(stage_shear, inverse)
            q_rhs = -0.5 * omega**2 * value_q**2 - norm_sq / omega**2
            metric_rhs = (
                omega[..., None, None] ** 2
                * value_q[..., None, None]
                * value_metric
                + 2.0 * stage_shear
            )
            q_rhs = _project_scalar(
                angular,
                q_rhs,
                projection_tails,
                "raychaudhuri_sdc_rhs",
            )
            metric_rhs = _project_sym2(
                angular,
                metric_rhs,
                projection_tails,
                "metric_sdc_rhs",
            )
            local_metric_rhs = np.einsum(
                "nia,n...ij,njb->n...ab",
                grid.frames,
                metric_rhs,
                grid.frames,
            )
            try:
                inverse_transpose = np.swapaxes(
                    np.linalg.inv(value_factor), -1, -2
                )
            except np.linalg.LinAlgError as error:
                raise FloatingPointError(
                    f"{label}: metric factor became singular"
                ) from error
            factor_rhs = 0.5 * np.matmul(
                local_metric_rhs, inverse_transpose
            )
            jacobian = 2.0 * scalar_coordinates.v1 * value_s
            packed_rhs = np.concatenate(
                (
                    q_rhs[..., None],
                    factor_rhs.reshape((*value_q.shape, 4)),
                ),
                axis=-1,
            )
            _require_finite(label, packed_rhs=packed_rhs)
            return jacobian * packed_rhs

        initial = np.concatenate(
            (
                q[:, :, index[0], None],
                factor[:, :, index[0]].reshape(
                    (*q[:, :, index[0]].shape, 4)
                ),
            ),
            axis=-1,
        )
        try:
            result = solve_sdc(
                segment,
                initial,
                rhs,
                tolerance=tolerance,
                overgrid_tolerance=overgrid_tolerance,
                maximum_corrections=maximum_corrections,
            )
        except (FloatingPointError, np.linalg.LinAlgError) as error:
            diagnostics.append(
                {
                    "element": element + 1,
                    "s_left": float(segment.left),
                    "s_right": float(segment.right),
                    "v_left": float(scalar_coordinates.v1 * segment.left**2),
                    "v_right": float(scalar_coordinates.v1 * segment.right**2),
                    "status": "failed",
                    "reason": str(error),
                }
            )
            raise MetricSDCFailure(str(error), diagnostics.copy()) from error
        diagnostics.append(
            {
                "element": element + 1,
                "s_left": float(segment.left),
                "s_right": float(segment.right),
                "v_left": float(scalar_coordinates.v1 * segment.left**2),
                "v_right": float(scalar_coordinates.v1 * segment.right**2),
                "corrections": result.corrections,
                "converged": result.converged,
                "resolved": result.resolved,
                "accepted": result.accepted,
                "status": "completed",
                "collocation_defect": result.collocation_defect,
                "overgrid_defect": result.overgrid_defect,
            }
        )
        if not result.converged:
            raise MetricSDCFailure(
                "metric/Raychaudhuri SDC did not converge on "
                f"v=[{scalar_coordinates.v1 * segment.left**2:.12g}, "
                f"{scalar_coordinates.v1 * segment.right**2:.12g}]: "
                f"collocation defect={result.collocation_defect:.6g}, "
                f"overgrid defect={result.overgrid_defect:.6g}",
                diagnostics.copy(),
            )
        local_q = result.values[..., 0]
        local_factor = result.values[..., 1:].reshape(
            (len(segment.nodes), *q[:, :, 0].shape, 2, 2)
        )
        q[:, :, index] = np.moveaxis(local_q, 0, 2)
        factor[:, :, index] = np.moveaxis(local_factor, 0, 2)
        metric[:, :, index] = ambient_metric(factor[:, :, index])

    underresolved = [item for item in diagnostics if not item["resolved"]]
    if underresolved:
        worst = max(
            underresolved, key=lambda item: float(item["overgrid_defect"])
        )
        raise MetricSDCFailure(
            f"{len(underresolved)} metric/Raychaudhuri SDC elements are "
            "underresolved; worst element "
            f"{worst['element']} on v=[{worst['v_left']:.12g}, "
            f"{worst['v_right']:.12g}] has independent overgrid defect "
            f"{worst['overgrid_defect']:.6g} > {overgrid_tolerance:.6g}",
            diagnostics.copy(),
        )

    transfer = np.matmul(np.matmul(half, old_inverse), metric)
    transfer = 0.5 * (transfer + np.swapaxes(transfer, -1, -2))
    inverse = tangent_inverse(grid, metric)
    shear = tensor_tracefree(transfer, metric, inverse)
    if angular is not None:
        _record_tail(
            angular.sym2,
            shear,
            projection_tails,
            "transferred_shear_sdc",
        )
    return metric, q, shear, diagnostics


def solve_weighted_omega(
    grid: PointSphereGrid,
    old_state: FirstOrderState,
    source: Array,
    new_omega: Array,
    u: Array,
    initial_value: Array | None = None,
    angular: AngularGalerkin | None = None,
    projection_tails: ProjectionTails | None = None,
    transport_cfl: float | None = None,
) -> Array:
    """Solve the attachment's D3(Omega omega) equation along u."""

    value = np.zeros_like(old_state.weighted_omega)
    if initial_value is not None:
        boundary_value = np.broadcast_to(
            np.asarray(initial_value), value[:, 0].shape
        )
        value[:, 0] = _project_scalar(
            angular,
            boundary_value,
            projection_tails,
            "weighted_omega_boundary",
        )
    grad_log_new_omega = scalar_gradient(grid, np.log(new_omega))
    rhs_source = source - 2.0 * old_state.omega**2 * np.einsum(
        "n...i,n...i->n...", old_state.zeta_up, grad_log_new_omega
    )
    midpoint_shift = midpoint_values(old_state.shift, u, axis=1)
    midpoint_source = midpoint_values(rhs_source, u, axis=1)
    for i in range(len(u) - 1):
        step = float(u[i + 1] - u[i])
        substeps = _transport_substeps(
            grid,
            old_state.shift,
            midpoint_shift,
            u,
            i,
            transport_cfl,
        )

        def rhs(stage_value_scalar: Array, alpha: float) -> Array:
            shift = _quadratic_stage_value(
                old_state.shift, midpoint_shift, i, alpha, axis=1
            )
            forcing = _quadratic_stage_value(
                rhs_source, midpoint_source, i, alpha, axis=1
            )
            complete = forcing - np.einsum(
                "n...i,n...i->n...",
                shift,
                scalar_gradient(grid, stage_value_scalar),
            )
            return _project_scalar(
                angular,
                complete,
                projection_tails,
                "weighted_omega_rhs",
            )

        current = value[:, i]
        local_step = step / substeps
        for substep in range(substeps):
            alpha_left = substep / substeps
            alpha_middle = (substep + 0.5) / substeps
            alpha_right = (substep + 1.0) / substeps
            k1 = rhs(current, alpha_left)
            k2 = rhs(current + 0.5 * local_step * k1, alpha_middle)
            k3 = rhs(current + 0.5 * local_step * k2, alpha_middle)
            k4 = rhs(current + local_step * k3, alpha_right)
            current = current + local_step * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            ) / 6.0
        value[:, i + 1] = current
    return value


def complete_weighted_omegab(
    grid: PointSphereGrid,
    old_state: FirstOrderState,
    weighted_omegab_half: Array,
    omegab_half_source: Array,
    new_omega: Array,
    new_weighted_omega: Array,
    new_zeta_up: Array,
    new_shift: Array,
    angular: AngularGalerkin | None = None,
    projection_tails: ProjectionTails | None = None,
) -> tuple[Array, Array]:
    """Return the full integer ``Omega*omegab`` and its v derivative.

    The half iterate solves ``(partial_u+b^i)log(Omega)=-2 B_half``.
    The geometric coefficient uses ``b^(i+1)`` instead, so

    ``B_full = B_half - .5 (b_new-b_old).grad(log(Omega_new))``.

    The returned derivative applies the product rule using the two shift
    construction sources and ``partial_v log(Omega)=-2 Omega*omega``.
    """

    shift_difference = new_shift - old_state.shift
    grad_log_new_omega = scalar_gradient(grid, np.log(new_omega))
    weighted_omegab_full = weighted_omegab_half - 0.5 * np.einsum(
        "n...i,n...i->n...", shift_difference, grad_log_new_omega
    )
    new_shift_source = _project_vector(
        angular,
        -4.0 * new_omega[..., None] ** 2 * new_zeta_up,
        projection_tails,
        "weighted_omegab_new_shift_source",
    )
    old_shift_source = _project_vector(
        angular,
        -4.0 * old_state.omega[..., None] ** 2 * old_state.zeta_up,
        projection_tails,
        "weighted_omegab_old_shift_source",
    )
    shift_v_difference = new_shift_source - old_shift_source
    omegab_full_source = omegab_half_source - 0.5 * (
        np.einsum(
            "n...i,n...i->n...",
            shift_v_difference,
            grad_log_new_omega,
        )
        - 2.0
        * np.einsum(
            "n...i,n...i->n...",
            shift_difference,
            scalar_gradient(grid, new_weighted_omega),
        )
    )
    return (
        _project_scalar(
            angular,
            weighted_omegab_full,
            projection_tails,
            "weighted_omegab_full",
        ),
        _project_scalar(
            angular,
            omegab_full_source,
            projection_tails,
            "weighted_omegab_full_source",
        ),
    )


def solve_weighted_chib(
    grid: PointSphereGrid,
    metric: Array,
    omega: Array,
    zeta_up: Array,
    shift: Array,
    q: Array,
    shear: Array,
    u: Array,
    v: Array,
    angular: AngularGalerkin | None = None,
    projection_tails: ProjectionTails | None = None,
    scalar_coordinates: CharacteristicLGLMesh | None = None,
    initial_value: Array | None = None,
) -> tuple[Array, Array]:
    """Advance Omega*chib in v using equation 6 of the attachment."""

    inverse = tangent_inverse(grid, metric)
    difference, _ = connection_difference(grid, metric, inverse)
    weighted_tr_chi = omega**2 * q
    weighted_chi = shear + 0.5 * weighted_tr_chi[..., None, None] * metric
    d3_weighted_chi = _differentiate_u(
        weighted_chi, u, axis=1, scalar_coordinates=scalar_coordinates
    )
    d3_weighted_chi += lie_covariant_tensor(grid, shift, weighted_chi)

    zeta = np.einsum("n...ij,n...j->n...i", metric, zeta_up)
    nabla_zeta = one_form_covariant_derivative(grid, zeta, difference)
    sym_nabla_zeta = nabla_zeta + np.swapaxes(nabla_zeta, -1, -2)
    grad_log_omega = scalar_gradient(grid, np.log(omega))
    sym_zeta_grad = np.einsum(
        "n...i,n...j->n...ij", zeta, grad_log_omega
    )
    sym_zeta_grad += np.swapaxes(sym_zeta_grad, -1, -2)
    source = (
        d3_weighted_chi
        - 2.0 * omega[..., None, None] ** 2 * sym_nabla_zeta
        - 4.0 * omega[..., None, None] ** 2 * sym_zeta_grad
    )
    source = _project_sym2(
        angular,
        source,
        projection_tails,
        "weighted_chib_rhs",
    )
    if initial_value is None:
        projector = sphere_broadcast(grid, 1)
        initial_face = np.broadcast_to(
            u[None, :, None, None] * projector,
            metric[:, :, 0].shape,
        )
    else:
        initial_face = np.broadcast_to(
            np.asarray(initial_value), metric[:, :, 0].shape
        )
    initial = np.broadcast_to(
        initial_face[:, :, None], metric.shape
    )
    weighted_chib = initial + _integrate_v(
        source, v, axis=2, scalar_coordinates=scalar_coordinates
    )
    weighted_chib = 0.5 * (
        weighted_chib + np.swapaxes(weighted_chib, -1, -2)
    )
    weighted_chib = _project_sym2(
        angular,
        weighted_chib,
        projection_tails,
        "weighted_chib_state",
    )
    return weighted_chib, source


def solve_incoming_metric(
    grid: PointSphereGrid,
    metric_on_hminus1: Array,
    weighted_chib: Array,
    shift: Array,
    u: Array,
    kinematic_factor: float = 2.0,
    angular: AngularGalerkin | None = None,
    projection_tails: ProjectionTails | None = None,
    transport_cfl: float | None = None,
) -> Array:
    """Integrate D3 g_tilde = 2 Omega chib from H_-1.

    ``kinematic_factor=0.5`` is intentionally supported only for the mutation
    test of the inconsistent coefficient in the attachment's Task 1.
    """

    result = np.zeros_like(weighted_chib)
    result[:, 0] = _project_sym2(
        angular,
        metric_on_hminus1,
        projection_tails,
        "incoming_metric_boundary",
    )
    midpoint_shift = midpoint_values(shift, u, axis=1)
    midpoint_chib = midpoint_values(weighted_chib, u, axis=1)
    for i in range(len(u) - 1):
        step = float(u[i + 1] - u[i])
        substeps = _transport_substeps(
            grid,
            shift,
            midpoint_shift,
            u,
            i,
            transport_cfl,
        )

        def rhs(value_metric: Array, alpha: float) -> Array:
            stage_shift = _quadratic_stage_value(
                shift, midpoint_shift, i, alpha, axis=1
            )
            stage_chib = _quadratic_stage_value(
                weighted_chib, midpoint_chib, i, alpha, axis=1
            )
            complete = kinematic_factor * stage_chib - lie_covariant_tensor(
                grid, stage_shift, value_metric
            )
            return _project_sym2(
                angular,
                complete,
                projection_tails,
                "incoming_metric_rhs",
            )

        current = result[:, i]
        local_step = step / substeps
        for substep in range(substeps):
            alpha_left = substep / substeps
            alpha_middle = (substep + 0.5) / substeps
            alpha_right = (substep + 1.0) / substeps
            k1 = rhs(current, alpha_left)
            k2 = rhs(current + 0.5 * local_step * k1, alpha_middle)
            k3 = rhs(current + 0.5 * local_step * k2, alpha_middle)
            k4 = rhs(current + local_step * k3, alpha_right)
            current = current + local_step * (
                k1 + 2.0 * k2 + 2.0 * k3 + k4
            ) / 6.0
        result[:, i + 1] = current
    return result


def metric_closure(
    grid: PointSphereGrid,
    state: FirstOrderState,
    reconstructed: Array,
    u: Array,
    scalar_coordinates: CharacteristicLGLMesh | None = None,
) -> dict[str, Array | float]:
    inverse = tangent_inverse(grid, state.metric)
    difference = reconstructed - state.metric
    pointwise = np.sqrt(np.maximum(tensor_norm_sq(difference, inverse), 0.0))
    differential = _differentiate_u(
        state.metric, u, axis=1, scalar_coordinates=scalar_coordinates
    )
    differential += lie_covariant_tensor(grid, state.shift, state.metric)
    differential -= 2.0 * state.weighted_chib
    differential_norm = np.sqrt(
        np.maximum(tensor_norm_sq(differential, inverse), 0.0)
    )
    return {
        "pointwise": pointwise,
        "differential_pointwise": differential_norm,
        "maximum": float(np.max(pointwise)),
        "differential_maximum": float(np.max(differential_norm)),
    }


def picard_step(
    grid: PointSphereGrid,
    state: FirstOrderState,
    boundary: dict[str, Array | float],
    u: Array,
    v: Array,
    metric_substeps: int = 2,
    angular: AngularGalerkin | None = None,
    scalar_coordinates: CharacteristicLGLMesh | None = None,
    metric_parameterization: str = "direct",
    metric_integrator: str = "rk4",
    sdc_tolerance: float = 1.0e-10,
    sdc_overgrid_tolerance: float = np.inf,
    sdc_maximum_corrections: int = 12,
    u_integrator: str = "rk4",
    u_sdc_tolerance: float = 1.0e-10,
    u_sdc_overgrid_tolerance: float = 1.0e-7,
    u_sdc_maximum_corrections: int = 12,
    u_sdc_half_trace_tolerance: float = 1.0e-9,
    u_sdc_half_trace_absolute_floor: float = 1.0e-14,
    incoming: dict[str, Array | float] | None = None,
    impose_previous_outgoing_boundary: bool = True,
    transport_cfl: float | None = None,
    compute_incoming_metric_closure: bool = True,
    expansion_excision_threshold: float | None = None,
) -> tuple[FirstOrderState, dict[str, object]]:
    """One complete numerical sweep.

    ``incoming`` supplies non-Minkowski data on ``v=0`` for the direct-RK
    benchmark path.  Missing entries retain the flat-cone values,
    so every production Q1 caller is unchanged.  The dictionary may contain
    ``metric``, ``expansion``, ``log_omega`` (or ``omega``),
    ``weighted_omegab``, ``weighted_chib``, ``zeta_up``, and ``shift``.
    For a restarted incoming face, ``weighted_tr_chi`` is the theory-native
    datum ``Omega*tr(chi)``.  It is converted to the stored
    ``q=Omega^{-1}tr(chi)`` only after the new lapse has been constructed.
    """

    if u_integrator not in {"rk4", "sdc"}:
        raise ValueError("u_integrator must be 'rk4' or 'sdc'")
    if u_integrator == "sdc" and scalar_coordinates is None:
        raise ValueError("the SDC u integrator requires an LGL mesh")
    if incoming is not None and metric_integrator == "sdc":
        raise NotImplementedError(
            "general incoming characteristic data are currently implemented "
            "only for the direct-RK metric march"
        )
    u_sdc_diagnostics: dict[str, list[dict[str, object]]] = {}
    u_sdc_maps: dict[str, Array] = {}
    if u_integrator == "sdc":
        # Local import is required: the equation-specific wrappers reuse this
        # module's state/projection helpers, so importing them at module load
        # time would create a cycle.
        from .tau_solvers import (
            solve_half_shear_tau_sdc,
            solve_incoming_metric_tau_sdc,
            solve_log_omega_tau_sdc,
            solve_weighted_omega_tau_sdc,
        )

        def record_u_sdc(field: str, result: object) -> None:
            u_sdc_diagnostics[field] = list(result.diagnostics)
            for name, value in result.defect_maps.items():
                key = f"u_sdc_{field}_{name}"
                array = np.asarray(value, dtype=float)
                if array.shape != (len(u), len(v)):
                    raise ValueError(
                        f"{key} has shape {array.shape}, expected "
                        f"({len(u)},{len(v)})"
                    )
                u_sdc_maps[key] = array

    projection_tails: ProjectionTails = {}
    # The literal zeroth iterate in the supplied construction is extended
    # from v=0 and need not already agree with the nontrivial data on H_{-1}.
    # Later integer iterates do agree there by construction.  Keep the
    # established repair as the default for existing callers, but allow the
    # theory-from-zero driver to preserve S^(0) exactly.
    if impose_previous_outgoing_boundary:
        impose_outgoing_boundary(state, boundary)
    geometry = section_geometry(grid, state)
    _require_finite(
        "section geometry",
        inverse=geometry["inverse"],
        connection_difference=geometry["difference"],
        eta=geometry["eta"],
        etab=geometry["etab"],
    )
    if u_integrator == "rk4":
        half = solve_half_shear(
            grid,
            state,
            geometry,
            boundary,
            u,
            angular=angular,
            projection_tails=projection_tails,
            scalar_coordinates=scalar_coordinates,
            transport_cfl=transport_cfl,
        )
    else:
        half_result = solve_half_shear_tau_sdc(
            grid,
            state,
            boundary,
            scalar_coordinates,
            angular=angular,
            projection_tails=projection_tails,
            tolerance=u_sdc_tolerance,
            overgrid_tolerance=u_sdc_overgrid_tolerance,
            trace_tolerance=u_sdc_half_trace_tolerance,
            trace_absolute_floor=u_sdc_half_trace_absolute_floor,
            maximum_corrections=u_sdc_maximum_corrections,
        )
        half = half_result.values
        record_u_sdc("half_shear", half_result)
    _require_finite("half-shear solve", half_shear=half)
    inverse = geometry["inverse"]
    half_trace = tensor_trace(half, inverse)
    half_norm = np.sqrt(
        np.maximum(tensor_norm_sq(half, inverse), 0.0)
    )
    half_trace_diagnostics = {
        "absolute_maximum": float(np.max(np.abs(half_trace))),
        "relative_maximum": float(
            np.max(np.abs(half_trace) / np.maximum(half_norm, 1.0e-14))
        ),
    }
    if angular is not None:
        half_trace_coefficients = angular.scalar.analyze(half_trace)
        half_trace_diagnostics.update(
            {
                "retained_coefficient_maximum": float(
                    np.max(
                        np.abs(
                            half_trace_coefficients[
                                angular.scalar.retained
                            ]
                        )
                    )
                ),
                "discarded_work_coefficient_maximum": float(
                    np.max(
                        np.abs(
                            half_trace_coefficients[
                                ~angular.scalar.retained
                            ]
                        )
                    )
                ),
            }
        )
    eta_etab = np.einsum(
        "n...i,n...ij,n...j->n...",
        geometry["eta"],
        inverse,
        geometry["etab"],
    )
    shear_dot = np.einsum(
        "n...ik,n...jl,n...ij,n...kl->n...",
        inverse,
        inverse,
        half,
        geometry["weighted_hatchib"],
    )
    weighted_tr_chi = state.omega**2 * state.q
    weighted_tr_chib = geometry["weighted_tr_chib"]
    eta_up = np.einsum("n...ij,n...j->n...i", inverse, geometry["eta"])
    div_eta = vector_divergence(grid, eta_up, geometry["difference"])
    d3_weighted_tr_chi = _differentiate_u(
        weighted_tr_chi, u, axis=1, scalar_coordinates=scalar_coordinates
    ) + np.einsum(
        "n...i,n...i->n...",
        state.shift,
        scalar_gradient(grid, weighted_tr_chi),
    )
    omegab_source = 0.25 * (
        shear_dot
        + 0.5 * weighted_tr_chi * weighted_tr_chib
        - 4.0 * state.omega**2 * eta_etab
        + d3_weighted_tr_chi
        - 2.0 * state.omega**2 * div_eta
    )
    omegab_source = _project_scalar(
        angular,
        omegab_source,
        projection_tails,
        "weighted_omegab_half_source",
    )
    _require_finite(
        "incoming-lapse half source", omegab_source=omegab_source
    )
    incoming_data = {} if incoming is None else incoming

    def incoming_face(name: str, template: Array, default: float = 0.0) -> Array:
        value = np.asarray(incoming_data.get(name, default))
        return np.broadcast_to(value, template[:, :, 0].shape)

    new_weighted_omegab_half = incoming_face(
        "weighted_omegab", state.weighted_omegab
    )[:, :, None] + _integrate_v(
        omegab_source, v, axis=2, scalar_coordinates=scalar_coordinates
    )
    new_weighted_omegab_half = _project_scalar(
        angular,
        new_weighted_omegab_half,
        projection_tails,
        "weighted_omegab_half_state",
    )
    _require_finite(
        "incoming-lapse half integration",
        weighted_omegab_half=new_weighted_omegab_half,
    )
    if "weighted_omegab" in boundary:
        # On the imposed outgoing boundary b^(i+1)=b^i, so the full and
        # half-step coefficients coincide.
        new_weighted_omegab_half[:, 0] = np.asarray(
            boundary["weighted_omegab"]
        )
    boundary_omega = np.asarray(boundary.get("omega", 1.0))
    if u_integrator == "rk4":
        new_omega = solve_log_omega(
            grid,
            state,
            new_weighted_omegab_half,
            u,
            initial_log_omega=np.log(boundary_omega),
            angular=angular,
            projection_tails=projection_tails,
            transport_cfl=transport_cfl,
        )
        new_log_omega = np.log(new_omega)
    else:
        log_omega_result = solve_log_omega_tau_sdc(
            grid,
            state,
            new_weighted_omegab_half,
            scalar_coordinates,
            np.log(boundary_omega),
            angular=angular,
            projection_tails=projection_tails,
            tolerance=u_sdc_tolerance,
            overgrid_tolerance=u_sdc_overgrid_tolerance,
            maximum_corrections=u_sdc_maximum_corrections,
        )
        if log_omega_result.auxiliary is None:
            raise AssertionError("the tau log(Omega) solve omitted Omega")
        new_log_omega = log_omega_result.values
        new_omega = np.asarray(log_omega_result.auxiliary["omega"])
        record_u_sdc("log_omega", log_omega_result)
    incoming_log_omega = incoming_data.get("log_omega")
    if incoming_log_omega is None and "omega" in incoming_data:
        incoming_log_omega = np.log(np.asarray(incoming_data["omega"]))
    incoming_lapse_mismatch = 0.0
    if incoming_log_omega is not None:
        prescribed_log_omega = np.broadcast_to(
            np.asarray(incoming_log_omega), new_log_omega[:, :, 0].shape
        )
        incoming_lapse_mismatch = float(
            np.max(
                np.abs(
                    new_log_omega[:, :, 0] - prescribed_log_omega
                )
            )
        )
        # A restarted characteristic slab iterates log(Omega) itself.  Its
        # accepted incoming trace is data, not a value to be reconstructed
        # from a separately relaxed Omega*omegab coefficient.
        new_log_omega[:, :, 0] = prescribed_log_omega
        new_omega[:, :, 0] = np.exp(prescribed_log_omega)
    _require_finite("lapse solve", omega=new_omega)
    if float(np.min(new_omega)) <= 0.0:
        raise FloatingPointError(
            "lapse solve left the positive cone: "
            f"min(Omega)={float(np.min(new_omega)):.12g}"
        )
    if u_integrator == "rk4":
        new_weighted_omega = solve_weighted_omega(
            grid,
            state,
            omegab_source,
            new_omega,
            u,
            initial_value=np.asarray(boundary.get("weighted_omega", 0.0)),
            angular=angular,
            projection_tails=projection_tails,
            transport_cfl=transport_cfl,
        )
    else:
        weighted_omega_result = solve_weighted_omega_tau_sdc(
            grid,
            state,
            half,
            new_log_omega,
            scalar_coordinates,
            np.asarray(boundary.get("weighted_omega", 0.0)),
            angular=angular,
            projection_tails=projection_tails,
            tolerance=u_sdc_tolerance,
            overgrid_tolerance=u_sdc_overgrid_tolerance,
            reference_v_source=omegab_source,
            source_cross_equation_tolerance=u_sdc_overgrid_tolerance,
            interface_tolerance=u_sdc_overgrid_tolerance,
            maximum_corrections=u_sdc_maximum_corrections,
        )
        new_weighted_omega = weighted_omega_result.values
        record_u_sdc("weighted_omega", weighted_omega_result)
        if weighted_omega_result.auxiliary is None:
            raise AssertionError(
                "the tau weighted-omega solve omitted its fresh source traces"
            )
    _require_finite(
        "outgoing-lapse-coefficient solve",
        weighted_omega=new_weighted_omega,
    )

    div_half = tensor_divergence(
        grid, half, geometry["difference"], geometry["inverse"]
    )
    grad_weighted_omega = scalar_gradient(grid, new_weighted_omega)
    grad_weighted_tr_chi = scalar_gradient(grid, weighted_tr_chi)
    grad_log_new_omega = scalar_gradient(grid, np.log(new_omega))
    half_zeta = np.einsum("n...ij,n...j->n...i", half, state.zeta_up)
    zeta_source = (
        -2.0 * weighted_tr_chi[..., None] * state.zeta_up
        - 2.0 * np.einsum("n...ij,n...j->n...i", inverse, half_zeta)
        + 2.0 * np.einsum(
            "n...ij,n...j->n...i", inverse, grad_weighted_omega
        )
        + np.einsum("n...ij,n...j->n...i", inverse, div_half)
        - 0.5
        * np.einsum("n...ij,n...j->n...i", inverse, grad_weighted_tr_chi)
        + weighted_tr_chi[..., None]
        * np.einsum("n...ij,n...j->n...i", inverse, grad_log_new_omega)
    )
    zeta_source = _project_vector(
        angular,
        zeta_source,
        projection_tails,
        "zeta_rhs",
    )
    _require_finite("zeta source", zeta_source=zeta_source)
    new_zeta = incoming_face(
        "zeta_up", state.zeta_up
    )[:, :, None] + _integrate_v(
        zeta_source, v, axis=2, scalar_coordinates=scalar_coordinates
    )
    if angular is None:
        projector = sphere_broadcast(grid, 2)
        new_zeta = np.einsum(
            "n...ij,n...j->n...i", projector, new_zeta
        )
    else:
        new_zeta = _project_vector(
            angular,
            new_zeta,
            projection_tails,
            "zeta_state",
        )
    _require_finite("zeta integration", zeta=new_zeta)
    shift_source = _project_vector(
        angular,
        -4.0 * new_omega[..., None] ** 2 * new_zeta,
        projection_tails,
        "shift_rhs",
    )
    new_shift = incoming_face(
        "shift", state.shift
    )[:, :, None] + _integrate_v(
        shift_source, v, axis=2, scalar_coordinates=scalar_coordinates
    )
    if angular is not None:
        new_shift = _project_vector(
            angular,
            new_shift,
            projection_tails,
            "shift_state",
        )
    _require_finite("shift integration", shift=new_shift)
    incoming_q = incoming_data.get("expansion")
    if "weighted_tr_chi" in incoming_data:
        prescribed_weighted_tr_chi = np.broadcast_to(
            np.asarray(incoming_data["weighted_tr_chi"]),
            state.q[:, :, 0].shape,
        )
        incoming_q = (
            prescribed_weighted_tr_chi / new_omega[:, :, 0] ** 2
        )
    if metric_integrator == "rk4":
        excision_diagnostics: dict[str, object] = {}
        new_metric, new_q, new_shear = solve_metric_and_expansion(
            grid,
            state,
            half,
            new_omega,
            u,
            v,
            substeps=metric_substeps,
            angular=angular,
            projection_tails=projection_tails,
            scalar_coordinates=scalar_coordinates,
            metric_parameterization=metric_parameterization,
            initial_metric=incoming_data.get("metric"),
            initial_q=incoming_q,
            expansion_excision_threshold=expansion_excision_threshold,
            excision_diagnostics=excision_diagnostics,
        )
        metric_sdc_diagnostics: list[dict[str, float | int | bool]] = []
    elif metric_integrator == "sdc":
        if scalar_coordinates is None:
            raise ValueError("the SDC metric integrator requires an LGL mesh")
        if metric_parameterization != "cholesky":
            raise ValueError(
                "the SDC pilot currently requires the Cholesky metric "
                "parameterization"
            )
        (
            new_metric,
            new_q,
            new_shear,
            metric_sdc_diagnostics,
        ) = solve_metric_and_expansion_sdc(
            grid,
            state,
            half,
            new_omega,
            u,
            v,
            scalar_coordinates,
            angular=angular,
            projection_tails=projection_tails,
            tolerance=sdc_tolerance,
            overgrid_tolerance=sdc_overgrid_tolerance,
            maximum_corrections=sdc_maximum_corrections,
        )
    else:
        raise ValueError("metric_integrator must be 'rk4' or 'sdc'")
    _require_finite(
        "metric/Raychaudhuri solve",
        metric=new_metric,
        q=new_q,
        shear=new_shear,
    )
    boundary_inverse = np.asarray(boundary["inverse"])
    metric_boundary_difference = new_metric[:, 0] - np.asarray(
        boundary["metric"]
    )
    shear_boundary_difference = new_shear[:, 0] - np.asarray(
        boundary["shear"]
    )
    boundary_mismatch = {
        "metric_maximum": float(
            np.max(
                np.sqrt(
                    np.maximum(
                        tensor_norm_sq(
                            metric_boundary_difference, boundary_inverse
                        ),
                        0.0,
                    )
                )
            )
        ),
        "expansion_maximum": float(
            np.max(
                np.abs(new_q[:, 0] - np.asarray(boundary["expansion"]))
            )
        ),
        "shear_maximum": float(
            np.max(
                np.sqrt(
                    np.maximum(
                        tensor_norm_sq(
                            shear_boundary_difference, boundary_inverse
                        ),
                        0.0,
                    )
                )
            )
        ),
        "omega_maximum": float(
            np.max(np.abs(new_omega[:, 0] - boundary_omega))
        ),
        "zeta_maximum": (
            float(
                np.max(
                    np.sqrt(
                        np.maximum(
                            np.einsum(
                                "nvi,nvij,nvj->nv",
                                new_zeta[:, 0]
                                - np.asarray(boundary["zeta_up"]),
                                np.asarray(boundary["metric"]),
                                new_zeta[:, 0]
                                - np.asarray(boundary["zeta_up"]),
                            ),
                            0.0,
                        )
                    )
                )
            )
            if "zeta_up" in boundary
            else 0.0
        ),
        "shift_maximum": (
            float(
                np.max(
                    np.sqrt(
                        np.maximum(
                            np.einsum(
                                "nvi,nvij,nvj->nv",
                                new_shift[:, 0]
                                - np.asarray(boundary["shift"]),
                                np.asarray(boundary["metric"]),
                                new_shift[:, 0]
                                - np.asarray(boundary["shift"]),
                            ),
                            0.0,
                        )
                    )
                )
            )
            if "shift" in boundary
            else 0.0
        ),
    }
    new_metric[:, 0] = np.asarray(boundary["metric"])
    new_q[:, 0] = np.asarray(boundary["expansion"])
    new_shear[:, 0] = np.asarray(boundary["shear"])
    new_omega[:, 0] = boundary_omega
    if "zeta_up" in boundary:
        new_zeta[:, 0] = np.asarray(boundary["zeta_up"])
    if "shift" in boundary:
        new_shift[:, 0] = np.asarray(boundary["shift"])
    new_weighted_omega[:, 0] = np.asarray(
        boundary.get("weighted_omega", 0.0)
    )
    # Convert the half iterate in D3 log(Omega) into the full geometric
    # coefficient.  This is the value that must be stored and transferred to
    # the next characteristic slab.
    new_weighted_omegab, omegab_full_source = complete_weighted_omegab(
        grid,
        state,
        new_weighted_omegab_half,
        omegab_source,
        new_omega,
        new_weighted_omega,
        new_zeta,
        new_shift,
        angular=angular,
        projection_tails=projection_tails,
    )
    _require_finite(
        "full incoming-lapse coefficient",
        weighted_omegab=new_weighted_omegab,
        omegab_full_source=omegab_full_source,
    )

    new_weighted_chib, chib_source = solve_weighted_chib(
        grid,
        new_metric,
        new_omega,
        new_zeta,
        new_shift,
        new_q,
        new_shear,
        u,
        v,
        angular=angular,
        projection_tails=projection_tails,
        scalar_coordinates=scalar_coordinates,
        initial_value=incoming_data.get("weighted_chib"),
    )
    _require_finite(
        "incoming-second-form solve",
        weighted_chib=new_weighted_chib,
        chib_source=chib_source,
    )
    if "weighted_chib" in boundary:
        new_weighted_chib[:, 0] = np.asarray(boundary["weighted_chib"])
    new_state = FirstOrderState(
        metric=new_metric,
        omega=new_omega,
        zeta_up=new_zeta,
        shift=new_shift,
        q=new_q,
        shear=new_shear,
        weighted_chib=new_weighted_chib,
        weighted_omega=new_weighted_omega,
        weighted_omegab=new_weighted_omegab,
    )
    validate_state(grid, new_state)
    # Retain the actual semidiscrete endpoint sources used by the Galerkin
    # system.  They differ from the unprojected continuum right-hand sides by
    # the angular truncation defect.  Construction-aware Ricci diagnostics
    # must use these sources and must not silently set that defect to zero.
    new_inverse = tangent_inverse(grid, new_metric)
    new_shear_norm_sq = tensor_norm_sq(new_shear, new_inverse)
    raychaudhuri_source = _project_scalar(
        angular,
        -0.5 * new_omega**2 * new_q**2
        - new_shear_norm_sq / new_omega**2,
    )
    metric_source = _project_sym2(
        angular,
        new_omega[..., None, None] ** 2
        * new_q[..., None, None]
        * new_metric
        + 2.0 * new_shear,
    )
    if not compute_incoming_metric_closure:
        reconstructed = np.array(new_metric, copy=True)
    elif u_integrator == "rk4":
        reconstructed = solve_incoming_metric(
            grid,
            new_metric[:, 0],
            new_weighted_chib,
            new_shift,
            u,
            angular=angular,
            projection_tails=projection_tails,
            transport_cfl=transport_cfl,
        )
    else:
        incoming_metric_result = solve_incoming_metric_tau_sdc(
            grid,
            new_metric[:, 0],
            new_weighted_chib,
            new_shift,
            scalar_coordinates,
            angular=angular,
            projection_tails=projection_tails,
            tolerance=u_sdc_tolerance,
            overgrid_tolerance=u_sdc_overgrid_tolerance,
            physical_overgrid_tolerance=u_sdc_overgrid_tolerance,
            maximum_corrections=u_sdc_maximum_corrections,
        )
        reconstructed = incoming_metric_result.values
        record_u_sdc("incoming_metric", incoming_metric_result)
    return new_state, {
        "half_shear": half,
        "weighted_omegab_half": new_weighted_omegab_half,
        "weighted_omegab_full": new_weighted_omegab,
        "omegab_half_source": omegab_source,
        "omegab_half_source_semantics": (
            "v-march source using the characteristic mesh's average tau "
            "derivative at shared interfaces"
        ),
        "omegab_full_source": omegab_full_source,
        # Backward-compatible name for scripts that deliberately mutate the
        # half-step construction equation.
        "omegab_source": omegab_source,
        "zeta_source": zeta_source,
        "chib_source": chib_source,
        "raychaudhuri_source": raychaudhuri_source,
        "metric_source": metric_source,
        "incoming_metric": reconstructed,
        "incoming_metric_closure_computed": compute_incoming_metric_closure,
        "projection_tails": projection_tails,
        "half_shear_trace_drift": half_trace_diagnostics,
        "preoverwrite_boundary_mismatch": boundary_mismatch,
        "incoming_log_omega_preoverwrite_mismatch": (
            incoming_lapse_mismatch
        ),
        "expansion_excision": (
            excision_diagnostics
            if metric_integrator == "rk4"
            else {
                "threshold": expansion_excision_threshold,
                "event_v": np.full(len(u), np.nan),
                "event_angular_index": np.full(
                    len(u), -1, dtype=int
                ),
                "active_u_at_terminal_v": np.ones(
                    len(u), dtype=bool
                ),
            }
        ),
        "metric_sdc_diagnostics": metric_sdc_diagnostics,
        "u_integrator": u_integrator,
        "u_sdc_diagnostics": u_sdc_diagnostics,
        "u_sdc_maps": u_sdc_maps,
        "weighted_omega_fresh_v_source_left_trace": (
            None
            if u_integrator == "rk4"
            else weighted_omega_result.auxiliary[
                "fresh_v_source_left_trace"
            ]
        ),
        "weighted_omega_fresh_v_source_right_trace": (
            None
            if u_integrator == "rk4"
            else weighted_omega_result.auxiliary[
                "fresh_v_source_right_trace"
            ]
        ),
        "weighted_omega_fresh_v_source_semantics": (
            "distinct left/right one-sided tau traces actually used by the "
            "weighted-omega composite-element RHS; never interface-averaged"
        ),
    }


def validate_state(grid: PointSphereGrid, state: FirstOrderState) -> None:
    local = np.einsum(
        "nia,n...ij,njb->n...ab", grid.frames, state.metric, grid.frames
    )
    eigenvalues = np.linalg.eigvalsh(local)
    arrays = (
        state.metric,
        state.omega,
        state.q,
        state.weighted_chib,
        state.weighted_omega,
        state.weighted_omegab,
    )
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise FloatingPointError("numerical state contains nonfinite values")
    if float(np.min(eigenvalues)) <= 0.0 or float(np.min(state.omega)) <= 0.0:
        raise FloatingPointError(
            "numerical state left the positive region: "
            f"min(g)={float(np.min(eigenvalues)):.6g}, "
            f"min(Omega)={float(np.min(state.omega)):.6g}"
        )


def update_norm(new: FirstOrderState, old: FirstOrderState) -> float:
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
    ):
        current = getattr(new, name)
        previous = getattr(old, name)
        scale = np.maximum(1.0, np.maximum(np.abs(current), np.abs(previous)))
        values.append(float(np.mean(((current - previous) / scale) ** 2)))
    return math.sqrt(sum(values))


def update_map(new: FirstOrderState, old: FirstOrderState) -> Array:
    """Return a normalized Picard-update RMS on every (u,v) section."""

    result = np.zeros(new.q.shape[1:], dtype=float)
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
    ):
        current = getattr(new, name)
        previous = getattr(old, name)
        scale = np.maximum(1.0, np.maximum(np.abs(current), np.abs(previous)))
        relative_sq = ((current - previous) / scale) ** 2
        reduction_axes = (0, *range(3, relative_sq.ndim))
        result += np.mean(relative_sq, axis=reduction_axes)
    return np.sqrt(result)
