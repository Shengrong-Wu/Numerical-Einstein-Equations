"""Independent curvature and wave residual from reconstructed primitives.

Only ``(gamma, log(Omega), b, phi)`` enter this module.  No Picard source,
previous iterate, null coefficient, or construction context is accepted.
Curvature is computed from the four-g connection difference relative to
the flat ``(u,v)`` product with the unit round-sphere connection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray

from nee.numerics.coordinate_differentiation import high_order_differentiate  # noqa: E402
from nee.numerics.sphere import PointSphereGrid, tangent_inverse  # noqa: E402
from nee.state.fields import PrimitiveFields


def _differentiate_u(
    values: Array,
    u: Array,
    *,
    axis: int,
    stencil: int,
    coordinates: Any | None,
) -> Array:
    if coordinates is not None:
        return coordinates.differentiate_u(values, axis=axis)
    return high_order_differentiate(
        values, u, axis=axis, stencil=min(stencil, len(u))
    )


def _differentiate_v(
    values: Array,
    v: Array,
    *,
    axis: int,
    stencil: int,
    coordinates: Any | None,
) -> Array:
    if coordinates is not None:
        return coordinates.differentiate_v(values, axis=axis)
    return high_order_differentiate(
        values, v, axis=axis, stencil=min(stencil, len(v))
    )


def hybrid_projector(grid: PointSphereGrid) -> Array:
    projector = np.zeros((grid.count, 5, 5), dtype=float)
    projector[:, 0, 0] = 1.0
    projector[:, 1, 1] = 1.0
    projector[:, 2:5, 2:5] = grid.projector
    return projector


def angular_reference_derivative(
    grid: PointSphereGrid, values: Array, tensor_rank: int
) -> Array:
    """Round-reference derivative of hybrid R2 x TS2 covariant tensors."""

    if tensor_rank not in (1, 2, 3):
        raise ValueError("hybrid derivative supports covariant ranks 1--3")
    first, second = grid.directional_derivatives(values)
    batch_rank = values.ndim - tensor_rank - 1
    frame_shape = (
        (grid.count,)
        + (1,) * batch_rank
        + (3,)
        + (1,) * tensor_rank
    )
    raw = (
        grid.first.reshape(frame_shape)
        * np.expand_dims(first, axis=values.ndim - tensor_rank)
        + grid.second.reshape(frame_shape)
        * np.expand_dims(second, axis=values.ndim - tensor_rank)
    )
    projector = hybrid_projector(grid)
    if tensor_rank == 1:
        projected = np.einsum("npa,n...ra->n...rp", projector, raw)
    elif tensor_rank == 2:
        projected = np.einsum(
            "npa,nqb,n...rab->n...rpq", projector, projector, raw
        )
    else:
        projected = np.einsum(
            "npa,nqb,nsc,n...rabc->n...rpqs",
            projector,
            projector,
            projector,
            raw,
        )
    result = np.zeros(
        values.shape[:-tensor_rank] + (5,) + (5,) * tensor_rank
    )
    result[..., 2:5, *(slice(None),) * tensor_rank] = projected
    return result


def spacetime_reference_derivative(
    grid: PointSphereGrid,
    values: Array,
    tensor_rank: int,
    u: Array,
    v: Array,
    *,
    stencil: int = 9,
    coordinates: Any | None = None,
) -> Array:
    result = angular_reference_derivative(grid, values, tensor_rank)
    slots = (slice(None),) * tensor_rank
    result[..., 0, *slots] = _differentiate_u(
        values,
        u,
        axis=1,
        stencil=stencil,
        coordinates=coordinates,
    )
    result[..., 1, *slots] = _differentiate_v(
        values,
        v,
        axis=2,
        stencil=stencil,
        coordinates=coordinates,
    )
    return result


def build_metric_and_inverse(
    grid: PointSphereGrid,
    metric_sphere: Array,
    log_Omega: Array,
    b: Array,
) -> tuple[Array, Array, Array]:
    Omega = np.exp(log_Omega)
    shape = Omega.shape
    g = np.zeros(shape + (5, 5), dtype=float)
    inverse_g = np.zeros_like(g)
    sphere_inverse = tangent_inverse(grid, metric_sphere)
    shift_cov = np.einsum("n...ij,n...j->n...i", metric_sphere, b)
    shift_norm_sq = np.einsum("n...i,n...i->n...", shift_cov, b)

    g[..., 0, 0] = shift_norm_sq
    g[..., 0, 1] = -2.0 * Omega**2
    g[..., 1, 0] = g[..., 0, 1]
    g[..., 0, 2:5] = -shift_cov
    g[..., 2:5, 0] = -shift_cov
    g[..., 2:5, 2:5] = metric_sphere

    null_inverse = -0.5 / Omega**2
    inverse_g[..., 0, 1] = null_inverse
    inverse_g[..., 1, 0] = null_inverse
    inverse_g[..., 1, 2:5] = null_inverse[..., None] * b
    inverse_g[..., 2:5, 1] = inverse_g[..., 1, 2:5]
    inverse_g[..., 2:5, 2:5] = sphere_inverse
    return g, inverse_g, sphere_inverse


def connection_difference(
    grid: PointSphereGrid,
    g: Array,
    inverse_g: Array,
    u: Array,
    v: Array,
    *,
    stencil: int = 9,
    coordinates: Any | None = None,
) -> Array:
    derivative = spacetime_reference_derivative(
        grid,
        g,
        2,
        u,
        v,
        stencil=stencil,
        coordinates=coordinates,
    )
    lower = 0.5 * (
        derivative
        + np.swapaxes(derivative, -3, -2)
        - np.einsum("n...kij->n...ijk", derivative)
    )
    return np.einsum("n...lk,n...ijk->n...lij", inverse_g, lower)


def direct_ricci(
    grid: PointSphereGrid,
    g: Array,
    inverse_g: Array,
    difference: Array,
    u: Array,
    v: Array,
    *,
    stencil: int = 9,
    coordinates: Any | None = None,
) -> Array:
    """Compute Ricci from ``D C + C*C`` without null-equation intermediates."""

    ricci = np.zeros_like(g)
    # Ricci of the unit round reference sphere.
    ricci[..., 2:5, 2:5] = grid.projector[:, None, None]
    derivative_u = _differentiate_u(
        difference,
        u,
        axis=1,
        stencil=stencil,
        coordinates=coordinates,
    )
    derivative_v = _differentiate_v(
        difference,
        v,
        axis=2,
        stencil=stencil,
        coordinates=coordinates,
    )
    for direction, derivative_direction in enumerate(
        (derivative_u, derivative_v)
    ):
        ricci += derivative_direction[..., direction, :, :]
        ricci[..., :, direction] -= np.einsum(
            "n...iji->n...j", derivative_direction
        )

    projector = hybrid_projector(grid)
    for angular in range(3):
        direction = angular + 2
        matrix = (
            grid.first[:, angular, None] * grid.derivative_first
            + grid.second[:, angular, None] * grid.derivative_second
        )
        raw = np.tensordot(matrix, difference, axes=(1, 0))
        ricci += np.einsum(
            "na,njb,nkc,n...abc->n...jk",
            projector[:, direction, :],
            projector,
            projector,
            raw,
        )
        ricci[..., :, direction] -= np.einsum(
            "nia,njb,nic,n...abc->n...j",
            projector,
            projector,
            projector,
            raw,
        )

    # C^i_im C^m_jk - C^i_km C^m_ji
    trace = np.einsum("n...iim->n...m", difference)
    ricci += np.einsum("n...m,n...mjk->n...jk", trace, difference)
    ricci -= np.einsum(
        "n...ikm,n...mji->n...jk", difference, difference
    )
    return ricci


def scalar_covector(
    grid: PointSphereGrid,
    phi: Array,
    u: Array,
    v: Array,
    *,
    stencil: int = 9,
    coordinates: Any | None = None,
) -> Array:
    covector = np.zeros(phi.shape + (5,), dtype=float)
    covector[..., 0] = _differentiate_u(
        phi,
        u,
        axis=1,
        stencil=stencil,
        coordinates=coordinates,
    )
    covector[..., 1] = _differentiate_v(
        phi,
        v,
        axis=2,
        stencil=stencil,
        coordinates=coordinates,
    )
    covector[..., 2:5] = grid.reference_derivative(phi, tensor_rank=0)
    return covector


def wave_operator(
    grid: PointSphereGrid,
    phi: Array,
    inverse_g: Array,
    difference: Array,
    u: Array,
    v: Array,
    *,
    stencil: int = 9,
    coordinates: Any | None = None,
) -> tuple[Array, Array]:
    dphi = scalar_covector(
        grid,
        phi,
        u,
        v,
        stencil=stencil,
        coordinates=coordinates,
    )
    derivative = spacetime_reference_derivative(
        grid,
        dphi,
        1,
        u,
        v,
        stencil=stencil,
        coordinates=coordinates,
    )
    hessian = derivative - np.einsum(
        "n...lij,n...l->n...ij", difference, dphi
    )
    wave = np.einsum("n...ij,n...ij->n...", inverse_g, hessian)
    return wave, dphi


def null_components(
    tensor: Array, Omega: Array, b: Array
) -> dict[str, Array]:
    angular = tensor[..., 2:5, 2:5]
    return {
        "33": (
            tensor[..., 0, 0]
            + 2.0
            * np.einsum("n...i,n...i->n...", b, tensor[..., 0, 2:5])
            + np.einsum("n...i,n...ij,n...j->n...", b, angular, b)
        )
        / Omega**2,
        "44": tensor[..., 1, 1] / Omega**2,
        "34": (
            tensor[..., 0, 1]
            + np.einsum("n...i,n...i->n...", b, tensor[..., 2:5, 1])
        )
        / Omega**2,
        "3A": (
            tensor[..., 0, 2:5]
            + np.einsum("n...i,n...ij->n...j", b, angular)
        )
        / Omega[..., None],
        "4A": tensor[..., 1, 2:5] / Omega[..., None],
        "AB": angular,
    }


def positive_null_norm(
    components: dict[str, Array], sphere_inverse: Array
) -> Array:
    value = (
        components["33"] ** 2
        + components["44"] ** 2
        + 2.0 * components["34"] ** 2
    )
    value += np.einsum(
        "n...i,n...ij,n...j->n...",
        components["3A"],
        sphere_inverse,
        components["3A"],
    )
    value += np.einsum(
        "n...i,n...ij,n...j->n...",
        components["4A"],
        sphere_inverse,
        components["4A"],
    )
    value += np.einsum(
        "n...ik,n...jl,n...ij,n...kl->n...",
        sphere_inverse,
        sphere_inverse,
        components["AB"],
        components["AB"],
    )
    return np.sqrt(np.maximum(value, 0.0))


@dataclass(frozen=True)
class ResidualResult:
    coordinate_residual: Array
    null_components: dict[str, Array]
    pointwise_norm: Array
    section_l2: Array
    einstein_section_l2: Array
    wave_section_l2: Array | None
    wave: Array | None
    summary: dict[str, Any]


def summarize_result_on_mask(
    result: ResidualResult,
    mask: Array,
    *,
    label: str,
) -> dict[str, Any]:
    """Summarize a completed audit on an explicit coordinate reliability mask."""

    selected = np.asarray(mask, dtype=bool)
    if selected.shape != result.section_l2.shape or not np.any(selected):
        raise ValueError("residual summary mask has the wrong shape or is empty")
    summary: dict[str, Any] = {
        "label": label,
        "cell_count": int(np.count_nonzero(selected)),
        "combined_Linf_uv_L2_sphere": float(
            np.max(result.section_l2[selected])
        ),
        "einstein_Linf_uv_L2_sphere": float(
            np.max(result.einstein_section_l2[selected])
        ),
        "pointwise_maximum": float(
            np.max(result.pointwise_norm[:, selected])
        ),
        "components": {
            name: float(np.max(np.abs(value[:, selected])))
            for name, value in result.null_components.items()
        },
    }
    if result.wave is not None:
        assert result.wave_section_l2 is not None
        summary["wave_Linf_uv_L2_sphere"] = float(
            np.max(result.wave_section_l2[selected])
        )
        summary["wave_pointwise_maximum"] = float(
            np.max(np.abs(result.wave[:, selected]))
        )
        summary["ESE_acceptance_sum"] = (
            summary["einstein_Linf_uv_L2_sphere"]
            + summary["wave_Linf_uv_L2_sphere"]
        )
    return summary


def _slice_fields(
    fields: PrimitiveFields, u_slice: slice, v_slice: slice
) -> PrimitiveFields:
    return PrimitiveFields(
        g=fields.g[:, u_slice, v_slice],
        log_Omega=fields.log_Omega[:, u_slice, v_slice],
        b=fields.b[:, u_slice, v_slice],
        phi=(
            None
            if fields.phi is None
            else fields.phi[:, u_slice, v_slice]
        ),
    )


def evaluate(
    grid: PointSphereGrid,
    fields: PrimitiveFields,
    u: Array,
    v: Array,
    *,
    stencil: int = 9,
    mask_halo: int | None = None,
    coordinates: Any | None = None,
) -> ResidualResult:
    """Evaluate the independent EVE/ESE residual on one audit grid."""

    g, inverse_g, sphere_inverse = build_metric_and_inverse(
        grid, fields.g, fields.log_Omega, fields.b
    )
    difference = connection_difference(
        grid,
        g,
        inverse_g,
        u,
        v,
        stencil=stencil,
        coordinates=coordinates,
    )
    ricci = direct_ricci(
        grid,
        g,
        inverse_g,
        difference,
        u,
        v,
        stencil=stencil,
        coordinates=coordinates,
    )
    wave = None
    if fields.phi is None:
        residual = ricci
    else:
        wave, dphi = wave_operator(
            grid,
            fields.phi,
            inverse_g,
            difference,
            u,
            v,
            stencil=stencil,
            coordinates=coordinates,
        )
        residual = ricci - np.einsum(
            "n...i,n...j->n...ij", dphi, dphi
        )
    Omega = np.exp(fields.log_Omega)
    components = null_components(residual, Omega, fields.b)
    einstein_pointwise = positive_null_norm(components, sphere_inverse)
    pointwise = einstein_pointwise
    if wave is not None:
        pointwise = np.sqrt(pointwise**2 + wave**2)

    local_metric = np.einsum(
        "nia,n...ij,njb->n...ab", grid.frames, fields.g, grid.frames
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local_metric), 0.0))
    section_l2 = np.sqrt(
        np.maximum(
            4.0 * math.pi * np.mean(pointwise**2 * area_ratio, axis=0),
            0.0,
        )
    )
    einstein_section_l2 = np.sqrt(
        np.maximum(
            4.0
            * math.pi
            * np.mean(einstein_pointwise**2 * area_ratio, axis=0),
            0.0,
        )
    )
    wave_section_l2 = None
    if wave is not None:
        wave_section_l2 = np.sqrt(
            np.maximum(
                4.0 * math.pi * np.mean(wave**2 * area_ratio, axis=0),
                0.0,
            )
        )
    halo = stencil - 1 if mask_halo is None else mask_halo
    if 2 * halo >= min(len(u), len(v)):
        halo = max(1, min(len(u), len(v)) // 4)
    mask = np.zeros(section_l2.shape, dtype=bool)
    mask[halo : len(u) - halo, halo : len(v) - halo] = True
    if not np.any(mask):
        mask[:] = True
    component_summary = {
        name: {
            "raw_maximum": float(np.max(np.abs(value))),
            "masked_maximum": float(
                np.max(np.abs(value[:, mask]))
            ),
        }
        for name, value in components.items()
    }
    summary: dict[str, Any] = {
        "method": (
            "four-g connection difference from primitive "
            "(gamma,logOmega,b,phi); no construction context"
        ),
        "stencil": stencil,
        "mask_halo": halo,
        "raw_pointwise_maximum": float(np.max(pointwise)),
        "masked_pointwise_maximum": float(np.max(pointwise[:, mask])),
        "raw_Linf_uv_L2_sphere": float(np.max(section_l2)),
        "masked_Linf_uv_L2_sphere": float(np.max(section_l2[mask])),
        "raw_einstein_Linf_uv_L2_sphere": float(
            np.max(einstein_section_l2)
        ),
        "masked_einstein_Linf_uv_L2_sphere": float(
            np.max(einstein_section_l2[mask])
        ),
        "components": component_summary,
    }
    if wave is not None:
        assert wave_section_l2 is not None
        summary["wave_raw_maximum"] = float(np.max(np.abs(wave)))
        summary["wave_masked_maximum"] = float(np.max(np.abs(wave[:, mask])))
        summary["raw_wave_Linf_uv_L2_sphere"] = float(
            np.max(wave_section_l2)
        )
        summary["masked_wave_Linf_uv_L2_sphere"] = float(
            np.max(wave_section_l2[mask])
        )
        summary["raw_ESE_acceptance_sum"] = (
            summary["raw_einstein_Linf_uv_L2_sphere"]
            + summary["raw_wave_Linf_uv_L2_sphere"]
        )
        summary["masked_ESE_acceptance_sum"] = (
            summary["masked_einstein_Linf_uv_L2_sphere"]
            + summary["masked_wave_Linf_uv_L2_sphere"]
        )
    return ResidualResult(
        coordinate_residual=residual,
        null_components=components,
        pointwise_norm=pointwise,
        section_l2=section_l2,
        einstein_section_l2=einstein_section_l2,
        wave_section_l2=wave_section_l2,
        wave=wave,
        summary=summary,
    )


def evaluate_blocked(
    grid: PointSphereGrid,
    fields: PrimitiveFields,
    u: Array,
    v: Array,
    *,
    stencil: int = 9,
    block_size: int = 5,
    derivative_halo: int | None = None,
    mask_halo: int | None = None,
    coordinates: Any | None = None,
    minimum_s: float | None = None,
) -> dict[str, Any]:
    """Memory-bounded independent audit retaining every coordinate cell."""

    halo = (stencil - 1) if derivative_halo is None else derivative_halo
    safe_halo = halo if mask_halo is None else mask_halo
    section = np.empty((len(u), len(v)), dtype=float)
    einstein_section = np.empty_like(section)
    wave_section = (
        np.empty_like(section) if fields.phi is not None else None
    )
    raw_pointwise = 0.0
    masked_pointwise = 0.0
    component_raw: dict[str, float] = {}
    component_masked: dict[str, float] = {}
    wave_raw = 0.0
    wave_masked = 0.0
    for u_start in range(0, len(u), block_size):
        u_stop = min(len(u), u_start + block_size)
        local_u_start = max(0, u_start - halo)
        local_u_stop = min(len(u), u_stop + halo)
        for v_start in range(0, len(v), block_size):
            v_stop = min(len(v), v_start + block_size)
            local_v_start = max(0, v_start - halo)
            local_v_stop = min(len(v), v_stop + halo)
            local = evaluate(
                grid,
                _slice_fields(
                    fields,
                    slice(local_u_start, local_u_stop),
                    slice(local_v_start, local_v_stop),
                ),
                u[local_u_start:local_u_stop],
                v[local_v_start:local_v_stop],
                stencil=stencil,
                mask_halo=0,
                coordinates=(
                    None
                    if coordinates is None
                    else coordinates.subset(
                        slice(local_u_start, local_u_stop),
                        slice(local_v_start, local_v_stop),
                    )
                ),
            )
            iu = slice(u_start - local_u_start, u_stop - local_u_start)
            iv = slice(v_start - local_v_start, v_stop - local_v_start)
            core_pointwise = local.pointwise_norm[:, iu, iv]
            core_section = local.section_l2[iu, iv]
            section[u_start:u_stop, v_start:v_stop] = core_section
            einstein_section[u_start:u_stop, v_start:v_stop] = (
                local.einstein_section_l2[iu, iv]
            )
            if wave_section is not None:
                assert local.wave_section_l2 is not None
                wave_section[u_start:u_stop, v_start:v_stop] = (
                    local.wave_section_l2[iu, iv]
                )
            raw_pointwise = max(
                raw_pointwise, float(np.max(core_pointwise))
            )
            global_u = np.arange(u_start, u_stop)
            global_v = np.arange(v_start, v_stop)
            safe_u = (global_u >= safe_halo) & (
                global_u < len(u) - safe_halo
            )
            if minimum_s is None:
                safe_v = global_v >= safe_halo
            else:
                if coordinates is None or not hasattr(coordinates, "s"):
                    raise ValueError(
                        "minimum_s requires mapped coordinates with s nodes"
                    )
                safe_v = (
                    np.asarray(coordinates.s)[global_v] >= minimum_s
                )
            safe_v &= global_v < len(v) - safe_halo
            safe = safe_u[:, None] & safe_v[None, :]
            if np.any(safe):
                masked_pointwise = max(
                    masked_pointwise,
                    float(np.max(core_pointwise[:, safe])),
                )
            for name, value in local.null_components.items():
                core = np.abs(value[:, iu, iv])
                component_raw[name] = max(
                    component_raw.get(name, 0.0), float(np.max(core))
                )
                if np.any(safe):
                    component_masked[name] = max(
                        component_masked.get(name, 0.0),
                        float(np.max(core[:, safe])),
                    )
            if local.wave is not None:
                core_wave = np.abs(local.wave[:, iu, iv])
                wave_raw = max(wave_raw, float(np.max(core_wave)))
                if np.any(safe):
                    wave_masked = max(
                        wave_masked, float(np.max(core_wave[:, safe]))
                    )
    safe_mask = np.zeros(section.shape, dtype=bool)
    if 2 * safe_halo < min(section.shape):
        safe_u_global = np.zeros(len(u), dtype=bool)
        safe_u_global[safe_halo : len(u) - safe_halo] = True
        safe_v_global = np.zeros(len(v), dtype=bool)
        if minimum_s is None:
            safe_v_global[safe_halo : len(v) - safe_halo] = True
        else:
            if coordinates is None or not hasattr(coordinates, "s"):
                raise ValueError(
                    "minimum_s requires mapped coordinates with s nodes"
                )
            safe_v_global = np.asarray(coordinates.s) >= minimum_s
            safe_v_global[len(v) - safe_halo :] = False
        safe_mask[:] = safe_u_global[:, None] & safe_v_global[None, :]
        if not np.any(safe_mask):
            raise ValueError("the protected mapped audit mask is empty")
    else:
        safe_mask[:] = True
        masked_pointwise = raw_pointwise
        component_masked = dict(component_raw)
        wave_masked = wave_raw
    summary: dict[str, Any] = {
        "method": (
            "blocked four-g connection difference from primitive "
            "(gamma,logOmega,b,phi); no construction context"
        ),
        "stencil": stencil,
        "derivative_halo": halo,
        "mask_halo": safe_halo,
        "block_size": block_size,
        "minimum_s": minimum_s,
        "raw_pointwise_maximum": raw_pointwise,
        "masked_pointwise_maximum": masked_pointwise,
        "raw_Linf_uv_L2_sphere": float(np.max(section)),
        "masked_Linf_uv_L2_sphere": float(np.max(section[safe_mask])),
        "raw_einstein_Linf_uv_L2_sphere": float(
            np.max(einstein_section)
        ),
        "masked_einstein_Linf_uv_L2_sphere": float(
            np.max(einstein_section[safe_mask])
        ),
        "components": {
            name: {
                "raw_maximum": component_raw[name],
                "masked_maximum": component_masked.get(
                    name, component_raw[name]
                ),
            }
            for name in component_raw
        },
        "section_L2_map": section.tolist(),
        "einstein_section_L2_map": einstein_section.tolist(),
    }
    if fields.phi is not None:
        assert wave_section is not None
        summary["wave_raw_maximum"] = wave_raw
        summary["wave_masked_maximum"] = wave_masked
        summary["raw_wave_Linf_uv_L2_sphere"] = float(
            np.max(wave_section)
        )
        summary["masked_wave_Linf_uv_L2_sphere"] = float(
            np.max(wave_section[safe_mask])
        )
        summary["raw_ESE_acceptance_sum"] = (
            summary["raw_einstein_Linf_uv_L2_sphere"]
            + summary["raw_wave_Linf_uv_L2_sphere"]
        )
        summary["masked_ESE_acceptance_sum"] = (
            summary["masked_einstein_Linf_uv_L2_sphere"]
            + summary["masked_wave_Linf_uv_L2_sphere"]
        )
        summary["wave_section_L2_map"] = wave_section.tolist()
    return summary
