"""Independent four-g audit on native power-coordinate LGL meshes."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from nee.diagnostics.independent_audit import (
    PrimitiveFields,
    evaluate,
    summarize_result_on_mask,
)
from nee.discretization.overgrid import resample_primitives_power


def evaluate_mapped_overgrid(
    grid: Any,
    fields: PrimitiveFields,
    coordinates: Any,
    *,
    retained_degree: int,
    protected_s_values: tuple[float, ...],
) -> dict[str, Any]:
    """Audit primitives on a higher-degree mapped-coordinate overgrid."""

    if not protected_s_values:
        raise ValueError("at least one protected s threshold is required")
    if any(not math.isfinite(value) for value in protected_s_values) or any(
        right <= left
        for left, right in zip(
            protected_s_values, protected_s_values[1:], strict=False
        )
    ):
        raise ValueError(
            "protected s thresholds must be finite and strictly increasing"
        )
    point_count = (retained_degree + 1) ** 2 + 8
    overgrid = resample_primitives_power(
        grid,
        fields,
        coordinates,
        u_count=len(coordinates.u) + 4,
        v_count=len(coordinates.v) + 4,
        point_count=point_count,
        harmonic_degree=retained_degree,
        stencil=7,
        spectral_degree_increment=3,
    )
    result = evaluate(
        overgrid.grid,
        overgrid.fields,
        overgrid.u,
        overgrid.v,
        stencil=7,
        mask_halo=3,
        coordinates=overgrid.coordinates,
    )
    target = overgrid.coordinates
    assert target is not None
    safe_u = np.ones(len(target.u), dtype=bool)
    safe_v_base = np.ones(len(target.v), dtype=bool)
    outer_halo = 3
    interface_halo = 3
    safe_u[:outer_halo] = False
    safe_u[-outer_halo:] = False
    safe_v_base[-outer_halo:] = False
    for source_mesh, target_nodes, safe in (
        (coordinates.tau, target.tau, safe_u),
        (coordinates.s, target.s, safe_v_base),
    ):
        for segment in source_mesh.segments[:-1]:
            center = int(np.argmin(np.abs(target_nodes - segment.right)))
            safe[
                max(0, center - interface_halo) : min(
                    len(safe), center + interface_halo + 1
                )
            ] = False
    protected: dict[str, Any] = {}
    for minimum_s in protected_s_values:
        safe_v = safe_v_base & (target.s >= minimum_s)
        mask = safe_u[:, None] & safe_v[None, :]
        protected[f"s_ge_{minimum_s:.2f}"] = summarize_result_on_mask(
            result,
            mask,
            label=(
                f"s >= {minimum_s:.2f}, outer halo {outer_halo}, "
                f"interface halo {interface_halo}"
            ),
        )
    summary = dict(result.summary)
    summary.update(
        {
            "method": (
                "four-g connection difference on an independent "
                "higher-degree (tau,s) LGL overgrid"
            ),
            "overgrid": overgrid.diagnostics,
            "protected": protected,
            "primary_protected_region": (
                f"s_ge_{protected_s_values[0]:.2f}"
            ),
            "section_L2_map": result.section_l2.tolist(),
            "einstein_section_L2_map": (
                result.einstein_section_l2.tolist()
            ),
            "wave_section_L2_map": (
                None
                if result.wave_section_l2 is None
                else result.wave_section_l2.tolist()
            ),
        }
    )
    return summary
