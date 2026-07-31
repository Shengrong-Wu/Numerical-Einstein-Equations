"""Vacuum data with shear on both characteristic faces.

The public state is the official full weighted state.  The repository-local
first-order transport kernel is used only inside one Picard map
``U^(i) -> U^(i+1)``.  Characteristic data are generated and saved
independently of every iterate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[4]

from nee.io.boundary_artifact import load_boundary_data, save_boundary_data  # noqa: E402
from nee.solver.backend import weighted_update_map, weighted_update_norm  # noqa: E402
from nee.state.boundary import BoundaryData
from nee.state.iterate import PicardState as WeightedState  # noqa: E402
from .picard_contract import (  # noqa: E402
    assert_characteristic_traces,
    incoming_trace_errors,
    initial_iterate,
    raw_picard_candidate,
    relaxed_next_iterate,
)
from .reliability import composite_uv_mask  # noqa: E402
from .slab_continuation import (  # noqa: E402
    concatenate_states,
    outgoing_segment,
    slab_boundary,
    terminal_incoming_data,
)


from nee.numerics.spherical_harmonics import AngularGalerkin  # noqa: E402
from nee.numerics.lgl import CompositeLGLMesh  # noqa: E402
from nee.numerics.vacuum_residual import components  # noqa: E402
from nee.numerics.ricci_residual import (  # noqa: E402
    one_form_lie_derivative,
)
from nee.numerics.sphere import (  # noqa: E402
    PointSphereGrid,
    connection_difference,
    gaussian_curvature,
    lie_covariant_tensor,
    scalar_gradient,
    tangent_inverse,
    tensor_divergence,
    tensor_norm_sq,
    tensor_tracefree,
    vector_divergence,
)


Array = np.ndarray


def source_revision() -> str:
    """Hash the maintained driver and the repository-local numerical kernel."""

    digest = hashlib.sha256()
    files = [Path(__file__).resolve()]
    for directory in (
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parents[2] / "numerics",
    ):
        files.extend(sorted(directory.glob("*.py")))
    for path in files:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class DoubleSqrtLGLMesh:
    """Composite LGL mesh in ``t=sqrt(2(u+1))`` and ``s=sqrt(2v)``."""

    t: CompositeLGLMesh
    s: CompositeLGLMesh
    u: Array
    v: Array

    @classmethod
    def create(
        cls,
        t_breakpoints: Array,
        t_degrees: int | Array,
        s_breakpoints: Array,
        s_degrees: int | Array,
    ) -> "DoubleSqrtLGLMesh":
        t = CompositeLGLMesh.create(t_breakpoints, t_degrees)
        s = CompositeLGLMesh.create(s_breakpoints, s_degrees)
        return cls(
            t=t,
            s=s,
            u=-1.0 + 0.5 * t.nodes**2,
            v=0.5 * s.nodes**2,
        )

    @staticmethod
    def _divide_by_corner_coordinate(
        derivative: Array, coordinate: Array, axis: int
    ) -> Array:
        moved = np.moveaxis(derivative, axis, 0)
        result = np.empty_like(moved)
        if coordinate[0] == 0.0:
            shape = (len(coordinate) - 1,) + (1,) * (moved.ndim - 1)
            result[1:] = moved[1:] / coordinate[1:].reshape(shape)
            # The data are only assumed smooth in the square-root coordinate.
            # The physical derivative may diverge at the corner.  Copying the
            # first positive trace keeps array diagnostics finite; all
            # reported protected maxima exclude that endpoint.
            result[0] = result[1]
        else:
            shape = (len(coordinate),) + (1,) * (moved.ndim - 1)
            result[:] = moved / coordinate.reshape(shape)
        return np.moveaxis(result, 0, axis)

    def differentiate_u(self, values: Array, axis: int = 1) -> Array:
        return self._divide_by_corner_coordinate(
            self.t.differentiate(values, axis=axis),
            self.t.nodes,
            axis,
        )

    def differentiate_v(self, values: Array, axis: int = 2) -> Array:
        return self._divide_by_corner_coordinate(
            self.s.differentiate(values, axis=axis),
            self.s.nodes,
            axis,
        )

    def integrate_v(self, source: Array, axis: int = 2) -> Array:
        shape = [1] * source.ndim
        shape[axis] = len(self.s.nodes)
        return self.s.integrate(
            self.s.nodes.reshape(shape) * source,
            axis=axis,
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "coordinate_system": "t=sqrt(2(u+1)), s=sqrt(2v)",
            "t_breakpoints": [
                self.t.segments[0].left,
                *[segment.right for segment in self.t.segments],
            ],
            "t_degrees": [segment.degree for segment in self.t.segments],
            "s_breakpoints": [
                self.s.segments[0].left,
                *[segment.right for segment in self.s.segments],
            ],
            "s_degrees": [segment.degree for segment in self.s.segments],
            "u_count": len(self.u),
            "v_count": len(self.v),
        }


@dataclass(frozen=True)
class Resolution:
    name: str
    elements: int
    degree: int
    retained: int
    work: int
    points: int
    neighbors: int
    sweeps: int
    boundary_substeps: int
    metric_substeps: int
    relaxation: float
    tolerance: float


def rotation_shift(points: Array) -> Array:
    """Return ``0.1 partial_phi`` as an ambient tangent vector."""

    x, y, _ = points.T
    return 0.1 * np.column_stack((-y, x, np.zeros_like(x)))


def quadratic_hessian_tensors(grid: PointSphereGrid) -> tuple[Array, Array]:
    """Return the two explicitly normalized trace-free Hessians.

    For ``f(n)=n^T A n`` with ``tr(A)=0`` on the unit sphere,
    ``tf Hess(f)=2 P A P+fP``.  The exact normalizers below make the
    pointwise tensor norm at most one.
    """

    projector = grid.projector
    points = grid.points
    matrices = (
        np.diag([-0.5, -0.5, 1.0]),
        np.diag([1.0, -1.0, 0.0]),
    )
    normalizers = (math.sqrt(2.0) / 3.0, 1.0 / (2.0 * math.sqrt(2.0)))
    tensors = []
    for matrix, normalizer in zip(matrices, normalizers, strict=True):
        function = np.einsum("ni,ij,nj->n", points, matrix, points)
        raw = (
            2.0
            * np.einsum(
                "nik,kl,nlj->nij",
                projector,
                matrix,
                projector,
            )
            + function[:, None, None] * projector
        )
        tensors.append(normalizer * raw)
    return tensors[0], tensors[1]


def tensor_norm_certificates(
    grid: PointSphereGrid, tensors: tuple[Array, Array]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for index, tensor in enumerate(tensors, start=1):
        norm_sq = np.maximum(
            np.einsum("nij,nij->n", tensor, tensor),
            0.0,
        )
        result[f"X{index}"] = {
            "L2": float(math.sqrt(4.0 * math.pi * np.mean(norm_sq))),
            "Linf": float(math.sqrt(np.max(norm_sq))),
            "maximum_round_trace": float(
                np.max(np.abs(np.einsum("nij,nij->n", grid.projector, tensor)))
            ),
        }
    return result


def _transferred_shear(
    grid: PointSphereGrid,
    reference: Array,
    metric: Array,
    amplitude: float,
) -> Array:
    raw = np.matmul(np.matmul(reference, grid.projector), metric)
    raw = 0.5 * (raw + np.swapaxes(raw, -1, -2))
    inverse = tangent_inverse(grid, metric)
    return amplitude * tensor_tracefree(raw, metric, inverse)


def _rk4_nodes(
    nodes: Array,
    initial: tuple[Array, ...],
    rhs: Callable[[float, tuple[Array, ...]], tuple[Array, ...]],
    substeps: int,
) -> tuple[Array, ...]:
    outputs = [
        np.empty((len(nodes), *value.shape), dtype=float) for value in initial
    ]
    for output, value in zip(outputs, initial, strict=True):
        output[0] = value
    current = tuple(np.asarray(value, dtype=float).copy() for value in initial)
    for index in range(len(nodes) - 1):
        left = float(nodes[index])
        step = float(nodes[index + 1] - nodes[index]) / substeps
        for substep in range(substeps):
            x0 = left + substep * step
            k1 = rhs(x0, current)
            middle = tuple(
                value + 0.5 * step * derivative
                for value, derivative in zip(current, k1, strict=True)
            )
            k2 = rhs(x0 + 0.5 * step, middle)
            middle = tuple(
                value + 0.5 * step * derivative
                for value, derivative in zip(current, k2, strict=True)
            )
            k3 = rhs(x0 + 0.5 * step, middle)
            right = tuple(
                value + step * derivative
                for value, derivative in zip(current, k3, strict=True)
            )
            k4 = rhs(x0 + step, right)
            current = tuple(
                value
                + (step / 6.0)
                * (first + 2.0 * second + 2.0 * third + fourth)
                for value, first, second, third, fourth in zip(
                    current, k1, k2, k3, k4, strict=True
                )
            )
        for output, value in zip(outputs, current, strict=True):
            output[index + 1] = value
    return tuple(np.moveaxis(output, 0, 1) for output in outputs)


def solve_outgoing_face(
    grid: PointSphereGrid,
    v: Array,
    x1: Array,
    *,
    substeps: int,
) -> dict[str, Array]:
    round_metric = grid.projector
    initial_metric = round_metric.copy()
    initial_expansion = np.full(grid.count, 2.0)

    def metric_rhs(
        value_v: float, state: tuple[Array, ...]
    ) -> tuple[Array, ...]:
        metric, expansion = state
        shear = _transferred_shear(
            grid, x1, metric, math.sqrt(max(value_v, 0.0))
        )
        inverse = tangent_inverse(grid, metric)
        shear_norm_sq = tensor_norm_sq(shear, inverse)
        return (
            expansion[:, None, None] * metric + 2.0 * shear,
            -0.5 * expansion**2 - shear_norm_sq,
        )

    metric, expansion = _rk4_nodes(
        v,
        (initial_metric, initial_expansion),
        metric_rhs,
        substeps,
    )
    inverse = tangent_inverse(grid, metric)
    shear = np.stack(
        [
            _transferred_shear(grid, x1, metric[:, index], math.sqrt(value))
            for index, value in enumerate(v)
        ],
        axis=1,
    )
    initial_zeta = np.zeros((grid.count, 3))
    initial_shift = rotation_shift(grid.points)

    def interpolate(field: Array, value_v: float) -> Array:
        if value_v <= v[0]:
            return field[:, 0]
        if value_v >= v[-1]:
            return field[:, -1]
        right = int(np.searchsorted(v, value_v))
        left = right - 1
        fraction = (value_v - v[left]) / (v[right] - v[left])
        return (1.0 - fraction) * field[:, left] + fraction * field[:, right]

    def torsion_rhs(
        value_v: float, state: tuple[Array, ...]
    ) -> tuple[Array, ...]:
        zeta_up, _shift = state
        value_metric = interpolate(metric, value_v)
        value_inverse = tangent_inverse(grid, value_metric)
        value_expansion = interpolate(expansion, value_v)
        value_shear = interpolate(shear, value_v)
        difference, _ = connection_difference(
            grid, value_metric, value_inverse
        )
        divergence = tensor_divergence(
            grid, value_shear, difference, value_inverse
        )
        gradient = scalar_gradient(grid, value_expansion)
        mixed = np.matmul(value_shear, value_metric)
        # Sigma^A_B zeta^B; the first factor is raised with g^{-1}.
        mixed = np.matmul(value_inverse, value_shear)
        derivative_zeta = (
            -2.0 * value_expansion[:, None] * zeta_up
            - 2.0 * np.einsum("nij,nj->ni", mixed, zeta_up)
            + np.einsum("nij,nj->ni", value_inverse, divergence)
            - 0.5 * np.einsum("nij,nj->ni", value_inverse, gradient)
        )
        derivative_shift = -4.0 * zeta_up
        return derivative_zeta, derivative_shift

    zeta_up, shift = _rk4_nodes(
        v,
        (initial_zeta, initial_shift),
        torsion_rhs,
        substeps,
    )
    return {
        "metric": metric,
        "inverse": inverse,
        "expansion": expansion,
        "shear": shear,
        "omega": np.ones_like(expansion),
        "weighted_omega": np.zeros_like(expansion),
        "zeta_up": zeta_up,
        "shift": shift,
    }


def solve_incoming_face(
    grid: PointSphereGrid,
    u: Array,
    x2: Array,
    *,
    substeps: int,
) -> dict[str, Array]:
    round_metric = grid.projector
    shift = rotation_shift(grid.points)
    initial_metric = round_metric.copy()
    initial_in_expansion = np.full(grid.count, -2.0)
    initial_zeta = np.zeros((grid.count, 3))
    initial_out_expansion = np.full(grid.count, 2.0)

    def rhs(
        value_u: float, state: tuple[Array, ...]
    ) -> tuple[Array, ...]:
        metric, in_expansion, zeta, out_expansion = state
        inverse = tangent_inverse(grid, metric)
        shear = _transferred_shear(
            grid, x2, metric, math.sqrt(max(value_u + 1.0, 0.0))
        )
        weighted_chib = (
            0.5 * in_expansion[:, None, None] * metric + shear
        )
        difference, _ = connection_difference(grid, metric, inverse)
        lie_metric = lie_covariant_tensor(grid, shift, metric)
        gradient_in = scalar_gradient(grid, in_expansion)
        advected_in = np.einsum(
            "ni,ni->n", shift, gradient_in
        )
        metric_rhs = (
            in_expansion[:, None, None] * metric
            + 2.0 * shear
            - lie_metric
        )
        in_rhs = (
            -0.5 * in_expansion**2
            - tensor_norm_sq(shear, inverse)
            - advected_in
        )

        divergence_shear = tensor_divergence(
            grid, shear, difference, inverse
        )
        shear_mixed = np.matmul(shear, inverse)
        target_d3_zeta = (
            -1.5 * in_expansion[:, None] * zeta
            - np.einsum("nij,nj->ni", shear_mixed, zeta)
            - divergence_shear
            + 0.5 * gradient_in
        )
        coordinate_zeta_rhs = (
            target_d3_zeta
            - one_form_lie_derivative(grid, shift, zeta)
            + np.einsum(
                "nij,nj->ni",
                np.matmul(weighted_chib, inverse),
                zeta,
            )
        )

        eta_up = np.einsum("nij,nj->ni", inverse, zeta)
        div_eta = vector_divergence(grid, eta_up, difference)
        eta_norm_sq = np.einsum(
            "ni,nij,nj->n", zeta, inverse, zeta
        )
        curvature, _, _ = gaussian_curvature(grid, metric)
        gradient_out = scalar_gradient(grid, out_expansion)
        out_rhs = (
            -out_expansion * in_expansion
            + 2.0 * div_eta
            + 2.0 * eta_norm_sq
            - 2.0 * curvature
            - np.einsum("ni,ni->n", shift, gradient_out)
        )
        return metric_rhs, in_rhs, coordinate_zeta_rhs, out_rhs

    metric, in_expansion, zeta, out_expansion = _rk4_nodes(
        u,
        (
            initial_metric,
            initial_in_expansion,
            initial_zeta,
            initial_out_expansion,
        ),
        rhs,
        substeps,
    )
    inverse = tangent_inverse(grid, metric)
    shear = np.stack(
        [
            _transferred_shear(
                grid, x2, metric[:, index], math.sqrt(value + 1.0)
            )
            for index, value in enumerate(u)
        ],
        axis=1,
    )
    weighted_chib = (
        0.5 * in_expansion[..., None, None] * metric + shear
    )
    zeta_up = np.einsum("nuij,nuj->nui", inverse, zeta)
    return {
        "metric": metric,
        "weighted_tr_chib": in_expansion,
        "weighted_hatchib": shear,
        "weighted_chib": weighted_chib,
        "weighted_tr_chi": out_expansion,
        "omega": np.ones_like(in_expansion),
        "weighted_omegab": np.zeros_like(in_expansion),
        "zeta_up": zeta_up,
        "shift": np.broadcast_to(
            shift[:, None], (grid.count, len(u), 3)
        ).copy(),
    }


def construct_boundary_data(
    grid: PointSphereGrid,
    mesh: DoubleSqrtLGLMesh,
    *,
    substeps: int,
) -> tuple[BoundaryData, dict[str, Any]]:
    tensors = quadratic_hessian_tensors(grid)
    certificates = tensor_norm_certificates(grid, tensors)
    for name, certificate in certificates.items():
        if certificate["L2"] < 0.5 - 1.0e-12:
            raise ValueError(f"{name} violates the L2 lower bound")
        if certificate["Linf"] > 1.0 + 1.0e-12:
            raise ValueError(f"{name} violates the Linf upper bound")
    outgoing = solve_outgoing_face(
        grid, mesh.v, tensors[0], substeps=substeps
    )
    incoming = solve_incoming_face(
        grid, mesh.u, tensors[1], substeps=substeps
    )
    for name in ("metric", "zeta_up", "shift"):
        mismatch = float(
            np.max(np.abs(outgoing[name][:, 0] - incoming[name][:, 0]))
        )
        if mismatch > 2.0e-10:
            raise ValueError(f"corner mismatch in {name}: {mismatch:.3e}")
    return BoundaryData.create(outgoing, incoming), certificates


def _l2_maps(
    grid: PointSphereGrid,
    state: WeightedState,
    values: dict[str, Array],
) -> dict[str, Array]:
    inverse = tangent_inverse(grid, state.metric)
    local_metric = np.einsum(
        "nia,n...ij,njb->n...ab",
        grid.frames,
        state.metric,
        grid.frames,
    )
    area_ratio = np.sqrt(np.maximum(np.linalg.det(local_metric), 0.0))

    def integrate(density_sq: Array) -> Array:
        return np.sqrt(
            np.maximum(
                4.0 * math.pi * np.mean(density_sq * area_ratio, axis=0),
                0.0,
            )
        )

    def form_sq(value: Array) -> Array:
        return np.maximum(
            np.einsum(
                "n...i,n...ij,n...j->n...",
                value,
                inverse,
                value,
            ),
            0.0,
        )

    def tensor_sq(value: Array) -> Array:
        return np.maximum(tensor_norm_sq(value, inverse), 0.0)

    return {
        "r33": integrate(values["Omega2_Ric33"] ** 2),
        "r34": integrate(values["Omega2_Ric34"] ** 2),
        "r3": integrate(form_sq(values["Omega_Ric3A"])),
        "r4": integrate(form_sq(values["Omega_Ric4A"])),
        "rhat": integrate(tensor_sq(values["Omega2_hat_RicAB"])),
        "rR": integrate(values["Omega2_R"] ** 2),
        "r44": integrate(
            (state.omega**2 * values["Ric44_fresh"]) ** 2
        ),
    }


def residual_regions(total: Array) -> Array:
    """Return category indices for R_{<=1}, R_2, R_3, R_4, R_{>=5}."""

    result = np.zeros_like(total, dtype=np.int8)
    result[(total >= 1.0e-2) & (total < 1.0e-1)] = 1
    result[(total >= 1.0e-3) & (total < 1.0e-2)] = 2
    result[(total >= 1.0e-4) & (total < 1.0e-3)] = 3
    result[total < 1.0e-4] = 4
    return result


def plot_regions(
    path: Path,
    mesh: DoubleSqrtLGLMesh,
    total: Array,
    reliable: Array,
) -> dict[str, int]:
    categories = residual_regions(total)
    colors = ["#2b283d", "#6f6689", "#9a89b3", "#ccb8dc", "#f3e8f5"]
    width, height = 1500, 980
    left, right, top, bottom = 150, 370, 100, 130
    plot_width = width - left - right
    plot_height = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=28)
    small = ImageFont.load_default(size=23)
    u_bounds = np.concatenate(
        (
            [mesh.u[0]],
            0.5 * (mesh.u[:-1] + mesh.u[1:]),
            [mesh.u[-1]],
        )
    )
    v_bounds = np.concatenate(
        (
            [mesh.v[0]],
            0.5 * (mesh.v[:-1] + mesh.v[1:]),
            [mesh.v[-1]],
        )
    )

    def pixel_x(value: float) -> int:
        return left + round(plot_width * 2.0 * (value + 1.0))

    def pixel_y(value: float) -> int:
        return top + plot_height - round(plot_height * 2.0 * value)

    # Q is open.  The characteristic-face and terminal-face nodes are not
    # classified as interior convergence regions.
    for i in range(1, len(mesh.u) - 1):
        x0 = pixel_x(float(u_bounds[i]))
        x1 = pixel_x(float(u_bounds[i + 1]))
        for j in range(1, len(mesh.v) - 1):
            y0 = pixel_y(float(v_bounds[j + 1]))
            y1 = pixel_y(float(v_bounds[j]))
            draw.rectangle(
                (x0, y0, x1, y1),
                fill=(
                    colors[int(categories[i, j])]
                    if reliable[i, j]
                    else "#d9d9d9"
                ),
            )
    draw.rectangle(
        (left, top, left + plot_width, top + plot_height),
        outline="black",
        width=3,
    )
    draw.text(
        (left + 70, 30),
        "Six-component vacuum Ricci residual regions",
        fill="black",
        font=font,
    )
    draw.text(
        (left + plot_width // 2 - 10, height - 75),
        "u",
        fill="black",
        font=font,
    )
    draw.text((55, top + plot_height // 2), "v", fill="black", font=font)
    for fraction, label in (
        (0.0, "-1"),
        (0.5, "-0.75"),
        (1.0, "-0.5"),
    ):
        x = left + round(plot_width * fraction)
        draw.line((x, top + plot_height, x, top + plot_height + 12), fill="black")
        draw.text(
            (x - 30, top + plot_height + 20), label, fill="black", font=small
        )
    for fraction, label in ((0.0, "0"), (0.5, "0.25"), (1.0, "0.5")):
        y = top + plot_height - round(plot_height * fraction)
        draw.line((left - 12, y, left, y), fill="black")
        draw.text((55, y - 12), label, fill="black", font=small)
    legend_labels = (
        "R <= 1: r >= 1e-1",
        "R2: 1e-2 <= r < 1e-1",
        "R3: 1e-3 <= r < 1e-2",
        "R4: 1e-4 <= r < 1e-3",
        "R >= 5: r < 1e-4",
        "excluded diagnostic stencil",
    )
    legend_colors = (*colors, "#d9d9d9")
    legend_x = left + plot_width + 55
    for index, (color, label) in enumerate(
        zip(legend_colors, legend_labels, strict=True)
    ):
        y = top + 35 + 80 * index
        draw.rectangle((legend_x, y, legend_x + 48, y + 48), fill=color)
        draw.rectangle(
            (legend_x, y, legend_x + 48, y + 48), outline="black", width=1
        )
        draw.text((legend_x + 65, y + 8), label, fill="black", font=small)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, dpi=(220, 220))
    interior = categories[reliable]
    return {
        label: int(np.count_nonzero(interior == index))
        for index, label in enumerate(
            ("R_le_1", "R_2", "R_3", "R_4", "R_ge_5")
        )
    }


def run_level(
    output: Path,
    resolution: Resolution,
) -> dict[str, Any]:
    breakpoints = np.linspace(0.0, 1.0, resolution.elements + 1)
    mesh = DoubleSqrtLGLMesh.create(
        breakpoints,
        resolution.degree,
        breakpoints,
        resolution.degree,
    )
    grid = PointSphereGrid.create(
        resolution.points,
        neighbor_count=resolution.neighbors,
        degree=min(5, resolution.retained + 1),
        spectral_degree=resolution.work + 1,
    )
    angular = AngularGalerkin(
        grid,
        retained_degree=resolution.retained,
        work_degree=resolution.work,
    )
    boundary, certificates = construct_boundary_data(
        grid,
        mesh,
        substeps=resolution.boundary_substeps,
    )
    boundary_path = output / "boundary-data.npz"
    save_boundary_data(
        boundary_path,
        boundary,
        u=mesh.u,
        v=mesh.v,
    )
    # The iteration consumes the serialized, content-verified artifact rather
    # than retaining an implicit dependency on the generator's live objects.
    boundary, artifact_u, artifact_v = load_boundary_data(boundary_path)
    np.testing.assert_array_equal(artifact_u, mesh.u)
    np.testing.assert_array_equal(artifact_v, mesh.v)
    records: list[dict[str, float | int]] = []
    context: dict[str, Any] | None = None
    slab_states: list[WeightedState] = []
    slab_summaries: list[dict[str, Any]] = []
    incoming = dict(boundary.incoming)
    global_s_breakpoints = np.asarray(
        [
            mesh.s.segments[0].left,
            *[segment.right for segment in mesh.s.segments],
        ]
    )
    t_breakpoints = np.asarray(
        [
            mesh.t.segments[0].left,
            *[segment.right for segment in mesh.t.segments],
        ]
    )
    global_sweep = 0
    for slab_index in range(len(global_s_breakpoints) - 1):
        local_mesh = DoubleSqrtLGLMesh.create(
            t_breakpoints,
            resolution.degree,
            global_s_breakpoints[slab_index : slab_index + 2],
            resolution.degree,
        )
        local_outgoing = outgoing_segment(
            boundary.outgoing,
            mesh.v,
            local_mesh.v,
        )
        local_boundary = slab_boundary(local_outgoing, incoming)
        state = initial_iterate(grid, local_mesh.v, local_boundary)
        slab_records: list[dict[str, float | int]] = []
        for sweep in range(1, resolution.sweeps + 1):
            previous = state
            candidate, context = raw_picard_candidate(
                grid,
                local_mesh,
                angular,
                state,
                local_boundary,
                metric_substeps=resolution.metric_substeps,
            )
            candidate_update = weighted_update_norm(candidate, previous)
            state = relaxed_next_iterate(
                previous,
                candidate,
                local_boundary,
                resolution.relaxation,
            )
            state.validate(grid.frames)
            local_boundary.verify_unchanged()
            assert_characteristic_traces(state, local_boundary)
            global_sweep += 1
            record = {
                "slab": slab_index + 1,
                "slab_sweep": sweep,
                "sweep": global_sweep,
                "unrelaxed_candidate_update": candidate_update,
                "weighted_update": weighted_update_norm(state, previous),
                "maximum_update_map": float(
                    np.max(weighted_update_map(state, previous))
                ),
            }
            records.append(record)
            slab_records.append(record)
            if record["weighted_update"] <= resolution.tolerance:
                break
        if context is None:
            raise AssertionError("at least one Picard sweep is required")
        terminal_update = float(slab_records[-1]["weighted_update"])
        if terminal_update > resolution.tolerance:
            raise RuntimeError(
                f"slab {slab_index + 1} did not settle: "
                f"update={terminal_update:.6e}, "
                f"tolerance={resolution.tolerance:.6e}"
            )
        slab_states.append(state)
        slab_summaries.append(
            {
                "slab": slab_index + 1,
                "v_interval": [
                    float(local_mesh.v[0]),
                    float(local_mesh.v[-1]),
                ],
                "boundary_digest": local_boundary.digest,
                "terminal_weighted_update": terminal_update,
                "settled": True,
                "incoming_trace_errors": incoming_trace_errors(
                    state, local_boundary
                ),
                "last_raw_trace_errors_before_restore": context[
                    "trace_errors_before_restore"
                ],
                "last_raw_trace_errors_after_restore": context[
                    "trace_errors_after_restore"
                ],
            }
        )
        print(
            f"{resolution.name}: settled slab {slab_index + 1}/"
            f"{len(global_s_breakpoints) - 1} at update "
            f"{terminal_update:.3e}",
            flush=True,
        )
        incoming = terminal_incoming_data(state)

    state = concatenate_states(slab_states)
    state.validate(grid.frames)
    assert_characteristic_traces(state, boundary)
    np.testing.assert_allclose(
        np.concatenate(
            [
                (
                    0.5
                    * CompositeLGLMesh.create(
                        global_s_breakpoints[index : index + 2],
                        resolution.degree,
                    ).nodes**2
                )
                if index == 0
                else (
                    0.5
                    * CompositeLGLMesh.create(
                        global_s_breakpoints[index : index + 2],
                        resolution.degree,
                    ).nodes[1:] ** 2
                )
                for index in range(len(global_s_breakpoints) - 1)
            ]
        ),
        mesh.v,
        atol=5.0e-15,
        rtol=0.0,
    )
    values = components(
        grid,
        state,
        mesh.u,
        mesh.v,
        mode="fresh",
        scalar_coordinates=mesh,
        include_gauss_curvature=True,
    )
    maps = _l2_maps(grid, state, values)
    requested_names = ("r33", "r34", "r3", "r4", "rhat", "rR")
    total = sum(maps[name] for name in requested_names)
    full_total = total + maps["r44"]
    halo = max(1, resolution.degree // 3)
    protected = composite_uv_mask(
        total.shape[0],
        total.shape[1],
        element_degree=resolution.degree,
        interface_halo=halo,
    )
    open_grid = np.zeros(total.shape, dtype=bool)
    open_grid[1:-1, 1:-1] = True
    maxima = {
        name: {
            "all_nodes_including_characteristic_faces": float(np.max(value)),
            "open_grid": float(np.max(value[open_grid])),
            "protected_interior": float(np.max(value[protected])),
        }
        for name, value in {**maps, "r": total, "r_with_r44": full_total}.items()
    }
    state.save(
        output / "final-state.npz",
        u=mesh.u,
        v=mesh.v,
        extra={
            **{f"residual_{name}": value for name, value in maps.items()},
            "residual_requested_sum": total,
            "residual_sum_with_r44": full_total,
            "residual_reliability_mask": protected,
        },
    )
    np.savez_compressed(
        output / "residual-maps.npz",
        u=mesh.u,
        v=mesh.v,
        **maps,
        r=total,
        r_with_r44=full_total,
        reliability_mask=protected,
    )
    region_counts = plot_regions(
        output / "convergence-regions.png", mesh, total, protected
    )
    summary = {
        "schema": "nee-official-experiment-04b-level-v1",
        "case_id": resolution.name,
        "resolution": resolution.__dict__,
        "mesh": mesh.diagnostics(),
        "angular": angular.diagnostics(),
        "tensor_certificates": certificates,
        "boundary_digest": boundary.digest,
        "incoming_trace_errors": incoming_trace_errors(state, boundary),
        "slabs": slab_summaries,
        "last_raw_trace_errors_before_restore": context[
            "trace_errors_before_restore"
        ],
        "last_raw_trace_errors_after_restore": context[
            "trace_errors_after_restore"
        ],
        "picard": records,
        "residual_semantics": {
            "r": "r33+r34+r3+r4+rhat+rR",
            "r4_weight": "Omega Ric_4A (the displayed identity's weight)",
            "r44": (
                "fresh independent Raychaudhuri residual, reported separately "
                "and not included in the requested six-term region plot"
            ),
            "protected_halo_nodes": halo,
            "reliability_mask": (
                "exclude the complete first and last coordinate elements "
                "and the stated halo on both sides of every remaining "
                "composite-element interface"
            ),
            "reliability_node_count": int(np.count_nonzero(protected)),
        },
        "residual_maxima": maxima,
        "region_counts": region_counts,
        "minimum_metric_eigenvalue": state.validate(grid.frames)[
            "minimum_metric_eigenvalue"
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def production_resolution() -> Resolution:
    """Return the reported 61x61x300, retained-L=8 configuration."""

    return Resolution(
        "level-3",
        6,
        10,
        8,
        14,
        300,
        40,
        30,
        12,
        6,
        1.0,
        1.0e-8,
    )


def run(
    output: Path,
    *,
    quick: bool,
    reported_only: bool = False,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"immutable run directory exists: {output}")
    output.mkdir(parents=True)
    revision = source_revision()
    campaign_levels = (
        Resolution(
            "level-1", 6, 4, 3, 6, 100, 24, 20, 4, 2, 1.0, 1.0e-8
        ),
        Resolution(
            "level-2", 6, 5, 4, 8, 180, 32, 20, 6, 3, 1.0, 1.0e-8
        ),
        (
            Resolution(
                "level-3", 6, 7, 5, 10, 260, 36, 20, 8, 4, 1.0, 1.0e-8
            )
            if quick
            else production_resolution()
        ),
    )
    levels = (
        (production_resolution(),)
        if reported_only
        else campaign_levels
    )
    summaries = []
    for resolution in levels:
        summaries.append(run_level(output / resolution.name, resolution))
    high = summaries[-1]
    aggregate = {
        "schema": "nee-vacuum-crossed-pulses-aggregate-1",
        "experiment": 5,
        "configuration": "crossed characteristic shears",
        "domain": {
            "u": [-1.0, -0.5],
            "v": [0.0, 0.5],
        },
        "levels": summaries,
        "reported_level": high["case_id"],
        "reported_summary": high,
        "source_revision": revision,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    (output / "aggregate-summary.json").write_text(
        json.dumps(aggregate, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--reported-only",
        action="store_true",
        help="run only the reported 61x61x300, retained-L=8 configuration",
    )
    args = parser.parse_args()
    run(
        args.output,
        quick=args.quick,
        reported_only=args.reported_only,
    )


if __name__ == "__main__":
    main()
