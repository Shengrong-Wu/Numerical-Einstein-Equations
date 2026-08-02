import numpy as np

from nee.numerics.lgl import (
    CharacteristicLGLMesh,
    CompositeLGLMesh,
    LGLSegment,
)
from nee.numerics.sphere import (
    PointSphereGrid,
    tangent_inverse,
    tensor_trace,
    tensor_tracefree,
)


def test_lgl_differentiates_polynomials_exactly() -> None:
    segment = LGLSegment.create(-0.7, 1.2, 7)
    values = segment.nodes**6 - 2.0 * segment.nodes**3 + 1.0
    expected = 6.0 * segment.nodes**5 - 6.0 * segment.nodes**2
    np.testing.assert_allclose(segment.derivative @ values, expected, atol=3e-12)


def test_lgl_integrates_polynomials_exactly() -> None:
    segment = LGLSegment.create(0.2, 1.4, 6)
    values = 3.0 * segment.nodes**4 - segment.nodes + 2.0
    antiderivative = 0.6 * segment.nodes**5 - 0.5 * segment.nodes**2 + 2.0 * segment.nodes
    expected = antiderivative - antiderivative[0]
    np.testing.assert_allclose(segment.integral @ values, expected, atol=2e-13)


def test_composite_interface_derivative_average() -> None:
    mesh = CompositeLGLMesh.create(np.array([0.0, 0.3, 1.0]), np.array([5, 5]))
    values = mesh.nodes**4
    derivative = mesh.differentiate(values, interface_rule="average")
    np.testing.assert_allclose(derivative, 4.0 * mesh.nodes**3, atol=3e-12)


def test_characteristic_lgl_mesh_declares_square_root_exponent() -> None:
    mesh = CharacteristicLGLMesh.create(
        np.asarray([0.0, 0.5]),
        3,
        np.asarray([0.0, 0.25]),
        3,
        0.5,
    )

    assert mesh.delta == 0.5


def test_tracefree_projection_is_tangent_symmetric_and_traceless() -> None:
    grid = PointSphereGrid.create(50, neighbor_count=24, degree=4, spectral_degree=6)
    rng = np.random.default_rng(4)
    raw = rng.normal(size=(grid.count, 3, 3))
    raw = 0.5 * (raw + np.swapaxes(raw, -1, -2))
    raw = np.einsum("nik,nkl,nlj->nij", grid.projector, raw, grid.projector)
    g = grid.projector.copy()
    inverse_g = tangent_inverse(grid, g)
    projected = tensor_tracefree(raw, g, inverse_g)
    np.testing.assert_allclose(projected, np.swapaxes(projected, -1, -2), atol=2e-14)
    np.testing.assert_allclose(np.einsum("ni,nij->nj", grid.points, projected), 0.0, atol=3e-14)
    np.testing.assert_allclose(tensor_trace(projected, inverse_g), 0.0, atol=3e-14)
