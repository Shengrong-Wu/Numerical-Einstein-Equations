import numpy as np
from nee.discretization.overgrid import TypedAngularResampler, _angular_resample
from nee.numerics.sphere import PointSphereGrid


def fields(grid):
    z = grid.points[:, 2]
    polynomial = (231*z**6 - 315*z**4 + 105*z**2 - 5)/16
    derivative = (1386*z**5 - 1260*z**3 + 210*z)/16
    gradient = derivative[:, None]*(np.array([0.,0.,1.])-z[:,None]*grid.points)
    return (polynomial[:,None,None], gradient[:,None,None,:],
            polynomial[:,None,None,None,None]*grid.projector[:,None,None,:,:])


def test_typed_audit_keeps_highest_vector_and_tensor_modes():
    source = PointSphereGrid.create(100, spectral_degree=8)
    target = PointSphereGrid.create(112, spectral_degree=8)
    transfer = TypedAngularResampler(6)
    for value, exact in zip(fields(source), fields(target), strict=True):
        result, condition = transfer(value, source, target, 6)
        assert condition < 5
        np.testing.assert_allclose(result, exact, rtol=0, atol=2e-11)
    # A degree-six scalar fit cannot represent this degree-six tensor's
    # degree-eight ambient components, even on these well-resolved grids.
    wrong, _ = _angular_resample(fields(source)[2], source, target, 6)
    assert np.max(np.abs(wrong-fields(target)[2])) > 1e-2


def test_common_audit_grid_preserves_polynomials_from_different_source_meshes():
    import math
    from nee.discretization.overgrid import resample_primitives_power
    from nee.numerics.scalar_coordinates import CharacteristicPowerMesh
    from nee.numerics.lgl import CompositeLGLMesh
    from nee.state.fields import PrimitiveFields
    grid = PointSphereGrid.create(100, spectral_degree=8)
    fractions = np.unique(np.concatenate([np.linspace(0, 1, n+1) for n in (2, 3)]))
    operators = (CompositeLGLMesh.create(math.log(2)*fractions, 5),
                 CompositeLGLMesh.create(fractions, 5))
    results = []
    for elements in (2, 3):
        mesh = CharacteristicPowerMesh.create(np.linspace(0, math.log(2), elements+1), 4,
            np.linspace(0, 1, elements+1), 4, .04, .1)
        shape = (grid.count, len(mesh.u), len(mesh.v))
        phi = np.broadcast_to(mesh.tau.nodes[None,:,None]**3 + mesh.s.nodes[None,None,:]**2, shape)
        fields = PrimitiveFields(g=np.broadcast_to(grid.projector[:,None,None,:,:], shape+(3,3)),
            log_Omega=np.zeros(shape), b=np.zeros(shape+(3,)), phi=phi)
        result = resample_primitives_power(grid, fields, mesh, u_count=0, v_count=0,
            point_count=112, harmonic_degree=4, angular_transfer=TypedAngularResampler(4),
            differentiation_degree=5, audit_operators=operators)
        results.append(result)
        exact = result.coordinates.tau[None,:,None]**3 + result.coordinates.s[None,None,:]**2
        np.testing.assert_allclose(result.fields.phi, np.broadcast_to(exact,result.fields.phi.shape), atol=1e-12, rtol=0)
    left, right = results
    np.testing.assert_array_equal(left.u, right.u)
    np.testing.assert_array_equal(left.v, right.v)
    safe = left.coordinates.s >= .6
    np.testing.assert_allclose(left.coordinates.differentiate_v(left.fields.phi)[:,:,safe],
                               right.coordinates.differentiate_v(right.fields.phi)[:,:,safe], rtol=1e-10, atol=1e-9)
