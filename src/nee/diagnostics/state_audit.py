"""Independent differentiation of the complete weighted state on an overgrid.

No null derivative of a metric-derived connection, or outgoing derivative of
outgoing shear, is used. Intrinsic angular curvature uses the smooth section
metric. Endpoint values of singular coordinate Jacobians are never reported.
"""
from __future__ import annotations

import numpy as np

from nee.discretization.overgrid import (
    _coordinate_resample, _composite_interpolate,
    _project_tensor, _project_vector, resample_primitives, resample_primitives_power,
    TypedAngularResampler,
)
from nee.state.fields import PrimitiveFields
from nee.state.iterate import PicardState
from nee.diagnostics.ricci_components import components, section_maps
from nee.diagnostics.construction_residuals import evaluate as closure_audit


def audit(grid, state, u, v, *, retained_degree, coordinates=None,
          protected_s_values=(0.6, 0.7, 0.8), point_count=None,
          degree_increment=3, coordinate_increment=4, halo=3,
          audit_operators=None):
    if not isinstance(state, PicardState):
        from nee.solver.backend import from_numerical
        state = from_numerical(state, grid)
    primitives = PrimitiveFields(g=state.g, log_Omega=state.log_Omega, b=state.b, phi=state.phi)
    # Full weighted forms include products such as (Omega trchi)*g.
    # A scalar degree-L fit of ambient tensor components discards geometric
    # modes on regular grids as well as mapped ones.
    transfer_degree = min(2*retained_degree, int(np.sqrt(grid.count-1))-2)
    if transfer_degree < retained_degree:
        raise ValueError('source sphere cannot resolve the retained tensor audit band')
    angular_transfer = TypedAngularResampler(transfer_degree)
    count = point_count or max(grid.count+12, (transfer_degree+2)**2+8)
    if coordinates is None:
        target = resample_primitives(grid, primitives, u, v,
            u_count=len(u)+coordinate_increment, v_count=len(v)+coordinate_increment,
            point_count=count, harmonic_degree=transfer_degree,
            angular_transfer=angular_transfer,
            differentiation_degree=transfer_degree+1)
    else:
        target = resample_primitives_power(grid, primitives, coordinates,
            u_count=len(u)+coordinate_increment, v_count=len(v)+coordinate_increment,
            point_count=count, harmonic_degree=transfer_degree,
            spectral_degree_increment=degree_increment,
            angular_transfer=angular_transfer,
            differentiation_degree=transfer_degree+1,
            audit_operators=audit_operators)
    target.diagnostics['source_retained_degree'] = retained_degree
    target.diagnostics['differentiation_degree'] = transfer_degree+1

    def transfer(value):
        if coordinates is None:
            value = _coordinate_resample(value, u, v, target.u, target.v)
        else:
            value = _composite_interpolate(coordinates.tau, value, target.coordinates.tau, axis=1)
            value = _composite_interpolate(coordinates.s, value, target.coordinates.s, axis=2)
        return angular_transfer(value, grid, target.grid, retained_degree)[0]

    arrays = {name: transfer(value) for name, value in state.arrays().items()
              if name not in {'g', 'b', 'log_Omega', 'phi'}}
    for name in ('Omega_chi', 'Omega_chib'):
        arrays[name] = _project_tensor(target.grid, arrays[name])
    for name in ('zeta', 'nabla_phi'):
        if name in arrays:
            arrays[name] = _project_vector(target.grid, arrays[name])
    current = PicardState(g=target.fields.g, b=target.fields.b,
        log_Omega=target.fields.log_Omega, phi=target.fields.phi, **arrays)
    current.validate(target.grid.frames)
    values = components(target.grid, current, target.u, target.v, coordinates=target.coordinates)
    maps = section_maps(target.grid, current, target.u, values, scale_invariant=False)
    wave = maps.get('box_phi', np.zeros_like(maps['combined']))
    weights = {'Ric34': 2.0, 'E34': 2.0, 'trace_RicAB': 0.5, 'trace_EAB': 0.5}
    einstein = np.sqrt(sum(weights.get(name, 1.0)*value**2 for name, value in maps.items() if name not in {'combined', 'box_phi'}))
    total = np.hypot(einstein, wave)
    from nee.diagnostics.independent_audit import positive_null_norm
    prefix = 'E' if current.is_scalar else 'Ric'
    omega2 = current.Omega**2
    trace_key = 'Omega2_trace_EAB' if current.is_scalar else 'Omega2_R_plus_Ric34'
    null = {'33': values['Omega2_'+prefix+'33']/omega2,
            '44': values[prefix+'44'], '34': values['Omega2_'+prefix+'34']/omega2,
            '3A': values['Omega_'+prefix+'3A']/current.Omega[..., None],
            '4A': values['Omega_'+prefix+'4A']/current.Omega[..., None],
            'AB': (values['Omega2_hat_'+prefix+'AB'] + 0.5*values[trace_key][..., None, None]*current.g)/omega2[..., None, None]}
    pointwise = positive_null_norm(null, current.inverse_g)
    if current.is_scalar:
        pointwise = np.hypot(pointwise, values['minus_Omega2_box_phi']/omega2)
    open_mask = np.zeros(total.shape, dtype=bool)
    open_mask[1:-1, 1:-1] = True
    h = min(halo, max(1, (min(total.shape)-1)//4))
    safe_u, safe_v = np.zeros(len(target.u), bool), np.zeros(len(target.v), bool)
    safe_u[h:-h] = True
    safe_v[h:-h] = True
    if coordinates is not None:
        tau_mask_mesh, s_mask_mesh = ((coordinates.tau, coordinates.s)
                                    if audit_operators is None else audit_operators)
        for mesh, nodes, safe in ((tau_mask_mesh, target.coordinates.tau, safe_u),
                                  (s_mask_mesh, target.coordinates.s, safe_v)):
            for segment in mesh.segments[:-1]:
                center = int(np.argmin(np.abs(nodes-segment.right)))
                safe[max(0, center-h):min(len(safe), center+h+1)] = False
    mask = safe_u[:, None] & safe_v[None, :]

    def summary(selected):
        if not np.any(selected):
            raise ValueError('first-order audit has an empty reliability region')
        if not all(np.all(np.isfinite(value[selected])) for value in maps.values()):
            raise FloatingPointError('nonfinite first-order residual in the reliable region')
        result = {'cell_count': int(np.sum(selected)), 'pointwise_maximum': float(np.max(pointwise[:, selected])),
            'combined_Linf_uv_L2_sphere': float(np.max(total[selected])),
            'einstein_Linf_uv_L2_sphere': float(np.max(einstein[selected])),
            'components': {name: float(np.max(value[selected])) for name, value in maps.items()}}
        if current.is_scalar:
            result['wave_Linf_uv_L2_sphere'] = float(np.max(wave[selected]))
            result['ESE_acceptance_sum'] = result['einstein_Linf_uv_L2_sphere'] + result['wave_Linf_uv_L2_sphere']
        return result

    raw, masked = summary(open_mask), summary(mask)
    closure_mask = mask.copy()
    if coordinates is not None:
        closure_mask &= target.coordinates.s[None, :] >= min(protected_s_values)
    result = {'method': 'first null derivatives of independently resampled weighted state; no construction sources',
        'norm': 'sphere L2 of Euclidean sum of squared null component norms',
        'endpoint_policy': 'exclude both characteristic endpoints; intrinsic angular derivatives use smooth sphere data',
        'overgrid': target.diagnostics,
        'raw_pointwise_maximum': raw['pointwise_maximum'],
        'masked_pointwise_maximum': masked['pointwise_maximum'],
        'raw_Linf_uv_L2_sphere': raw['combined_Linf_uv_L2_sphere'],
        'masked_Linf_uv_L2_sphere': masked['combined_Linf_uv_L2_sphere'],
        'raw_einstein_Linf_uv_L2_sphere': raw['einstein_Linf_uv_L2_sphere'],
        'masked_einstein_Linf_uv_L2_sphere': masked['einstein_Linf_uv_L2_sphere'],
        'section_L2_map': total.tolist(), 'einstein_section_L2_map': einstein.tolist(),
        'wave_section_L2_map': wave.tolist() if current.is_scalar else None,
        'components': masked['components'],
        'metric_connection_closures': closure_audit(target.grid, current, target.u, target.v,
                                                   coordinates=target.coordinates, halo=h, reliability_mask=closure_mask)}
    result['metric_connection_closures']['mask_cell_count'] = int(np.sum(closure_mask))
    if coordinates is not None:
        result['metric_connection_closures']['minimum_s'] = float(min(protected_s_values))
    if current.is_scalar:
        for name, row in (('raw', raw), ('masked', masked)):
            result[name+'_wave_Linf_uv_L2_sphere'] = row['wave_Linf_uv_L2_sphere']
            result[name+'_ESE_acceptance_sum'] = row['ESE_acceptance_sum']
    if coordinates is not None:
        result['protected'] = {f's_ge_{minimum:.2f}': summary(mask & (target.coordinates.s[None, :] >= minimum))
                               for minimum in protected_s_values}
    return result
