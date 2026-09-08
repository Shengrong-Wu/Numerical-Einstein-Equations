"""Check first-derivative consistency of completed pulse states without evolution."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from nee.config import load_config
from nee.diagnostics.construction_residuals import evaluate
from nee.diagnostics.convergence_regions import composite_uv_mask
from nee.discretization.double_null_mesh import DoubleSqrtLGLMesh
from nee.io.state_artifact import load_state, file_hash
from nee.io.provenance import runtime_provenance
from nee.numerics.scalar_coordinates import CharacteristicPowerMesh, mesh_from_config
from nee.numerics.scalar_config import ExperimentConfig
from nee.numerics.sphere import PointSphereGrid


def cases(run, public):
    exp = public.experiment.identifier
    if exp == 'exp04':
        if public.experiment.mode != 'standard':
            raise ValueError('the saved Experiment 4 mesh check requires standard mode')
        from nee.experiments.exp04_vacuum_strong_short_pulse.campaign import _audit_mesh
        from nee.numerics.vacuum_state_io import load_state as load_numerical
        from nee.solver.backend import from_numerical
        grid, mesh = _audit_mesh(public)
        for name in ('strong-pulse', 'zero-control'):
            path = run/'data'/'results'/'Q1'/name/'final-state.npz'
            numerical, u, v = load_numerical(path)
            yield path, from_numerical(numerical, grid), u, v, grid, mesh, None, None
    elif exp == 'exp05':
        for directory in sorted((run/'data').iterdir()):
            if not (directory/'final-state.npz').is_file():
                continue
            info = json.loads((directory/'summary.json').read_text())['resolution']
            breaks = np.linspace(0, 1, info['elements']+1)
            mesh = DoubleSqrtLGLMesh.create(breaks, info['degree'], breaks, info['degree'])
            grid = PointSphereGrid.create(info['points'], neighbor_count=info['neighbors'],
                degree=min(5, info['retained']+1), spectral_degree=info['work']+1)
            path = directory/'final-state.npz'
            state, u, v, _ = load_state(path)
            mask = composite_uv_mask(len(u), len(v), element_degree=info['degree'], interface_halo=max(1,info['degree']//3))
            yield path, state, u, v, grid, mesh, mask, None
    elif exp in {'exp07', 'exp08'}:
        parent = run/'data'/('cases' if exp == 'exp07' else 'patches')
        for path in sorted(parent.glob('*/final-state.npz')):
            info = json.loads((path.parent/'summary.json').read_text())
            state, u, v, _ = load_state(path)
            if exp == 'exp07':
                c = info['scalar_coordinates']
                mesh = CharacteristicPowerMesh.create(np.array(c['tau_breakpoints']),np.array(c['tau_degrees']),
                    np.array(c['s_breakpoints']),np.array(c['s_degrees']),c['v_max'],c['fractional_power'])
                work = info['angular']['work_degree']
                neighbors = public.angular.neighbor_count
            else:
                config = ExperimentConfig.from_dict(info['config'])
                mesh = mesh_from_config(config.scalar_coordinates)
                work = config.angular.work_degree
                neighbors = config.angular.neighbor_count
            grid = PointSphereGrid.create(state.g.shape[0], neighbor_count=neighbors, spectral_degree=work+1)
            yield path, state, u, v, grid, mesh, None, 0.6
    else:
        raise ValueError('saved pulse consistency checks support exp04 through exp08, excluding exp06')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('the consistency report already exists')
    summary = json.loads((args.run/'summary.json').read_text())
    if summary.get('terminal_status') != 'completed':
        raise ValueError('the source experiment has not completed its gates')
    public = load_config(args.run/'resolved-config.toml')
    rows = []
    for path, state, u, v, grid, mesh, mask, minimum_s in cases(args.run, public):
        np.testing.assert_array_equal(u, mesh.u)
        np.testing.assert_array_equal(v, mesh.v)
        fingerprint = file_hash(path)
        closures = evaluate(grid,state,u,v,coordinates=mesh,halo=2,
                            reliability_mask=mask,protected_s_minimum=minimum_s)
        if fingerprint != file_hash(path):
            raise RuntimeError('the saved state changed during its read-only audit')
        rows.append({'state':str(path.relative_to(args.run)), 'state_sha256':fingerprint,
                     'closures':closures})
        print(path.parent.name, flush=True)
    if not rows:
        raise ValueError('no saved states were checked')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    report = {'method':'native-grid first derivatives; no evolution or second null derivatives',
              'source_summary_sha256':file_hash(args.run/'summary.json'),
              'software':runtime_provenance(Path(__file__).resolve().parents[1]),'cases':rows}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
