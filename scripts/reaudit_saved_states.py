"""Recompute mapped first-order audits without modifying or re-evolving a run."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from check_saved_constraints import cases
from nee.config import load_config
from nee.diagnostics.state_audit import audit
from nee.io.provenance import runtime_provenance
from nee.io.state_artifact import file_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('immutable diagnostic output already exists')
    summary_path = args.run/'summary.json'
    summary = json.loads(summary_path.read_text())
    if summary.get('terminal_status') != 'completed':
        raise ValueError('source run has not completed its gates')
    public = load_config(args.run/'resolved-config.toml')
    exp = public.experiment.identifier
    if exp not in {'exp04', 'exp08'}:
        raise ValueError('saved mapped re-audits support Experiments 4 and 8; rerun Experiment 7 with its common comparison grid')
    rows = {}
    for path, state, u, v, grid, mesh, _, _ in cases(args.run, public):
        np.testing.assert_array_equal(u, mesh.u)
        np.testing.assert_array_equal(v, mesh.v)
        state_hash = file_hash(path)
        case_summary = path.parent/'summary.json'
        if exp == 'exp04':
            retained = public.angular.retained_degree
            thresholds = tuple(float(mesh.s.nodes[-1])*f for f in (.2, .4, .6, .8))
        else:
            retained = json.loads(case_summary.read_text())['config']['angular']['retained_degree']
            thresholds = (.6,)
        result = audit(grid, state, u, v, coordinates=mesh, retained_degree=retained,
                       protected_s_values=thresholds)
        if file_hash(path) != state_hash:
            raise RuntimeError('source state changed during its audit')
        rows[path.parent.name] = {'state':str(path.relative_to(args.run)), 'state_sha256':state_hash,
                                 'case_summary_sha256':file_hash(case_summary), 'audit':result}
        print(path.parent.name, result['protected'], flush=True)
    if not rows:
        raise ValueError('no complete states available')
    if exp == 'exp04':
        from nee.experiments.exp04_vacuum_strong_short_pulse.campaign import _exact_zero_control_audit
        exact_control = _exact_zero_control_audit(public)
    else:
        exact_control = None
    report = {'schema':'nee-saved-first-order-audits-1', 'experiment':exp,
              'method':'new diagnostics from unchanged saved states; no evolution',
              'terminal_status':'completed', 'source_summary_sha256':file_hash(summary_path),
              'software':runtime_provenance(Path(__file__).resolve().parents[1]),
              'cases':rows, 'exact_zero_control':exact_control}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
