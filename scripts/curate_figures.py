"""Regenerate the three article figures from explicitly selected run outputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import inspect
import shutil
import tempfile
from pathlib import Path

from render_exp05_residual_spectrum import render as render_exp05
from render_exp07_curvature_spectrum import render as render_exp07
from nee.experiments.exp08_scalar_trapped_section.campaign import _summary_figure
from nee.io.provenance import runtime_provenance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('docs/experiments'))
    parser.add_argument('--experiments', nargs='+', choices=('exp05', 'exp07', 'exp08'), default=('exp05', 'exp07', 'exp08'))
    args = parser.parse_args()
    provenance = {}
    for exp in args.experiments:
        source = args.results_root/exp
        summary_path = source/'summary.json'
        summary = json.loads(summary_path.read_text())
        if summary.get('terminal_status') != 'completed':
            raise ValueError(f'{exp} has not completed its declared gates')
        destination = args.output_root/exp/'results'
        destination.mkdir(parents=True, exist_ok=True)
        inputs = [summary_path]
        if exp == 'exp05':
            inputs.extend([source/'data'/'level-3'/'summary.json', source/'data'/'level-3'/'residual-maps.npz'])
            render_exp05(source/'data'/'level-3', destination/'residual-spectrum.png', samples_per_element=96)
            extra = source/'data'/'level-3'/'convergence-regions.png'
            shutil.copy2(extra, destination/extra.name)
            inputs.append(extra)
        elif exp == 'exp07':
            inputs.append(source/'data'/'cases'/'coordinate-level-2'/'summary.json')
            render_exp07(source/'data'/'cases'/'coordinate-level-2'/'summary.json', destination/'curvature-residual-spectrum.png', samples_per_element=96)
        else:
            if not summary['apparent_horizon']['verified']:
                raise ValueError('the horizon has unverified sections')
            extra = source/'data'/'apparent-horizon'/'degree4'/'apparent-horizon.png'
            shutil.copy2(extra, destination/extra.name)
            inputs.append(extra)
            with tempfile.TemporaryDirectory(prefix='nee-figure-') as temporary:
                _summary_figure(Path(temporary), summary)
                shutil.copy2(Path(temporary)/'figures'/'curved-domain-and-horizon.png', destination/'curved-domain-and-horizon.png')
        provenance[exp] = {'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                           'software': json.loads((source/'manifest.json').read_text())['software'],
                           'generator_software': runtime_provenance(Path(__file__).resolve().parents[1]),
                           'source_artifacts': {str(path.relative_to(source)):hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
                           'generator_scripts': {path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (Path(__file__), Path(__file__).with_name('residual_plotting.py'), Path(__file__).with_name('render_exp05_residual_spectrum.py'), Path(__file__).with_name('render_exp07_curvature_spectrum.py'), Path(inspect.getfile(_summary_figure)))}}
        (destination/'figure-provenance.json').write_text(json.dumps(provenance[exp], indent=2)+'\n')


if __name__ == '__main__':
    main()
