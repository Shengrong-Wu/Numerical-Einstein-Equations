"""Regenerate the three article figures from explicitly selected run outputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
from pathlib import Path

from render_exp05_residual_spectrum import render as render_exp05
from render_exp07_curvature_spectrum import render as render_exp07
from nee.experiments.exp08_scalar_trapped_section.campaign import _summary_figure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, default=Path('docs/experiments'))
    args = parser.parse_args()
    provenance = {}
    for exp in ('exp05', 'exp07', 'exp08'):
        source = args.results_root/exp
        summary_path = source/'summary.json'
        summary = json.loads(summary_path.read_text())
        if summary.get('terminal_status') != 'completed':
            raise ValueError(f'{exp} has not completed its declared gates')
        destination = args.output_root/exp/'results'
        destination.mkdir(parents=True, exist_ok=True)
        if exp == 'exp05':
            render_exp05(source/'data'/'level-3', destination/'residual-spectrum.png', samples_per_element=96)
        elif exp == 'exp07':
            render_exp07(source/'data'/'cases'/'coordinate-level-2'/'summary.json', destination/'curvature-residual-spectrum.png', samples_per_element=96)
        else:
            _summary_figure(source/'data', summary)
            shutil.copy2(source/'data'/'figures'/'curved-domain-and-horizon.png', destination/'curved-domain-and-horizon.png')
        provenance[exp] = {'summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
                           'software': json.loads((source/'manifest.json').read_text())['software']}
        (destination/'figure-provenance.json').write_text(json.dumps(provenance[exp], indent=2)+'\n')


if __name__ == '__main__':
    main()
