"""Explicit supported controls and fixed protocol descriptors per experiment."""
from __future__ import annotations
import json
from importlib.resources import files


def flatten(value, prefix=''):
    for key, item in value.items():
        name = prefix+'.'+key if prefix else key
        if isinstance(item, dict):
            yield from flatten(item, name)
        else:
            yield name, item


def validate_supported(config):
    contracts = json.loads(files('nee.experiments').joinpath('config_contracts.json').read_text())
    key = config.experiment.identifier+'/'+config.experiment.mode
    contract = contracts[key]
    actual = dict(flatten(json.loads(json.dumps(config.resolved()))))
    expected = contract['protocol']
    adjustable = set(contract['adjustable'])
    for name in set(actual) | set(expected):
        if name.startswith('experiment.') or name.endswith(('_node_count', '_dimension')):
            continue
        if name in adjustable:
            continue
        if name not in actual or name not in expected or actual[name] != expected[name]:
            raise ValueError(f'{name} is not an adjustable control of {key}; fixed protocol value: {expected.get(name)!r}')
    return {'adjustable': sorted(adjustable), 'fixed_protocol': {k:v for k,v in expected.items() if k not in adjustable}}
