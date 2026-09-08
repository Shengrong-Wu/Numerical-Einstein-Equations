"""Explicit supported controls and fixed protocol descriptors per experiment."""
from __future__ import annotations
import json
import math
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
    if key not in contracts:
        raise ValueError(f'no experiment protocol is defined for {key}')
    contract = contracts[key]
    actual = dict(flatten(json.loads(json.dumps(config.resolved()))))
    expected = contract['protocol']
    adjustable = set(contract['adjustable'])
    for name in set(actual) | set(expected):
        if name.startswith('experiment.') or name in {'coordinates.u_node_count', 'coordinates.v_node_count', 'angular.scalar_retained_dimension', 'angular.scalar_work_dimension'}:
            continue
        if name in adjustable:
            continue
        if name not in actual or name not in expected or actual[name] != expected[name]:
            raise ValueError(f'{name} is not an adjustable control of {key}; fixed protocol value: {expected.get(name)!r}')
    exp, mode = config.experiment.identifier, config.experiment.mode
    if (exp == 'exp04' and mode == 'standard') or (exp in {'exp06','exp07'} and mode == 'smoke'):
        if len(set(config.coordinates.u_degrees)) != 1 or len(set(config.coordinates.v_degrees)) != 1:
            raise ValueError(f'{key} requires a uniform polynomial degree within each coordinate')
    if exp == 'exp04':
        if not math.isclose(float(config.physics['cap']), config.coordinates.v_max, rel_tol=0, abs_tol=1e-13):
            raise ValueError('physics.cap must equal coordinates.v_max')
        if mode == 'standard':
            nodes = [-math.log(-value) for value in config.coordinates.u_breakpoints]
            spacing = (nodes[-1]-nodes[0])/(len(nodes)-1)
            if any(not math.isclose(value,nodes[0]+i*spacing,rel_tol=0,abs_tol=1e-13) for i,value in enumerate(nodes)):
                raise ValueError('exp04 requires uniformly spaced u breakpoints in tau=-log(-u)')
    if exp == 'exp06' and mode == 'smoke' and len(config.physics['nu_values']) != 1:
        raise ValueError('exp06 smoke accepts exactly one nu value; use standard mode for a family')
    return {'adjustable': sorted(adjustable), 'fixed_protocol': {k:v for k,v in expected.items() if k not in adjustable}}
