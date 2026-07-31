"""Lazy registry for the eight official experiments."""

from __future__ import annotations

from importlib import import_module

from .base import ExperimentDefinition


MODULES = {
    "exp01": "exp01_regular_vacuum",
    "exp02": "exp02_schwarzschild_horizon",
    "exp03": "exp03_schwarzschild_interior",
    "exp04": "exp04_vacuum_strong_short_pulse",
    "exp05": "exp05_vacuum_crossed_pulses",
    "exp06": "exp06_regular_exact_scalar",
    "exp07": "exp07_nonspherical_scalar",
    "exp08": "exp08_scalar_trapped_section",
}


def get(identifier: str) -> ExperimentDefinition:
    try:
        module_name = MODULES[identifier]
    except KeyError as error:
        raise ValueError(f"unknown experiment {identifier!r}") from error
    module = import_module(f"nee.experiments.{module_name}.definition")
    definition = module.DEFINITION
    if not isinstance(definition, ExperimentDefinition):
        raise TypeError(f"{module_name} does not expose an ExperimentDefinition")
    return definition

