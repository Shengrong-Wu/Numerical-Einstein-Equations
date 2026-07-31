"""Public experiment definition shared by all module entrances."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from nee.config import ExperimentConfig


Campaign = Callable[[ExperimentConfig, Path], dict[str, Any]]


@dataclass(frozen=True)
class ExperimentDefinition:
    identifier: str
    title: str
    equation_system: str
    campaign: Campaign

