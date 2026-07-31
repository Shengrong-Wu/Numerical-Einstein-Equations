"""Numerical characteristic Einstein and Einstein--scalar equations."""

from .api import picard_sweep
from .state.boundary import BoundaryData
from .state.iterate import PicardState

__all__ = ["BoundaryData", "PicardState", "picard_sweep"]
__version__ = "0.1.0"

