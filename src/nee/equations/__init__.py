"""Vacuum and Einstein--scalar equation providers."""

from .base import EquationSystem
from .scalar_field import ScalarFieldEquations
from .vacuum import VacuumEquations

__all__ = ["EquationSystem", "ScalarFieldEquations", "VacuumEquations"]

