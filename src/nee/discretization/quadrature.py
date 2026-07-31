"""Coordinate spectral quadrature interfaces."""

from nee.discretization.lgl import CompositeLGLMesh, LGLSegment
from nee.numerics.coordinate_quadrature import cumulative_polynomial_quadrature

__all__ = [
    "CompositeLGLMesh",
    "LGLSegment",
    "cumulative_polynomial_quadrature",
]
