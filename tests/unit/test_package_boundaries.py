from __future__ import annotations

import ast
import importlib
from pathlib import Path

import numpy as np
import nee

from nee.solver.stopping import has_converged
from nee.state.fields import PrimitiveFields


RESPONSIBILITY_MODULES = (
    "nee.geometry.connection",
    "nee.geometry.curvature",
    "nee.geometry.null_geometry",
    "nee.geometry.tensors",
    "nee.discretization.projection",
    "nee.discretization.quadrature",
    "nee.initial_data.constraints",
    "nee.initial_data.crossed_shears",
    "nee.initial_data.exact_extraction",
    "nee.initial_data.scalar_pulse",
    "nee.initial_data.short_pulse",
    "nee.solver.picard",
    "nee.solver.relaxation",
    "nee.solver.slab_continuation",
    "nee.solver.stopping",
    "nee.diagnostics.convergence_regions",
    "nee.diagnostics.exact_comparison",
    "nee.diagnostics.ricci_residuals",
    "nee.io.checkpoint",
    "nee.exact_solutions.kerr",
    "nee.exact_solutions.minkowski",
    "nee.exact_solutions.schwarzschild",
)


def test_responsibility_modules_import_from_installed_package() -> None:
    for name in RESPONSIBILITY_MODULES:
        assert importlib.import_module(name).__name__ == name


def _absolute_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    result: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            result.add(node.module)
        elif isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
    return result


def test_inward_dependency_boundaries() -> None:
    package = Path(nee.__file__).resolve().parent
    for area in ("solver", "diagnostics", "initial_data", "io"):
        for path in (package / area).glob("*.py"):
            imports = _absolute_imports(path)
            assert not any(name.startswith("nee.experiments") for name in imports)
    for path in (package / "initial_data").glob("*.py"):
        imports = _absolute_imports(path)
        assert not any(name.startswith("nee.solver") for name in imports)
    for path in (package / "discretization").glob("*.py"):
        imports = _absolute_imports(path)
        assert not any(name.startswith("nee.diagnostics") for name in imports)


def test_primitive_fields_and_stopping_policy() -> None:
    fields = PrimitiveFields(
        metric=np.eye(3)[None, None, None],
        log_omega=np.zeros((1, 1, 1)),
        shift=np.zeros((1, 1, 1, 3)),
    )
    assert fields.phi is None
    assert has_converged(1.0e-9, 1.0e-8)
    assert not has_converged(float("nan"), 1.0e-8)
