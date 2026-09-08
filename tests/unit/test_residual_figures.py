from pathlib import Path
import importlib.util
import numpy as np


def test_open_grid_plot_ignores_endpoint_placeholders():
    path = Path(__file__).resolve().parents[2] / 'scripts/residual_plotting.py'
    spec = importlib.util.spec_from_file_location('residual_plotting', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    nodes = np.linspace(0, 1, 9)
    elements = [np.arange(5), np.arange(4, 9)]
    values = 10**(nodes[:, None] + 2*nodes[None, :])
    values[[0, -1], :] = np.nan
    values[:, [0, -1]] = np.inf
    first, second, dense = module.dense_open_log_residual(
        values, nodes, nodes, elements, elements, 16)
    assert 0 < first.min() < first.max() < 1
    assert 0 < second.min() < second.max() < 1
    np.testing.assert_allclose(dense, first[:, None] + 2*second[None, :], atol=2e-14)
