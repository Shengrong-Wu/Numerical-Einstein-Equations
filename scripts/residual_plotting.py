"""Elementwise interpolation for figures, omitting unreported outer endpoints."""
from __future__ import annotations

import numpy as np


def _open_axis(nodes, element_indices, samples_per_element):
    pieces, matrices, indices = [], [], []
    for number, source in enumerate(element_indices):
        index = np.asarray(source)
        index = index[(index > 0) & (index < len(nodes) - 1)]
        if len(index) < 2:
            raise ValueError('an open plotting element needs two interior nodes')
        local = nodes[index]
        targets = np.linspace(local[0], local[-1], samples_per_element,
                              endpoint=number == len(element_indices) - 1)
        distances = local[:, None] - local[None, :]
        np.fill_diagonal(distances, 1.0)
        weights = 1.0 / np.prod(distances, axis=1)
        matrix = np.empty((len(targets), len(local)))
        for row, target in enumerate(targets):
            difference = target - local
            exact = np.flatnonzero(np.abs(difference) <= 16*np.finfo(float).eps)
            if exact.size:
                matrix[row] = 0.0
                matrix[row, exact[0]] = 1.0
            else:
                terms = weights / difference
                matrix[row] = terms / np.sum(terms)
        pieces.append(targets)
        matrices.append(matrix)
        indices.append(index)
    return np.concatenate(pieces), matrices, indices


def dense_open_log_residual(residual, first_nodes, second_nodes,
                            first_indices, second_indices, samples_per_element):
    """Interpolate log residual only from the open grid; never extend to endpoints."""
    if not np.all(np.isfinite(residual[1:-1, 1:-1])):
        raise ValueError('the open residual map contains nonfinite values')
    if np.any(residual[1:-1, 1:-1] < 0):
        raise ValueError('a residual norm cannot be negative')
    first, first_matrices, first_indices = _open_axis(
        first_nodes, first_indices, samples_per_element)
    second, second_matrices, second_indices = _open_axis(
        second_nodes, second_indices, samples_per_element)
    dense = np.empty((len(first), len(second)))
    a = 0
    for first_index, first_matrix in zip(first_indices, first_matrices, strict=True):
        b = 0
        for second_index, second_matrix in zip(second_indices, second_matrices, strict=True):
            # Read selected nodes only, so endpoint placeholders cannot enter the figure.
            local = np.log10(np.maximum(residual[np.ix_(first_index, second_index)], 1e-7))
            dense[a:a+len(first_matrix), b:b+len(second_matrix)] = first_matrix @ local @ second_matrix.T
            b += len(second_matrix)
        a += len(first_matrix)
    return first, second, dense
