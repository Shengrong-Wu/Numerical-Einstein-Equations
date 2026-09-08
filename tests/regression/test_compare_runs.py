import json
from pathlib import Path

import numpy as np

from scripts.compare_runs import compare_trees


def test_same_configuration_comparison(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    value = {"terminal_status": "completed", "residual": 2.0e-4, "count": 7}
    for root in (reference, candidate):
        (root / "summary.json").write_text(json.dumps(value))
        np.savez_compressed(root / "final-state.npz", field=np.arange(6.0).reshape(2, 3))
    report = compare_trees(reference, candidate)
    assert report["passed"]
    assert report["json_files"] == 1
    assert report["npz_files"] == 1


def test_boolean_and_integer_arrays_use_exact_comparison(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    candidate = tmp_path / "candidate"
    reference.mkdir()
    candidate.mkdir()
    for root in (reference, candidate):
        (root / "summary.json").write_text(
            json.dumps({"terminal_status": "completed"})
        )
    np.savez_compressed(
        reference / "final-state.npz",
        mask=np.asarray([True, False]),
        counts=np.asarray([1, 2]),
    )
    np.savez_compressed(
        candidate / "final-state.npz",
        mask=np.asarray([True, True]),
        counts=np.asarray([1, 3]),
    )

    report = compare_trees(reference, candidate)

    assert not report["passed"]
    assert report["failure_count"] == 2
    failures = report["npz"][0]["failures"]
    assert {item["array"] for item in failures} == {"mask", "counts"}


def test_missing_content_cannot_pass(tmp_path: Path) -> None:
    reference, candidate = tmp_path / 'reference', tmp_path / 'candidate'
    reference.mkdir()
    candidate.mkdir()
    (reference / 'summary.json').write_text(json.dumps({'residual': 0.0, 'count': 2}))
    (candidate / 'summary.json').write_text('{}')
    np.savez(reference / 'final-state.npz', field=np.ones(3))
    report = compare_trees(reference, candidate)
    assert not report['passed']
    assert 'final-state.npz' in report['missing_files']
    np.savez(candidate / 'final-state.npz', different=np.ones(3))
    assert not compare_trees(reference, candidate)['passed']


def test_empty_comparison_cannot_pass(tmp_path: Path) -> None:
    assert not compare_trees(tmp_path, tmp_path)['passed']


def test_array_dtype_and_unsigned_values_are_checked(tmp_path: Path) -> None:
    from scripts.compare_runs import compare_npz
    left, right = tmp_path / 'left.npz', tmp_path / 'right.npz'
    np.savez(left, count=np.array([1], dtype=np.uint64))
    np.savez(right, count=np.array([2], dtype=np.uint64))
    assert compare_npz(left, right)['failures']
    np.savez(right, count=np.array([1], dtype=np.int64))
    assert compare_npz(left, right)['failures']


def test_json_identity_and_nonfinite_values_are_checked(tmp_path: Path) -> None:
    from scripts.compare_runs import compare_json
    left, right = tmp_path / 'left.json', tmp_path / 'right.json'
    left.write_text('{"family":"vacuum","value":Infinity}')
    right.write_text('{"family":"scalar","value":Infinity}')
    assert len(compare_json(left, right)['failures']) == 2
