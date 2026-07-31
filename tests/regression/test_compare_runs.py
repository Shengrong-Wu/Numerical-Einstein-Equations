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
