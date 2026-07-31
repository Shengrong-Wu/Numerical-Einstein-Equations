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
