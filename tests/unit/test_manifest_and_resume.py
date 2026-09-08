from __future__ import annotations

import json
from pathlib import Path

import pytest

from nee.config import dump_resolved_config, load_config
from nee.experiments.runner import _digest, _validate_resume
from nee.io import manifest


def test_peak_memory_converts_macos_bytes_to_kib(monkeypatch: pytest.MonkeyPatch) -> None:
    class Usage:
        ru_maxrss = 8 * 1024

    monkeypatch.setattr(manifest.resource, "getrusage", lambda _: Usage())
    monkeypatch.setattr(manifest.sys, "platform", "darwin")
    assert manifest.peak_memory_kib() == 8


def test_resume_validates_config_and_artifact_hashes(tmp_path: Path) -> None:
    source = Path("configs/experiments/exp01/smoke.toml")
    config = load_config(source)
    dump_resolved_config(config, tmp_path / "resolved-config.toml")
    artifact = tmp_path / "summary.json"
    artifact.write_text("{}\n", encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "output_hashes": {"summary.json": _digest(artifact)},
            }
        ),
        encoding="utf-8",
    )
    assert _validate_resume(tmp_path, config) == "completed"
    artifact.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="content-hash"):
        _validate_resume(tmp_path, config)


def test_manifest_keeps_multiple_case_artifacts(tmp_path: Path) -> None:
    import numpy as np
    from nee.experiments.runner import _collect_artifacts
    for case in ('a', 'b'):
        directory = tmp_path / 'data' / case
        directory.mkdir(parents=True)
        np.savez(directory / 'final-state.npz', radius=np.ones((2, 3)))
    hashes, schemas = _collect_artifacts(tmp_path, tmp_path / 'data')
    assert set(schemas) == {'data/a/final-state.npz', 'data/b/final-state.npz'}
    assert schemas['data/a/final-state.npz']['radius']['shape'] == [2, 3]
    assert not (tmp_path / 'final-state.npz').exists()
    assert 'artifacts.json' in hashes
