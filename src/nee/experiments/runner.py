"""Uniform immutable runner used by every experiment entrance."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Sequence, TextIO

import numpy as np

from nee.config import dump_resolved_config, load_config
from nee.io.manifest import build_manifest, write_json

from .analysis import terminal_status
from .registry import MODULES, get


class _Tee:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _first(root: Path, names: tuple[str, ...]) -> Path | None:
    candidates = [path for name in names for path in root.rglob(name)]
    return min(candidates, key=lambda path: (len(path.parts), str(path))) if candidates else None


def _promote_artifacts(output: Path, data: Path) -> dict[str, str]:
    choices = {
        "boundary-data.npz": ("boundary-data.npz", "boundary-data-official.npz"),
        "final-state.npz": ("final-state.npz", "official-state.npz", "official-mapped-state.npz"),
        "residual-maps.npz": ("residual-maps.npz",),
    }
    hashes: dict[str, str] = {}
    for target_name, source_names in choices.items():
        target = output / target_name
        if not target.exists():
            source = _first(data, source_names)
            if source is not None:
                shutil.copy2(source, target)
        if target.exists():
            hashes[target_name] = _digest(target)
    if not (output / "residual-maps.npz").exists():
        state = output / "final-state.npz"
        arrays: dict[str, np.ndarray] = {}
        if state.exists():
            with np.load(state, allow_pickle=False) as archive:
                arrays = {
                    name: np.asarray(archive[name])
                    for name in archive.files
                    if "residual" in name or "update_map" in name
                }
        np.savez_compressed(output / "residual-maps.npz", **arrays)
        hashes["residual-maps.npz"] = _digest(output / "residual-maps.npz")
    return hashes


def public_main(identifier: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m nee.experiments.{MODULES[identifier]}")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    definition = get(identifier)
    config = load_config(args.config)
    if config.experiment.identifier != identifier:
        parser.error(
            f"configuration is for {config.experiment.identifier}, not {identifier}"
        )
    output = args.output.resolve()
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"immutable run directory exists: {output}")
        resolved = output / "resolved-config.toml"
        if not resolved.exists():
            raise ValueError("resume target has no resolved configuration")
        print(f"resume validation complete: {output}")
        return 0
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    dump_resolved_config(config, output / "resolved-config.toml")
    data = output / "data"
    started = time.time()
    try:
        with (output / "run.log").open("w", encoding="utf-8") as log:
            tee = _Tee(sys.stdout, log)
            with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
                print(json.dumps(config.resolved(), indent=2, sort_keys=True))
                print(f"target output: {output}")
                summary = definition.campaign(config, data)
                status = terminal_status(summary)
                print(f"terminal status: {status}")
        summary["terminal_status"] = status
        write_json(output / "summary.json", summary)
        outputs = _promote_artifacts(output, data)
        outputs["summary.json"] = _digest(output / "summary.json")
        boundary_hash = outputs.get("boundary-data.npz")
        manifest = build_manifest(
            root=Path(__file__).resolve().parents[3],
            experiment=identifier,
            started_at=started,
            config=config.resolved(),
            inputs={} if boundary_hash is None else {"boundary-data.npz": boundary_hash},
            outputs=outputs,
            arrays={},
        )
        manifest["status"] = status
        write_json(output / "manifest.json", manifest)
        print(f"summary: {output / 'summary.json'}")
        print(f"retained state: {output / 'final-state.npz'}")
        return 0 if status == "completed" else 2
    except Exception as error:
        failure = {
            "schema": "nee-run-failure-1",
            "experiment": identifier,
            "terminal_status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json(output / "summary.json", failure)
        print(f"terminal status: failed ({type(error).__name__}: {error})", file=sys.stderr)
        return 2
