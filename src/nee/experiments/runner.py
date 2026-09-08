"""Uniform immutable runner used by every experiment entrance."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import shutil
import sys
import time
import tomllib
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


def _collect_artifacts(output: Path, data: Path):
    """Index every case explicitly; never promote an arbitrary first match."""
    hashes, schemas, cases = {}, {}, {}
    for artifact in sorted(data.rglob('*')):
        if not artifact.is_file() or artifact.suffix not in {'.npz', '.json', '.png', '.toml'}:
            continue
        name = artifact.relative_to(output).as_posix()
        hashes[name] = _digest(artifact)
        case = artifact.parent.relative_to(data).as_posix()
        cases.setdefault(case, []).append(name)
        if artifact.suffix == '.npz':
            with np.load(artifact, allow_pickle=False) as archive:
                schemas[name] = {}
                for key in archive.files:
                    value = archive[key]
                    schemas[name][key] = {'shape': list(value.shape), 'dtype': str(value.dtype)}
    if not schemas:
        raise ValueError('campaign produced no numerical array artifacts')
    write_json(output / 'artifacts.json', {'cases': cases, 'array_schemas': schemas})
    hashes['artifacts.json'] = _digest(output / 'artifacts.json')
    return hashes, schemas


def _validate_resume(
    output: Path,
    config: Any,
) -> str:
    resolved = output / "resolved-config.toml"
    manifest_path = output / "manifest.json"
    if not resolved.exists():
        raise ValueError("resume target has no resolved configuration")
    with resolved.open("rb") as stream:
        recorded_config = tomllib.load(stream)
    expected_config = json.loads(json.dumps(config.resolved()))
    if recorded_config != expected_config:
        raise ValueError("resume configuration does not match the existing run")
    if not manifest_path.exists():
        raise ValueError("resume target has no completed run manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hashes = manifest.get("output_hashes")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("resume manifest has no output content hashes")
    for name, expected in hashes.items():
        artifact = output / name
        if not artifact.is_file():
            raise ValueError(f"resume artifact is missing: {name}")
        if _digest(artifact) != expected:
            raise ValueError(f"resume artifact failed its content-hash check: {name}")
    status = manifest.get("status")
    if status not in {"completed", "failed"}:
        raise ValueError(f"resume manifest has invalid terminal status: {status!r}")
    return status


def public_main(identifier: str, argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog=f"python -m nee.experiments.{MODULES[identifier]}")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    definition = get(identifier)
    config = load_config(args.config)
    from .config_contract import validate_supported
    contract = validate_supported(config)
    if config.experiment.identifier != identifier:
        parser.error(
            f"configuration is for {config.experiment.identifier}, not {identifier}"
        )
    output = args.output.resolve()
    if output.exists():
        if not args.resume:
            raise FileExistsError(f"immutable run directory exists: {output}")
        status = _validate_resume(output, config)
        print(f"resume validation complete: {output} ({status})")
        return 0 if status == "completed" else 2
    output.mkdir(parents=True)
    (output / "figures").mkdir()
    dump_resolved_config(config, output / "resolved-config.toml")
    write_json(output / "parameter-contract.json", contract)
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
        outputs, schemas = _collect_artifacts(output, data)
        outputs["summary.json"] = _digest(output / "summary.json")
        boundary_hashes = {name: value for name, value in outputs.items() if "boundary" in name and name.endswith(".npz")}
        manifest = build_manifest(
            root=Path(__file__).resolve().parents[3],
            experiment=identifier,
            started_at=started,
            config=config.resolved(),
            inputs=boundary_hashes,
            outputs=outputs,
            arrays={},
        )
        manifest["array_schemas"] = schemas
        manifest["status"] = status
        write_json(output / "manifest.json", manifest)
        print(f"summary: {output / 'summary.json'}")
        print(f"case artifacts: {output / 'artifacts.json'}")
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
