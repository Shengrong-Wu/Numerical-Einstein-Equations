"""Crash-safe, provenance-bound checkpoints for interrupted solver runs.

Operational checkpoints are deliberately distinct from scientific warm-start
states.  They restore the controller of one interrupted invocation (including
its accumulated diagnostics), whereas ``--resume-state`` and ``--resume-cap``
start a new invocation from an already computed numerical state.

Each generation is immutable.  It is assembled in a hidden directory and the
directory is renamed into place only after its state, progress journal, and
manifest have all been flushed.  Consequently a torn newest generation never
damages the preceding valid checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Callable


OPERATIONAL_CHECKPOINT_SCHEMA_VERSION = 1
OPERATIONAL_CHECKPOINT_KIND = "numerical-operational-sweep"


class OperationalCheckpointError(ValueError):
    """Raised when an operational checkpoint is incomplete or incompatible."""


def prepare_output_directory(
    path: Path,
    *,
    operational_restart: bool,
) -> Path:
    """Create a clean scientific output directory or admit an exact restart.

    Reusing a populated directory for a nominally fresh run can mix old
    reliability artifacts with a new state.  Only an explicitly validated
    operational restart may enter a nonempty directory.
    """

    output = Path(path).resolve()
    if output.exists() and not operational_restart and any(output.iterdir()):
        raise OperationalCheckpointError(
            f"fresh run requires an empty output directory: {output}; choose "
            "a new output label or use --restart-checkpoint for the same run"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability after an atomic rename."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _strict_json_text(value: Any) -> str:
    return json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"


def atomic_write_json(path: Path, value: Any) -> None:
    destination = Path(path)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(_strict_json_text(value))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _read_strict_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant {token}")
            ),
        )
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise OperationalCheckpointError(
            f"cannot read strict checkpoint JSON {path}: {error}"
        ) from error
    if not isinstance(value, dict):
        raise OperationalCheckpointError(
            f"operational checkpoint JSON is not an object: {path}"
        )
    return value


def _artifact_record(path: Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "filename": resolved.name,
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _source_is_current(source_guard: Callable[[], bool] | None) -> None:
    if source_guard is not None and not bool(source_guard()):
        raise RuntimeError(
            "a fingerprinted source changed during the solver run; refusing "
            "to publish an operational checkpoint"
        )


def write_operational_checkpoint(
    root: Path,
    generation: str,
    *,
    state_writer: Callable[[Path], None],
    producer: str,
    solver_semantics: str,
    solver_source_fingerprint: str,
    producer_source_fingerprint: str,
    runtime_provenance: dict[str, str],
    run_id: str,
    run_contract: dict[str, Any],
    position: dict[str, Any],
    progress: dict[str, Any],
    lineage: dict[str, Any],
    retain_valid_generations: int = 2,
    source_guard: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Publish one immutable checkpoint generation.

    ``state_writer`` must write the complete numerical archive to the path it
    receives.  The generation directory itself is not visible under its final
    name until every artifact and the manifest are durable.
    """

    if not generation or Path(generation).name != generation:
        raise ValueError("checkpoint generation must be one path component")
    if retain_valid_generations < 2:
        raise ValueError(
            "operational checkpoint retention must keep at least two generations"
        )
    checkpoint_root = Path(root).resolve()
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    destination = checkpoint_root / generation
    quarantined: Path | None = None
    if destination.exists():
        try:
            validate_operational_checkpoint(
                destination,
                producer=producer,
                solver_semantics=solver_semantics,
                solver_source_fingerprint=solver_source_fingerprint,
                producer_source_fingerprint=producer_source_fingerprint,
                runtime_provenance=runtime_provenance,
                run_contract=run_contract,
                unsafe_allow_unverified=True,
            )
        except OperationalCheckpointError:
            quarantined = checkpoint_root / (
                f".{generation}.invalid-{uuid.uuid4().hex}"
            )
            destination.replace(quarantined)
        else:
            raise FileExistsError(
                f"operational checkpoint generation is immutable: {destination}"
            )

    _source_is_current(source_guard)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{generation}.", dir=checkpoint_root)
    )
    try:
        state_path = temporary / "state.npz"
        progress_path = temporary / "progress.json"
        manifest_path = temporary / "manifest.json"

        state_writer(state_path)
        if not state_path.is_file():
            raise OperationalCheckpointError(
                "operational checkpoint state writer produced no state archive"
            )
        progress_payload = {
            **progress,
            "position": position,
            "run_id": run_id,
        }
        atomic_write_json(progress_path, progress_payload)
        _source_is_current(source_guard)

        manifest_payload = {
            "schema_version": OPERATIONAL_CHECKPOINT_SCHEMA_VERSION,
            "manifest_kind": OPERATIONAL_CHECKPOINT_KIND,
            "producer": producer,
            "solver_semantics": solver_semantics,
            "solver_source_fingerprint": solver_source_fingerprint,
            "producer_source_fingerprint": producer_source_fingerprint,
            "runtime_provenance": runtime_provenance,
            "run_id": run_id,
            "run_contract": run_contract,
            "position": position,
            "lineage": lineage,
            "retention": {
                "valid_generations": retain_valid_generations,
                "minimum_for_fallback": 2,
            },
            "artifacts": {
                "state": _artifact_record(state_path),
                "progress": _artifact_record(progress_path),
            },
        }
        atomic_write_json(manifest_path, manifest_payload)
        _source_is_current(source_guard)
        temporary.replace(destination)
        _fsync_directory(checkpoint_root)

        final_manifest = destination / "manifest.json"
        pointer = {
            "schema_version": 2,
            "run_id": run_id,
            "generation": generation,
            "manifest_sha256": file_sha256(final_manifest),
        }
        atomic_write_json(checkpoint_root / "latest.json", pointer)
        pruned = prune_operational_generations(
            checkpoint_root,
            run_id=run_id,
            retain_valid_generations=retain_valid_generations,
        )
        # Only after the replacement generation and latest pointer are durable
        # may we discard torn assembly/quarantine directories for this ordinal.
        for stale in checkpoint_root.glob(f".{generation}.*"):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
        return {
            "directory": str(destination),
            "state": str(destination / "state.npz"),
            "progress": str(destination / "progress.json"),
            "manifest": str(final_manifest),
            "manifest_sha256": pointer["manifest_sha256"],
            "payload": manifest_payload,
            "pruned_generations": pruned,
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prune_operational_generations(
    root: Path,
    *,
    run_id: str,
    retain_valid_generations: int = 2,
) -> list[str]:
    """Prune only older, internally valid generations of the same run.

    This runs only after a new manifest and ``latest.json`` are durable.  A
    corrupt or foreign generation is never deleted by inference.  Keeping two
    valid generations preserves one fallback if the newest artifact is later
    found damaged.
    """

    if retain_valid_generations < 2:
        raise ValueError("must retain at least two valid generations")
    checkpoint_root = Path(root).resolve()
    valid: list[Path] = []
    for candidate in checkpoint_root.iterdir():
        if not candidate.is_dir() or candidate.name.startswith("."):
            continue
        manifest = candidate / "manifest.json"
        try:
            payload = _read_strict_json(manifest)
        except OperationalCheckpointError:
            continue
        if (
            payload.get("manifest_kind") != OPERATIONAL_CHECKPOINT_KIND
            or payload.get("run_id") != run_id
        ):
            continue
        artifacts = payload.get("artifacts")
        if not isinstance(artifacts, dict):
            continue
        intact = True
        for name in ("state", "progress"):
            record = artifacts.get(name)
            if not isinstance(record, dict):
                intact = False
                break
            filename = record.get("filename")
            if not isinstance(filename, str) or Path(filename).name != filename:
                intact = False
                break
            artifact = candidate / filename
            if not artifact.is_file() or _artifact_record(artifact) != record:
                intact = False
                break
        if intact:
            valid.append(candidate)
    valid.sort(key=lambda item: item.name, reverse=True)
    removed: list[str] = []
    for candidate in valid[retain_valid_generations:]:
        shutil.rmtree(candidate)
        removed.append(candidate.name)
    if removed:
        _fsync_directory(checkpoint_root)
    return removed


def _manifest_from_path(path: Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir():
        return candidate / "manifest.json"
    if candidate.name == "latest.json":
        pointer = _read_strict_json(candidate)
        generation = pointer.get("generation")
        if not isinstance(generation, str) or Path(generation).name != generation:
            raise OperationalCheckpointError(
                f"invalid checkpoint generation in {candidate}"
            )
        manifest = candidate.parent / generation / "manifest.json"
        if manifest.is_file() and pointer.get("manifest_sha256") != file_sha256(
            manifest
        ):
            raise OperationalCheckpointError(
                f"latest pointer hash does not match {manifest}"
            )
        return manifest
    if candidate.name in {"state.npz", "progress.json"}:
        return candidate.parent / "manifest.json"
    return candidate


def validate_operational_checkpoint(
    path: Path,
    *,
    producer: str,
    solver_semantics: str,
    solver_source_fingerprint: str,
    producer_source_fingerprint: str,
    runtime_provenance: dict[str, str],
    run_contract: dict[str, Any],
    unsafe_allow_unverified: bool = False,
) -> dict[str, Any]:
    """Validate provenance and every artifact before numerical deserialization."""

    manifest_path = _manifest_from_path(path)
    if not manifest_path.is_file():
        raise OperationalCheckpointError(
            f"operational checkpoint has no manifest: {manifest_path}"
        )
    payload = _read_strict_json(manifest_path)
    expected = {
        "schema_version": OPERATIONAL_CHECKPOINT_SCHEMA_VERSION,
        "manifest_kind": OPERATIONAL_CHECKPOINT_KIND,
        "producer": producer,
        "solver_semantics": solver_semantics,
        "solver_source_fingerprint": solver_source_fingerprint,
        "producer_source_fingerprint": producer_source_fingerprint,
        "runtime_provenance": runtime_provenance,
        "run_contract": run_contract,
    }
    differences = [
        name for name, value in expected.items() if payload.get(name) != value
    ]
    if differences:
        raise OperationalCheckpointError(
            "operational checkpoint does not match this invocation: "
            + ", ".join(differences)
        )

    run_id = payload.get("run_id")
    position = payload.get("position")
    lineage = payload.get("lineage")
    artifacts = payload.get("artifacts")
    if not isinstance(run_id, str) or not run_id:
        raise OperationalCheckpointError("checkpoint manifest has no run id")
    if not isinstance(position, dict):
        raise OperationalCheckpointError("checkpoint manifest has no position")
    if not isinstance(lineage, dict):
        raise OperationalCheckpointError("checkpoint manifest has no lineage")
    if not isinstance(artifacts, dict):
        raise OperationalCheckpointError("checkpoint manifest has no artifacts")
    lineage_status = lineage.get("status")
    if lineage_status != "verified" and not unsafe_allow_unverified:
        raise OperationalCheckpointError(
            f"checkpoint lineage is not verified: {lineage_status!r}"
        )

    resolved_artifacts: dict[str, str] = {}
    for name in ("state", "progress"):
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise OperationalCheckpointError(
                f"checkpoint manifest lacks {name} artifact"
            )
        filename = record.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise OperationalCheckpointError(
                f"invalid checkpoint artifact filename for {name}"
            )
        artifact = manifest_path.parent / filename
        if not artifact.is_file():
            raise OperationalCheckpointError(
                f"checkpoint artifact is missing: {artifact}"
            )
        actual = _artifact_record(artifact)
        if actual != record:
            raise OperationalCheckpointError(
                f"checkpoint {name} artifact does not match its manifest"
            )
        resolved_artifacts[name] = str(artifact)

    progress = _read_strict_json(Path(resolved_artifacts["progress"]))
    if progress.get("run_id") != run_id or progress.get("position") != position:
        raise OperationalCheckpointError(
            "checkpoint progress journal disagrees with its manifest position"
        )
    return {
        "status": (
            "verified" if lineage_status == "verified" else "unsafe-unverified"
        ),
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "state": resolved_artifacts["state"],
        "progress_path": resolved_artifacts["progress"],
        "progress": progress,
        "payload": payload,
        "run_id": run_id,
        "position": position,
        "lineage": lineage,
    }


def resolve_operational_restart_checkpoint(
    path: Path,
    **validation: Any,
) -> dict[str, Any]:
    """Resolve an exact generation or a fallback-capable checkpoint root.

    A generation directory, its manifest, state, or progress file is exact and
    therefore fails if damaged.  A checkpoint-root directory or its
    ``latest.json`` pointer requests the newest *valid* generation and scans
    backward when that generation is torn.
    """

    candidate = Path(path).expanduser().resolve()
    if candidate.name == "latest.json":
        return find_latest_valid_operational_checkpoint(
            candidate.parent, **validation
        )
    if candidate.is_dir() and not (candidate / "manifest.json").is_file():
        return find_latest_valid_operational_checkpoint(candidate, **validation)
    return validate_operational_checkpoint(candidate, **validation)


def find_latest_valid_operational_checkpoint(
    root: Path,
    **validation: Any,
) -> dict[str, Any]:
    """Return the newest valid immutable generation, falling back if torn.

    Hidden assembly directories and ``latest.json`` are not trusted as the
    only source of truth.  Every visible generation is validated in descending
    lexical order, which is chronological for the zero-padded names used by
    both Q1 drivers.
    """

    checkpoint_root = Path(root).expanduser().resolve()
    failures: list[str] = []
    anchor_run_id: str | None = None
    pointer_path = checkpoint_root / "latest.json"
    if pointer_path.is_file():
        try:
            pointer = _read_strict_json(pointer_path)
        except OperationalCheckpointError as error:
            failures.append(f"latest.json: {error}")
        else:
            pointer_run_id = pointer.get("run_id")
            generation = pointer.get("generation")
            if (
                pointer.get("schema_version") == 2
                and isinstance(pointer_run_id, str)
                and bool(pointer_run_id)
                and isinstance(generation, str)
                and Path(generation).name == generation
            ):
                referenced_manifest = (
                    checkpoint_root / generation / "manifest.json"
                )
                if referenced_manifest.is_file():
                    try:
                        referenced = _read_strict_json(referenced_manifest)
                    except OperationalCheckpointError as error:
                        failures.append(
                            f"latest manifest ownership: {error}"
                        )
                    else:
                        if referenced.get("run_id") != pointer_run_id:
                            raise OperationalCheckpointError(
                                "latest pointer and referenced manifest "
                                "disagree on run_id"
                            )
                anchor_run_id = pointer_run_id
            else:
                failures.append(
                    "latest.json has no schema-2 run ownership anchor"
                )
    candidates = sorted(
        (
            item
            for item in checkpoint_root.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        ),
        key=lambda item: item.name,
        reverse=True,
    )
    valid: list[dict[str, Any]] = []
    for candidate in candidates:
        try:
            result = validate_operational_checkpoint(candidate, **validation)
        except (OSError, OperationalCheckpointError) as error:
            failures.append(f"{candidate.name}: {error}")
            continue
        if anchor_run_id is not None and result["run_id"] != anchor_run_id:
            failures.append(
                f"{candidate.name}: foreign run_id {result['run_id']!r} "
                f"does not match latest anchor {anchor_run_id!r}"
            )
            continue
        valid.append(result)
    if anchor_run_id is None:
        run_ids = {str(result["run_id"]) for result in valid}
        if len(run_ids) > 1:
            raise OperationalCheckpointError(
                "checkpoint root contains multiple valid run_ids but no "
                "trusted schema-2 latest anchor; pass an exact generation"
            )
    if valid:
        result = valid[0]
        result["fallback_diagnostics"] = failures
        return result
    detail = "; ".join(failures[:6]) if failures else "no generations found"
    raise OperationalCheckpointError(
        f"no valid operational checkpoint under {checkpoint_root}: {detail}"
    )


def finite_json_number(value: float) -> float | None:
    """Small helper for progress journals that must remain strict JSON."""

    converted = float(value)
    return converted if math.isfinite(converted) else None
