"""Compare two same-configuration numerical result trees."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


IGNORED_KEYS = {
    "seconds",
    "wall_time_seconds",
    "peak_memory_kib",
    "runtime",
    "software",
    "source_revision",
    "numerical_fingerprint",
    "final_state_hash",
}


def leaves(value: Any, prefix: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(value, dict):
        if not value:
            yield prefix, {}
        for key, child in value.items():
            if key not in IGNORED_KEYS:
                yield from leaves(child, f"{prefix}.{key}" if prefix else key)
    elif isinstance(value, list):
        if not value:
            yield prefix, []
        for index, child in enumerate(value):
            yield from leaves(child, f"{prefix}[{index}]")
    else:
        yield prefix, value


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0e-300)


def compare_json(reference: Path, candidate: Path) -> dict[str, Any]:
    expected = dict(leaves(json.loads(reference.read_text())))
    actual = dict(leaves(json.loads(candidate.read_text())))
    common = sorted(set(expected).intersection(actual))
    failures: list[dict[str, Any]] = [
        {"field": key, "reason": "missing" if key in expected else "unexpected"}
        for key in sorted(set(expected).symmetric_difference(actual))
    ]
    maximum_relative = 0.0
    maximum_absolute = 0.0
    compared = 0
    for key in common:
        left, right = expected[key], actual[key]
        if type(left) is not type(right):
            failures.append({"field": key, "reason": "type mismatch"})
            continue
        if isinstance(left, bool) or isinstance(right, bool):
            if left != right:
                failures.append({"field": key, "reference": left, "candidate": right})
            compared += 1
        elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if not (math.isfinite(float(left)) and math.isfinite(float(right))):
                failures.append({"field": key, "reason": "nonfinite numerical value"})
            elif isinstance(left, int) and isinstance(right, int):
                if left != right:
                    failures.append({"field": key, "reference": left, "candidate": right})
            else:
                absolute = abs(float(left) - float(right))
                relative = _relative_error(float(left), float(right))
                maximum_absolute = max(maximum_absolute, absolute)
                maximum_relative = max(maximum_relative, relative)
                accepted = absolute <= 1.0e-12 if max(abs(float(left)), abs(float(right))) < 1.0e-12 else relative <= 1.0e-9
                if not accepted:
                    failures.append({"field": key, "absolute": absolute, "relative": relative})
            compared += 1
        else:
            if left != right:
                failures.append({"field": key, "reference": left, "candidate": right})
            compared += 1
    if not compared:
        failures.append({"reason": "empty comparison"})
    return {
        "compared_values": compared,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "failures": failures,
    }


def compare_npz(reference: Path, candidate: Path) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    maximum_absolute = 0.0
    maximum_relative = 0.0
    compared_arrays = 0
    with np.load(reference, allow_pickle=False) as left, np.load(candidate, allow_pickle=False) as right:
        failures.extend(
            {"array": name, "reason": "missing" if name in left.files else "unexpected"}
            for name in sorted(set(left.files).symmetric_difference(right.files))
        )
        names = sorted(set(left.files).intersection(right.files))
        for name in names:
            expected, actual = np.asarray(left[name]), np.asarray(right[name])
            if expected.shape != actual.shape:
                failures.append({"array": name, "shape_reference": list(expected.shape), "shape_candidate": list(actual.shape)})
                continue
            if expected.dtype != actual.dtype:
                failures.append({"array": name, "reason": "dtype mismatch"})
                continue
            compared_arrays += 1
            if not expected.size:
                failures.append({"array": name, "reason": "empty array"})
                continue
            if expected.dtype.kind not in "fc":
                if not np.array_equal(expected, actual):
                    failures.append(
                        {
                            "array": name,
                            "exact_mismatch_count": int(
                                np.count_nonzero(expected != actual)
                            ),
                        }
                    )
                continue
            if not (np.all(np.isfinite(expected)) and np.all(np.isfinite(actual))):
                failures.append({"array": name, "reason": "nonfinite numerical value"})
                continue
            difference = np.abs(expected - actual)
            absolute = float(np.max(difference)) if difference.size else 0.0
            scale = np.maximum(np.maximum(np.abs(expected), np.abs(actual)), 1.0e-300)
            relative = float(np.max(difference / scale)) if difference.size else 0.0
            maximum_absolute = max(maximum_absolute, absolute)
            maximum_relative = max(maximum_relative, relative)
            if not np.allclose(expected, actual, rtol=5.0e-12, atol=5.0e-13, equal_nan=False):
                failures.append({"array": name, "absolute": absolute, "relative": relative})
    if not compared_arrays:
        failures.append({"reason": "empty array comparison"})
    return {
        "compared_arrays": compared_arrays,
        "maximum_absolute_difference": maximum_absolute,
        "maximum_relative_difference": maximum_relative,
        "failures": failures,
    }


def compare_trees(reference: Path, candidate: Path) -> dict[str, Any]:
    json_results = []
    npz_results = []
    missing = []
    for source in sorted(reference.rglob("summary.json")):
        relative = source.relative_to(reference)
        target = candidate / relative
        if target.exists():
            json_results.append({"file": str(relative), **compare_json(source, target)})
        else:
            missing.append(str(relative))
    for pattern in ("*.npz",):
        for source in sorted(reference.rglob(pattern)):
            relative = source.relative_to(reference)
            target = candidate / relative
            if target.exists():
                npz_results.append({"file": str(relative), **compare_npz(source, target)})
            else:
                missing.append(str(relative))
    expected_files = {str(p.relative_to(reference)) for pattern in ("summary.json", "*.npz") for p in reference.rglob(pattern)}
    actual_files = {str(p.relative_to(candidate)) for pattern in ("summary.json", "*.npz") for p in candidate.rglob(pattern)}
    unexpected = sorted(actual_files - expected_files)
    failure_count = sum(len(item["failures"]) for item in (*json_results, *npz_results)) + len(missing)
    empty = not json_results or not npz_results
    failure_count += len(unexpected) + int(empty)
    return {
        "schema": "nee-run-comparison-1",
        "json_files": len(json_results),
        "npz_files": len(npz_results),
        "missing_files": missing,
        "unexpected_files": unexpected,
        "incomplete_comparison": empty,
        "failure_count": failure_count,
        "passed": failure_count == 0,
        "json": json_results,
        "npz": npz_results,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report = compare_trees(args.reference, args.candidate)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in ("json_files", "npz_files", "failure_count", "passed")}))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
