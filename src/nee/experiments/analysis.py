"""Common terminal-status analysis for experiment aggregates."""

from __future__ import annotations

from typing import Any, Iterator


def _statuses(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        status = value.get("terminal_status")
        if isinstance(status, str):
            yield status
        for child in value.values():
            yield from _statuses(child)
    elif isinstance(value, list):
        for child in value:
            yield from _statuses(child)


def terminal_status(summary: dict[str, Any]) -> str:
    statuses = tuple(_statuses(summary))
    return "failed" if any(value != "completed" for value in statuses) else "completed"

