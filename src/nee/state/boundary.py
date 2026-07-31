"""Immutable characteristic data on the two intersecting null faces."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


Array = np.ndarray


def _digest(
    outgoing: Mapping[str, Array],
    incoming: Mapping[str, Array],
    metadata: Mapping[str, Any],
) -> str:
    digest = hashlib.sha256()
    for side, values in (("outgoing", outgoing), ("incoming", incoming)):
        for name, value in sorted(values.items()):
            contiguous = np.ascontiguousarray(value)
            digest.update(side.encode())
            digest.update(name.encode())
            digest.update(str(contiguous.dtype).encode())
            digest.update(json.dumps(contiguous.shape).encode())
            digest.update(contiguous.tobytes())
    digest.update(json.dumps(dict(metadata), sort_keys=True, separators=(",", ":")).encode())
    return f"sha256:{digest.hexdigest()}"


@dataclass(frozen=True)
class BoundaryData:
    outgoing: Mapping[str, Array]
    incoming: Mapping[str, Array]
    metadata: Mapping[str, Any]
    content_hash: str

    @classmethod
    def create(
        cls,
        outgoing: Mapping[str, Array],
        incoming: Mapping[str, Array],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "BoundaryData":
        frozen: dict[str, dict[str, Array]] = {"outgoing": {}, "incoming": {}}
        for side, values in (("outgoing", outgoing), ("incoming", incoming)):
            for name, value in values.items():
                copied = np.asarray(value, dtype=np.float64).copy()
                copied.setflags(write=False)
                frozen[side][name] = copied
        meta = dict(metadata or {})
        content_hash = _digest(frozen["outgoing"], frozen["incoming"], meta)
        return cls(
            MappingProxyType(frozen["outgoing"]),
            MappingProxyType(frozen["incoming"]),
            MappingProxyType(meta),
            content_hash,
        )

    def verify_unchanged(self) -> None:
        actual = _digest(self.outgoing, self.incoming, self.metadata)
        if actual != self.content_hash:
            raise RuntimeError("immutable characteristic data were modified")

    @property
    def digest(self) -> str:
        return self.content_hash
