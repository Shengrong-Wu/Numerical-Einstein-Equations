"""Complete first-order state evolved by one Picard sweep."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .fields import tangent_inverse, tensor_trace, tracefree


Array = np.ndarray
STATE_SCHEMA = "nee-theory-state-2"


@dataclass
class PicardState:
    g: Array
    b: Array
    log_Omega: Array
    Omega_chi: Array
    Omega_chib: Array
    zeta: Array
    Omega_omega: Array
    Omega_omegab: Array
    phi: Array | None = None
    Omega_e3phi: Array | None = None
    Omega_e4phi: Array | None = None
    nabla_phi: Array | None = None

    @property
    def Omega(self) -> Array:
        return np.exp(self.log_Omega)

    @property
    def inverse_g(self) -> Array:
        return tangent_inverse(self.g)

    @property
    def Omega_trchi(self) -> Array:
        return tensor_trace(self.Omega_chi, self.inverse_g)

    @property
    def Omega_trchib(self) -> Array:
        return tensor_trace(self.Omega_chib, self.inverse_g)

    @property
    def Omega_chih(self) -> Array:
        return tracefree(self.Omega_chi, self.g, self.inverse_g)

    @property
    def Omega_chibh(self) -> Array:
        return tracefree(self.Omega_chib, self.g, self.inverse_g)

    @property
    def trchi(self) -> Array:
        return self.Omega_trchi / self.Omega

    @property
    def trchib(self) -> Array:
        return self.Omega_trchib / self.Omega

    @property
    def chih(self) -> Array:
        return self.Omega_chih / self.Omega[..., None, None]

    @property
    def chibh(self) -> Array:
        return self.Omega_chibh / self.Omega[..., None, None]

    @property
    def is_scalar(self) -> bool:
        return self.phi is not None

    def arrays(self) -> dict[str, Array]:
        return {
            item.name: np.asarray(value)
            for item in fields(self)
            if (value := getattr(self, item.name)) is not None
        }

    def copy(self) -> "PicardState":
        return PicardState(
            **{
                item.name: None if (value := getattr(self, item.name)) is None else np.asarray(value).copy()
                for item in fields(self)
            }
        )

    def validate(self, frames: Array | None = None) -> dict[str, float]:
        from .validation import validate_state

        return validate_state(self, frames)

    def save(
        self,
        path: Path,
        *,
        u: Array,
        v: Array,
        extra: dict[str, Any] | None = None,
    ) -> None:
        arrays: dict[str, Any] = {
            "state_schema": np.asarray(STATE_SCHEMA),
            "u": np.asarray(u),
            "v": np.asarray(v),
            **self.arrays(),
        }
        if extra:
            overlap = set(arrays).intersection(extra)
            if overlap:
                raise ValueError(f"extra arrays overlap state fields: {sorted(overlap)}")
            arrays.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> tuple["PicardState", Array, Array, dict[str, Array]]:
        with np.load(path) as artifact:
            schema = str(artifact["state_schema"])
            if schema != STATE_SCHEMA:
                raise ValueError(f"unsupported state schema {schema!r}")
            names = {item.name for item in fields(cls)}
            values = {name: artifact[name].copy() if name in artifact.files else None for name in names}
            extra = {
                name: artifact[name].copy()
                for name in artifact.files
                if name not in names | {"state_schema", "u", "v"}
            }
            return cls(**values), artifact["u"].copy(), artifact["v"].copy(), extra
