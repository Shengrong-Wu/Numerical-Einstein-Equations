"""Complete first-order state evolved by one Picard sweep."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from .fields import tangent_inverse, tensor_trace, tracefree


Array = np.ndarray
STATE_SCHEMA = "nee-full-weighted-state-1"


@dataclass
class PicardState:
    sphere_metric: Array
    shift: Array
    log_lapse: Array
    outgoing_null_form: Array
    incoming_null_form: Array
    torsion: Array
    outgoing_weighted_omega: Array
    incoming_weighted_omega: Array
    scalar: Array | None = None
    scalar_e3: Array | None = None
    scalar_e4: Array | None = None
    scalar_sphere_gradient: Array | None = None

    @property
    def lapse(self) -> Array:
        return np.exp(self.log_lapse)

    @property
    def inverse_metric(self) -> Array:
        return tangent_inverse(self.sphere_metric)

    @property
    def outgoing_expansion(self) -> Array:
        return tensor_trace(self.outgoing_null_form, self.inverse_metric)

    @property
    def incoming_expansion(self) -> Array:
        return tensor_trace(self.incoming_null_form, self.inverse_metric)

    @property
    def outgoing_shear(self) -> Array:
        return tracefree(self.outgoing_null_form, self.sphere_metric, self.inverse_metric)

    @property
    def incoming_shear(self) -> Array:
        return tracefree(self.incoming_null_form, self.sphere_metric, self.inverse_metric)

    @property
    def is_scalar(self) -> bool:
        return self.scalar is not None

    # Algebraic kernel views. These are derived aliases, never additional
    # stored state variables.
    @property
    def metric(self) -> Array:
        return self.sphere_metric

    @property
    def omega(self) -> Array:
        return self.lapse

    @property
    def log_omega(self) -> Array:
        return self.log_lapse

    @property
    def x_out(self) -> Array:
        return self.outgoing_null_form

    @property
    def x_in(self) -> Array:
        return self.incoming_null_form

    @property
    def zeta_up(self) -> Array:
        return self.torsion

    @property
    def w_out(self) -> Array:
        return self.outgoing_weighted_omega

    @property
    def w_in(self) -> Array:
        return self.incoming_weighted_omega

    @property
    def phi(self) -> Array | None:
        return self.scalar

    @property
    def p3(self) -> Array | None:
        return self.scalar_e3

    @property
    def p4(self) -> Array | None:
        return self.scalar_e4

    @property
    def grad_phi(self) -> Array | None:
        return self.scalar_sphere_gradient

    @property
    def q(self) -> Array:
        return self.outgoing_expansion / self.lapse**2

    @property
    def a_out(self) -> Array:
        return self.outgoing_expansion

    @property
    def a_in(self) -> Array:
        return self.incoming_expansion

    @property
    def sigma_out(self) -> Array:
        return self.outgoing_shear

    @property
    def sigma_in(self) -> Array:
        return self.incoming_shear

    @property
    def shear(self) -> Array:
        return self.outgoing_shear

    @property
    def weighted_chib(self) -> Array:
        return self.incoming_null_form

    @property
    def weighted_omega(self) -> Array:
        return self.outgoing_weighted_omega

    @property
    def weighted_omegab(self) -> Array:
        return self.incoming_weighted_omega

    @property
    def scalar_p(self) -> Array:
        if self.scalar_e4 is None:
            raise AttributeError("vacuum state has no outgoing scalar derivative")
        return self.scalar_e4

    @property
    def incoming_scalar(self) -> Array:
        if self.scalar_e3 is None:
            raise AttributeError("vacuum state has no incoming scalar derivative")
        return self.scalar_e3

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
