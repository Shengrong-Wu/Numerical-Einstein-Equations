from pathlib import Path

import numpy as np
import pytest

from nee.io.boundary_artifact import load_boundary_data, save_boundary_data
from nee.state.boundary import BoundaryData
from nee.state.iterate import PicardState


def state() -> PicardState:
    shape = (4, 3, 5)
    metric = np.zeros(shape + (3, 3))
    metric[..., 0, 0] = 1.0
    metric[..., 1, 1] = 1.0
    return PicardState(
        sphere_metric=metric,
        shift=np.zeros(shape + (3,)),
        log_lapse=np.zeros(shape),
        outgoing_null_form=metric.copy(),
        incoming_null_form=-metric.copy(),
        torsion=np.zeros(shape + (3,)),
        outgoing_weighted_omega=np.zeros(shape),
        incoming_weighted_omega=np.zeros(shape),
    )


def test_state_copy_has_no_shared_arrays() -> None:
    original = state()
    copied = original.copy()
    for name, value in original.arrays().items():
        assert not np.shares_memory(value, copied.arrays()[name])


def test_state_round_trip(tmp_path: Path) -> None:
    original = state()
    path = tmp_path / "state.npz"
    u = np.linspace(-1.0, -0.5, 3)
    v = np.linspace(0.0, 0.5, 5)
    original.save(path, u=u, v=v, extra={"update_map": np.ones((3, 5))})
    loaded, loaded_u, loaded_v, extra = PicardState.load(path)
    np.testing.assert_array_equal(loaded_u, u)
    np.testing.assert_array_equal(loaded_v, v)
    np.testing.assert_array_equal(loaded.sphere_metric, original.sphere_metric)
    np.testing.assert_array_equal(extra["update_map"], 1.0)


def test_boundary_arrays_are_read_only_and_hashed() -> None:
    boundary = BoundaryData.create({"x": np.arange(4.0)}, {"y": np.arange(3.0)})
    with pytest.raises(ValueError):
        boundary.outgoing["x"][0] = 9.0
    boundary.verify_unchanged()
    assert boundary.content_hash.startswith("sha256:")


def test_boundary_artifact_round_trip(tmp_path: Path) -> None:
    boundary = BoundaryData.create(
        {"x": np.arange(4.0)}, {"y": np.arange(3.0)}, metadata={"case": "unit"}
    )
    path = tmp_path / "boundary.npz"
    save_boundary_data(path, boundary, u=np.arange(3.0), v=np.arange(4.0))
    loaded, u, v = load_boundary_data(path)
    assert loaded.content_hash == boundary.content_hash
    np.testing.assert_array_equal(u, np.arange(3.0))
    np.testing.assert_array_equal(v, np.arange(4.0))

