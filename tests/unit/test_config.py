from pathlib import Path

import pytest

from nee.config import load_config


ROOT = Path(__file__).resolve().parents[2]


def test_all_declared_configs_validate() -> None:
    paths = sorted((ROOT / "configs" / "experiments").glob("exp*/*.toml"))
    assert len(paths) == 17
    for path in paths:
        config = load_config(path)
        assert config.coordinates.node_count(config.coordinates.u_degrees) > 1
        assert config.coordinates.node_count(config.coordinates.v_degrees) > 1


def test_reported_crossed_pulse_dimensions() -> None:
    config = load_config(ROOT / "configs/experiments/exp05/standard.toml")
    assert config.coordinates.node_count(config.coordinates.u_degrees) == 61
    assert config.coordinates.node_count(config.coordinates.v_degrees) == 61
    assert (config.angular.point_count, config.angular.retained_degree) == (300, 8)
    assert (config.angular.work_degree, config.angular.differentiation_degree) == (14, 15)


def test_trapped_section_control_changes_only_angular_resolution() -> None:
    standard = load_config(ROOT / "configs/experiments/exp08/standard.toml")
    control = load_config(ROOT / "configs/experiments/exp08/angular-control.toml")
    assert standard.coordinates == control.coordinates
    assert standard.physics == control.physics
    assert standard.initial_data == control.initial_data
    assert standard.initial_data["scalar_amplitude"] == -2.0
    assert (standard.angular.retained_degree, standard.angular.work_degree) == (5, 10)
    assert (control.angular.retained_degree, control.angular.work_degree) == (7, 14)


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    original = (ROOT / "configs/experiments/exp01/smoke.toml").read_text()
    path = tmp_path / "bad.toml"
    path.write_text("unknown = 1\n" + original)
    with pytest.raises(ValueError):
        load_config(path)
