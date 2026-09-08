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


def test_apparent_horizon_control_changes_only_angular_resolution() -> None:
    standard = load_config(ROOT / "configs/experiments/exp08/standard.toml")
    control = load_config(ROOT / "configs/experiments/exp08/angular-control.toml")
    assert standard.coordinates == control.coordinates
    assert standard.physics == control.physics
    assert standard.initial_data == control.initial_data
    assert standard.initial_data["scalar_amplitude"] == -2.0
    assert (standard.angular.retained_degree, standard.angular.work_degree) == (4, 7)
    assert (control.angular.retained_degree, control.angular.work_degree) == (7, 14)
    assert standard.physics["curved_exponent"] == pytest.approx(25.0 / 24.0)
    assert standard.physics["target_v"] < standard.physics["v_max"]


def test_unknown_top_level_key_is_rejected(tmp_path: Path) -> None:
    original = (ROOT / "configs/experiments/exp01/smoke.toml").read_text()
    path = tmp_path / "bad.toml"
    path.write_text("unknown = 1\n" + original)
    with pytest.raises(ValueError):
        load_config(path)


def test_renamed_smoke_file_keeps_mode_and_passes_physical_parameters(tmp_path, monkeypatch) -> None:
    from dataclasses import replace
    from nee.experiments import campaigns
    path = tmp_path / 'renamed.toml'
    path.write_text((ROOT/'configs/experiments/exp01/smoke.toml').read_text())
    config = load_config(path)
    config = replace(config, physics={**config.physics, 'mass': 7.0},
                     solver=replace(config.solver, maximum_sweeps=3))
    captured = {}
    def exact(grid, u, v, mass):
        captured['mass'] = mass
        return None, {}
    def case(**kwargs):
        captured.update(kwargs)
        kwargs['builder'](None, kwargs['u'], kwargs['v'])
        return {'terminal_status': 'completed'}
    monkeypatch.setattr(campaigns.vacuum_benchmarks, 'regular_schwarzschild_state', exact)
    monkeypatch.setattr(campaigns.support, 'vacuum_case', case)
    campaigns.regular_vacuum(config, tmp_path/'output')
    assert config.experiment.mode == 'smoke'
    assert captured['mass'] == 7.0
    assert captured['iterations'] == 3


def test_fixed_protocol_settings_are_not_silently_ignored() -> None:
    from dataclasses import replace
    from nee.experiments.config_contract import validate_supported
    config = load_config(ROOT/'configs/experiments/exp01/standard.toml')
    validate_supported(config)
    with pytest.raises(ValueError, match='not an adjustable control'):
        validate_supported(replace(config, solver=replace(config.solver, relaxation=0.5)))
    with pytest.raises(ValueError, match='not an adjustable control'):
        validate_supported(replace(config, physics={**config.physics, 'misspelled_mass': 7}))
