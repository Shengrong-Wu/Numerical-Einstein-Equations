from pathlib import Path

from nee.experiments.exp04_vacuum_strong_short_pulse import campaign


def test_standard_pulse_campaign_uses_requested_sweep_count(
    monkeypatch, tmp_path: Path
) -> None:
    requested: list[tuple[str, int]] = []

    def fake_run(**kwargs):
        requested.append((kwargs["label"], kwargs["iterations"]))
        return {"status": "completed"}

    monkeypatch.setattr(campaign, "_run_spectral_case", fake_run)
    monkeypatch.setattr(
        campaign, "_mapped_audit", lambda output, label, public: {"label": label}
    )
    monkeypatch.setattr(
        campaign, "_exact_zero_control_audit", lambda public: {"exact": True}
    )
    from nee.config import load_config
    public = load_config("configs/experiments/exp04/standard.toml")
    campaign.run_standard(tmp_path / "run", public=public, iterations=4)

    assert requested == [("strong-pulse", 4), ("zero-control", 4)]


def test_standard_audit_replays_canonical_evolution_grid():
    import math
    import numpy as np
    from nee.config import load_config
    from nee.numerics.lgl import CharacteristicLGLMesh

    public = load_config("configs/experiments/exp04/standard.toml")
    _, audit = campaign._audit_mesh(public)
    evolution = CharacteristicLGLMesh.create(
        np.linspace(0.0, math.log(2.0), 3), 8,
        np.sqrt(np.asarray(public.coordinates.v_breakpoints) / 0.5), 8, 0.5)
    np.testing.assert_array_equal(audit.u, evolution.u)
    np.testing.assert_array_equal(audit.v, evolution.v)


def test_standard_pulse_disables_unrequested_operational_checkpoints(monkeypatch, tmp_path):
    from nee.config import load_config
    observed = {}

    def fake_run(arguments):
        observed['checkpoints'] = arguments.checkpoint_every_sweep
        return {}

    monkeypatch.setattr(campaign.q1, 'run', fake_run)
    campaign._run_spectral_case(
        output_root=tmp_path, label='test', strength=1.0, iterations=6,
        public=load_config('configs/experiments/exp04/standard.toml'))
    assert observed['checkpoints'] is False
