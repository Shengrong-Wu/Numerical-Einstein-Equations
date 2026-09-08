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
