"""Deterministic, non-graphics checks for the advanced 3D integration."""
from pathlib import Path
from visual3d.server import simulate, load_selector, HERE


def test_real_synthetic_episode_has_three_steps():
    outcome = simulate(42, "random")

    assert len(outcome["events"]) == 3
    assert [event["step"] for event in outcome["events"]] == [1, 2, 3]
    assert isinstance(outcome["total_reward"], (int, float))
    assert isinstance(outcome["connected"], bool)
    assert 0.05 <= outcome["attendance_probability"] <= 0.95


def test_seeded_profile_stable():
    a, b = simulate(143, "random"), simulate(143, "random")
    assert a["profile"] == b["profile"]
    assert a["events"] == b["events"]


def test_missing_checkpoint_is_rejected():
    try:
        load_selector("dqn", "models/not-a-real-model.pth")
    except FileNotFoundError:
        return
    raise AssertionError("A nonexistent trained checkpoint should not silently use random decisions")


def test_threejs_frontend_is_included():
    text = (HERE / "index.html").read_text(encoding="utf-8")
    assert 'three.module.js' in text
    assert '/api/episode?seed=' in text
    assert 'WebGLRenderer' in text

