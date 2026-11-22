"""Check the serialized Python-to-Three.js episode contract (no browser required)."""

from visual3d.server import HERE, simulate


def test_episode_payload_has_frontend_contract():
    payload = simulate(42, "random")
    assert set(("algorithm", "seed", "profile", "events", "total_reward", "connected", "attendance_probability", "note")) <= payload.keys()
    assert len(payload["events"]) == 3
    assert set(("phone_access", "travel_barrier", "telehealth_access", "missed_visit_risk")) <= payload["profile"].keys()
    assert all(set(("step", "action_id", "action", "reward", "flags")) <= e.keys() for e in payload["events"])
    assert all(len(e["flags"]) == 4 for e in payload["events"])
    assert set(("attended", "probability")) <= payload["events"][-1].keys()
    assert isinstance(payload["connected"], bool)


def test_supplied_selector_controls_each_action():
    # Stub selector proves each event comes from the action selector, not a scripted browser route.
    payload = simulate(42, "test-policy", selector=lambda observation: 4)
    assert [event["action_id"] for event in payload["events"]] == [4, 4, 4]
    assert [event["action"] for event in payload["events"]] == ["No additional action"] * 3
    assert [event["flags"] for event in payload["events"]] == [[0, 0, 0, 0]] * 3
    assert payload["algorithm"] == "TEST-POLICY"


def test_threejs_consumes_episode_endpoint_and_displays_outcomes():
    html = (HERE / "index.html").read_text(encoding="utf-8")
    assert "fetch('/api/episode?seed='" in html
    assert "THREE.WebGLRenderer" in html
    assert "episode.events" in html
    assert "episode.total_reward" in html
    assert "episode.connected" in html
