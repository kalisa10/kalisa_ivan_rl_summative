"""Simple checks for renderer mapping; no display needed."""
from environment.rendering import lerp, target_for_action


def test_action_destinations():
    assert [target_for_action(i) for i in range(5)] == [
        "OUTREACH", "OUTREACH", "TRANSPORT", "TELEHEALTH", "HOME"
    ]


def test_animation_interpolation():
    assert lerp((0, 0), (10, 20), -1) == (0, 0)
    assert lerp((0, 0), (10, 20), 1) == (10, 20)
    assert lerp((0, 0), (10, 20), 0.5) == (5, 10)
