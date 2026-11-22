"""No checkpoint or trained TorchRL runtime required for these focused tests."""

import numpy as np
import pytest

from diagnostics.generalization import (
    AccessBarrierStressEnv, paired_bootstrap, run_episode, summarize,
)
from environment import PatientNavigationEnv


def test_stress_profile_is_documented_and_reproducible():
    env = AccessBarrierStressEnv()
    try:
        a, _ = env.reset(seed=123)
        b, _ = env.reset(seed=123)
        assert np.allclose(a, b)
        assert np.array_equal(a[:3], [0.0, 1.0, 0.0])
        assert 0.80 <= float(a[3]) < 0.90
        assert env.observation_space.contains(a)
        _, _, _, _, _ = env.step(2)
        assert env.flags[2] == 1
    finally:
        env.close()


def test_frozen_three_step_evaluation_is_repeatable():
    act = lambda obs, rng: 4
    a = run_episode(PatientNavigationEnv, 7000, act)
    b = run_episode(PatientNavigationEnv, 7000, act)
    assert a == b
    assert a["actions"] == [4, 4, 4]
    assert a["connected"] in (0, 1)


def test_bootstrap_is_paired_and_reproducible():
    a = [1.0, 2.0, 3.0, 4.0]
    b = [0.0, 1.0, 2.0, 3.0]
    one = paired_bootstrap(a, b, resamples=50, seed=11)
    two = paired_bootstrap(a, b, resamples=50, seed=11)
    assert one == two
    assert one["mean_difference"] == pytest.approx(1.0)
    assert one["ci95_lower"] == pytest.approx(1.0)
    assert one["ci95_upper"] == pytest.approx(1.0)


def test_summary_reports_unfabricated_means():
    agent = [{"reward": 1.0, "connected": 1, "probability": 0.7},
             {"reward": -1.0, "connected": 0, "probability": 0.2}]
    random = [{"reward": 0.0, "connected": 0, "probability": 0.3},
              {"reward": -1.0, "connected": 0, "probability": 0.3}]
    out = summarize(agent, baseline=random, resamples=30)
    assert out["mean_reward"] == pytest.approx(0.0)
    assert out["connection_rate"] == pytest.approx(0.5)
    assert out["reward_vs_random"]["mean_difference"] == pytest.approx(0.5)


def test_training_seed_summary_uses_between_seed_variance():
    from diagnostics.seed_stability import summarize_seed_runs
    rows = [
        {"training_seed": 101, "trained_mean_reward": 0.5, "trained_connection_rate": 0.6},
        {"training_seed": 202, "trained_mean_reward": 1.0, "trained_connection_rate": 0.7},
        {"training_seed": 303, "trained_mean_reward": 1.5, "trained_connection_rate": 0.8},
    ]
    result = summarize_seed_runs(rows)
    assert result["n_training_seeds"] == 3
    assert result["mean_reward_across_seeds"] == pytest.approx(1.0)
    assert result["sd_reward_across_seeds"] == pytest.approx(0.5)
