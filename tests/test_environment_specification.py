"""Regression checks for the documented observation, transition, and reward contract."""

import numpy as np
import pytest

from environment import PatientNavigationEnv


def test_observation_provenance_and_cadence():
    env = PatientNavigationEnv()
    try:
        obs, _ = env.reset(seed=42)
        assert obs.shape == (9,)
        assert obs.dtype == np.float32
        assert env.observation_space.contains(obs)
        assert all(v in (0.0, 1.0) for v in obs[:3])
        assert 0.3 <= obs[3] < 0.9
        assert np.array_equal(obs[4:8], np.zeros(4))
        assert obs[8] == 0.0
        initial_profile = obs[:4].copy()
        for step_index in range(1, 4):
            obs, reward, terminated, truncated, info = env.step(4)
            assert np.array_equal(obs[:4], initial_profile)
            assert np.array_equal(obs[4:8], np.zeros(4))
            assert obs[8] == pytest.approx(step_index / 3)
            assert isinstance(reward, float)
            assert terminated is (step_index == 3)
            assert truncated is False
        assert isinstance(info["attended"], bool)
        assert 0.05 <= info["attendance_probability"] <= 0.95
    finally:
        env.close()


def test_action_dependent_persistent_state_and_intervention_cost():
    env = PatientNavigationEnv()
    try:
        env.reset(seed=42)
        obs, reward, done, _, _ = env.step(2)
        assert reward == pytest.approx(-0.20)
        assert obs[6] == 1.0
        assert not done
        obs, repeated_reward, done, _, _ = env.step(2)
        assert repeated_reward == pytest.approx(-0.27)
        assert obs[6] == 1.0
        assert not done
    finally:
        env.close()


def test_reset_with_same_seed_reproduces_patient_profile():
    env = PatientNavigationEnv()
    try:
        first, _ = env.reset(seed=2026)
        second, _ = env.reset(seed=2026)
        assert np.array_equal(first, second)
    finally:
        env.close()
