"""Small tests for the policy network and TorchRL environment integration."""

import pytest
import numpy as np

pytest.importorskip("torchrl")

from training.dqn_training import QNetwork, greedy_action, make_torchrl_env, transition


def test_q_network_action():
    model = QNetwork()
    action = greedy_action(model, np.zeros(9, dtype=np.float32))
    assert 0 <= action < 5


def test_torchrl_episodes():
    env = make_torchrl_env()
    try:
        td = env.reset()
        assert td.get("observation").numel() == 9
        for _ in range(3):
            td, state, reward, done = transition(env, td, 4)
            assert state.shape == (9,)
            assert isinstance(reward, float)
        assert done
    finally:
        env.close()
