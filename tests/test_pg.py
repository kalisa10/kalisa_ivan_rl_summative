"""Policy gradient checks; small and fast, without full 2,000-episode runs."""
import numpy as np
import pytest
import torch

pytest.importorskip("torchrl")

from training.pg_training import Actor, Critic, discounted_returns, greedy_action, collect_episode
from training.dqn_training import make_torchrl_env


def test_policy_actions():
    actor = Actor()
    critic = Critic()
    observation = torch.zeros(1, 9)
    assert actor(observation).shape == (1, 5)
    assert critic(observation).shape == (1,)
    assert 0 <= greedy_action(actor, np.zeros(9, dtype=np.float32)) < 5


def test_discounted_returns():
    assert torch.allclose(discounted_returns([1.0, 2.0], 0.5), torch.tensor([2.0, 2.0]))


def test_pg_collect_episode():
    env = make_torchrl_env()
    try:
        s, a, r, done = collect_episode(env, Actor())
        assert len(s) == len(a) == len(r) == len(done) == 3
        assert done[-1] is True
    finally:
        env.close()
