from environment import PatientNavigationEnv


def test_environment_episode():
    env = PatientNavigationEnv()
    observation, _ = env.reset(seed=42)
    assert env.observation_space.contains(observation)
    for _ in range(3):
        observation, reward, done, truncated, info = env.step(4)
        assert env.observation_space.contains(observation)
        assert isinstance(reward, float)
        assert not truncated
    assert done
    assert "attended" in info
    env.close()
