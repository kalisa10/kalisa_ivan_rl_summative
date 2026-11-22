"""Ten independently trained hyperparameter configurations per algorithm.

The first configuration is the baseline. Each remaining configuration changes
one setting to make its effect easier to discuss.
"""

BASE = {
    "dqn": {"learning_rate": 0.001, "gamma": 0.99, "batch_size": 64,
            "epsilon_end": 0.05, "replay_capacity": 10000, "warmup": 128,
            "target_every": 100},
    "reinforce": {"lr": 0.001, "gamma": 0.99, "batch_episodes": 16,
                  "entropy_coef": 0.01},
    "ppo": {"lr": 0.001, "gamma": 0.99, "batch_episodes": 16,
            "ppo_epochs": 4, "clip_epsilon": 0.2, "value_coef": 0.5,
            "entropy_coef": 0.01},
    "a2c": {"lr": 0.001, "gamma": 0.99, "batch_episodes": 16,
            "value_coef": 0.5, "entropy_coef": 0.01},
}

CHANGES = {
    "dqn": [
        ("learning_rate", 0.0005), ("learning_rate", 0.002),
        ("gamma", 0.95), ("gamma", 0.995),
        ("batch_size", 32), ("batch_size", 128),
        ("epsilon_end", 0.1), ("target_every", 200), ("warmup", 256),
    ],
    "reinforce": [
        ("lr", 0.0005), ("lr", 0.002),
        ("gamma", 0.95), ("gamma", 0.995),
        ("batch_episodes", 8), ("batch_episodes", 32),
        ("entropy_coef", 0.0), ("entropy_coef", 0.03), ("gamma", 0.9),
    ],
    "ppo": [
        ("lr", 0.0005), ("lr", 0.002),
        ("gamma", 0.95), ("gamma", 0.995),
        ("batch_episodes", 8), ("batch_episodes", 32),
        ("clip_epsilon", 0.1), ("clip_epsilon", 0.3), ("ppo_epochs", 8),
    ],
    "a2c": [
        ("lr", 0.0005), ("lr", 0.002),
        ("gamma", 0.95), ("gamma", 0.995),
        ("batch_episodes", 8), ("batch_episodes", 32),
        ("entropy_coef", 0.0), ("entropy_coef", 0.03), ("value_coef", 1.0),
    ],
}

EXPERIMENTS = []
for algorithm, baseline in BASE.items():
    EXPERIMENTS.append({"id": f"{algorithm.upper()}_01", "algorithm": algorithm,
                        "change": "Baseline", "parameters": baseline.copy()})
    for number, (key, value) in enumerate(CHANGES[algorithm], start=2):
        params = baseline.copy()
        params[key] = value
        EXPERIMENTS.append({"id": f"{algorithm.upper()}_{number:02d}",
                            "algorithm": algorithm,
                            "change": f"{key} = {value}", "parameters": params})
