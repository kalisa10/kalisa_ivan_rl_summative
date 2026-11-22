"""Synthetic patient navigation environment for an RL assignment.

This simulates appointment access, not medical decisions or real patient care.
"""

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class PatientNavigationEnv(gym.Env):
    """An agent gets up to three actions to help one simulated patient."""

    ACTION_NAMES = (
        "Send appointment reminder",
        "Call patient",
        "Arrange transport",
        "Coordinate telehealth",
        "No additional action",
    )

    def __init__(self):
        super().__init__()
        self.action_space = spaces.Discrete(len(self.ACTION_NAMES))
        # Features are normalized to [0, 1]; see _observation() below.
        self.observation_space = spaces.Box(0.0, 1.0, shape=(9,), dtype=np.float32)
        self.max_steps = 3
        self.steps = 0
        self.flags = np.zeros(4, dtype=np.float32)
        self.profile = np.zeros(4, dtype=np.float32)

    def _observation(self):
        # phone_access, travel_barrier, telehealth_access, missed-visit risk,
        # reminder_sent, called, transport_arranged, telehealth_coordinated,
        # fraction of navigation window elapsed.
        return np.concatenate((
            self.profile,
            self.flags,
            np.array([self.steps / self.max_steps], dtype=np.float32),
        )).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.flags[:] = 0
        phone = float(self.np_random.integers(0, 2))
        travel_barrier = float(self.np_random.integers(0, 2))
        telehealth_access = float(self.np_random.integers(0, 2))
        risk = float(self.np_random.uniform(0.3, 0.9))
        self.profile = np.array(
            [phone, travel_barrier, telehealth_access, risk], dtype=np.float32
        )
        return self._observation(), {}

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        if self.steps >= self.max_steps:
            raise RuntimeError("Episode finished; call reset() before step().")

        self.steps += 1
        reward = -0.02  # Small cost for each navigation action/time step.
        info = {"action": self.ACTION_NAMES[int(action)]}

        if action < 4:
            if self.flags[action]:
                reward -= 0.25  # Repeating the same intervention wastes resources.
            elif action == 0 and not self.profile[0]:
                reward -= 0.35  # Reminder cannot be delivered without phone access.
            elif action == 3 and not self.profile[2]:
                reward -= 0.35  # Telehealth unavailable for this patient.
            else:
                self.flags[action] = 1.0
                reward -= [0.05, 0.10, 0.18, 0.15][action]

        terminated = self.steps == self.max_steps
        if terminated:
            phone, travel_barrier, telehealth_access, risk = self.profile
            reminder, called, transport, telehealth = self.flags
            # Synthetic probability of attending/connecting to care.
            probability = (
                0.60 - 0.40 * risk - 0.20 * travel_barrier
                + 0.15 * reminder * phone + 0.25 * called
                + 0.30 * transport * travel_barrier
                + 0.30 * telehealth * telehealth_access
            )
            probability = float(np.clip(probability, 0.05, 0.95))
            attended = bool(self.np_random.random() < probability)
            reward += 2.0 if attended else -1.0
            info.update(attended=attended, attendance_probability=probability)

        return self._observation(), float(reward), terminated, False, info
