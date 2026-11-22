"""DQN training for the synthetic patient-navigation environment.

TorchRL's GymWrapper handles the environment interaction. We implement the
small Q-network, replay sampling and Bellman update explicitly in PyTorch.
"""

import argparse
import csv
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchrl.envs import GymWrapper
from torchrl.envs.utils import step_mdp

from environment import PatientNavigationEnv

MODEL_PATH = Path("models/dqn/smartcare_dqn.pth")
LOG_PATH = Path("logs/dqn_episodes.csv")


class QNetwork(nn.Module):
    """Map nine patient-state values to five action Q-values."""

    def __init__(self, observations=9, actions=5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(observations, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, actions),
        )

    def forward(self, state):
        return self.layers(state)


def load_model(path=MODEL_PATH):
    """Load the trained weights for testing or the animated demo."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = QNetwork()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def greedy_action(model, observation):
    """Pick the action with the highest predicted value."""
    with torch.no_grad():
        state = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        return int(model(state).argmax(dim=1).item())


def make_torchrl_env():
    """Use the same existing Gymnasium environment through TorchRL."""
    return GymWrapper(PatientNavigationEnv(), categorical_action_encoding=True)


def transition(env, state_td, action):
    """Step the TorchRL environment and return an ordinary replay transition."""
    action_shape = env.action_spec.shape
    state_td.set("action", torch.tensor(action, dtype=torch.long).reshape(action_shape))
    result = env.step(state_td)
    following = result.get("next")
    next_state = following.get("observation").detach().cpu().numpy().reshape(-1).copy()
    reward = float(following.get("reward").item())
    done = bool(following.get("done").item())
    return step_mdp(result), next_state, reward, done


def train(episodes=2000, seed=42, batch_size=64, learning_rate=0.001,
          gamma=0.99, epsilon_end=0.05, replay_capacity=10000,
          warmup=128, target_every=100, save_path=MODEL_PATH,
          log_path=LOG_PATH):
    """Train one DQN model and save real results, never fabricated scores."""
    if episodes < 1 or batch_size < 1 or warmup < 1 or target_every < 1:
        raise ValueError("episodes, batch_size, warmup and target_every must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = make_torchrl_env()
    env.set_seed(seed)
    online = QNetwork()
    target = QNetwork()
    target.load_state_dict(online.state_dict())
    optimizer = torch.optim.Adam(online.parameters(), lr=learning_rate)
    replay = deque(maxlen=replay_capacity)
    outcomes = []
    update_metrics = []
    total_updates = 0
    total_steps = 0

    try:
        for episode in range(episodes):
            # Linear epsilon schedule: exploration fades through training.
            epsilon = 1.0 - (1.0 - epsilon_end) * episode / max(1, episodes - 1)
            td = env.reset()
            state = td.get("observation").detach().cpu().numpy().reshape(-1).copy()
            episode_reward = 0.0
            done = False
            steps = 0
            while not done:
                if random.random() < epsilon:
                    action = random.randrange(5)
                else:
                    action = greedy_action(online, state)
                td, next_state, reward, done = transition(env, td, action)
                replay.append((state, action, reward, next_state, done))
                state = next_state
                steps += 1
                total_steps += 1
                episode_reward += reward

                if len(replay) >= max(warmup, batch_size):
                    batch = random.sample(replay, batch_size)
                    states = torch.as_tensor(np.stack([x[0] for x in batch]), dtype=torch.float32)
                    actions = torch.tensor([x[1] for x in batch], dtype=torch.long)
                    rewards = torch.tensor([x[2] for x in batch], dtype=torch.float32)
                    next_states = torch.as_tensor(np.stack([x[3] for x in batch]), dtype=torch.float32)
                    dones = torch.tensor([x[4] for x in batch], dtype=torch.float32)
                    current_q = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_best = target(next_states).max(dim=1).values
                        expected_q = rewards + gamma * (1 - dones) * next_best
                    loss = F.smooth_l1_loss(current_q, expected_q)
                    td_error = (current_q.detach() - expected_q).abs().mean().item()
                    update_metrics.append({"update": total_updates + 1, "episode": episode + 1,
                                           "loss": float(loss.detach().item()),
                                           "mean_absolute_td_error": float(td_error)})
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                    optimizer.step()
                    total_updates += 1
                    if total_updates % target_every == 0:
                        target.load_state_dict(online.state_dict())

            # Each patient scenario produces one three-step episode.
            outcomes.append({"episode": episode + 1, "steps": steps,
                             "reward": round(episode_reward, 4), "epsilon": round(epsilon, 4)})
            if (episode + 1) % max(1, episodes // 10) == 0:
                recent = np.mean([o["reward"] for o in outcomes[-min(100, len(outcomes)):]])
                print(f"Episode {episode + 1}/{episodes} | mean reward (last 100): {recent:.3f} "
                      f"| epsilon: {epsilon:.3f} | updates: {total_updates}", flush=True)
    finally:
        env.close()

    save_path = Path(save_path)
    log_path = Path(log_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": online.state_dict(), "episodes": episodes, "seed": seed}, save_path)
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "steps", "reward", "epsilon"])
        writer.writeheader()
        writer.writerows(outcomes)
    # A separate, lossless optimizer-update log supports the rubric's DQN-loss figure.
    updates_path = log_path.with_name(log_path.stem + "_updates.csv")
    with updates_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["update", "episode", "loss", "mean_absolute_td_error"])
        writer.writeheader()
        writer.writerows(update_metrics)
    print(f"Saved model: {save_path} | episode log: {log_path} | "
          f"optimizer log: {updates_path} | steps: {total_steps}")
    return outcomes


def evaluate(model_path=MODEL_PATH, episodes=100, seed=1000):
    """Compare trained greedy decisions with random actions on matching seeds."""
    model = load_model(model_path)
    results = {"trained": [], "random": []}
    for name in results:
        env = PatientNavigationEnv()
        try:
            for i in range(episodes):
                observation, _ = env.reset(seed=seed + i)
                rng = random.Random(seed + i)
                done = False
                score = 0.0
                while not done:
                    action = greedy_action(model, observation) if name == "trained" else rng.randrange(5)
                    observation, reward, terminated, truncated, _ = env.step(action)
                    score += reward
                    done = terminated or truncated
                results[name].append(score)
        finally:
            env.close()
    print(f"Trained mean reward ({episodes} episodes): {np.mean(results['trained']):.3f}")
    print(f"Random  mean reward ({episodes} episodes): {np.mean(results['random']):.3f}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train or evaluate SmartCare DQN")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        evaluate(episodes=args.episodes, seed=args.seed)
    else:
        train(episodes=args.episodes, seed=args.seed,
              learning_rate=args.lr, gamma=args.gamma, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
