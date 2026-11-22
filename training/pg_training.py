"""Train REINFORCE, PPO or A2C on the same synthetic SmartCare environment.

The environment is wrapped by TorchRL. All three policies use a small
PyTorch actor; PPO and A2C also use a value network (critic).
"""

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torchrl.envs import GymWrapper
from torchrl.envs.utils import step_mdp

from environment import PatientNavigationEnv

ALGORITHMS = ("reinforce", "ppo", "a2c")


class Actor(nn.Module):
    """Return five action logits from the nine patient-state features."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(9, 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh(),
                                     nn.Linear(64, 5))

    def forward(self, observations):
        return self.network(observations)


class Critic(nn.Module):
    """Estimate the expected return from a patient state."""

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(9, 64), nn.Tanh(),
                                     nn.Linear(64, 64), nn.Tanh(),
                                     nn.Linear(64, 1))

    def forward(self, observations):
        return self.network(observations).squeeze(-1)


def load_policy(algorithm, path=None):
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Choose one of {ALGORITHMS}")
    path = Path(path or f"models/pg/{algorithm}.pth")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["algorithm"] != algorithm:
        raise ValueError("Model algorithm does not match --algorithm")
    actor = Actor()
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor


def greedy_action(actor, observation):
    with torch.no_grad():
        obs = torch.as_tensor(observation, dtype=torch.float32).reshape(1, -1)
        return int(actor(obs).argmax(dim=-1).item())


def discounted_returns(rewards, gamma):
    values = []
    future = 0.0
    for reward in reversed(rewards):
        future = reward + gamma * future
        values.append(future)
    return torch.tensor(list(reversed(values)), dtype=torch.float32)


def collect_episode(env, actor):
    """Gather one complete three-decision patient-navigation trajectory."""
    td = env.reset()
    states, actions, rewards, dones = [], [], [], []
    finished = False
    while not finished:
        obs = td.get("observation").detach().cpu().reshape(-1).clone()
        with torch.no_grad():
            action = Categorical(logits=actor(obs.unsqueeze(0))).sample().item()
        td.set("action", torch.tensor(action, dtype=torch.long).reshape(env.action_spec.shape))
        result = env.step(td)
        following = result.get("next")
        reward = float(following.get("reward").item())
        finished = bool(following.get("done").item())
        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        dones.append(finished)
        td = step_mdp(result)
    return states, actions, rewards, dones


def train(algorithm="ppo", episodes=2000, seed=42, lr=0.001, gamma=0.99,
          batch_episodes=16, ppo_epochs=4, clip_epsilon=0.2,
          value_coef=0.5, entropy_coef=0.01, save_path=None, log_path=None):
    """Use the same environment, episode budget and observation/action spaces."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    if episodes < 1 or batch_episodes < 1 or ppo_epochs < 1 or lr <= 0:
        raise ValueError("Episode counts and learning rate must be positive")
    if not (0 <= gamma <= 1) or not (0 < clip_epsilon < 1):
        raise ValueError("gamma or clip_epsilon is out of range")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = GymWrapper(PatientNavigationEnv(), categorical_action_encoding=True)
    env.set_seed(seed)
    actor = Actor()
    critic = Critic() if algorithm != "reinforce" else None
    parameters = list(actor.parameters()) + (list(critic.parameters()) if critic else [])
    optimizer = torch.optim.Adam(parameters, lr=lr)
    history = []
    update_metrics = []
    updates = 0
    try:
        for start in range(0, episodes, batch_episodes):
            states, actions, returns = [], [], []
            episode_scores = []
            for _ in range(min(batch_episodes, episodes - start)):
                s, a, r, _ = collect_episode(env, actor)
                states.extend(s)
                actions.extend(a)
                returns.extend(discounted_returns(r, gamma).tolist())
                episode_scores.append(sum(r))
                history.append({"episode": len(history) + 1, "reward": round(sum(r), 5)})

            states = torch.stack(states)
            actions = torch.tensor(actions, dtype=torch.long)
            returns = torch.tensor(returns, dtype=torch.float32)
            with torch.no_grad():
                old_log_probs = Categorical(logits=actor(states)).log_prob(actions)
                old_values = critic(states) if critic else None
                advantages = returns - old_values if critic else returns
                # Center advantages to reduce variance across sampled patients.
                if advantages.numel() > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            epochs = ppo_epochs if algorithm == "ppo" else 1
            for _ in range(epochs):
                distribution = Categorical(logits=actor(states))
                log_probs = distribution.log_prob(actions)
                entropy = distribution.entropy().mean()
                if algorithm == "ppo":
                    ratio = torch.exp(log_probs - old_log_probs)
                    unclipped = ratio * advantages
                    clipped = ratio.clamp(1 - clip_epsilon, 1 + clip_epsilon) * advantages
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                else:
                    policy_loss = -(log_probs * advantages).mean()

                loss = policy_loss - entropy_coef * entropy
                value_loss = None
                if critic is not None:
                    value_loss = nn.functional.mse_loss(critic(states), returns)
                    loss = loss + value_coef * value_loss
                update_metrics.append({
                    "update": updates + 1,
                    "episode": len(history),
                    "policy_loss": float(policy_loss.detach().item()),
                    "value_loss": float(value_loss.detach().item()) if value_loss is not None else "",
                    "entropy": float(entropy.detach().item()),
                    "total_loss": float(loss.detach().item()),
                })
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                updates += 1

            if len(history) % max(1, episodes // 10) < batch_episodes or len(history) == episodes:
                recent = np.mean([row["reward"] for row in history[-min(100, len(history)):]])
                print(f"{algorithm.upper()} episode {len(history)}/{episodes} | "
                      f"mean reward (last 100): {recent:.3f} | updates: {updates}", flush=True)
    finally:
        env.close()

    save_path = Path(save_path or f"models/pg/{algorithm}.pth")
    log_path = Path(log_path or f"logs/{algorithm}_episodes.csv")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"algorithm": algorithm, "actor": actor.state_dict(),
                "episodes": episodes, "seed": seed}, save_path)
    with log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["episode", "reward"])
        writer.writeheader()
        writer.writerows(history)
    # Entropy and loss values are logged per optimizer update, not fabricated from rewards.
    updates_path = log_path.with_name(log_path.stem + "_updates.csv")
    with updates_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["update", "episode", "policy_loss", "value_loss", "entropy", "total_loss"])
        writer.writeheader()
        writer.writerows(update_metrics)
    print(f"Saved: {save_path} | log: {log_path} | optimizer log: {updates_path}")
    return history


def evaluate(algorithm, episodes=100, seed=1000, path=None):
    actor = load_policy(algorithm, path)
    scores = {"trained": [], "random": []}
    successes = {"trained": [], "random": []}
    for label in scores:
        env = PatientNavigationEnv()
        try:
            for i in range(episodes):
                state, _ = env.reset(seed=seed + i)
                rng = random.Random(seed + i)
                reward_total = 0.0
                finished = False
                while not finished:
                    action = greedy_action(actor, state) if label == "trained" else rng.randrange(5)
                    state, reward, terminated, truncated, info = env.step(action)
                    reward_total += reward
                    finished = terminated or truncated
                scores[label].append(reward_total)
                successes[label].append(int(info["attended"]))
        finally:
            env.close()
    for label in scores:
        print(f"{label:7s} | mean reward: {np.mean(scores[label]):.3f} "
              f"| appointment connection: {np.mean(successes[label])*100:.1f}%")
    return scores


def main():
    parser = argparse.ArgumentParser(description="SmartCare policy-gradient training")
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch-episodes", type=int, default=16)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        evaluate(args.algorithm, episodes=args.episodes, seed=args.seed)
    else:
        train(args.algorithm, episodes=args.episodes, seed=args.seed, lr=args.lr,
              gamma=args.gamma, batch_episodes=args.batch_episodes,
              ppo_epochs=args.ppo_epochs, clip_epsilon=args.clip_epsilon)


if __name__ == "__main__":
    main()
