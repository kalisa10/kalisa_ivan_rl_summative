"""Run and compare hyperparameter experiments on the same SmartCare environment.

Usage:
    uv run python -m experiments.run_experiments --list
    uv run python -m experiments.run_experiments --experiment PPO_01
    uv run python -m experiments.run_experiments --algorithm ppo
    uv run python -m experiments.run_experiments --all

Results are only recorded after a run finishes successfully.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np

from experiments.settings import EXPERIMENTS

ROOT = Path("experiments/results")
EVAL_SEED = 1000


def evaluate_checkpoint(algorithm, model_path, episodes=100, seed=EVAL_SEED):
    """Evaluate greedy actions and random actions on matching patient seeds."""
    from environment import PatientNavigationEnv

    if algorithm == "dqn":
        from training.dqn_training import load_model, greedy_action
        model = load_model(model_path)
    else:
        from training.pg_training import load_policy, greedy_action
        model = load_policy(algorithm, model_path)

    scores = {"trained": [], "random": []}
    successes = {"trained": [], "random": []}
    for label in ("trained", "random"):
        env = PatientNavigationEnv()
        try:
            for i in range(episodes):
                observation, _ = env.reset(seed=seed + i)
                rng = random.Random(seed + i)
                total = 0.0
                finished = False
                while not finished:
                    action = (greedy_action(model, observation) if label == "trained"
                              else rng.randrange(env.action_space.n))
                    observation, reward, terminated, truncated, info = env.step(action)
                    total += reward
                    finished = terminated or truncated
                scores[label].append(total)
                successes[label].append(int(info["attended"]))
        finally:
            env.close()
    return {
        "trained_mean_reward": float(np.mean(scores["trained"])),
        "trained_reward_std": float(np.std(scores["trained"])),
        "trained_connection_rate": float(np.mean(successes["trained"])),
        "random_mean_reward": float(np.mean(scores["random"])),
        "random_connection_rate": float(np.mean(successes["random"])),
    }


def train_one(spec, episodes, eval_episodes, output_root=ROOT):
    """Train independently, evaluate, and write one experiment's artifacts."""
    name = spec["id"]
    algorithm = spec["algorithm"]
    folder = Path(output_root) / algorithm / name
    folder.mkdir(parents=True, exist_ok=True)
    model_path = folder / "model.pth"
    log_path = folder / "episodes.csv"
    params = spec["parameters"].copy()
    params.update(episodes=episodes, seed=42, save_path=model_path, log_path=log_path)
    if algorithm == "dqn":
        from training.dqn_training import train
        history = train(**params)
    else:
        from training.pg_training import train
        history = train(algorithm=algorithm, **params)

    metrics = evaluate_checkpoint(algorithm, model_path, episodes=eval_episodes)
    result = {
        "experiment": name,
        "algorithm": algorithm,
        "episodes": episodes,
        "seed": 42,
        "evaluation_episodes": eval_episodes,
        "evaluation_seed_start": EVAL_SEED,
        "parameters": spec["parameters"],
        "model_path": str(model_path),
        "episode_log": str(log_path),
        "final_100_training_mean_reward": float(np.mean([row["reward"] for row in history[-100:]])),
        **metrics,
    }
    # Write the completion marker LAST; interrupted runs are not treated as complete.
    (folder / "summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def load_results(root=ROOT):
    """Only accept completed runs with their corresponding model and episode log."""
    rows = []
    for path in sorted(Path(root).glob("*/*/summary.json")):
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if Path(entry["model_path"]).exists() and Path(entry["episode_log"]).exists():
                rows.append(entry)
        except (ValueError, KeyError, OSError):
            print(f"Ignoring unreadable summary: {path}")
    return rows


def write_reports(root=ROOT):
    """Build report-ready tables and bar charts from real completed results."""
    results = load_results(root)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fields = ["experiment", "algorithm", "episodes", "trained_mean_reward",
              "trained_reward_std", "trained_connection_rate", "random_mean_reward",
              "random_connection_rate", "final_100_training_mean_reward"]
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result.get(key) for key in fields})

    for algorithm in ("dqn", "reinforce", "ppo", "a2c"):
        runs = [row for row in results if row["algorithm"] == algorithm]
        if not runs:
            continue
        best = max(runs, key=lambda x: x["trained_mean_reward"])
        (root / f"{algorithm}_best.json").write_text(
            json.dumps(best, indent=2) + "\n", encoding="utf-8")

    lines = ["# SmartCare RL — Hyperparameter Experiment Results", "",
             "All results are measured on synthetic patient scenarios. Each row is one completed run.",
             "The same evaluation seeds are used across experiments. One training seed per setting is preliminary evidence.", ""]
    for algorithm in ("dqn", "reinforce", "ppo", "a2c"):
        runs = [row for row in results if row["algorithm"] == algorithm]
        lines += [f"## {algorithm.upper()} ({len(runs)}/10 completed)", "",
                  "| Experiment | Changed setting | Mean reward | Connection rate | Reward std |",
                  "|---|---|---:|---:|---:|"]
        for entry in runs:
            cfg = next(item for item in EXPERIMENTS if item["id"] == entry["experiment"])
            lines.append(f"| {entry['experiment']} | {cfg['change']} | "
                         f"{entry['trained_mean_reward']:.3f} | "
                         f"{entry['trained_connection_rate']*100:.1f}% | "
                         f"{entry['trained_reward_std']:.3f} |")
        if not runs:
            lines.append("| Not yet run | — | — | — | — |")
        lines.append("")
    (root / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    if results:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        algorithms = ("dqn", "reinforce", "ppo", "a2c")
        for algorithm in algorithms:
            runs = [r for r in results if r["algorithm"] == algorithm]
            if not runs:
                continue
            fig, ax = plt.subplots(figsize=(9, 4.5))
            ax.bar([r["experiment"] for r in runs], [r["trained_mean_reward"] for r in runs])
            ax.set_title(f"{algorithm.upper()} — mean evaluation reward")
            ax.set_ylabel("Mean reward (same synthetic evaluation seeds)")
            ax.tick_params(axis="x", labelrotation=35)
            fig.tight_layout()
            fig.savefig(root / f"{algorithm}_comparison.png", dpi=150)
            plt.close(fig)
        best_runs = [max((r for r in results if r["algorithm"] == alg),
                         key=lambda x: x["trained_mean_reward"], default=None)
                     for alg in algorithms]
        best_runs = [r for r in best_runs if r is not None]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([r["algorithm"].upper() for r in best_runs],
               [r["trained_mean_reward"] for r in best_runs])
        ax.set_title("Best evaluated configuration per algorithm")
        ax.set_ylabel("Mean reward")
        fig.tight_layout()
        fig.savefig(root / "best_algorithms.png", dpi=150)
        plt.close(fig)
    return results


def main():
    parser = argparse.ArgumentParser(description="SmartCare 40-experiment runner")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="List 40 configurations")
    group.add_argument("--experiment", help="Run one ID, e.g. PPO_01")
    group.add_argument("--algorithm", choices=("dqn", "reinforce", "ppo", "a2c"),
                       help="Run 10 configs for one algorithm")
    group.add_argument("--all", action="store_true", help="Run all 40")
    group.add_argument("--report-only", action="store_true", help="Rebuild reports from completed runs")
    parser.add_argument("--episodes", type=int, default=2000,
                        help="Training episodes per experiment (default 2000)")
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--smoke", action="store_true",
                        help="Separate quick 20-episode test, not counted in 40 runs")
    parser.add_argument("--force", action="store_true", help="Overwrite completed results")
    args = parser.parse_args()
    if args.list:
        for cfg in EXPERIMENTS:
            print(f"{cfg['id']:15s} {cfg['change']:26s} {cfg['parameters']}")
        return
    root = Path("experiments/smoke") if args.smoke else ROOT
    if args.report_only:
        results = write_reports(ROOT)
        print(f"Wrote reports using {len(results)} finished experiment(s)")
        return
    if args.episodes < 1 or args.eval_episodes < 1:
        parser.error("--episodes and --eval-episodes must be positive")
    if args.experiment:
        selected = [cfg for cfg in EXPERIMENTS if cfg["id"] == args.experiment.upper()]
        if not selected:
            parser.error("Unknown experiment; run --list")
    elif args.algorithm:
        selected = [cfg for cfg in EXPERIMENTS if cfg["algorithm"] == args.algorithm]
    else:
        selected = EXPERIMENTS
    episodes = 64 if args.smoke else args.episodes
    eval_episodes = 10 if args.smoke else args.eval_episodes
    if not args.smoke and episodes != 2000:
        print("WARNING: nonstandard training budget; compare runs only at the same episode budget")

    for spec in selected:
        summary_path = root / spec["algorithm"] / spec["id"] / "summary.json"
        model_path = summary_path.parent / "model.pth"
        episode_log = summary_path.parent / "episodes.csv"
        if summary_path.exists() and model_path.exists() and episode_log.exists() and not args.force:
            print(f"SKIP {spec['id']}: completed summary and model exist (use --force to rerun)")
            continue
        # A stale summary should not survive a rerun that later gets interrupted.
        if summary_path.exists():
            summary_path.unlink()
        print(f"\nRUN {spec['id']} | {spec['change']} | {episodes} training episodes", flush=True)
        result = train_one(spec, episodes, eval_episodes, root)
        print(f"DONE {spec['id']}: eval reward {result['trained_mean_reward']:.3f}; "
              f"connection {result['trained_connection_rate']*100:.1f}%", flush=True)
        if not args.smoke:
            write_reports(ROOT)
    if args.smoke:
        print("Smoke-test outputs are in experiments/smoke and do NOT count as completed runs.")
    else:
        print("Updated experiments/results/comparison.csv, comparison.md and comparison charts")


if __name__ == "__main__":
    main()
