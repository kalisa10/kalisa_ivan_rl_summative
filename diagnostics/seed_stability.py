"""Optional repeated-training-seed validation for selected SmartCare configurations.

This is an *additional* experiment, not a replacement for the original 40 runs.
Usage:
    uv run python -m diagnostics.seed_stability --algorithm dqn --seeds 101 202 303
    uv run python -m diagnostics.seed_stability --algorithm ppo --seeds 101 202 303
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.settings import EXPERIMENTS
from experiments.run_experiments import evaluate_checkpoint
from diagnostics.generalization import BEST

DEFAULT_ROOT = Path("diagnostics/seed_stability_results")


def summarize_seed_runs(rows):
    """Between-training-seed dispersion, not episode-level reward variation."""
    if not rows:
        raise ValueError("No completed training seeds")
    rewards = np.asarray([x["trained_mean_reward"] for x in rows], dtype=float)
    rates = np.asarray([x["trained_connection_rate"] for x in rows], dtype=float)
    return {
        "training_seeds": [int(x["training_seed"]) for x in rows],
        "n_training_seeds": len(rows),
        "mean_reward_across_seeds": float(rewards.mean()),
        "sd_reward_across_seeds": float(rewards.std(ddof=1)) if len(rows) > 1 else None,
        "mean_connection_rate_across_seeds": float(rates.mean()),
        "sd_connection_rate_across_seeds": float(rates.std(ddof=1)) if len(rows) > 1 else None,
        "interpretation": "Across independent train seeds; scores share a fixed synthetic evaluation cohort",
    }


def train_seed(algorithm, seed, episodes, eval_episodes, root, eval_seed):
    experiment_id, _ = BEST[algorithm]
    config = next(x for x in EXPERIMENTS if x["id"] == experiment_id)
    destination = root / algorithm / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint = destination / "model.pth"
    log_path = destination / "episodes.csv"
    summary_path = destination / "summary.json"
    if summary_path.exists() and checkpoint.exists() and log_path.exists():
        previous = json.loads(summary_path.read_text(encoding="utf-8"))
        if (previous.get("training_episodes") == episodes and
            previous.get("evaluation_episodes") == eval_episodes and
            previous.get("evaluation_seed_start") == eval_seed):
            print(f"SKIP {algorithm.upper()} seed={seed}: completed matching run")
            return previous
        raise RuntimeError(f"Existing run uses different settings: {summary_path}; use a different output folder")
    if summary_path.exists():
        summary_path.unlink()
    params = config["parameters"].copy()
    params.update(episodes=episodes, seed=seed, save_path=checkpoint, log_path=log_path)
    if algorithm == "dqn":
        from training.dqn_training import train
        train(**params)
    else:
        from training.pg_training import train
        train(algorithm=algorithm, **params)
    metrics = evaluate_checkpoint(algorithm, checkpoint, episodes=eval_episodes, seed=eval_seed)
    result = {
        "algorithm": algorithm,
        "experiment": experiment_id,
        "training_seed": seed,
        "training_episodes": episodes,
        "evaluation_episodes": eval_episodes,
        "evaluation_seed_start": eval_seed,
        "checkpoint": str(checkpoint),
        "log_path": str(log_path),
        "parameters": config["parameters"],
        **metrics,
    }
    summary_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def write_results(root, algorithm, rows):
    output = root / algorithm
    output.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda x: x["training_seed"])
    aggregate = summarize_seed_runs(rows)
    (output / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    fields = ["experiment", "training_seed", "training_episodes", "evaluation_episodes",
              "trained_mean_reward", "trained_connection_rate", "trained_reward_std",
              "random_mean_reward", "random_connection_rate"]
    with (output / "training_seed_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(row["training_seed"]) for row in rows],
           [row["trained_mean_reward"] for row in rows])
    ax.set_xlabel("Independent training seed")
    ax.set_ylabel("Mean reward over fixed evaluation cohort")
    ax.set_title(f"{algorithm.upper()}: variability across training seeds")
    fig.tight_layout()
    fig.savefig(output / "training_seed_rewards.png", dpi=160)
    plt.close(fig)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Train extra copies of best configs with independent seeds")
    parser.add_argument("--algorithm", choices=tuple(BEST), required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[101, 202, 303])
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--eval-seed", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_episodes < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("episodes must be positive and seeds must be unique")
    if args.episodes != 2000:
        print("WARNING: Reduced training budget; do not compare directly to original 2000-episode runs")
    if len(args.seeds) < 3:
        print("WARNING: Fewer than 3 independent training seeds gives weak stability evidence")
    results = []
    for seed in args.seeds:
        print(f"RUN {args.algorithm.upper()} seed {seed} ({args.episodes} episodes)", flush=True)
        results.append(train_seed(args.algorithm, seed, args.episodes,
                                  args.eval_episodes, args.output, args.eval_seed))
    aggregate = write_results(args.output, args.algorithm, results)
    print(f"Saved training-seed stability to {args.output / args.algorithm}")
    print(f"Across-seed mean reward: {aggregate['mean_reward_across_seeds']:.3f}")
    if aggregate["sd_reward_across_seeds"] is not None:
        print(f"Across-seed reward SD: {aggregate['sd_reward_across_seeds']:.3f}")


if __name__ == "__main__":
    main()
