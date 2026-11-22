"""Generate required rubric plots from measured logs; never synthesize missing values.

    uv run python -m diagnostics.plots

Uses existing experiment reward logs for convergence and evaluation; uses
new, independently trained diagnostic runs for optimizer-loss and entropy.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def moving_average(values, window=100):
    # Inclusive trailing mean, with shorter windows at the start.
    output = []
    total = 0.0
    for index, value in enumerate(values):
        total += value
        if index >= window:
            total -= values[index - window]
        output.append(total / min(index + 1, window))
    return output


def write_figures(experiments=Path("experiments/results"),
                  diagnostics=Path("diagnostics/runs"),
                  output=Path("diagnostics/figures")):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiments, diagnostics, output = map(Path, (experiments, diagnostics, output))
    output.mkdir(parents=True, exist_ok=True)
    produced = []
    algorithms = ("dqn", "ppo", "a2c", "reinforce")

    # Complete experiment evaluation data (already measured by run_experiments).
    comparison_path = experiments / "comparison.csv"
    if comparison_path.exists():
        rows = read_csv(comparison_path)
        grouped = defaultdict(list)
        for row in rows:
            if row.get("algorithm") in algorithms:
                grouped[row["algorithm"]].append(float(row["trained_mean_reward"]))
        if grouped:
            fig, ax = plt.subplots(figsize=(9, 5))
            labels = [alg.upper() for alg in algorithms if grouped[alg]]
            means = [sum(grouped[alg]) / len(grouped[alg]) for alg in algorithms if grouped[alg]]
            ax.bar(labels, means)
            ax.set_title("Evaluation reward averaged over tested configurations")
            ax.set_ylabel("Mean evaluation reward")
            ax.set_xlabel("Algorithm")
            ax.axhline(0, linewidth=0.7)
            fig.tight_layout()
            path = output / "algorithm_evaluation.png"
            fig.savefig(path, dpi=170)
            plt.close(fig)
            produced.append(path)

    # The actual 40-configuration training logs, convergences for selected winners.
    best = {"dqn": "DQN_09", "ppo": "PPO_06", "a2c": "A2C_06", "reinforce": "REINFORCE_03"}
    fig, ax = plt.subplots(figsize=(10, 5))
    present = 0
    for alg in algorithms:
        path = experiments / alg / best[alg] / "episodes.csv"
        if not path.exists():
            print(f"Missing reward log (skip): {path}")
            continue
        rows = read_csv(path)
        if not rows:
            continue
        rewards = [float(r["reward"]) for r in rows]
        episodes = [int(r["episode"]) for r in rows]
        ax.plot(episodes, moving_average(rewards), label=f"{alg.upper()} ({best[alg]})")
        present += 1
    if present:
        ax.set_title("Training convergence: trailing 100-episode mean reward")
        ax.set_xlabel("Training episode")
        ax.set_ylabel("Mean episode reward")
        ax.legend()
        fig.tight_layout()
        path = output / "training_convergence.png"
        fig.savefig(path, dpi=170)
        produced.append(path)
    plt.close(fig)

    # Genuine DQN SmoothL1 loss recorded during a fresh diagnostic training run.
    path = diagnostics / "dqn" / "DQN_09" / "episodes_updates.csv"
    if path.exists():
        rows = read_csv(path)
        if rows:
            x = [int(r["update"]) for r in rows]
            losses = [float(r["loss"]) for r in rows]
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x, moving_average(losses, 100), label="100-update rolling DQN SmoothL1 loss")
            ax.set_title("DQN optimization objective (new diagnostic training run)")
            ax.set_xlabel("Optimizer update")
            ax.set_ylabel("SmoothL1 Bellman loss")
            ax.legend()
            fig.tight_layout()
            result = output / "dqn_loss.png"
            fig.savefig(result, dpi=170)
            plt.close(fig)
            produced.append(result)
    else:
        print("No DQN optimization log yet; train with diagnostics.run_diagnostics --algorithm dqn")

    # Stochastic-policy entropy. Existing 40-run reward CSVs cannot recreate it.
    fig, ax = plt.subplots(figsize=(10, 5))
    count = 0
    for alg in ("ppo", "a2c", "reinforce"):
        path = diagnostics / alg / best[alg] / "episodes_updates.csv"
        if not path.exists():
            print(f"No entropy log yet for {alg.upper()}; train diagnostic run to populate")
            continue
        rows = read_csv(path)
        if not rows:
            continue
        x = [int(r["update"]) for r in rows]
        y = [float(r["entropy"]) for r in rows]
        ax.plot(x, moving_average(y, 20), label=alg.upper())
        count += 1
    if count:
        ax.set_title("Policy-gradient entropy (new diagnostic training runs)")
        ax.set_xlabel("Optimizer update")
        ax.set_ylabel("Categorical action entropy (nats)")
        ax.legend()
        fig.tight_layout()
        result = output / "policy_entropy.png"
        fig.savefig(result, dpi=170)
        produced.append(result)
    plt.close(fig)

    for path in produced:
        print(f"Wrote {path}")
    return produced


def main():
    parser = argparse.ArgumentParser(description="Generate SmartCare rubric figures from real measurements")
    parser.add_argument("--experiments-root", default="experiments/results")
    parser.add_argument("--diagnostics-root", default="diagnostics/runs")
    parser.add_argument("--output", default="diagnostics/figures")
    args = parser.parse_args()
    produced = write_figures(args.experiments_root, args.diagnostics_root, args.output)
    if not produced:
        print("No valid data found. Copy the original experiment results or run diagnostic training.")


if __name__ == "__main__":
    main()
