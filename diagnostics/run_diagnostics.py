"""Re-train selected configurations solely to log diagnostic quantities.

Existing 40 experiment summaries and checkpoints are NEVER overwritten.

Usage:
    uv run python -m diagnostics.run_diagnostics --algorithm dqn
    uv run python -m diagnostics.run_diagnostics --all
    uv run python -m diagnostics.run_diagnostics --algorithm ppo --episodes 64  # test
"""
import argparse
import json
from pathlib import Path

from experiments.settings import EXPERIMENTS
from experiments.run_experiments import train_one

ROOT = Path("diagnostics/runs")
BEST = {"dqn": "DQN_09", "ppo": "PPO_06", "a2c": "A2C_06", "reinforce": "REINFORCE_03"}


def main():
    parser = argparse.ArgumentParser(description="Produce genuine training diagnostics for best-known configurations")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--algorithm", choices=tuple(BEST))
    selector.add_argument("--all", action="store_true")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--force", action="store_true", help="Replace an existing diagnostic run")
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_episodes < 1:
        parser.error("--episodes and --eval-episodes must be positive")
    specs = {item["id"]: item for item in EXPERIMENTS}
    names = list(BEST.values()) if args.all else [BEST[args.algorithm]]
    if args.episodes != 2000:
        print("DIAGNOSTIC ONLY: reduced training budget; do not compare its scores with the full 40 experiments.")
    for name in names:
        spec = specs[name]
        dest = ROOT / spec["algorithm"] / name
        summary = dest / "summary.json"
        updates = dest / "episodes_updates.csv"
        if summary.exists() and updates.exists() and not args.force:
            print(f"SKIP {name}: diagnostic metrics already exist; use --force to rerun")
            continue
        # Delete only an obsolete diagnostic summary, never the original experiment summary.
        if summary.exists():
            summary.unlink()
        print(f"RUN {name}: {args.episodes} episodes; diagnostic output -> {dest}", flush=True)
        outcome = train_one(spec, args.episodes, args.eval_episodes, output_root=ROOT)
        if not updates.exists():
            raise RuntimeError(f"Optimizer metrics were not written: {updates}")
        print(f"DONE {name}: evaluation reward {outcome['trained_mean_reward']:.3f}, "
              f"connection rate {outcome['trained_connection_rate']:.0%}")
    print("Diagnostic files are under diagnostics/runs; existing 40 runs are unchanged.")


if __name__ == "__main__":
    main()
