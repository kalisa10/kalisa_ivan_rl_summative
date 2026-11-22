"""Reproducible out-of-sample and stress-scenario evaluation for SmartCare RL.

Run from project root:
    uv run python -m diagnostics.generalization --all
    uv run python -m diagnostics.generalization --algorithm dqn --episodes 30

No model training and no modification of experiments/results/ occur here.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from environment import PatientNavigationEnv

BEST = {
    "dqn": ("DQN_09", "dqn"),
    "ppo": ("PPO_06", "ppo"),
    "a2c": ("A2C_06", "a2c"),
    "reinforce": ("REINFORCE_03", "reinforce"),
}
OUTPUT_ROOT = Path("diagnostics/generalization_results")


class AccessBarrierStressEnv(PatientNavigationEnv):
    """Synthetic *distribution-shift stress test*, not real-world validation.

    The same reward and transition function is used as PatientNavigationEnv.
    Only initial patient characteristics differ: no phone/telehealth access,
    travel barrier present, and missed-visit risk in [0.80, 0.90).
    """

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.profile = np.array(
            [0.0, 1.0, 0.0, float(self.np_random.uniform(0.80, 0.90))],
            dtype=np.float32,
        )
        return self._observation(), {}


def load_selector(algorithm, checkpoint):
    """Import model code only when evaluating trained checkpoints."""
    if algorithm == "dqn":
        from training.dqn_training import greedy_action, load_model
        model = load_model(checkpoint)
    else:
        from training.pg_training import greedy_action, load_policy
        model = load_policy(algorithm, checkpoint)
    return lambda observation: greedy_action(model, observation)


def run_episode(env_type, seed, selector):
    """Use the same environment seed for two policies (paired comparison)."""
    env = env_type()
    try:
        observation, _ = env.reset(seed=seed)
        profile = env.profile.copy()
        random_rng = np.random.default_rng(seed + 109729)
        total_reward = 0.0
        action_ids = []
        for _ in range(env.max_steps):
            action = int(selector(observation, random_rng))
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            action_ids.append(action)
            if terminated or truncated:
                break
        if not (terminated or truncated):
            raise RuntimeError("Episode did not terminate")
        return {
            "reward": float(total_reward),
            "connected": int(info["attended"]),
            "probability": float(info["attendance_probability"]),
            "profile": profile.tolist(),
            "actions": action_ids,
        }
    finally:
        env.close()


def paired_bootstrap(values, baseline, *, resamples=1500, seed=7):
    """Bootstrap paired mean difference and percentile 95% interval."""
    delta = np.asarray(values, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if delta.ndim != 1 or delta.size < 2:
        raise ValueError("Need at least two paired observations")
    rng = np.random.default_rng(seed)
    # Use bounded memory independent of large resample counts.
    draws = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        draws[i] = float(delta[rng.integers(0, len(delta), len(delta))].mean())
    return {
        "mean_difference": float(delta.mean()),
        "ci95_lower": float(np.percentile(draws, 2.5)),
        "ci95_upper": float(np.percentile(draws, 97.5)),
    }


def summarize(episodes, *, baseline, resamples=1500, bootstrap_seed=7):
    """Compute transparent descriptive metrics and paired uncertainty."""
    rew = np.asarray([item["reward"] for item in episodes], dtype=np.float64)
    con = np.asarray([item["connected"] for item in episodes], dtype=np.float64)
    base_rew = np.asarray([item["reward"] for item in baseline], dtype=np.float64)
    base_con = np.asarray([item["connected"] for item in baseline], dtype=np.float64)
    if len(rew) != len(base_rew):
        raise ValueError("Paired cohorts must have matching episode counts")
    return {
        "episodes": len(episodes),
        "mean_reward": float(rew.mean()),
        "reward_sd": float(rew.std(ddof=1)),
        "connection_rate": float(con.mean()),
        "mean_connection_probability": float(np.mean([item["probability"] for item in episodes])),
        "reward_vs_random": paired_bootstrap(rew, base_rew, resamples=resamples, seed=bootstrap_seed),
        "connection_vs_random": paired_bootstrap(con, base_con, resamples=resamples, seed=bootstrap_seed + 1),
    }


def evaluate(algorithm, checkpoint, episodes=200, seed=5000, shift_seed=8000, bootstrap=1500):
    """Evaluate one frozen model on two novel, disjoint seed cohorts."""
    if episodes < 2 or bootstrap < 10:
        raise ValueError("episodes must be >= 2 and bootstrap >= 10")
    if algorithm not in BEST:
        raise ValueError(f"Unknown algorithm {algorithm}")
    if Path(checkpoint).is_file() is False:
        raise FileNotFoundError(f"Checkpoint missing: {checkpoint}")
    if set(range(seed, seed + episodes)).intersection(range(shift_seed, shift_seed + episodes)):
        raise ValueError("Standard and shifted evaluation seed ranges must not overlap")
    agent = load_selector(algorithm, checkpoint)
    results = {"algorithm": algorithm, "checkpoint": str(checkpoint), "training_seed": 42,
               "evaluation_only": True, "cohorts": {}, "protocol": {
                   "episodes_per_cohort": episodes, "standard_seed_start": seed,
                   "stress_seed_start": shift_seed, "bootstrap_resamples": bootstrap,
                   "policy": "greedy deterministic", "random_policy": "uniform across five actions",
                   "paired_random_outcomes": "same environment episode seeds; each policy has an independent action RNG",
                   "stress_test": "access barrier present; phone=0; telehealth=0; risk in [0.80,0.90)",
                   "claim_limit": "Synthetic distribution shift, NOT real-world generalization or training-seed stability",
               }}
    episode_rows = []
    for label, env_type, first_seed in (
        ("standard", PatientNavigationEnv, seed),
        ("access_barrier_stress", AccessBarrierStressEnv, shift_seed),
    ):
        trained = []
        random_policy = []
        for i in range(episodes):
            episode_seed = first_seed + i
            trained_episode = run_episode(env_type, episode_seed, lambda obs, rng: agent(obs))
            random_episode = run_episode(env_type, episode_seed, lambda obs, rng: rng.integers(0, 5))
            trained.append(trained_episode)
            random_policy.append(random_episode)
            for policy, episode in (("trained", trained_episode), ("random", random_episode)):
                episode_rows.append({
                    "algorithm": algorithm, "cohort": label, "episode_seed": episode_seed,
                    "policy": policy, "reward": episode["reward"],
                    "connected": episode["connected"], "attendance_probability": episode["probability"],
                    "actions": "/".join(map(str, episode["actions"])),
                })
        results["cohorts"][label] = {
            "trained": summarize(trained, baseline=random_policy, resamples=bootstrap),
            "random": {
                "episodes": len(random_policy),
                "mean_reward": float(np.mean([r["reward"] for r in random_policy])),
                "connection_rate": float(np.mean([r["connected"] for r in random_policy])),
            },
        }
    standard = results["cohorts"]["standard"]["trained"]
    stressed = results["cohorts"]["access_barrier_stress"]["trained"]
    results["stress_minus_standard"] = {
        "mean_reward": stressed["mean_reward"] - standard["mean_reward"],
        "connection_rate": stressed["connection_rate"] - standard["connection_rate"],
        "interpretation": "Descriptive between-cohort difference; not a paired controlled effect",
    }
    return results, episode_rows


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_figures(output, summaries):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(summaries)
    cohorts = ("standard", "access_barrier_stress")
    for key, name, ylabel, filename in (
        ("mean_reward", "Mean evaluation reward", "Mean reward", "reward_by_cohort.png"),
        ("connection_rate", "Appointment connection rate", "Connection rate (%)", "connection_by_cohort.png"),
    ):
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for index, cohort in enumerate(cohorts):
            vals = [summaries[a]["cohorts"][cohort]["trained"][key] for a in names]
            if key == "connection_rate":
                vals = [100 * v for v in vals]
            ax.bar(x + (index - 0.5) * 0.36, vals, 0.36, label=cohort.replace("_", " "))
        ax.set_xticks(x, [name.upper() for name in names])
        ax.set_ylabel(ylabel)
        ax.set_title(name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def write_report(output, summaries, *, standard_seed=5000, stress_seed=8000):
    lines = [
        "# SmartCare RL — Out-of-sample and stress-scenario evaluation", "",
        "These scores are measured on **synthetic** patients using frozen checkpoints.",
        f"The standard cohort uses patient seeds beginning at {standard_seed}, while the stress cohort uses",
        f"seeds beginning at {stress_seed}, with low access/high travel barriers. The reward function and transition",
        "mechanics do not change. This is not evidence of real-world clinical generalization.", "",
        "| Model | Standard reward | Standard connected | Stress reward | Stress connected |", "|---|---:|---:|---:|---:|",
    ]
    for alg, result in summaries.items():
        first = result["cohorts"]["standard"]["trained"]
        second = result["cohorts"]["access_barrier_stress"]["trained"]
        lines.append(f"| {alg.upper()} | {first['mean_reward']:.3f} | {first['connection_rate']:.0%} | "
                     f"{second['mean_reward']:.3f} | {second['connection_rate']:.0%} |")
    lines += ["", "## Reward improvement over random actions (paired bootstrap 95% CI)", "",
              "Intervals resample synthetic patient episodes, **not** independent training runs.",
              "They quantify evaluation-scenario uncertainty and do not establish model training stability.", "",
              "| Model | Cohort | Reward delta | 95% CI | Connection delta (pp) | 95% CI (pp) |",
              "|---|---|---:|---|---:|---|" ]
    for alg, result in summaries.items():
        for cohort, content in result["cohorts"].items():
            rew = content["trained"]["reward_vs_random"]
            con = content["trained"]["connection_vs_random"]
            lines.append(f"| {alg.upper()} | {cohort.replace('_',' ')} | {rew['mean_difference']:.3f} | "
                         f"[{rew['ci95_lower']:.3f}, {rew['ci95_upper']:.3f}] | "
                         f"{con['mean_difference']*100:+.1f} | "
                         f"[{con['ci95_lower']*100:+.1f}, {con['ci95_upper']*100:+.1f}] |")
    lines += ["", "## Limitations", "",
              "- One training seed (42) per original configuration: no between-training-seed variance estimate.",
              "- Two synthetic patient distributions; neither represents validated clinical data.",
              "- Terminal attendance uses the simulator's stochastic Bernoulli outcome.",
              "- Checkpoints were selected using earlier evaluation scores; this is additional validation,",
              "  not a guarantee of unbiased performance on untested real populations.",
              "- The stress cohort deliberately concentrates high-risk, low-access cases, so its raw",
              "  mean scores should not be interpreted as a matched causal effect of distribution shift.", "",
              "Run `uv run python -m diagnostics.generalization --all` to reproduce.\n"]
    (output / "GENERALIZATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Evaluate frozen SmartCare agents on novel synthetic cohorts")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Evaluate all four best checkpoints (default)")
    group.add_argument("--algorithm", choices=tuple(BEST), help="Evaluate one algorithm")
    parser.add_argument("--episodes", type=int, default=200, help="Patient scenarios per cohort")
    parser.add_argument("--seed", type=int, default=5000, help="Standard cohort starting seed")
    parser.add_argument("--stress-seed", type=int, default=8000, help="Stress cohort starting seed")
    parser.add_argument("--bootstrap", type=int, default=1500, help="Paired bootstrap samples")
    parser.add_argument("--checkpoint-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    names = [args.algorithm] if args.algorithm else list(BEST)
    summaries = {}
    episode_rows = []
    args.output.mkdir(parents=True, exist_ok=True)
    for alg in names:
        experiment, directory = BEST[alg]
        checkpoint = args.checkpoint_root / directory / experiment / "model.pth"
        print(f"Evaluating {experiment} on {args.episodes} standard and stress scenarios", flush=True)
        result, episodes = evaluate(alg, checkpoint, args.episodes, args.seed, args.stress_seed, args.bootstrap)
        summaries[alg] = result
        episode_rows.extend(episodes)
        for cohort, stats in result["cohorts"].items():
            print(f"  {cohort:21s} reward={stats['trained']['mean_reward']:+.3f} "
                  f"connected={stats['trained']['connection_rate']:.1%} "
                  f"random reward={stats['random']['mean_reward']:+.3f}")
        (args.output / f"{alg}_evaluation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    write_csv(args.output / "episodes.csv", episode_rows)
    write_report(args.output, summaries, standard_seed=args.seed, stress_seed=args.stress_seed)
    write_figures(args.output, summaries)
    print(f"Wrote evaluation evidence to {args.output}")


if __name__ == "__main__":
    main()
