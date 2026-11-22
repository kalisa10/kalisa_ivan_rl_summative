"""Checks for experiment setup and report generation (no training required)."""

import json
from pathlib import Path

from experiments.settings import BASE, EXPERIMENTS
from experiments.run_experiments import write_reports, load_results


def test_exactly_ten_distinct_configs_per_algorithm():
    assert len(EXPERIMENTS) == 40
    for name in BASE:
        group = [cfg for cfg in EXPERIMENTS if cfg["algorithm"] == name]
        assert len(group) == 10
        configs = [json.dumps(cfg["parameters"], sort_keys=True) for cfg in group]
        assert len(set(configs)) == 10
        assert group[0]["change"] == "Baseline"
        baseline = group[0]["parameters"]
        for item in group[1:]:
            differences = [k for k in baseline if baseline[k] != item["parameters"][k]]
            assert len(differences) == 1


def test_report_only_uses_finished_runs(tmp_path):
    root = tmp_path / "results"
    folder = root / "ppo" / "PPO_01"
    folder.mkdir(parents=True)
    summary = {"experiment": "PPO_01", "algorithm": "ppo", "episodes": 20,
               "model_path": str(folder / "model.pth"),
               "episode_log": str(folder / "episodes.csv"),
               "trained_mean_reward": 1.0, "trained_reward_std": 0.3,
               "trained_connection_rate": 0.8, "random_mean_reward": 0.2,
               "random_connection_rate": 0.55,
               "final_100_training_mean_reward": 0.7}
    (folder / "summary.json").write_text(json.dumps(summary))
    assert not load_results(root)  # Never report a summary with missing artifacts.
    (folder / "model.pth").write_bytes(b"test")
    (folder / "episodes.csv").write_text("episode,reward\n1,1.0\n")
    rows = write_reports(root)
    assert len(rows) == 1
    assert (root / "comparison.csv").exists()
    assert (root / "comparison.md").exists()
    assert (root / "ppo_comparison.png").exists()
    assert (root / "ppo_best.json").exists()
