# SmartCare RL — Rubric evidence update

This is an **overlay**, not a replacement project. Copy its `training/`, `diagnostics/` and `tests/test_diagnostics.py` into the existing `smartcare_rl_summative` directory. Keep all `experiments/results` files and trained model checkpoints unchanged.

## What it adds

- `training/dqn_training.py`: CSV record of the **actual SmoothL1 Bellman loss** and absolute TD error at **each optimizer update**. Next to `episodes.csv`, the output is `episodes_updates.csv`.
- `training/pg_training.py`: CSV record of actual **policy loss, critic loss (when applicable), categorical entropy, and total loss** at each optimizer update.
- `diagnostics/run_diagnostics.py`: Re-trains the best-known configurations **into `diagnostics/runs/` only**, without changing the original 40 experiments, model files or their reported metrics.
- `diagnostics/plots.py`: Creates labeled training convergence, algorithm evaluation, DQN loss and policy entropy PNG figures from measured data. Missing measurements are explicitly skipped, never invented.
- `tests/test_diagnostics.py`: Simple tests for plotting helpers.

## Run the steps in your existing VS Code PowerShell terminal

```powershell
cd "$HOME\Downloads\smartcare_rl_summative"
uv run python -m pytest -q
uv run python -m diagnostics.plots
```

The first `diagnostics.plots` command should create training-convergence and algorithm-evaluation figures using your existing 40-run files. The DQN loss and entropy figures require new recorded optimization measurements:

```powershell
uv run python -m diagnostics.run_diagnostics --algorithm dqn
uv run python -m diagnostics.run_diagnostics --algorithm ppo
uv run python -m diagnostics.run_diagnostics --algorithm a2c
uv run python -m diagnostics.run_diagnostics --algorithm reinforce
uv run python -m diagnostics.plots
```

Run a quick smoke test of one diagnostic training pipeline first if desired:

```powershell
uv run python -m diagnostics.run_diagnostics --algorithm dqn --episodes 64
```

**IMPORTANT:** This 64-episode smoke-test uses the same diagnostic output folder. Before full training, rerun that algorithm with `--force` to replace the diagnostic smoke test, or delete only its directory under `diagnostics/runs`. Full diagnostic training uses 2,000 episodes. This is extra evidence; it is **not** part of the original 40-config experiment comparison.

## Git commit

```powershell
git add training/dqn_training.py training/pg_training.py diagnostics/ tests/test_diagnostics.py RUBRIC_UPDATE_GUIDE.md
git commit -m "feat: log training diagnostics and generate rubric figures"
git push
```

## Evidence limitations

Optimizer loss and entropy curves for the original 40 runs **cannot be recovered** from episode rewards and checkpoints. The new figures correspond to separately labeled training runs using the previously selected settings. Both datasets use the same SmartCare environment, but the new diagnostic scores should not silently replace historical evaluation results. Even when training uses the same seed, evaluate and report the datasets separately.

## Next rubric work

Add a complete observation provenance table and integration diagram to the report; document reward and terminal rules; capture a real Three.js screenshot; consider an additional out-of-distribution evaluation with changed simulation conditions and repeated seeds; and prepare the required five-minute recorded demonstration.
