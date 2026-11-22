# SmartCare RL — Generalization and stability evaluation protocol

## Purpose and scope

This is **additional evaluation** of frozen checkpoints from the 40 completed hyperparameter configurations. It does not overwrite or retrain any model. It assesses behavior on newly sampled simulated patient profiles and a deliberately shifted synthetic cohort. It is not a clinical validation study.

## Selected models

| Algorithm | Selected checkpoint | Original training budget | Original training seed |
|---|---|---|---|
| DQN | `experiments/results/dqn/DQN_09/model.pth` | 2,000 episodes | 42 |
| PPO | `experiments/results/ppo/PPO_06/model.pth` | 2,000 episodes | 42 |
| A2C | `experiments/results/a2c/A2C_06/model.pth` | 2,000 episodes | 42 |
| REINFORCE | `experiments/results/reinforce/REINFORCE_03/model.pth` | 2,000 episodes | 42 |

All four have the same five actions, nine observation values, three-step horizon and terminal reward mechanics. The same new evaluation episode seeds are used per model; no further gradient updates occur. Policies act greedily on their trained outputs. A uniformly random policy provides a within-cohort baseline.

## Patient cohorts

**Standard held-out simulator cohort.** Run with `--seed 5000 --episodes 200` by default. Each patient is sampled from the unchanged `PatientNavigationEnv` initial-state distribution. These new seeds differ from the earlier 1000–1099 evaluation cohort. A different seed does not prove disjointness from *all* previously generated training profiles because the low-dimensional synthetic simulator can generate statistically similar patients. It does provide new seeded draws.

**Access-barrier stress cohort.** Run with `--stress-seed 8000 --episodes 200` by default. Initial profiles are modified to have phone access=0, travel barrier=1, telehealth access=0 and risk uniformly sampled from `[0.80,0.90)`. The original reward function, actions, eligibility rules, step dynamics and terminal outcome randomness are preserved. This intentionally tests a challenging, different synthetic input distribution. Its results are not directly comparable to the standard cohort as a matched treatment effect.

## Metrics and uncertainty

- Mean episodic reward: average total reward over complete three-action episodes.
- Reward standard deviation: sample standard deviation across evaluation episodes.
- Connection rate: proportion of simulated episodes with `info['attended'] == True`.
- Mean simulated connection probability: the environment-calculated terminal probability, which helps distinguish policy-driven probabilities from finite-sample outcome noise.
- Improvement over random: **paired episode** difference, since the trained and random agents see the same seeded initial patient and use the same environment seed for the stochastic terminal draw. Each agent acts independently; outcomes differ with chosen actions.
- Uncertainty: percentile 95% bootstrap confidence intervals on the paired *evaluation episode* differences. Intervals capture variation over synthetic evaluation scenarios conditional on the selected frozen model; they **do not measure training stability across random initializations**.

**Training stability limitation:** all 40 original configurations were trained with seed 42. True training robustness would require retraining the same configuration with multiple independent seeds, recording distribution of scores and reporting mean ± standard deviation across seeds. This update does not imply those reruns have occurred. Episode-level reward SD and bootstrap confidence intervals are different quantities.

## Reproduction

From the repository root:

```powershell
uv run python -m pytest -q
uv run python -m diagnostics.generalization --algorithm dqn --episodes 20 --bootstrap 100
uv run python -m diagnostics.generalization --all --episodes 200 --bootstrap 1500
```

The short command validates the pipeline only; do not treat its estimates as your final report. The complete command writes `diagnostics/generalization_results/` with JSON model summaries, a long-form `episodes.csv`, a Markdown report and two labeled PNG figures.

Use `--help` to see options for custom seed ranges, checkpoint locations, output folder and bootstrap count. After evaluation, include the generated table and plots in the report, and discuss if performance drops in the access-barrier cohort. Do not state an outcome before actually running the command.

## Interpretation and ethical boundary

Differences between cohorts can reveal sensitivity to synthetic access assumptions. This does not establish real-patient effectiveness, safety, fairness or deployment suitability. The observation features and attendance probabilities are manually specified. The stress distribution does not represent observed healthcare prevalence. A stronger follow-up would use expert-reviewed distributions, repeated training seeds, external scenario rules and measured uncertainty across model retrainings.

## Optional independent-training-seed repeatability

Run `uv run python -m diagnostics.seed_stability --algorithm dqn --seeds 101 202 303` and repeat for PPO if time permits. This trains new copies of the best hyperparameter configuration in `diagnostics/seed_stability_results/<algorithm>/seed_<seed>/` with fresh initialization and learning randomness. All copies are evaluated on the same seeded standard cohort. The reportable statistic is **mean ± sample standard deviation of the evaluation mean reward across independently trained models**. Retained original DQN_09/PPO_06 checkpoints are not overwritten. Multiple independent training seeds are better evidence about training reliability than bootstrap intervals over episodes from one frozen model.
