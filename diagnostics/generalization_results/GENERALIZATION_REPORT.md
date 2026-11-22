# SmartCare RL — Out-of-sample and stress-scenario evaluation

These scores are measured on **synthetic** patients using frozen checkpoints.
The standard cohort uses patient seeds beginning at 5000, while the stress cohort uses
seeds beginning at 8000, with low access/high travel barriers. The reward function and transition
mechanics do not change. This is not evidence of real-world clinical generalization.

| Model | Standard reward | Standard connected | Stress reward | Stress connected |
|---|---:|---:|---:|---:|
| DQN | 1.237 | 86% | 0.490 | 61% |
| PPO | 1.030 | 84% | 0.140 | 61% |
| A2C | 1.028 | 86% | 0.240 | 61% |
| REINFORCE | 1.001 | 80% | 0.490 | 61% |

## Reward improvement over random actions (paired bootstrap 95% CI)

Intervals resample synthetic patient episodes, **not** independent training runs.
They quantify evaluation-scenario uncertainty and do not establish model training stability.

| Model | Cohort | Reward delta | 95% CI | Connection delta (pp) | 95% CI (pp) |
|---|---|---:|---|---:|---|
| DQN | standard | 1.144 | [0.953, 1.352] | +30.0 | [+24.0, +36.5] |
| DQN | access barrier stress | 1.024 | [0.850, 1.209] | +23.0 | [+17.5, +29.0] |
| PPO | standard | 0.937 | [0.737, 1.143] | +29.0 | [+22.5, +35.5] |
| PPO | access barrier stress | 0.674 | [0.500, 0.859] | +23.0 | [+17.5, +29.0] |
| A2C | standard | 0.935 | [0.741, 1.134] | +30.0 | [+23.7, +36.0] |
| A2C | access barrier stress | 0.774 | [0.600, 0.959] | +23.0 | [+17.5, +29.0] |
| REINFORCE | standard | 0.908 | [0.714, 1.103] | +24.5 | [+18.0, +31.0] |
| REINFORCE | access barrier stress | 1.024 | [0.850, 1.209] | +23.0 | [+17.5, +29.0] |

## Limitations

- One training seed (42) per original configuration: no between-training-seed variance estimate.
- Two synthetic patient distributions; neither represents validated clinical data.
- Terminal attendance uses the simulator's stochastic Bernoulli outcome.
- Checkpoints were selected using earlier evaluation scores; this is additional validation,
  not a guarantee of unbiased performance on untested real populations.
- The stress cohort deliberately concentrates high-risk, low-access cases, so its raw
  mean scores should not be interpreted as a matched causal effect of distribution shift.

Run `uv run python -m diagnostics.generalization --all` to reproduce.
