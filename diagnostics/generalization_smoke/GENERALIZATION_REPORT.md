# SmartCare RL — Out-of-sample and stress-scenario evaluation

These scores are measured on **synthetic** patients using frozen checkpoints.
The standard cohort uses patient seeds beginning at 5000, while the stress cohort uses
seeds beginning at 8000, with low access/high travel barriers. The reward function and transition
mechanics do not change. This is not evidence of real-world clinical generalization.

| Model | Standard reward | Standard connected | Stress reward | Stress connected |
|---|---:|---:|---:|---:|
| DQN | 1.529 | 95% | 0.160 | 50% |

## Reward improvement over random actions (paired bootstrap 95% CI)

Intervals resample synthetic patient episodes, **not** independent training runs.
They quantify evaluation-scenario uncertainty and do not establish model training stability.

| Model | Cohort | Reward delta | 95% CI | Connection delta (pp) | 95% CI (pp) |
|---|---|---:|---|---:|---|
| DQN | standard | 1.308 | [0.716, 2.007] | +35.0 | [+17.4, +55.0] |
| DQN | access barrier stress | 0.896 | [0.529, 1.441] | +20.0 | [+2.4, +35.0] |

## Limitations

- One training seed (42) per original configuration: no between-training-seed variance estimate.
- Two synthetic patient distributions; neither represents validated clinical data.
- Terminal attendance uses the simulator's stochastic Bernoulli outcome.
- Checkpoints were selected using earlier evaluation scores; this is additional validation,
  not a guarantee of unbiased performance on untested real populations.
- The stress cohort deliberately concentrates high-risk, low-access cases, so its raw
  mean scores should not be interpreted as a matched causal effect of distribution shift.

Run `uv run python -m diagnostics.generalization --all` to reproduce.
