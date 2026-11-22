# SmartCare RL — Hyperparameter Experiment Results

All results are measured on synthetic patient scenarios. Each row is one completed run.
The same evaluation seeds are used across experiments. One training seed per setting is preliminary evidence.

## DQN (10/10 completed)

| Experiment | Changed setting | Mean reward | Connection rate | Reward std |
|---|---|---:|---:|---:|
| DQN_01 | Baseline | 0.785 | 71.0% | 1.351 |
| DQN_02 | learning_rate = 0.0005 | 0.968 | 76.0% | 1.271 |
| DQN_03 | learning_rate = 0.002 | 0.872 | 74.0% | 1.310 |
| DQN_04 | gamma = 0.95 | 0.913 | 75.0% | 1.289 |
| DQN_05 | gamma = 0.995 | 1.091 | 82.0% | 1.148 |
| DQN_06 | batch_size = 32 | 0.652 | 66.0% | 1.425 |
| DQN_07 | batch_size = 128 | 0.913 | 75.0% | 1.296 |
| DQN_08 | epsilon_end = 0.1 | 0.856 | 73.0% | 1.315 |
| DQN_09 | target_every = 200 | 1.116 | 82.0% | 1.129 |
| DQN_10 | warmup = 256 | 0.885 | 75.0% | 1.320 |

## REINFORCE (10/10 completed)

| Experiment | Changed setting | Mean reward | Connection rate | Reward std |
|---|---|---:|---:|---:|
| REINFORCE_01 | Baseline | 0.332 | 63.0% | 1.479 |
| REINFORCE_02 | lr = 0.0005 | -0.370 | 43.0% | 1.485 |
| REINFORCE_03 | lr = 0.002 | 0.623 | 68.0% | 1.353 |
| REINFORCE_04 | gamma = 0.95 | 0.337 | 61.0% | 1.503 |
| REINFORCE_05 | gamma = 0.995 | 0.189 | 57.0% | 1.528 |
| REINFORCE_06 | batch_episodes = 8 | 0.390 | 62.0% | 1.484 |
| REINFORCE_07 | batch_episodes = 32 | -0.370 | 43.0% | 1.485 |
| REINFORCE_08 | entropy_coef = 0.0 | 0.293 | 61.0% | 1.487 |
| REINFORCE_09 | entropy_coef = 0.03 | 0.366 | 62.0% | 1.491 |
| REINFORCE_10 | gamma = 0.9 | 0.290 | 59.0% | 1.513 |

## PPO (10/10 completed)

| Experiment | Changed setting | Mean reward | Connection rate | Reward std |
|---|---|---:|---:|---:|
| PPO_01 | Baseline | 0.972 | 82.0% | 1.171 |
| PPO_02 | lr = 0.0005 | 0.631 | 72.0% | 1.349 |
| PPO_03 | lr = 0.002 | 0.764 | 75.0% | 1.332 |
| PPO_04 | gamma = 0.95 | 0.977 | 82.0% | 1.149 |
| PPO_05 | gamma = 0.995 | 0.941 | 82.0% | 1.175 |
| PPO_06 | batch_episodes = 8 | 0.998 | 82.0% | 1.157 |
| PPO_07 | batch_episodes = 32 | 0.684 | 73.0% | 1.345 |
| PPO_08 | clip_epsilon = 0.1 | 0.986 | 82.0% | 1.146 |
| PPO_09 | clip_epsilon = 0.3 | 0.799 | 77.0% | 1.266 |
| PPO_10 | ppo_epochs = 8 | 0.870 | 78.0% | 1.298 |

## A2C (10/10 completed)

| Experiment | Changed setting | Mean reward | Connection rate | Reward std |
|---|---|---:|---:|---:|
| A2C_01 | Baseline | 0.540 | 71.0% | 1.373 |
| A2C_02 | lr = 0.0005 | -0.370 | 43.0% | 1.485 |
| A2C_03 | lr = 0.002 | 0.783 | 77.0% | 1.261 |
| A2C_04 | gamma = 0.95 | 0.579 | 72.0% | 1.361 |
| A2C_05 | gamma = 0.995 | 0.535 | 71.0% | 1.372 |
| A2C_06 | batch_episodes = 8 | 0.783 | 77.0% | 1.261 |
| A2C_07 | batch_episodes = 32 | -0.369 | 43.0% | 1.485 |
| A2C_08 | entropy_coef = 0.0 | 0.579 | 72.0% | 1.361 |
| A2C_09 | entropy_coef = 0.03 | 0.585 | 72.0% | 1.364 |
| A2C_10 | value_coef = 1.0 | 0.579 | 72.0% | 1.361 |

