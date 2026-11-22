# SmartCare RL

SmartCare RL is a reinforcement learning project that explores how patient-support actions can improve appointment access. It uses a **synthetic healthcare navigation environment**; no real patient information or medical decisions are involved.

## Project overview

An agent has three steps to support a simulated patient by choosing among five actions: send an appointment reminder, call the patient, arrange transport, coordinate telehealth, or take no additional action. The environment provides nine observation values and rewards actions based on their cost, suitability, and the final appointment outcome.

Four algorithms are implemented: **DQN, REINFORCE, PPO, and A2C**. Each algorithm was tested with ten hyperparameter configurations (40 experiments in total).

## Setup

Requires Python and [uv](https://docs.astral.sh/uv/).

```powershell
uv sync
uv run python -m pytest -q
```

## Run the simulation

Launch the interactive Three.js visualization using the best recorded DQN checkpoint:

```powershell
uv run python main.py --visual3d --algorithm dqn --model experiments/results/dqn/DQN_09/model.pth
```

Open `http://127.0.0.1:8765/` if the browser does not open automatically. The 3D viewer requires WebGL and internet access to load Three.js. To try PPO instead:

```powershell
uv run python main.py --visual3d --algorithm ppo --model experiments/results/ppo/PPO_06/model.pth
```

The Pygame visualization is also available:

```powershell
uv run python main.py --visual --algorithm dqn --model experiments/results/dqn/DQN_09/model.pth
```

These commands require the corresponding `.pth` checkpoint files under `experiments/results/`.

## Experiments and results

```powershell
uv run python -m experiments.run_experiments --list
uv run python -m experiments.run_experiments --report-only
```

To run the full experiment suite from scratch:

```powershell
uv run python -m experiments.run_experiments --all
```

Each configuration trains for 2,000 episodes and is evaluated on 100 seeded synthetic patient scenarios. Reports and charts are saved under `experiments/results/`.

| Best configuration | Mean evaluation reward | Appointment connection rate |
| --- | ---: | ---: |
| DQN_09 | 1.1156 | 82% |
| PPO_06 | 0.9983 | 82% |
| A2C_06 | 0.7831 | 77% |
| REINFORCE_03 | 0.6232 | 68% |
| Random policy | 0.2306 | 59% |

These results come from a synthetic simulation with one training seed per configuration. They do not establish clinical effectiveness or statistically reliable superiority between algorithms.

## Project structure

- `environment/` — patient-navigation environment and 2D display
- `training/` — four training implementations and TorchRL environment integration
- `experiments/` — configurations, experiment runner, model checkpoints, and reports
- `visual3d/` — Three.js 3D visualization and local server
- `tests/` — automated checks
- `main.py` — simulation entry point

**Implementation note:** TorchRL is used for environment integration; algorithm losses and optimization are implemented in PyTorch.
