# SmartCare RL — Stage 4 (TorchRL + DQN)

**Mission:** Community Health Navigation & Telehealth Access. The project is a synthetic simulation for learning, not a medical decision system. No real patient information is used.

## 1. Install and test on Windows PowerShell

Open this folder in VS Code, then run:

```powershell
uv sync
uv run python -m pytest -q
```

`uv sync` installs Gymnasium, PyTorch, TorchRL, TensorDict, Pygame, and pytest in the project environment. If you already opened Stage 3, open this new Stage 4 folder instead; do not merge duplicate folders blindly.

## 2. Verify the TorchRL connection

```powershell
uv run python -m pytest tests/test_dqn.py -v
```

This checks that TorchRL's GymWrapper accepts the **same** custom patient-navigation environment as Stage 3.

## 3. Train your first DQN

Quick pipeline test (200 episodes):

```powershell
uv run python -m training.dqn_training --episodes 200
```

Fuller initial experiment (2000 episodes):

```powershell
uv run python -m training.dqn_training --episodes 2000
```

Training saves `models/dqn/smartcare_dqn.pth` and `logs/dqn_episodes.csv`. Each episode lasts at most three actions. **200 episodes is just a smoke test; it does not guarantee learning.**

## 4. Compare the trained agent with a random policy

```powershell
uv run python -m training.dqn_training --evaluate --episodes 100
```

Both policies are evaluated on the same 100 synthetic patient seeds. Higher mean reward is better. Results are measured, not hard-coded. The model is *not* necessarily better in every run.

## 5. Watch the trained agent in the animated simulator

```powershell
uv run python main.py --visual --model models/dqn/smartcare_dqn.pth
```

The top-right label should say **TRAINED DQN POLICY**, instead of **RANDOM TEST POLICY**. Keep `--model` absent to watch random actions:

```powershell
uv run python main.py --visual
```

GUI controls: Space = pause/autoplay, N = next action when paused, R = restart, Esc = quit.

## What the code does

- `environment/custom_env.py`: unchanged Stage 3 environment with nine observations, five actions, and three-step episodes.
- `environment/rendering.py`: unchanged care-pathway visuals plus an optional trained policy selector.
- `training/dqn_training.py`: TorchRL `GymWrapper`, small PyTorch Q-network, replay buffer, epsilon-greedy selection, Bellman target, Huber loss, target-network updates, model saving, and reward logs.
- `tests/test_dqn.py`: simple network and TorchRL environment checks.

**Next stages:** REINFORCE, PPO, and A2C; 10 real tuning runs each; proper comparison plots; improved GUI as needed; the 7–10-page report; the video; and a clean GitHub submission. **This package is not the complete summative assignment.**

## Notes

- `uv.lock` is intentionally not included because dependency resolution could not be performed in the preparation environment. Run `uv sync` and commit the resulting `uv.lock` once it succeeds on your computer.
- This DQN uses TorchRL for environment interactions (`GymWrapper` and TensorDict stepping), and implements the loss and optimization explicitly in PyTorch for clarity. This meets the Stage 4 interoperability goal; we will check any stricter TorchRL training-objective requirements before final submission.
- `pygame` provides a 2D visual demo. Confirm with the lecturer whether the required **advanced simulation** must use OpenGL/Three.js/MuJoCo before final submission.

## Stage 5: REINFORCE, PPO and A2C

This stage **adds three independent policy-gradient algorithms** without changing
`environment/custom_env.py`, its nine observations, five actions or reward rules.
The environment is wrapped with `torchrl.envs.GymWrapper`, as in our DQN stage.
The new models and logs have separate filenames; existing DQN weights are not touched.

From this project directory in **PowerShell**:

```powershell
uv sync
uv run python -m pytest -q
uv run python -m training.pg_training --algorithm reinforce --episodes 32
uv run python -m training.pg_training --algorithm ppo --episodes 32
uv run python -m training.pg_training --algorithm a2c --episodes 32
```

If these pass, train longer (the 32-episode runs are smoke tests, not final results):

```powershell
uv run python -m training.pg_training --algorithm reinforce --episodes 2000
uv run python -m training.pg_training --algorithm ppo --episodes 2000
uv run python -m training.pg_training --algorithm a2c --episodes 2000
```

Evaluate the saved models on 100 seeded synthetic patient episodes:

```powershell
uv run python -m training.pg_training --algorithm reinforce --evaluate --episodes 100
uv run python -m training.pg_training --algorithm ppo --evaluate --episodes 100
uv run python -m training.pg_training --algorithm a2c --evaluate --episodes 100
```

Show any trained policy in the animated GUI:

```powershell
uv run python main.py --visual --algorithm ppo --model models/pg/ppo.pth
uv run python main.py --visual --algorithm reinforce --model models/pg/reinforce.pth
uv run python main.py --visual --algorithm a2c --model models/pg/a2c.pth
uv run python main.py --visual --algorithm dqn --model models/dqn/smartcare_dqn.pth
```

**Algorithms in brief:** REINFORCE updates its actor from discounted episode
returns. A2C uses an actor plus a critic, with episodic return-minus-value
advantages. PPO uses an actor-critic and limits policy-ratio changes with a
clipped objective over multiple update passes. These are deliberately small,
readable educational implementations. `--batch-episodes` controls the number
of complete episodes per policy update; `--ppo-epochs` and `--clip-epsilon`
are PPO options. The `--lr` and `--gamma` options work for all three.

A score from a single episode is **not** evidence of general superiority.
Keep the DQN baseline result (0.738 mean reward versus random 0.196 over
100 episodes) for comparison, then evaluate the other models. You still need
**ten distinct hyperparameter runs per algorithm**, proper experiment records,
more than one training seed for strong claims, advanced visualization checks,
and the final report/video for the summative assignment.

**Note:** These algorithms are coded in PyTorch and interact with the same
TorchRL-wrapped environment. They do not use TorchRL's built-in policy-loss
classes. If your lecturer requires the algorithms themselves to be implemented
with TorchRL objectives, adapt these models before final submission.

## Stage 6: 40 hyperparameter experiments

See [STAGE6_GUIDE.md](STAGE6_GUIDE.md). You can list, smoke-test, and run all 40
configurations using `experiments.run_experiments`; each real run is saved
independently and automatically summarized in tables and charts.

## Stage 7 — Advanced Three.js 3D visualization

Stage 7 adds a **genuine WebGL/Three.js** 3D replay of the *real* SmartCare environment and any saved DQN/PPO/A2C/REINFORCE policies, without altering the completed 40-experiment results. See [`STAGE7_GUIDE.md`](STAGE7_GUIDE.md) for model-copy instructions and the important distinction between **TorchRL environment usage** and **TorchRL-native objective modules**.

```powershell
uv run python main.py --visual3d --algorithm dqn --model experiments/results/dqn/DQN_09/model.pth
uv run python -m training.torchrl_audit
```

The 3D frontend needs internet to load Three.js from jsDelivr and a WebGL-capable browser. The local Python HTTP server binds only to `127.0.0.1`.
