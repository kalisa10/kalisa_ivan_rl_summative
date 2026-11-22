# SmartCare RL — 3D Visualization and Agent Integration

## Purpose

SmartCare RL models a **synthetic** patient-appointment navigation decision: over three steps, an agent can send a reminder, call the patient, arrange transport, coordinate telehealth, or take no additional action. The simulator is for RL experimentation only; it contains no real patient records and does not make clinical decisions.

The Three.js frontend is **not** a standalone animation of scripted actions. A local Python HTTP service executes the same `PatientNavigationEnv` used for training, queries a saved policy, and returns each action, reward, and final episode outcome to the browser for visualization.

## Architecture and data flow

```text
Saved checkpoint (.pth)                   Synthetic patient seed
          |                                        |
          v                                        v
training.dqn_training.load_model()     PatientNavigationEnv.reset(seed)
 or training.pg_training.load_policy()             |
          |                                        v
          +---- greedy_action(model, obs) <---- observation (9 floats)
                          |                        |
                          +--> env.step(action) ---+
                                    |
                                3 transitions
                                    |
                          visual3d.server.simulate()
                                    |
                GET /api/episode?seed=42  (JSON episode)
                                    |
                           visual3d/index.html
                                    |
                         Three.js / WebGL renderer
```

The Python policy selects actions based on the latest observation, the environment updates intervention flags and step count, and a simulated terminal outcome is sampled at the third step. The frontend animates the returned events; its rendered animation does not calculate the rewards or outcomes. This is a local browser interface, not a deployed mobile service.

## Files and responsibilities

| Source | Responsibility |
|---|---|
| `environment/custom_env.py` | Defines five actions, nine observations, persistent intervention flags, three-step termination and reward/outcome calculation |
| `training/dqn_training.py` | Loads a DQN checkpoint and selects greedy actions |
| `training/pg_training.py` | Loads PPO, A2C or REINFORCE checkpoint and selects greedy actions |
| `visual3d/server.py` | Validates the checkpoint, performs seeded simulation, serializes actual transitions as JSON, serves a local API and static page |
| `visual3d/index.html` | Fetches local episode JSON, animates the pathway with Three.js/WebGL and displays the scenario, actions, rewards and outcome |
| `main.py` | Exposes the `--visual3d`, `--algorithm`, `--model` and `--port` command-line options |
| `tests/test_visual3d.py` | Existing three-step, reproducibility, checkpoint-error and frontend-presence checks |
| `tests/test_visualization_integration.py` | Additional API-payload contract and selected-policy replay tests |

## Checkpoint selection and serialization

From the **project root** with the existing experiment results present:

```powershell
uv sync
uv run python -m pytest -q
uv run python main.py --visual3d --algorithm dqn --model experiments/results/dqn/DQN_09/model.pth
```

For a PPO comparison, stop the running server with `Ctrl+C` and use:

```powershell
uv run python main.py --visual3d --algorithm ppo --model experiments/results/ppo/PPO_06/model.pth
```

The server loads the requested `.pth` checkpoint with the matching training module. If the path is missing, it raises `FileNotFoundError` rather than silently switching to a random policy. When no checkpoint is specified, the 3D route intentionally uses a **random-action demonstration**; the browser header alone is not evidence that a trained policy was loaded. Verify the terminal command and loaded model path when recording evidence.

Saved model paths in this document refer to your experiment folders. They must exist locally or be included in the submitted repository/archive. Do not rerun the 40 training experiments merely to demonstrate a saved model.

## Local HTTP interface

- URL: `http://127.0.0.1:8765/` (default port; change with `--port`)
- `GET /` or `GET /index.html`: serves the browser interface
- `GET /api/episode?seed=42`: computes and returns one real three-action episode as JSON
- Invalid seed (non-integer or outside `0..2147483647`): returns HTTP 400 with an error object
- Unknown paths: HTTP 404
- Other returned values are calculated in Python; the API is **local only** and does not use real patient data

Example JSON *shape* (field names, not invented outcome values):

```json
{
  "algorithm": "DQN",
  "seed": 42,
  "profile": {
    "phone_access": true,
    "travel_barrier": false,
    "telehealth_access": true,
    "missed_visit_risk": 0.5
  },
  "events": [
    { "step": 1, "action_id": 1, "action": "Call patient", "reward": -0.12, "flags": [0, 1, 0, 0] }
  ],
  "total_reward": 0.0,
  "connected": true,
  "attendance_probability": 0.5,
  "note": "Synthetic patient simulation, not medical advice or a real patient"
}
```

**Important:** The numbers and abbreviated single-event list above are only a schema illustration, **not recorded evaluation evidence**. Actual API responses contain exactly **three** events, with `attended` and `probability` added to the final event. The `connected` field is a sampled synthetic outcome, not a clinical fact.

## Frontend playback

The HTML imports Three.js from a CDN and creates a `THREE.WebGLRenderer`. The browser fetches `/api/episode?seed=<seed>` and uses the returned `events` to animate navigation through care locations. Controls provide Play/Pause, Next step, New scenario, editable scenario seed, and playback speed. The interface displays a patient profile, per-step intervention/reward, total reward, synthetic attendance probability and final connected/missed status. Rendering requires a WebGL-capable browser and access to the Three.js CDN; episode generation and model inference remain local.

**Reproducibility:** The same scenario seed produces a repeatable synthetic profile and sampled outcome under the same deterministic greedy checkpoint, simulator version and runtime. Different policies can be compared on the same seed. A single visual episode is illustrative, **not** a substitute for the 100-episode and cohort-level quantitative evaluations.

## Evidence checklist for the report and 5-minute demonstration

1. Capture the terminal command including `--algorithm`, `--model`, and the resolved checkpoint path.
2. Capture a full browser screenshot with the 3D environment, profile, interventions and final score visible.
3. Show the same seed with DQN_09 and PPO_06 if time permits, and note any differing choices.
4. Include an architecture diagram or the data-flow diagram above, and identify the `GET /api/episode` serialization boundary.
5. Cite evaluation results from `experiments/results/comparison.csv` and `diagnostics/generalization_results/` separately from the single replay.
6. Record camera and full-screen desktop as required by the assignment; the video itself still needs separate manual review.

## Limitations and interpretation

- The simulator has a fixed three-step horizon and hand-defined attendance probabilities; it is not validated against real healthcare outcomes.
- A successful episode does not imply a guaranteed appointment connection; the terminal result is sampled.
- The frontend receives a completed episode and **replays** it. It is not an online browser-side learning loop.
- Model checkpoints must be available locally. The Three.js CDN needs internet access.
- No API authentication is implemented because the server binds to `127.0.0.1` for local use; do not expose this development server publicly without additional controls.
- The saved results and demonstration establish behavior in the synthetic environment only. Avoid claims of clinical efficacy.
