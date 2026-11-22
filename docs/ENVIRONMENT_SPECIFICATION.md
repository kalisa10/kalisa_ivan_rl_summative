# SmartCare RL — Environment specification

## Purpose and justification for reinforcement learning

SmartCare RL is a **synthetic patient appointment-navigation** task, not a medical decision-making system. A navigation agent has a budget of three sequential decisions to select support interventions for one simulated patient. Interventions have immediate costs; their effects persist in the episode; and the final appointment-connected outcome is stochastic and only known after the third decision. A policy must therefore balance near-term costs against a delayed, uncertain terminal reward. Reinforcement learning is appropriate for exploring such sequential trade-offs in a simulated environment. The simulator does not establish clinical effectiveness or readiness for real patient deployment.

## Environment API, start, and end

- Implementation: `environment/custom_env.py`, class `PatientNavigationEnv`, using the Gymnasium `reset`/`step` interface.
- At `reset(seed=...)`: zero the step counter and four intervention flags; generate a synthetic patient profile; return an observation and an empty `info` dictionary.
- Observation space: `Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32)`.
- Action space: `Discrete(5)` with integer actions `0` through `4`.
- `step(action)` returns `(next_observation, reward, terminated, truncated, info)`.
- Terminal condition: exactly three actions (`terminated=True` on step 3). `truncated` is always `False` in the current implementation. `step` after termination raises an error.
- The four intervention flags persist across steps; patient profile is fixed within an episode; elapsed-time fraction increments by `1/3` per action.
- Agent input path: `reset()`/`step()` → nine-element observation → TorchRL `GymWrapper` for training → PyTorch policy/Q-network. During the 3D replay, the same observations go directly into a loaded PyTorch policy through `greedy_action`.

## Observation provenance and update cadence

Every observation is an in-memory **simulation value**; none is collected from patients, clinic records, connected devices, or external services.

| Index | Name | Meaning | Encoding / valid values | Provenance / generation | Update cadence |
|---:|---|---|---|---|---|
| 0 | `phone_access` | Simulated availability of phone access | `float32` binary: 0=no, 1=yes | Seeded `np_random.integers(0, 2)` at reset | Once per episode |
| 1 | `travel_barrier` | Simulated difficulty traveling to clinic | `float32` binary: 0=no, 1=yes | Seeded `np_random.integers(0, 2)` at reset | Once per episode |
| 2 | `telehealth_access` | Simulated ability to access remote care | `float32` binary: 0=no, 1=yes | Seeded `np_random.integers(0, 2)` at reset | Once per episode |
| 3 | `missed_visit_risk` | Synthetic baseline missed-appointment risk score (a model feature, **not** a calibrated probability or diagnosis) | `float32`, sampled in `[0.3, 0.9)` | Seeded `np_random.uniform(0.3, 0.9)` at reset | Once per episode |
| 4 | `reminder_sent` | Whether an eligible reminder has succeeded | `float32` binary: 0/1 | Initially 0; set to 1 when action 0 succeeds | After each action as applicable |
| 5 | `called` | Whether a call action has been registered | `float32` binary: 0/1 | Initially 0; set to 1 when action 1 succeeds | After each action as applicable |
| 6 | `transport_arranged` | Whether travel support was arranged | `float32` binary: 0/1 | Initially 0; set to 1 when action 2 succeeds | After each action as applicable |
| 7 | `telehealth_coordinated` | Whether an eligible remote-care intervention succeeded | `float32` binary: 0/1 | Initially 0; set to 1 when action 3 succeeds | After each action as applicable |
| 8 | `elapsed_fraction` | Fraction of the three-decision navigation window already used | `float32`: 0, 1/3, 2/3, 1 | `steps / max_steps` (`max_steps=3`) | Every action, including invalid/repeated actions |

**Access and action semantics:** The code rejects a reminder action if `phone_access=0`, and rejects telehealth coordination if `telehealth_access=0`. A call action is not blocked by `phone_access=0` in the current simulator; that simplification is a limitation rather than evidence of a real-world callable contact route. A repeated intervention is penalized and does not set a new flag. Action 4 (no additional action) leaves all flags unchanged but still uses one decision step.

## Actions, immediate reward, and state changes

Each action starts with a `-0.02` time/processing cost. If an eligible, not-yet-completed intervention is selected, its action-specific cost is added:

| Action ID | Agent action | Extra cost if successful | Persistent state effect |
|---:|---|---:|---|
| 0 | Send appointment reminder | `-0.05` | Set `reminder_sent=1` (requires phone access) |
| 1 | Call patient | `-0.10` | Set `called=1` |
| 2 | Arrange transport | `-0.18` | Set `transport_arranged=1` |
| 3 | Coordinate telehealth | `-0.15` | Set `telehealth_coordinated=1` (requires telehealth access) |
| 4 | No additional action | `0.00` extra | No flag changes |

Additional exceptions: **repeating** actions 0–3 after the matching flag is set adds a `-0.25` penalty instead of another intervention cost; attempting a reminder without phone access or telehealth without access adds a `-0.35` penalty. Neither failed action sets its flag. The `-0.02` step cost still applies.

## Delayed outcome and reward equation

At the end of the third action, the simulator computes the synthetic probability of connecting to care:

```text
p_raw = 0.60 - 0.40*risk - 0.20*travel_barrier
        + 0.15*reminder_sent*phone_access
        + 0.25*called
        + 0.30*transport_arranged*travel_barrier
        + 0.30*telehealth_coordinated*telehealth_access
p = clip(p_raw, 0.05, 0.95)
attended = seeded_bernoulli(p)
```

The third step then adds **`+2.00` if connected** or **`-1.00` if missed** to the ordinary step/intervention reward. This terminal outcome is delayed, stochastic and action-dependent. The `info` dictionary contains `attended` (Boolean) and `attendance_probability` (float) at termination. The episode return is the sum of the three step rewards.

## Example transition (deterministic pre-outcome behavior)

For a synthetic profile with phone access, a travel barrier and telehealth access, choosing `Arrange transport` as the first action increments `elapsed_fraction` to `1/3`, sets `transport_arranged=1`, and returns the immediate reward `-0.02 - 0.18 = -0.20`. The final result is **not** yet determined. This illustrates delayed credit assignment.

## Rendering and integration evidence

- The Python environment provides the actual observations, transitions, step rewards and terminal outcomes.
- `visual3d/server.py` loads a trained `.pth` checkpoint via `load_selector()`, applies a policy to the observation through `simulate()`, and serializes the complete episode as JSON on the local `/api/episode?seed=42` endpoint.
- `visual3d/index.html` animates those returned events using Three.js. The visualization is a replay of simulator-generated decisions, not a replacement for the environment.
- Recommended evidence: screenshot of 3D scene, terminal launch command, one API response, and comparison chart showing mean reward and connection rate.

## Model limitations and ethics

The environment is a deliberately small research demonstration: each patient gets only three decisions, availability is binary, interventions are hand-costed, attendance probability is an illustrative formula, no real clinical data is used, and no real-world outcome or fairness claim can be inferred. Training and evaluation share the same simulator's patient-generation distribution. Future work should include domain expert review, empirically validated transitions and rewards, broader cases and multiple training seeds before any real-world consideration.
