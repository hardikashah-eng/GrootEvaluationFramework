# Making the twin match the real robot: actuator model, log-clock replay and engine cross-check

A GR00T twin gets close to the real humanoid when two things change from Isaac Lab's defaults:

- its drives are an explicit model fitted to each physical robot;
- replays start from the recorded state just before the failure and run on the log's own clock.

When a failure still won't reproduce, replaying it in MuJoCo as well shows whether the miss comes from the replay, from Isaac Sim, or from the twin's model.

| Piece | Code | What it adds |
|---|---|---|
| Drive chain | [`actuators/drive_chain.py`](../src/groot_eval/actuators/drive_chain.py) | Nominal transport delay with jitter, dropped packets, current-loop lag, angle-dependent linkage torque limit |
| Stand-test identification | [`actuators/identify.py`](../src/groot_eval/actuators/identify.py) | Step metrics, frequency response, delay, friction, backlash, fit of the drive model, acceptance gates |
| Log-clock replay | [`timeline_replay.py`](../src/groot_eval/timeline_replay.py) | Log checks, commands scheduled by timestamp, restore before the failure, three replay modes, replay as a distribution |
| Engine cross-check | [`engine_crosscheck.py`](../src/groot_eval/engine_crosscheck.py) | MuJoCo env with exact state restore; repeatability and restore checks; a verdict on where the miss comes from |
| Real-to-sim format | [`real_format.py`](../src/groot_eval/real_format.py) | Conversion of real log streams with a report of what each step lost |

These build on what was already here: the wear- and fault-aware actuator core ([actuators.md](actuators.md)), open-loop replay and system identification (`replay.py`), failure reproduction with a knob search ([twin-failure-repro.md](twin-failure-repro.md)), determinism checks (`seeding.py`) and HDF5 to LeRobot conversion checks (`conversion_fidelity.py`).

## 1. Actuator model

### What the model chains

```text
command ─► transport delay (per env) + jitter ─► dropped packets (hold last)        DriveChain
        ─► PD on the encoder reading, wear, friction, stiction, torque-speed curve,
           heat derating, faults (motor_off, latency_spike, ...)                      WearAwareActuatorCore
        ─► current-loop lag ─► linkage torque limit by joint angle                    DriveChain
        ─► joint
```

| Stage | Parameters to fit |
|---|---|
| Command transport | Delay range in steps, jitter, drop rate per bus |
| Joint controller on the robot | Real gains, controller rate, encoder resolution and offset |
| Current loop | Time constant |
| Motor | Saturation torque, speed limit, heating and cooling, derating start |
| Gearbox | Backlash, Coulomb, viscous and stiction friction, reflected inertia (set as joint armature in the sim) |
| Linkage (ankles, knees) | Peak torque per joint angle |

In Isaac Lab, add the drive chain to the existing actuator config:

```python
WearAwareActuatorCfg(
    joint_names_expr=[".*_ankle_.*"], effort_limit=88.0, stiffness=150.0, damping=5.0,
    robot_profile="profiles/g1-0042.yaml",
    drive_chain={"delay_steps_min": 1, "delay_steps_max": 3, "jitter_steps": 1, "drop_prob": 0.002,
                 "current_tau_s": 0.002, "linkage": {0: [[-0.6, 40.0], [0.0, 88.0], [0.6, 55.0]]}},
)
```

For an explicit actuator in Isaac Lab, set the solver's own torque limit high so torque isn't clipped twice. Set PhysX joint friction to 0 when the model adds friction. Run physics at 1 to 2 ms.

### Fitting it from stand tests

With the robot lifted, run each joint through four tests:

1. **Steps:** `step_metrics` gives rise time, overshoot, settling time and steady-state error. `onset_delay` gives the pure dead time.
2. **Chirps:** `frequency_response` gives gain (dB), phase (degrees) and coherence per band. `bandwidth_hz` gives the −3 dB point.
3. **Constant-velocity sweeps** in both directions: `fit_friction` gives Coulomb, viscous and stiction friction.
4. **Direction reversals:** `estimate_backlash` gives the dead travel.

`fit_joint` then fits the remaining parameters by simulating the joint on its stand (`I q'' = tau − m g l sin q`) through the drive chain. Every candidate is one environment of a single batched run, refined by an evolution strategy with restarts.

```python
from groot_eval.actuators.identify import JointPlant, fit_joint, onset_delay, fit_friction, estimate_backlash

fixed = {"delay_steps": onset_delay(step_q, step_index), **fit_friction(qd, tau_needed),
         "backlash_rad": estimate_backlash(rev_cmd, rev_q)}
fixed.pop("stiction_torque")
fit = fit_joint([{"q_cmd": step_cmd, "q": step_q}, {"q_cmd": chirp_cmd, "q": chirp_q}],
                JointPlant(inertia=0.02, gravity_torque=0.5), dt=0.005, effort_limit=20.0, fixed=fixed)
```

**Measure delay, friction and backlash first.** Gains, friction, delay and backlash trade off against each other. A fit of all seven from position alone can match the behaviour with the wrong parameters: in the tests, an unconstrained fit matched the replay to 0.0016 rad but chose 15 times the true stiffness. With delay, friction and backlash fixed from their own tests, the fit recovers stiffness, damping and the current-loop constant to within about 4 %.

### Acceptance per joint

`acceptance` compares the real joint with the fitted model on the same commands:

| Check | Default gate |
|---|---|
| Step response | Rise time within ±10 % (or 10 ms); overshoot within ±10 % (or 1 point) |
| Frequency response | Gain within ±2 dB and phase within ±15° up to the real joint's bandwidth |
| Replay | Position RMSE under 0.01 rad |
| Saturation | Torque saturates on the same steps (overlap ≥ 0.7) |

The gates are starting points; tune them per joint class.

## 2. Log-clock replay

### Check the log first

`check_replay_log` refuses logs a replay can't trust:
- a missing policy, action or joint stream, or timestamps that aren't strictly increasing;
- a joint stream that doesn't cover the actions;
- streams whose clocks disagree by more than `max_skew_s` (measured offsets from PTP or a sync pulse are required);
- a missing version stamp (checkpoint, deployed build, firmware, gains, robot profile, twin assets).

### Commands at their logged times

`build_schedule` places each command at the sim step its timestamp falls in and holds it until the next one. Late commands, two commands in one step (the later wins) and missing ticks are replayed as they happened, and counted. On the test log (one late command, one duplicate), the log-clock replay matched the real run exactly. A replay that spaced the same commands evenly at the nominal period was more than 100 times worse.

### Start just before the failure

`restore_index(state_t, t_fail, lead_s)` picks the recorded state `lead_s` seconds before the failure. The replay sets that state, passes the commands sent just before it (so an env with actuation delay can refill its queue) and can run `settle_steps` before starting. Isaac Sim (PhysX) can't restore its internal contact state, so when contact history matters, settle or replay from the episode start.

### Three modes on one timeline

| Mode | What runs | What it tells you |
|---|---|---|
| `action` | Physics on, recorded commands at their logged times | Whether the twin's physics moves like the robot; compared at the log's timestamps |
| `kinematic` | Each recorded state set in turn and read back; `video.*` frames kept | Whether the twin can hold the real state (units, joint order, limits), and renders along the real path for comparison with real frames |
| `observation` | The policy fed the logged observations | Whether the policy acts the same on the logged inputs; isolates perception and model-build differences |

Each mode reports nRMSE per signal and the **divergence time**: the first time any signal's error passes 0.5 of its spread.

### Replay as a distribution

`replay_distribution` replays N times with the uncertain parameters drawn from their measured ranges. It reports:
- the reproduction rate, with a Wilson interval;
- outcome counts by failure mode;
- each parameter's effect: the normalised difference in its mean between reproducing and non-reproducing draws.

A `detect` function can name failures the env doesn't flag itself, such as slips or drops.

**Determinism.** Make the model and the sim repeatable, so the only variation is the uncertainty you chose to sample:
- seed the action head's noise;
- freeze the deployed TensorRT engine;
- use fixed stepping;
- set parameters only at setup.

## 3. Engine cross-check: Isaac Sim or the replay?

When the Isaac Sim twin doesn't reproduce a real failure, `cross_check` (or `cross_check_case` for a `FailureCase`) runs the same failure, with the same replay inputs, in Isaac Sim and in MuJoCo:

1. **Repeatability:** the same replay twice in each engine must give the same trajectory.
2. **State restore:** each engine must hold the recorded states (read back within 5 % of each signal's spread).
3. **Replay:** the log-clock replay in each engine, whether it reproduces the failure, and when it leaves the recording.
4. **Model parity:** joint names, ranges and body masses must match across the two engines' models.

| Verdict | Meaning | Next step |
|---|---|---|
| `replay_not_repeatable` | An engine gives different results for the same replay | Fix seeding, fixed stepping or GPU determinism before anything else |
| `replay_restore_issue` | Neither engine holds the recorded state | Check units, joint order, joint limits and the state mapping |
| `primary_restore_limitation` | MuJoCo holds the recorded state; Isaac doesn't | Replay from the episode start in Isaac |
| `reproduced_in_both` | Both engines reproduce it | The twin is fine for this case |
| `primary_engine_limitation` | MuJoCo reproduces it; Isaac doesn't | An Isaac contact, solver or restore limit. Use MuJoCo for this scenario class, or tune Isaac's contact settings |
| `reference_engine_gap` | Isaac reproduces it; MuJoCo doesn't | The MuJoCo model differs; the parity list names the differences |
| `shared_model_gap` | Neither reproduces it, and both leave the recording at about the same time | The twin's model or the replay inputs miss the cause. Run system identification and the repro knob search |
| `engine_dependent_gap` | Neither reproduces it, and they diverge at very different times | Engine physics differ here. Compare contact settings before trusting either |

`MuJoCoEnv` is a MuJoCo env with the harness interface (`reset`, `step`, `set_state`). It provides:
- exact full-state save and restore (`get_full_state` / `set_full_state`, including the solver's warm start); restored runs are bit-identical;
- scenario parameters applied through `param_setters`, rebuilt each episode so nothing carries over;
- a `torque` mode that runs actions through the drive chain, so the same actuator model drives both engines.

MuJoCo is an optional dependency (`pip install mujoco`); the tests skip without it.

## 4. Can real data become sim data without loss?

For numeric streams: **yes, if every step is chosen to be reversible.** `convert` measures each step and gives a verdict per stream:
- `lossless`: exact round trip;
- `rounding`: float rounding only, below a tolerance of the signal's spread;
- `lossy`: information was lost.

| Step | Lossless when | Lost otherwise |
|---|---|---|
| Column mapping | Every logged column has a slot in the sim | Dropped columns (reported by name) |
| Units | Always, up to ~1e-16 relative rounding | — |
| Quaternion order (w-first ↔ w-last) | Always: it's a reorder; q and −q are the same rotation | — |
| Value precision | float64 kept, or float32 error far below sensor noise | ~7 significant digits in float32 |
| Timestamps | float64 kept | float32 resolves 0.24 ms at one hour and 128 s at Unix epoch time |
| Rate | Native timestamps kept; replay on the log clock instead | Interpolation error and aliasing above the new Nyquist frequency |
| Images | Raw or a lossless codec (checked by `conversion_fidelity`) | Codec loss, colour order, resizing |

What was never logged is never in the data: contact forces, motor temperature, how an object sits in the gripper. No format conversion recovers it. `plan_lossless` gives the target settings for a lossless conversion and lists the columns the sim would need to add.

## Status and limits

- **Tested here** on a CPU: the drive chain and identification on a simulated stand joint; log-clock replay on a point mass with jittered, late and duplicated commands; the cross-check with MuJoCo pendulums standing in for both engines, each verdict produced by a deliberate defect; format conversion on synthetic joint and pose streams.
- **Not run here:** Isaac Sim and Isaac Lab, because there's no GPU. The cross-check calls only the harness env interface, so an Isaac Lab env wrapped like the others plugs in as the primary engine.
- **Not modelled:** joint compliance (series-elastic or harmonic-drive flex), which needs a motor-side state.
- **No command-line entry points yet:** these are Python APIs; `groot-twin` subcommands are a follow-up.
