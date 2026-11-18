# Robot adaptation kit

Describe a new robot once in `robot.yaml`. The kit generates every GR00T, simulator and robot-side config from
it, and two validators catch sign, unit, order and gear mistakes before the robot moves under a policy.

```
robot.yaml ──► kit validate (L0) ──► kit generate ──► kit roundtrip (L1) ──► kit jog (L2) ──► kit bringup --level N
               descriptor checks     10 config files    adapter end to end     twin vs robot      gate report
```

## 1. Start from a template

```bash
groot-robot kit init --template humanoid --name my_humanoid --out robots/my_humanoid.yaml
```

| Template | Groups | Dims | Cameras |
|---|---|---|---|
| `humanoid` | waist, left/right arm, left/right hand (legs left to the robot's own controller) | 19 | 1 |
| `arm_gripper` | arm (7), gripper | 8 | 2 |
| `bimanual` | two 6-joint arms, two grippers | 14 | 3 |
| `mobile_manipulator` | base (relative x, y, yaw), arm (6), gripper | 10 | 2 |

The numbers in the templates are **placeholders**. Replace every limit, gain, camera value and topic with your
robot's datasheet, URDF and calibration.

## 2. The descriptor

Joint groups are listed in the order the policy sees them. Model space is radians in each joint's own frame;
robot space is what the driver reports and accepts. For each joint:

```
robot = sign * gear_ratio * unit_scale * model        unit_scale = 1 (rad) or 180/pi (deg)
```

A gripper group maps `robot_range: [open, closed]` in driver units onto 0..1.

| Section | Holds |
|---|---|
| `groups[]` | `name`, `kind` (arm, hand, gripper, waist, leg, head, base), `action` (absolute or relative), `joint_defaults`, `joints` |
| `joints[]` | `name`, `limits` (rad), `unit`, `sign`, `gear_ratio`, `max_velocity`, `effort_limit`, `stiffness`, `damping` |
| `cameras[]` | `name`, `width`, `height`, `fps`, `color_order` (rgb or bgr), `intrinsics` (fx, fy, cx, cy), `parent_frame`, `xyz`, `quat`, `topic` |
| `timing` | `control_hz`, `policy_hz`, `action_horizon`, `max_latency_ms` |
| `frames` | `base`, `quat_order` (wxyz or xyzw) |
| `safety` | `max_action_rate` (per control step, model units), `velocity_scale` for bring-up |
| `assets` | `urdf`, `usd`, `mjcf` |

`groot-robot kit validate robot.yaml` reports **errors** for:

- duplicate joints or groups;
- limits where low ≥ high;
- a sign other than ±1, or a gear ratio ≤ 0;
- a policy rate faster than the control rate;
- a gripper without a range;
- a camera principal point outside the image;
- an extrinsic quaternion whose norm isn't 1;
- an unknown quaternion order.

It reports **warnings** for:

- limits above 4π, which look like degrees;
- a missing `max_velocity`;
- a control-to-policy rate ratio that isn't a whole number;
- a latency longer than one action chunk;
- a camera slower than the policy;
- missing intrinsics or assets.

## 3. Generated files

`groot-robot kit generate robot.yaml --out gen/`

| File | Used by |
|---|---|
| `modality.json` | LeRobot `meta/`; GR00T fine-tuning and the policy server. Relative groups carry `absolute: false` |
| `gr00t_modality.py` | GR00T fine-tuning: modality config with keys, delta indices and action representations (N1.6 layout) |
| `conversion.yaml` | `groot-twin convert-check`: HDF5 to LeRobot mapping and gates (HDF5 keys are guesses to edit) |
| `isaaclab_robot.py` | Isaac Lab `ArticulationCfg`, one `ImplicitActuatorCfg` per group; swap in `build_actuator` for the wear-aware drive model |
| `mujoco_actuators.xml` | MuJoCo position actuators and joint sensors to include in the robot's MJCF |
| `mujoco_stub.xml` | A joint-chain stand-in with the same names, order, limits and actuators, for testing before the real model exists |
| `ros2_topics.yaml` | `groot-robot bridge` / `rig`: state, command and camera topics (defaults to replace) |
| `logging_schema.yaml` | Policy, action and joint logging with the joint order, rates and chunk shape filled in |
| `robot_profile.yaml` | `groot-eval transfer --robot`: the limits the hardware accepts |
| `MANIFEST.json` | Descriptor hash and file list; `kit generate --check` exits 1 if the files are stale |

The GR00T and Isaac Lab files follow those projects' public config shapes. Their APIs move between releases, so
each file names the version it was written against. Check the imports against your install. Isaac Lab 3.0
orders quaternions XYZW; the file notes the descriptor's order.

## 4. Policy API adapter

`groot_eval.robot_kit.PolicyAdapter(desc)` holds every conversion in one place:

- `pack_obs(robot_readings, images, task)` → `video.<cam>` (1, H, W, 3) uint8 RGB, `state.<group>` (1, d)
  float32 radians, and the language key. BGR cameras are flipped, float frames scaled to uint8.
- `unpack_action(action, robot_now)` → one robot command per control step. Relative groups are added to the
  state at the start of the chunk. Each joint is clipped to its limits and each step to `max_action_rate`;
  the result reports which joints were clipped and how many steps were rate-limited.

## 5. Mapping round trip (L1)

`groot-robot kit roundtrip robot.yaml`

The round trip needs no robot or simulator:

1. Synthetic readings go through `pack_obs`.
2. A stand-in policy echoes the state back as the action.
3. `unpack_action` converts that back into robot commands.

The commands must equal the readings, within 1e-5 of each joint's range (the state passes through float32). The check catches:

- a conversion applied on the way in but not undone on the way out;
- groups packed in one order and unpacked in another;
- a gripper scaled twice;
- a camera's colour order wrong (a red test patch must arrive red);
- `modality.json` widths that disagree with the groups.

Pass `policy=` (for example a GR00T server client) to `mapping_round_trip` to also check the returned keys,
widths and horizon.

## 6. Joint jog test (L2)

The round trip can't catch a descriptor that is wrong the same way in both directions. Take a wrong `sign`:
the command goes out flipped, the reading comes back flipped, and everything looks consistent. Only an
observation that doesn't go through the descriptor can catch it.

The jog test moves one joint at a time by a small step in model space, on two targets:

- the twin, which uses the URDF/MJCF axes directly;
- the robot, through the descriptor.

It then compares how tracked points moved on each. A tracked point can be an end-effector or link pose from the
vendor controller, a mocap marker, or a fiducial seen by a camera.

| Robot vs twin | Diagnosis |
|---|---|
| same | ok |
| opposite | `sign` wrong |
| ×57.3 or ÷57.3 | `unit` wrong (degrees vs radians) |
| another ratio r | `gear_ratio` or a scale off by r |
| moves like twin joint k | joint order or name mapping swapped |
| twin moves, robot doesn't | joint not driven or at a limit |

It also checks each target's own readback (commanded step vs encoder step). A mismatch there means the driver's
state and command conventions differ.

The jog runs the smallest step first (default 0.01 rad, then 0.1 rad). If the small step shows a fault, it stops
before the larger one, so a unit error is caught at a 0.57 rad move rather than a 5.7 rad one. It starts each
joint from a pose inside its limits and jogs away from the nearer limit. **Run real jogs supervised, with the
controller's own limits and e-stop active.**

```bash
# twin only (checks the twin and the kit):
groot-robot kit jog robot.yaml --twin stub --out evidence/jog.json
# twin vs hardware: --robot names a factory that returns a JogTarget for your driver
groot-robot kit jog robot.yaml --twin assets/robot.xml --sites left_ee,right_ee \
    --robot my_driver.kit:jog_target --out evidence/jog.json
```

A `JogTarget` has:

- a `space` attribute: `"robot"` for hardware, `"model"` for a twin;
- `command(positions)`, which moves the joints and waits until they settle;
- `read()`, which returns joint readings by name;
- `points()`, which returns tracked points in metres in the base frame.

`MuJoCoJogTarget` wraps any MuJoCo model whose position actuators are named after the joints.

## 7. Staged bring-up

`groot-robot kit bringup robot.yaml --level 3 --evidence evidence/ --out reports/bringup`

| Level | Gate | Evidence |
|---|---|---|
| L0 | descriptor valid, adapters generate | run by the kit |
| L1 | data pipeline lossless | run by the kit (round trip); `convert_check.json` from `groot-twin convert-check` once real logs exist (warning until then) |
| L2 | twin matches the robot joint by joint | `jog.json`, `actuator_acceptance.json` ([twin actuator doc](twin-actuator-and-replay.md)) |
| L3 | policy runs in the twin | `open_loop.json`, `sim_eval.json` |
| L4 | hardware-in-the-loop timing passes | `timing.json` (`groot-robot timing-check`) |
| L5 | supervised real runs | `real_runs.json` (first LiveChoreBench cell) |

Evidence files are JSON with a top-level `passed`. A level counts only when every level below it passes. The
command writes `bringup.md` and `bringup.json` and exits 1 if the target level isn't reached. Above L1 the kit
only reads evidence: it never drives hardware on its own.

## Not in this version

- Generated LiveChoreBench starter tasks and cell template.
- The camera check against a calibration board (use `groot-eval real2sim` R3 meanwhile).
- An Isaac Lab jog target (the MuJoCo one covers the twin side today).
- A timing-budget runner on the target computer.
