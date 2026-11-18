"""Validators for a new robot: the mapping round trip and the joint jog test.

Mapping round trip
------------------
Synthetic robot readings go through the generated Policy API adapter into a
GR00T observation, a stand-in policy echoes the state back as the action, and
the adapter turns that into robot commands. The commands must equal the
readings. Any one-sided mistake (a sign applied on the way in but not on the
way out, groups packed in one order and unpacked in another, a gripper scaled
twice, a BGR camera passed through as RGB) shows up as a mismatch, with the
joint named. No robot or simulator is needed.

Joint jog test
--------------
The round trip cannot catch a descriptor that is wrong in the same way in
both directions: if ``sign`` is wrong, the command goes out flipped and the
reading comes back flipped, and everything looks consistent. Only something
that does not go through the descriptor can catch that. The jog test moves
one joint at a time by a small step in model space, on the twin (which uses
the URDF/MJCF axes directly) and on the robot (through the descriptor), and
compares how tracked points moved: an end effector or link pose reported by
the vendor controller, a mocap marker, or a fiducial seen by a camera. From
the comparison it names the fault:

=================  ===========================================================
Twin vs robot      Diagnosis
=================  ===========================================================
same               ok
opposite           ``sign`` wrong for this joint
x57.3 or /57.3     ``unit`` wrong (degrees vs radians)
other ratio r      ``gear_ratio`` (or a scale) off by r
like twin joint k  joint order: this name drives joint k on the robot
nothing moved      joint not driven, at a limit, or tracked points do not
                   depend on it
=================  ===========================================================

It also checks each target's own readback (commanded step vs encoder step):
a mismatch there means the driver's state and command conventions differ.

The jog runs smallest step first (default 0.01 rad, then 0.1 rad) and stops
before the larger step if the small one already shows a fault, so a unit
error is caught at a 0.57 rad move rather than a 5.7 rad one. Run real jogs
supervised, with the controller's own limits and e-stop active.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from groot_eval.robot_kit.adapter import PolicyAdapter
from groot_eval.robot_kit.descriptor import Descriptor
from groot_eval.robot_kit.generate import modality_json
from groot_eval.types import Finding

DEG = 180.0 / math.pi


# =====================================================================================
# Mapping round trip
# =====================================================================================
@dataclass
class RoundTrip:
    findings: list[Finding]
    max_error: float  # worst joint over all samples, as a fraction of that joint's range
    worst_joint: str | None = None

    @property
    def passed(self) -> bool:
        return not any(f.level == "error" for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "max_error": self.max_error, "worst_joint": self.worst_joint,
                "findings": [f.to_dict() for f in self.findings]}


def _sample_robot(desc: Descriptor, rng: np.random.Generator) -> dict[str, float]:
    """Robot-space readings well inside the limits, each joint a different value."""
    out: dict[str, float] = {}
    for g in desc.groups:
        for j in g.joints:
            u = float(rng.uniform(0.15, 0.85))
            if g.kind == "gripper":
                assert g.robot_range is not None
                o, c = g.robot_range
                out[j.name] = o + u * (c - o)
            else:
                lo, hi = j.limits
                out[j.name] = (lo + u * (hi - lo)) * j.scale
    return out


def _images(desc: Descriptor) -> dict[str, np.ndarray]:
    """Frames whose left half is pure red in RGB, written in each camera's own colour order."""
    out = {}
    for c in desc.cameras:
        img = np.zeros((c.height, c.width, 3), np.uint8)
        img[:, : c.width // 2, 0] = 255
        out[c.name] = img[..., ::-1].copy() if c.color_order == "bgr" else img
    return out


def echo_policy(desc: Descriptor) -> Callable[[dict[str, Any]], dict[str, np.ndarray]]:
    """A stand-in policy: absolute groups hold the current state, relative groups move by zero."""
    T = desc.timing.action_horizon

    def run(obs: dict[str, Any]) -> dict[str, np.ndarray]:
        out = {}
        for g in desc.groups:
            s = np.asarray(obs[f"state.{g.name}"], float)[0]
            a = np.zeros_like(s) if g.action == "relative" else s
            out[f"action.{g.name}"] = np.repeat(a[None], T, axis=0)[None]
        return out

    return run


def _spans(desc: Descriptor) -> dict[str, float]:
    """Each joint's range in robot units, for relative errors (state goes through float32)."""
    out = {}
    for g in desc.groups:
        for j in g.joints:
            if g.kind == "gripper" and g.robot_range is not None:
                out[j.name] = abs(g.robot_range[1] - g.robot_range[0])
            else:
                out[j.name] = abs((j.limits[1] - j.limits[0]) * j.scale)
    return out


def mapping_round_trip(desc: Descriptor, samples: int = 20, seed: int = 0, tol: float = 1e-5,
                       policy: Callable[[dict[str, Any]], Mapping[str, np.ndarray]] | None = None) -> RoundTrip:
    """Check the adapter end to end. With ``policy`` (for example a GR00T server client) the
    returned keys, widths and horizon are checked too; the value round trip always uses the echo
    policy, since a trained policy does not return its input."""
    f: list[Finding] = []
    ad = PolicyAdapter(desc)
    rng = np.random.default_rng(seed)
    images = _images(desc)
    expected_keys = ({f"state.{g.name}" for g in desc.groups} | {f"video.{c.name}" for c in desc.cameras}
                     | {desc.language_key})
    echo = echo_policy(desc)
    spans = _spans(desc)
    worst, worst_joint = 0.0, None
    obs: dict[str, Any] = {}
    for s in range(samples):
        robot = _sample_robot(desc, rng)
        obs = ad.pack_obs(robot, images, task="round trip")
        if s == 0:
            if set(obs) != expected_keys:
                f.append(Finding("error", "obs.keys", f"missing {sorted(expected_keys - set(obs))}, "
                                                      f"extra {sorted(set(obs) - expected_keys)}"))
            # the packed state must be the model vector in descriptor order
            want = np.concatenate([[(robot[j.name] - g.robot_range[0]) / (g.robot_range[1] - g.robot_range[0])
                                    if g.kind == "gripper" and g.robot_range else robot[j.name] / j.scale
                                    for j in g.joints] for g in desc.groups])
            got = np.concatenate([obs[f"state.{g.name}"][0] for g in desc.groups])
            if got.shape != want.shape or not np.allclose(got, want, atol=1e-5):
                f.append(Finding("error", "obs.state", "packed state is not the model vector in descriptor order"))
            for c in desc.cameras:
                v = obs[f"video.{c.name}"]
                if v.shape != (1, c.height, c.width, 3) or v.dtype != np.uint8:
                    f.append(Finding("error", "obs.video", f"{c.name}: {v.shape} {v.dtype}, want (1, {c.height}, "
                                                           f"{c.width}, 3) uint8"))
                elif not (v[0, 0, 0, 0] == 255 and v[0, 0, 0, 2] == 0):
                    f.append(Finding("error", "obs.video", f"{c.name}: red arrives as {v[0, 0, 0].tolist()}; "
                                                           f"colour order is wrong (descriptor: {c.color_order})"))
        un = ad.unpack_action(echo(obs), robot)
        for cmd in un.commands:
            for k, v in robot.items():
                e = abs(cmd[k] - v) / spans[k]
                if e > worst:
                    worst, worst_joint = e, k
        if un.clipped:
            f.append(Finding("error", "action.clip", f"in-range echo was clipped for {sorted(un.clipped)}"))
            break
    if worst > tol:
        f.append(Finding("error", "round_trip", f"command differs from reading by {worst:.3g} of the joint "
                                                f"range at {worst_joint}; a conversion is not inverted on the way out"))
    # modality.json must agree with the packed widths
    mj = modality_json(desc)
    for g in desc.groups:
        se = mj["state"][g.name]
        if se["end"] - se["start"] != g.dim:
            f.append(Finding("error", "modality", f"{g.name}: modality.json width {se['end'] - se['start']} != {g.dim}"))
    if policy is not None and obs:
        out = policy(obs)
        want_keys = {f"action.{g.name}" for g in desc.groups}
        if not want_keys <= set(out):
            f.append(Finding("error", "policy.keys", f"policy did not return {sorted(want_keys - set(out))}"))
        for g in desc.groups:
            a = np.asarray(out.get(f"action.{g.name}", np.zeros((0, 0))))
            a = a[0] if a.ndim == 3 else a
            if a.ndim == 2 and a.shape[1] != g.dim:
                f.append(Finding("error", "policy.width", f"action.{g.name}: width {a.shape[1]}, group has {g.dim}"))
            if a.ndim == 2 and a.shape[0] != desc.timing.action_horizon:
                f.append(Finding("warn", "policy.horizon", f"action.{g.name}: {a.shape[0]} steps, descriptor "
                                                           f"action_horizon {desc.timing.action_horizon}"))
    if not any(x.level == "error" for x in f):
        f.append(Finding("ok", "round_trip", f"{samples} samples, worst error {worst:.2g}"))
    return RoundTrip(f, worst, worst_joint)


# =====================================================================================
# Joint jog test
# =====================================================================================
class JogTarget(Protocol):
    """A robot or twin that the jog test can move. ``space`` is ``"robot"`` (commands and
    readings in robot units, converted through the descriptor) or ``"model"`` (the twin:
    radians in the model's own joint frames, no descriptor in the way)."""

    space: str

    def command(self, positions: Mapping[str, float]) -> None:
        """Command every joint to these positions and wait until settled."""

    def read(self) -> dict[str, float]:
        """Joint readings by name."""

    def points(self) -> dict[str, np.ndarray]:
        """Tracked points (end effectors, link frames, markers), metres, in the base frame."""


@dataclass
class JogRow:
    joint: str
    delta: float
    readback_ratio: float  # encoder step / commanded step (model space)
    displacement: np.ndarray  # tracked points, concatenated (3 * n_points)
    clipped: bool = False


@dataclass
class JogRun:
    target: str
    rows: dict[str, JogRow]
    findings: list[Finding] = field(default_factory=list)


def _to_model(desc: Descriptor, ad: PolicyAdapter, space: str, reading: Mapping[str, float]) -> np.ndarray:
    if space == "model":
        return np.array([float(reading[n]) for n in desc.joint_names])
    return ad.model_state(reading)


def _to_space(desc: Descriptor, ad: PolicyAdapter, space: str, model: np.ndarray) -> dict[str, float]:
    if space == "model":
        return {n: float(v) for n, v in zip(desc.joint_names, model)}
    out: dict[str, float] = {}
    for g, sl in ad.slices.items():
        out.update(ad._to_robot(desc.group(g), model[sl]))
    return out


def jog(desc: Descriptor, target: JogTarget, delta: float, name: str = "target",
        joints: Sequence[str] | None = None) -> JogRun:
    """Jog each joint by ``+delta`` (model space; grippers by ``delta`` of their 0..1 range) from
    the current pose, record the readback and tracked-point motion, and return to the pose."""
    ad = PolicyAdapter(desc)
    names = desc.joint_names
    lo = np.array([j.limits[0] for j in desc.joints])
    hi = np.array([j.limits[1] for j in desc.joints])
    for g, sl in ad.slices.items():
        if desc.group(g).kind == "gripper":
            lo[sl], hi[sl] = 0.0, 1.0
    # start from the current pose, moved inside the limits with room for the jog
    home = _to_model(desc, ad, target.space, target.read())
    margin = np.minimum(2 * abs(delta), (hi - lo) / 4)
    start = np.clip(home, lo + margin, hi - margin)
    if not np.allclose(start, home):
        target.command(_to_space(desc, ad, target.space, start))
        home = _to_model(desc, ad, target.space, target.read())
    p0 = target.points()
    keys = sorted(p0)
    v0 = np.concatenate([np.asarray(p0[k], float).ravel() for k in keys]) if keys else np.zeros(0)
    rows: dict[str, JogRow] = {}
    for i, n in enumerate(names):
        if joints is not None and n not in joints:
            continue
        step = delta if home[i] + delta <= hi[i] else -delta  # jog away from the nearer limit
        cmd = home.copy()
        cmd[i] += step
        clipped = not (lo[i] <= cmd[i] <= hi[i])
        cmd[i] = min(max(cmd[i], lo[i]), hi[i])
        target.command(_to_space(desc, ad, target.space, cmd))
        now = _to_model(desc, ad, target.space, target.read())
        p = target.points()
        v = np.concatenate([np.asarray(p[k], float).ravel() for k in keys]) if keys else np.zeros(0)
        moved = now[i] - home[i]
        rows[n] = JogRow(n, step, float(moved / step) if step else float("nan"), (v - v0) * np.sign(step), clipped)
        target.command(_to_space(desc, ad, target.space, home))
    run = JogRun(name, rows)
    for r in rows.values():
        if not math.isfinite(r.readback_ratio) or abs(r.readback_ratio - 1.0) > 0.2:
            run.findings.append(Finding("error", f"readback.{r.joint}",
                                        f"{name}: commanded {r.delta:+.3g}, encoder moved {r.readback_ratio * r.delta:+.3g} "
                                        f"(ratio {r.readback_ratio:.3g}); {_ratio_hint(r.readback_ratio)}"))
    return run


def _ratio_hint(r: float) -> str:
    if not math.isfinite(r) or abs(r) < 0.05:
        return "the joint did not move: not driven, at a limit, or command and state use different names/order"
    if abs(r + 1) < 0.2:
        return "command and state directions disagree"
    for k, label in ((DEG, "x57.3: degrees vs radians"), (1 / DEG, "/57.3: radians vs degrees")):
        if abs(abs(r) / k - 1) < 0.1:
            return label
    return f"command and state scales differ by {r:.3g} (gear ratio applied on one side only?)"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def compare_jogs(desc: Descriptor, twin: JogRun, robot: JogRun, still: float = 1e-4,
                 ratio_tol: float = 0.25) -> list[Finding]:
    """Diagnose each joint by comparing tracked-point motion on the robot with the twin's."""
    f: list[Finding] = []
    joints = [n for n in desc.joint_names if n in twin.rows and n in robot.rows]
    for n in joints:
        u_t, u_r = twin.rows[n].displacement, robot.rows[n].displacement
        nt, nr = float(np.linalg.norm(u_t)), float(np.linalg.norm(u_r))
        if nt < still and nr < still:
            f.append(Finding("warn", f"jog.{n}", "no tracked point moved in twin or robot; add a point that depends "
                                                 "on this joint"))
            continue
        if nr < still:
            f.append(Finding("error", f"jog.{n}", "twin moved, robot did not: joint not driven or at a limit"))
            continue
        if nt < still:
            f.append(Finding("error", f"jog.{n}", "robot moved, twin did not: twin joint missing or locked"))
            continue
        c = _cos(u_t, u_r)
        best_k, best_c = n, c
        for k in joints:
            ck = _cos(twin.rows[k].displacement, u_r)
            if abs(ck) > abs(best_c) + 0.05:
                best_k, best_c = k, ck
        if best_k != n and abs(best_c) > 0.9:
            f.append(Finding("error", f"jog.{n}", f"robot moves like twin joint {best_k!r}: joint order or name "
                                                  f"mapping is swapped"))
            continue
        r = nr / nt
        if c < -0.9:
            f.append(Finding("error", f"jog.{n}", f"robot moved opposite to the twin: flip sign (now "
                                                  f"{desc_sign(desc, n)})"))
        elif c < 0.9:
            f.append(Finding("warn", f"jog.{n}", f"robot motion direction differs from the twin (cos {c:.2f}): "
                                                 f"axis or kinematics mismatch"))
        elif abs(r - 1) <= ratio_tol:
            f.append(Finding("ok", f"jog.{n}", f"matches twin (ratio {r:.3g})"))
        elif any(abs(r / k - 1) < 0.15 for k in (DEG, 1 / DEG)):
            f.append(Finding("error", f"jog.{n}", f"robot moved {r:.3g}x the twin: unit is wrong (degrees vs radians)"))
        else:
            f.append(Finding("error", f"jog.{n}", f"robot moved {r:.3g}x the twin: gear_ratio or scale is off by "
                                                  f"that factor"))
    return f


def desc_sign(desc: Descriptor, joint: str) -> int:
    return next(j.sign for j in desc.joints if j.name == joint)


@dataclass
class JogReport:
    findings: list[Finding]
    deltas_run: list[float]

    @property
    def passed(self) -> bool:
        return not any(f.level == "error" for f in self.findings)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "deltas_run": self.deltas_run, "findings": [f.to_dict() for f in self.findings]}


def jog_test(desc: Descriptor, twin: JogTarget, robot: JogTarget, deltas: Sequence[float] = (0.01, 0.1),
             still: float | None = None) -> JogReport:
    """Run the jog on the twin and the robot, smallest step first; stop at the first step with a fault."""
    findings: list[Finding] = []
    run: list[float] = []
    for d in sorted(deltas):
        run.append(d)
        t = jog(desc, twin, d, "twin")
        r = jog(desc, robot, d, "robot")
        fs = t.findings + r.findings + compare_jogs(desc, t, r, still=still if still is not None else 1e-3 * d)
        findings = [Finding(x.level, x.check, f"[{d:g} rad] {x.detail}") for x in fs]
        if any(x.level == "error" for x in fs):
            break
    return JogReport(findings, run)


# =====================================================================================
# MuJoCo twin as a jog target
# =====================================================================================
class MuJoCoJogTarget:
    """A MuJoCo model driven by position actuators named after the joints. ``space`` is ``"model"``.

    ``points`` are the world positions of the model's sites (all of them, or ``sites``)."""

    space = "model"

    def __init__(self, xml: str, settle_steps: int = 1500, sites: Sequence[str] | None = None):
        import mujoco

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.settle_steps = settle_steps
        mujoco.mj_forward(self.model, self.data)
        n = self.model.nsite
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i) for i in range(n)]
        self.sites = [s for s in names if s and (sites is None or s in sites)]

    def _jid(self, name: str) -> int:
        return int(self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_JOINT, name))

    def command(self, positions: Mapping[str, float]) -> None:
        for k, v in positions.items():
            aid = self._mj.mj_name2id(self.model, self._mj.mjtObj.mjOBJ_ACTUATOR, k)
            if aid < 0:
                raise KeyError(f"no actuator named {k!r}")
            self.data.ctrl[aid] = v
        for _ in range(self.settle_steps):
            self._mj.mj_step(self.model, self.data)

    def read(self) -> dict[str, float]:
        out = {}
        for i in range(self.model.njnt):
            n = self._mj.mj_id2name(self.model, self._mj.mjtObj.mjOBJ_JOINT, i)
            if n:
                out[n] = float(self.data.qpos[self.model.jnt_qposadr[i]])
        return out

    def points(self) -> dict[str, np.ndarray]:
        return {s: self.data.site(s).xpos.copy() for s in self.sites}


def add_tip_sites(xml: str, desc: Descriptor) -> str:
    """Put a site at the tip of each link in :func:`generate.mujoco_stub_xml`, as tracked points."""
    for j in desc.joints:
        geom = '<geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02" mass="1.0"/>'
        anchor = f'<joint name="{j.name}" '
        i = xml.index(anchor)
        k = xml.index(geom, i) + len(geom)
        xml = xml[:k] + f'\n      <site name="tip_{j.name}" pos="0.1 0 0"/>' + xml[k:]
    return xml
