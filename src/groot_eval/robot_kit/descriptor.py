"""The robot descriptor: one ``robot.yaml`` that every adapter is generated from.

Most of the work of bringing a new robot into GR00T is writing the same facts
down several times: the joint order in the modality config, the joint names in
the Isaac Lab actuator config, the units in the converter, the topic names in
the ROS 2 bridge, the limits in the robot profile. Each copy is a chance for a
sign, unit or order mistake that only shows up when the robot moves the wrong
way. The descriptor holds those facts once; :mod:`groot_eval.robot_kit.generate`
writes the copies.

Conventions
-----------
* Model space is what the policy sees: radians, joint frame, in descriptor
  order, groups concatenated in the order they are listed.
* Robot space is what the driver reports and accepts. For a joint,
  ``robot = sign * gear_ratio * unit_scale * model`` where ``unit_scale`` is 1
  for ``rad`` and 180/pi for ``deg``. ``gear_ratio`` is 1 unless the driver
  reports motor-side angles.
* A gripper group is one value per gripper, ``robot_range`` in robot units
  mapped linearly onto ``[0, 1]`` (0 open, 1 closed) in model space.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from groot_eval.types import Finding

SCHEMA_VERSION = 1
UNIT_SCALE = {"rad": 1.0, "deg": 180.0 / math.pi}
GROUP_KINDS = {"arm", "hand", "gripper", "waist", "leg", "head", "base"}
ACTION_REPS = {"absolute", "relative"}
QUAT_ORDERS = {"wxyz", "xyzw"}
COLOR_ORDERS = {"rgb", "bgr"}


@dataclass
class Joint:
    name: str
    limits: tuple[float, float]  # model space, rad
    unit: str = "rad"
    sign: int = 1
    gear_ratio: float = 1.0
    max_velocity: float | None = None  # rad/s, model space
    effort_limit: float | None = None  # N*m
    stiffness: float | None = None  # PD gains for the sim actuator config
    damping: float | None = None

    @property
    def scale(self) -> float:
        """robot = scale * model."""
        return self.sign * self.gear_ratio * UNIT_SCALE.get(self.unit, 1.0)


@dataclass
class Group:
    name: str
    kind: str
    joints: list[Joint]
    action: str = "absolute"  # absolute | relative (delta from the current state)
    robot_range: tuple[float, float] | None = None  # gripper: robot units for open .. closed
    state_topic: str | None = None
    command_topic: str | None = None

    @property
    def dim(self) -> int:
        return len(self.joints)


@dataclass
class Camera:
    name: str
    width: int
    height: int
    fps: float
    color_order: str = "rgb"
    intrinsics: dict[str, float] | None = None  # fx, fy, cx, cy
    parent_frame: str | None = None
    xyz: tuple[float, float, float] | None = None
    quat: tuple[float, float, float, float] | None = None  # in frames.quat_order
    topic: str | None = None


@dataclass
class Timing:
    control_hz: float
    policy_hz: float
    action_horizon: int = 16
    max_latency_ms: float | None = None


@dataclass
class Safety:
    max_action_rate: float | None = None  # largest change per control step, model units
    velocity_scale: float = 1.0  # fraction of each joint's max_velocity allowed in bring-up
    workspace: dict[str, list[float]] | None = None


@dataclass
class Descriptor:
    name: str
    embodiment_tag: str
    groups: list[Group]
    cameras: list[Camera]
    timing: Timing
    safety: Safety = field(default_factory=Safety)
    quat_order: str = "wxyz"
    base_frame: str = "base_link"
    language_key: str = "annotation.human.action.task_description"
    assets: dict[str, str] = field(default_factory=dict)  # urdf, usd, mjcf, profile
    template: str | None = None
    schema_version: int = SCHEMA_VERSION

    # -- views ---------------------------------------------------------------------------------
    @property
    def joints(self) -> list[Joint]:
        return [j for g in self.groups for j in g.joints]

    @property
    def joint_names(self) -> list[str]:
        return [j.name for j in self.joints]

    def slices(self) -> dict[str, slice]:
        """Group name -> its slice of the concatenated model vector."""
        out, i = {}, 0
        for g in self.groups:
            out[g.name] = slice(i, i + g.dim)
            i += g.dim
        return out

    @property
    def dim(self) -> int:
        return sum(g.dim for g in self.groups)

    def group(self, name: str) -> Group:
        for g in self.groups:
            if g.name == name:
                return g
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for g in d["groups"]:
            for j in g["joints"]:
                j["limits"] = list(j["limits"])
        return d


# -- loading ---------------------------------------------------------------------------------------
def _tuple(v: Any, n: int | None = None) -> tuple[Any, ...] | None:
    if v is None:
        return None
    t = tuple(float(x) for x in v)
    if n is not None and len(t) != n:
        raise ValueError(f"expected {n} numbers, got {list(v)}")
    return t


def _joint(d: dict[str, Any] | str, defaults: dict[str, Any]) -> Joint:
    if isinstance(d, str):
        d = {"name": d}
    m = {**defaults, **d}
    lim = m.get("limits")
    if lim is None:
        raise ValueError(f"joint {m.get('name')!r} has no limits")
    return Joint(name=str(m["name"]), limits=(float(lim[0]), float(lim[1])), unit=str(m.get("unit", "rad")),
                 sign=int(m.get("sign", 1)), gear_ratio=float(m.get("gear_ratio", 1.0)),
                 max_velocity=m.get("max_velocity"), effort_limit=m.get("effort_limit"),
                 stiffness=m.get("stiffness"), damping=m.get("damping"))


def from_dict(raw: dict[str, Any]) -> Descriptor:
    """Build a :class:`Descriptor` from parsed YAML. Raises ``ValueError`` on shape errors;
    call :func:`validate` for the semantic checks."""
    if not isinstance(raw, dict):
        raise ValueError("descriptor must be a mapping")
    for key in ("name", "groups", "timing"):
        if key not in raw:
            raise ValueError(f"descriptor is missing {key!r}")
    groups = []
    for g in raw["groups"]:
        defaults = g.get("joint_defaults", {}) or {}
        rr = g.get("robot_range")
        groups.append(Group(name=str(g["name"]), kind=str(g.get("kind", "arm")),
                            joints=[_joint(j, defaults) for j in g.get("joints", [])],
                            action=str(g.get("action", "absolute")),
                            robot_range=(float(rr[0]), float(rr[1])) if rr is not None else None,
                            state_topic=g.get("state_topic"), command_topic=g.get("command_topic")))
    cams = []
    for c in raw.get("cameras", []) or []:
        cams.append(Camera(name=str(c["name"]), width=int(c["width"]), height=int(c["height"]), fps=float(c["fps"]),
                           color_order=str(c.get("color_order", "rgb")), intrinsics=c.get("intrinsics"),
                           parent_frame=c.get("parent_frame"), xyz=_tuple(c.get("xyz"), 3),
                           quat=_tuple(c.get("quat"), 4), topic=c.get("topic")))
    t = raw["timing"]
    timing = Timing(control_hz=float(t["control_hz"]), policy_hz=float(t["policy_hz"]),
                    action_horizon=int(t.get("action_horizon", 16)), max_latency_ms=t.get("max_latency_ms"))
    s = raw.get("safety", {}) or {}
    safety = Safety(max_action_rate=s.get("max_action_rate"), velocity_scale=float(s.get("velocity_scale", 1.0)),
                    workspace=s.get("workspace"))
    frames = raw.get("frames", {}) or {}
    return Descriptor(name=str(raw["name"]), embodiment_tag=str(raw.get("embodiment_tag", "new_embodiment")),
                      groups=groups, cameras=cams, timing=timing, safety=safety,
                      quat_order=str(frames.get("quat_order", "wxyz")), base_frame=str(frames.get("base", "base_link")),
                      language_key=str(raw.get("language_key", "annotation.human.action.task_description")),
                      assets=dict(raw.get("assets", {}) or {}), template=raw.get("template"),
                      schema_version=int(raw.get("schema_version", SCHEMA_VERSION)))


def load(path: str | Path) -> Descriptor:
    with open(path, encoding="utf-8") as fh:
        return from_dict(yaml.safe_load(fh))


# -- validation ------------------------------------------------------------------------------------
def validate(d: Descriptor) -> list[Finding]:
    """Semantic checks. ``error`` findings block bring-up level L0."""
    f: list[Finding] = []

    def err(check: str, msg: str) -> None:
        f.append(Finding("error", check, msg))

    def warn(check: str, msg: str) -> None:
        f.append(Finding("warn", check, msg))

    if d.schema_version != SCHEMA_VERSION:
        err("schema", f"schema_version {d.schema_version}, this kit reads {SCHEMA_VERSION}")
    if not d.groups:
        err("groups", "no joint groups")
    seen_groups: set[str] = set()
    seen: set[str] = set()
    for g in d.groups:
        if g.name in seen_groups:
            err("groups", f"group {g.name!r} listed twice")
        seen_groups.add(g.name)
        if g.kind not in GROUP_KINDS:
            err("groups", f"group {g.name!r}: kind {g.kind!r} not one of {sorted(GROUP_KINDS)}")
        if g.action not in ACTION_REPS:
            err("groups", f"group {g.name!r}: action {g.action!r} not one of {sorted(ACTION_REPS)}")
        if not g.joints:
            err("groups", f"group {g.name!r} has no joints")
        if g.kind == "gripper":
            if g.robot_range is None:
                err("gripper", f"group {g.name!r}: a gripper needs robot_range [open, closed]")
            elif g.robot_range[0] == g.robot_range[1]:
                err("gripper", f"group {g.name!r}: robot_range open == closed")
            if g.action == "relative":
                warn("gripper", f"group {g.name!r}: relative gripper commands are unusual; check the dataset")
        for j in g.joints:
            if j.name in seen:
                err("joints", f"joint {j.name!r} appears twice")
            seen.add(j.name)
            lo, hi = j.limits
            if not lo < hi:
                err("joints", f"joint {j.name!r}: limits {list(j.limits)} are not low < high")
            if j.unit not in UNIT_SCALE:
                err("joints", f"joint {j.name!r}: unit {j.unit!r} not one of {sorted(UNIT_SCALE)}")
            if j.sign not in (1, -1):
                err("joints", f"joint {j.name!r}: sign must be 1 or -1, got {j.sign}")
            if not j.gear_ratio > 0:
                err("joints", f"joint {j.name!r}: gear_ratio must be > 0 (use sign for direction)")
            if g.kind not in ("gripper", "base") and max(abs(lo), abs(hi)) > 4 * math.pi:
                warn("joints", f"joint {j.name!r}: limits {list(j.limits)} exceed 4*pi; were they written in degrees?")
            if g.kind != "gripper" and j.max_velocity is None:
                warn("joints", f"joint {j.name!r}: no max_velocity; bring-up jog speed cannot be capped")
    t = d.timing
    if t.control_hz <= 0 or t.policy_hz <= 0:
        err("timing", "control_hz and policy_hz must be > 0")
    elif t.policy_hz > t.control_hz:
        err("timing", f"policy_hz {t.policy_hz} is faster than control_hz {t.control_hz}")
    else:
        ratio = t.control_hz / t.policy_hz
        if abs(ratio - round(ratio)) > 1e-6:
            warn("timing", f"control_hz / policy_hz = {ratio:.3f} is not a whole number; action chunks will be resampled")
    if t.action_horizon < 1:
        err("timing", "action_horizon must be >= 1")
    if t.max_latency_ms is not None and t.policy_hz > 0 and t.max_latency_ms > 1000.0 / t.policy_hz * t.action_horizon:
        warn("timing", "max_latency_ms is longer than one action chunk; the robot will run out of actions")
    if d.quat_order not in QUAT_ORDERS:
        err("frames", f"quat_order {d.quat_order!r} not one of {sorted(QUAT_ORDERS)}")
    names: set[str] = set()
    for c in d.cameras:
        if c.name in names:
            err("cameras", f"camera {c.name!r} listed twice")
        names.add(c.name)
        if c.width <= 0 or c.height <= 0 or c.fps <= 0:
            err("cameras", f"camera {c.name!r}: width, height and fps must be > 0")
        if c.color_order not in COLOR_ORDERS:
            err("cameras", f"camera {c.name!r}: color_order {c.color_order!r} not one of {sorted(COLOR_ORDERS)}")
        if c.fps > 0 and t.policy_hz > c.fps:
            warn("cameras", f"camera {c.name!r} runs at {c.fps} fps, slower than the policy ({t.policy_hz} Hz)")
        if c.intrinsics:
            k = c.intrinsics
            if not (0 < k.get("cx", -1) < c.width and 0 < k.get("cy", -1) < c.height):
                err("cameras", f"camera {c.name!r}: principal point outside the image; intrinsics for another resolution?")
        else:
            warn("cameras", f"camera {c.name!r}: no intrinsics; the camera check cannot run")
        if c.quat is not None:
            n = math.sqrt(sum(q * q for q in c.quat))
            if abs(n - 1.0) > 1e-3:
                err("cameras", f"camera {c.name!r}: extrinsic quaternion norm {n:.4f}, not 1")
    if not d.cameras:
        warn("cameras", "no cameras: a GR00T policy needs at least one video stream")
    if not (d.assets.get("urdf") or d.assets.get("usd") or d.assets.get("mjcf")):
        warn("assets", "no URDF, USD or MJCF listed; the twin cannot be built from this descriptor")
    if d.safety.velocity_scale <= 0 or d.safety.velocity_scale > 1:
        err("safety", "velocity_scale must be in (0, 1]")
    if not any(x.level == "error" for x in f):
        f.append(Finding("ok", "descriptor", f"{d.name}: {len(d.groups)} groups, {d.dim} dims, {len(d.cameras)} cameras"))
    return f
