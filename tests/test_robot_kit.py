"""Robot adaptation kit: descriptor, generators, adapter, round trip, jog test, bring-up, CLI."""

from __future__ import annotations

import ast
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
import yaml

from groot_eval.robot_kit import TEMPLATES, TEMPLATES_DIR, PolicyAdapter, from_dict, load, validate
from groot_eval.robot_kit import bringup, generate
from groot_eval.robot_kit.validators import (
    DEG,
    MuJoCoJogTarget,
    add_tip_sites,
    echo_policy,
    jog_test,
    mapping_round_trip,
)

HUMANOID = TEMPLATES_DIR / "humanoid.yaml"


def raw(name: str = "humanoid") -> dict:
    return yaml.safe_load((TEMPLATES_DIR / f"{name}.yaml").read_text())


def levels(findings, level):
    return [f for f in findings if f.level == level]


# -- descriptor ------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", TEMPLATES)
def test_templates_validate_clean(name):
    d = load(TEMPLATES_DIR / f"{name}.yaml")
    assert not levels(validate(d), "error")


def test_humanoid_layout():
    d = load(HUMANOID)
    assert d.dim == 19
    assert [g.name for g in d.groups] == ["waist", "left_arm", "right_arm", "left_hand", "right_hand"]
    assert d.slices()["left_arm"] == slice(3, 10)


@pytest.mark.parametrize("mutate, check", [
    (lambda r: r["groups"][0]["joints"].append(dict(r["groups"][0]["joints"][0])), "joints"),
    (lambda r: r["groups"][1]["joints"][0].update(limits=[1.0, -1.0]), "joints"),
    (lambda r: r["groups"][1]["joints"][0].update(sign=2), "joints"),
    (lambda r: r["groups"][1]["joints"][0].update(unit="grad"), "joints"),
    (lambda r: r["timing"].update(policy_hz=100), "timing"),
    (lambda r: r["groups"][3].pop("robot_range"), "gripper"),
    (lambda r: r["cameras"][0]["intrinsics"].update(cx=900.0), "cameras"),
    (lambda r: r["cameras"][0].update(quat=[1.0, 0.5, 0.0, 0.0]), "cameras"),
    (lambda r: r.update(frames={"quat_order": "zyxw"}), "frames"),
])
def test_validate_errors(mutate, check):
    r = raw()
    mutate(r)
    errs = levels(validate(from_dict(r)), "error")
    assert any(f.check == check for f in errs), errs


def test_degrees_in_limits_warns():
    r = raw()
    r["groups"][1]["joints"][0]["limits"] = [-170, 170]
    assert any("degrees" in f.detail for f in levels(validate(from_dict(r)), "warn"))


def test_missing_limits_raises():
    r = raw()
    r["groups"][1]["joints"][0].pop("limits")
    with pytest.raises(ValueError):
        from_dict(r)


# -- generators ------------------------------------------------------------------------------------
def test_generate_all(tmp_path):
    d = load(HUMANOID)
    paths = generate.write_all(d, tmp_path)
    names = {p.name for p in paths}
    assert {"modality.json", "gr00t_modality.py", "isaaclab_robot.py", "mujoco_actuators.xml",
            "ros2_topics.yaml", "logging_schema.yaml", "robot_profile.yaml", "MANIFEST.json"} <= names
    for py in ("gr00t_modality.py", "isaaclab_robot.py"):
        ast.parse((tmp_path / py).read_text())
    mj = json.loads((tmp_path / "modality.json").read_text())
    assert mj["state"]["right_arm"] == {"start": 10, "end": 17}
    assert set(mj["video"]) == {"ego_view"}
    log = yaml.safe_load((tmp_path / "logging_schema.yaml").read_text())
    assert log["joint_order"] == d.joint_names and log["policy"]["action_chunk_shape"] == [16, 19]
    prof = yaml.safe_load((tmp_path / "robot_profile.yaml").read_text())
    assert prof["state_limits"]["left_hand"] == [[0.0, 1.0]]
    assert not generate.is_stale(d, tmp_path)
    d.groups[1].joints[0].sign = -1
    assert generate.is_stale(d, tmp_path)


def test_relative_group_marked_in_modality():
    d = load(TEMPLATES_DIR / "mobile_manipulator.yaml")
    mj = generate.modality_json(d)
    assert mj["action"]["base"]["absolute"] is False and "absolute" not in mj["action"]["arm"]
    assert "RELATIVE" in generate.gr00t_modality_py(d)


def test_mujoco_files_load():
    mujoco = pytest.importorskip("mujoco")
    d = load(HUMANOID)
    m = mujoco.MjModel.from_xml_string(generate.mujoco_stub_xml(d))
    assert m.njnt == d.dim and m.nu == d.dim
    # the actuator block includes into any model with the same joint names
    stub = generate.mujoco_stub_xml(d).replace("  <actuator>", "  <!--").replace("  </actuator>", "  -->")
    merged = stub.replace("</mujoco>", "\n".join(generate.mujoco_actuators_xml(d).splitlines()[1:-1]) + "\n</mujoco>")
    assert mujoco.MjModel.from_xml_string(merged).nsensor == d.dim


# -- adapter ---------------------------------------------------------------------------------------
def test_adapter_units_signs_gripper_and_bgr():
    r = raw()
    r["groups"][1]["joints"][3].update(unit="deg", sign=-1)  # left_elbow reported in degrees, reversed
    d = from_dict(r)
    ad = PolicyAdapter(d)
    robot = {n: 0.0 for n in d.joint_names}
    robot["left_elbow_joint"] = -30.0  # deg, reversed: model +30 deg
    robot["left_hand_joint"] = 25.0  # of 0..100
    img = np.zeros((480, 640, 3), np.uint8)
    img[..., 2] = 200  # red in BGR
    obs = ad.pack_obs(robot, {"ego_view": img}, task="pick")
    assert obs["state.left_arm"][0, 3] == pytest.approx(math.radians(30), rel=1e-6)
    assert obs["state.left_hand"][0, 0] == pytest.approx(0.25)
    assert obs["video.ego_view"][0, 0, 0].tolist() == [200, 0, 0]
    assert obs[d.language_key] == ["pick"]


def test_adapter_clips_and_rate_limits():
    d = load(HUMANOID)
    ad = PolicyAdapter(d)
    robot = {n: 0.0 for n in d.joint_names}
    act = {f"action.{g.name}": np.zeros((4, g.dim)) for g in d.groups}
    act["action.left_arm"][:, 3] = 5.0  # elbow far past its 2.09 limit
    un = ad.unpack_action(act, robot)
    assert un.clipped == {"left_elbow_joint": 4}
    assert un.rate_limited >= 1
    assert un.commands[0]["left_elbow_joint"] == pytest.approx(0.1)  # one rate step
    with pytest.raises(ValueError):
        ad.unpack_action({**act, "action.left_arm": np.zeros((4, 6))}, robot)


# -- round trip ------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", TEMPLATES)
def test_round_trip_passes_on_templates(name):
    res = mapping_round_trip(load(TEMPLATES_DIR / f"{name}.yaml"))
    assert res.passed, res.findings
    assert res.max_error < 1e-5


def test_round_trip_catches_one_sided_bug(monkeypatch):
    d = load(HUMANOID)
    orig = PolicyAdapter._to_robot

    def buggy(self, g, model):  # forgets the sign on the way out for one joint
        out = orig(self, g, model)
        if "left_elbow_joint" in out:
            out["left_elbow_joint"] = -out["left_elbow_joint"]
        return out

    monkeypatch.setattr(PolicyAdapter, "_to_robot", buggy)
    res = mapping_round_trip(d)
    assert not res.passed and res.worst_joint == "left_elbow_joint"


def test_round_trip_catches_colour_order(monkeypatch):
    import groot_eval.robot_kit.validators as v

    d = load(HUMANOID)
    real = v._images

    def rgb_frames(desc):  # the driver actually sends RGB although the descriptor says BGR
        out = real(desc)
        return {k: im[..., ::-1].copy() for k, im in out.items()}

    monkeypatch.setattr(v, "_images", rgb_frames)
    assert any(f.check == "obs.video" for f in mapping_round_trip(d).findings)


def test_round_trip_checks_policy_shape():
    d = load(HUMANOID)

    def short_policy(obs):
        out = echo_policy(d)(obs)
        out["action.left_arm"] = out["action.left_arm"][..., :6]
        return out

    res = mapping_round_trip(d, samples=2, policy=short_policy)
    assert any(f.check == "policy.width" for f in res.findings)


# -- jog test --------------------------------------------------------------------------------------
class FakeRobot:
    """Hardware stand-in: a MuJoCo twin behind a driver whose true conventions may differ
    from the descriptor (sign, unit, gear, swapped wiring)."""

    space = "robot"

    def __init__(self, desc, xml, scale: Mapping[str, float] | None = None, swap: tuple[str, str] | None = None):
        self.desc = desc
        self.twin = MuJoCoJogTarget(xml)
        self.scale = {j.name: j.scale for j in desc.joints}
        self.scale.update(scale or {})
        self.wire = {n: n for n in desc.joint_names}
        if swap:
            a, b = swap
            self.wire[a], self.wire[b] = b, a
        self.grip = {j.name: g.robot_range for g in desc.groups if g.kind == "gripper" for j in g.joints}

    def _model(self, n, v):
        if n in self.grip:
            o, c = self.grip[n]
            return (v - o) / (c - o)
        return v / self.scale[n]

    def _robot(self, n, q):
        if n in self.grip:
            o, c = self.grip[n]
            return o + q * (c - o)
        return q * self.scale[n]

    def command(self, positions):
        self.twin.command({self.wire[n]: self._model(n, v) for n, v in positions.items()})

    def read(self):
        q = self.twin.read()
        return {n: self._robot(n, q[self.wire[n]]) for n in self.desc.joint_names}

    def points(self):
        return self.twin.points()


@pytest.fixture(scope="module")
def arm():
    pytest.importorskip("mujoco")
    d = load(TEMPLATES_DIR / "arm_gripper.yaml")
    return d, add_tip_sites(generate.mujoco_stub_xml(d), d)


def run_jog(d, xml, **fault):
    return jog_test(d, MuJoCoJogTarget(xml), FakeRobot(d, xml, **fault))


def jog_errors(rep, joint):
    return [f.detail for f in rep.findings if f.level == "error" and f.check.endswith(joint)]


def test_jog_clean(arm):
    d, xml = arm
    rep = run_jog(d, xml)
    assert rep.passed, rep.findings
    assert rep.deltas_run == [0.01, 0.1]


def test_jog_finds_sign(arm):
    d, xml = arm
    rep = run_jog(d, xml, scale={"joint2": -1.0})
    assert any("flip sign" in e for e in jog_errors(rep, "joint2"))
    assert rep.deltas_run == [0.01]  # stopped before the larger step


def test_jog_finds_degrees(arm):
    d, xml = arm
    rep = run_jog(d, xml, scale={"joint3": 1.0 / DEG})  # driver takes rad, descriptor says nothing: 57x move
    assert any("degrees" in e for e in jog_errors(rep, "joint3"))


def test_jog_finds_gear(arm):
    d, xml = arm
    rep = run_jog(d, xml, scale={"joint5": 0.5})  # driver expects a 2:1 gear the descriptor lacks
    errs = jog_errors(rep, "joint5")
    assert any("gear_ratio" in e and "2" in e for e in errs), errs


def test_jog_finds_swapped_joints(arm):
    d, xml = arm
    rep = run_jog(d, xml, swap=("joint1", "joint3"))
    assert any("joint3" in e and "swapped" in e for e in jog_errors(rep, "joint1"))


def test_jog_readback_mismatch(arm):
    d, xml = arm

    class HalfState(FakeRobot):  # driver reports state at half the command scale
        def read(self):
            return {k: v * 0.5 for k, v in super().read().items()}

    rep = jog_test(d, MuJoCoJogTarget(xml), HalfState(d, xml))
    assert any(f.check.startswith("readback.") for f in rep.findings if f.level == "error")


# -- bring-up and CLI ------------------------------------------------------------------------------
def test_bringup_levels(tmp_path):
    d = load(HUMANOID)
    rep = bringup.run(d, 2, tmp_path)
    assert rep.reached == 1
    (tmp_path / "jog.json").write_text(json.dumps({"passed": True}))
    (tmp_path / "actuator_acceptance.json").write_text(json.dumps({"passed": True}))
    assert bringup.run(d, 2, tmp_path).reached == 2
    (tmp_path / "actuator_acceptance.json").write_text(json.dumps({"passed": False}))
    rep = bringup.run(d, 3, tmp_path)
    assert rep.reached == 1 and "FAIL" in rep.markdown()


def test_bringup_l0_fails_on_bad_descriptor():
    r = raw()
    r["timing"]["policy_hz"] = 500
    assert bringup.run(from_dict(r), 1).reached == -1


def test_cli_flow(tmp_path, capsys):
    pytest.importorskip("mujoco")
    from groot_eval.robot_cli import main

    robot = tmp_path / "robot.yaml"
    assert main(["kit", "init", "--template", "arm_gripper", "--name", "my_arm", "--out", str(robot)]) == 0
    assert load(robot).name == "my_arm"
    assert main(["kit", "validate", str(robot)]) == 0
    assert main(["kit", "generate", str(robot), "--out", str(tmp_path / "gen")]) == 0
    assert main(["kit", "generate", str(robot), "--out", str(tmp_path / "gen"), "--check"]) == 0
    assert main(["kit", "roundtrip", str(robot)]) == 0
    ev = tmp_path / "ev"
    assert main(["kit", "jog", str(robot), "--out", str(ev / "jog.json")]) == 0
    (ev / "actuator_acceptance.json").write_text(json.dumps({"passed": True}))
    assert main(["kit", "bringup", str(robot), "--level", "2", "--evidence", str(ev),
                 "--out", str(tmp_path / "rep")]) == 0
    assert "reached **L2**" in Path(tmp_path / "rep" / "bringup.md").read_text()
