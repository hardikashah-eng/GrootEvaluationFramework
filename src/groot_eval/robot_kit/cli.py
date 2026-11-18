"""``groot-robot kit``: init, validate, generate, roundtrip, jog, bringup."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from groot_eval.types import Finding


def _print(findings: list[Finding]) -> int:
    for f in findings:
        print(f"{f.level:5s} {f.check}: {f.detail}")
    return 1 if any(f.level == "error" for f in findings) else 0


def cmd_init(args: argparse.Namespace) -> int:
    from groot_eval.robot_kit import TEMPLATES, TEMPLATES_DIR

    if args.template not in TEMPLATES:
        print(f"unknown template {args.template!r}; choose from {TEMPLATES}", file=sys.stderr)
        return 2
    out = Path(args.out)
    if out.exists() and not args.force:
        print(f"{out} exists (use --force to overwrite)", file=sys.stderr)
        return 2
    text = (TEMPLATES_DIR / f"{args.template}.yaml").read_text(encoding="utf-8")
    if args.name:
        text = re.sub(r"^name: .*$", f"name: {args.name}", text, count=1, flags=re.M)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} from the {args.template} template; replace the placeholder numbers, then "
          f"`groot-robot kit validate {out}`")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from groot_eval.robot_kit import load, validate

    return _print(validate(load(args.descriptor)))


def cmd_generate(args: argparse.Namespace) -> int:
    from groot_eval.robot_kit import generate, load, validate

    desc = load(args.descriptor)
    if args.check:
        stale = generate.is_stale(desc, args.out)
        print(f"{args.out}: {'STALE, regenerate' if stale else 'up to date'}")
        return 1 if stale else 0
    findings = validate(desc)
    if any(f.level == "error" for f in findings):
        return _print(findings)
    for p in generate.write_all(desc, args.out):
        print(f"wrote {p}")
    return 0


def cmd_roundtrip(args: argparse.Namespace) -> int:
    from groot_eval.robot_kit import load
    from groot_eval.robot_kit.validators import mapping_round_trip

    res = mapping_round_trip(load(args.descriptor), samples=args.samples, seed=args.seed)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
    return _print(res.findings)


def _factory(spec: str) -> Any:
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def cmd_jog(args: argparse.Namespace) -> int:
    from groot_eval.robot_kit import generate, load
    from groot_eval.robot_kit.validators import MuJoCoJogTarget, add_tip_sites, jog_test

    desc = load(args.descriptor)
    if args.twin == "stub":
        xml = add_tip_sites(generate.mujoco_stub_xml(desc), desc)
    else:
        xml = Path(args.twin).read_text(encoding="utf-8")
    twin = MuJoCoJogTarget(xml, sites=args.sites.split(",") if args.sites else None)
    robot = _factory(args.robot)(desc) if args.robot else MuJoCoJogTarget(xml, sites=twin.sites)
    if not args.robot:
        print("no --robot given: jogging the twin against itself (checks the twin and the kit, not hardware)")
    deltas = [float(x) for x in args.deltas.split(",")]
    rep = jog_test(desc, twin, robot, deltas=deltas)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
    return _print(rep.findings)


def cmd_bringup(args: argparse.Namespace) -> int:
    from groot_eval.robot_kit import bringup, load

    rep = bringup.run(load(args.descriptor), args.level, args.evidence)
    md = rep.markdown()
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "bringup.md").write_text(md, encoding="utf-8")
        (out / "bringup.json").write_text(json.dumps(rep.to_dict(), indent=2), encoding="utf-8")
    print(md)
    return 0 if rep.reached >= args.level else 1


def register_kit(sub: Any) -> None:
    kit = sub.add_parser("kit", help="robot adaptation kit: one robot.yaml -> GR00T, sim and robot adapters, "
                                     "validators and staged bring-up")
    ks = kit.add_subparsers(dest="kit_command", required=True)

    sp = ks.add_parser("init", help="start a robot.yaml from a template")
    sp.add_argument("--template", required=True, help="humanoid | arm_gripper | bimanual | mobile_manipulator")
    sp.add_argument("--name")
    sp.add_argument("--out", default="robot.yaml")
    sp.add_argument("--force", action="store_true")
    sp.set_defaults(func=cmd_init)

    sp = ks.add_parser("validate", help="check the descriptor (bring-up L0)")
    sp.add_argument("descriptor")
    sp.set_defaults(func=cmd_validate)

    sp = ks.add_parser("generate", help="write modality, GR00T, Isaac Lab, MuJoCo, ROS 2, logging and profile files")
    sp.add_argument("descriptor")
    sp.add_argument("--out", required=True)
    sp.add_argument("--check", action="store_true", help="only report whether --out is stale")
    sp.set_defaults(func=cmd_generate)

    sp = ks.add_parser("roundtrip", help="robot readings -> GR00T obs -> echo policy -> robot commands")
    sp.add_argument("descriptor")
    sp.add_argument("--samples", type=int, default=20)
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--out", help="write the result as JSON")
    sp.set_defaults(func=cmd_roundtrip)

    sp = ks.add_parser("jog", help="jog each joint on the twin and the robot; name sign, unit, order and gear faults")
    sp.add_argument("descriptor")
    sp.add_argument("--twin", default="stub", help="MJCF path, or 'stub' for the generated joint-chain model")
    sp.add_argument("--sites", help="comma-separated MuJoCo sites to track (default: all)")
    sp.add_argument("--robot", help="module:factory returning a JogTarget for the hardware, given the descriptor")
    sp.add_argument("--deltas", default="0.01,0.1", help="jog steps in rad, run smallest first")
    sp.add_argument("--out", help="write the result as JSON (bring-up L2 evidence: jog.json)")
    sp.set_defaults(func=cmd_jog)

    sp = ks.add_parser("bringup", help="evaluate bring-up levels L0..LN and write a report")
    sp.add_argument("descriptor")
    sp.add_argument("--level", type=int, default=1, choices=range(6))
    sp.add_argument("--evidence", help="folder with jog.json, actuator_acceptance.json, open_loop.json, ...")
    sp.add_argument("--out", help="folder for bringup.md and bringup.json")
    sp.set_defaults(func=cmd_bringup)
