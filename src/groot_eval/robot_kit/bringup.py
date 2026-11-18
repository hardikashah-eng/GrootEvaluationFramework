"""Staged bring-up: L0 to L5, each a gate the robot must pass before the next.

=====  ==========================================  ==============================================
Level  Gate                                        Evidence
=====  ==========================================  ==============================================
L0     descriptor valid, adapters generate          run here: :func:`descriptor.validate`, generators
L1     data pipeline lossless                       run here: mapping round trip; plus
                                                    ``convert_check.json`` from ``groot-twin
                                                    convert-check`` once real logs exist
L2     twin matches the robot joint by joint        ``jog.json`` (``groot-robot kit jog``) and
                                                    ``actuator_acceptance.json`` (docs/twin-actuator-
                                                    and-replay.md)
L3     policy runs in the twin                      ``open_loop.json`` (``groot-eval open-loop``) and
                                                    ``sim_eval.json``
L4     hardware-in-the-loop timing passes           ``timing.json`` (``groot-robot timing-check``)
L5     supervised real runs                         ``real_runs.json``: first LiveChoreBench cell
=====  ==========================================  ==============================================

Evidence files are JSON with a top-level ``passed`` (bool). Levels above L1
only read evidence: the kit does not drive hardware on its own. A level
passes when its own gate passes and every level below it passes.
"""

from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from groot_eval.robot_kit import generate
from groot_eval.robot_kit.descriptor import Descriptor, validate
from groot_eval.robot_kit.validators import mapping_round_trip
from groot_eval.types import Finding

LEVELS: dict[int, tuple[str, tuple[str, ...]]] = {
    0: ("descriptor valid, adapters generate", ()),
    1: ("data pipeline lossless", ()),
    2: ("twin matches robot joint by joint", ("jog.json", "actuator_acceptance.json")),
    3: ("policy runs in the twin", ("open_loop.json", "sim_eval.json")),
    4: ("HIL timing passes", ("timing.json",)),
    5: ("supervised real runs", ("real_runs.json",)),
}
OPTIONAL = {1: ("convert_check.json",)}


@dataclass
class LevelResult:
    level: int
    title: str
    passed: bool
    findings: list[Finding] = field(default_factory=list)


@dataclass
class Bringup:
    robot: str
    target: int
    levels: list[LevelResult]

    @property
    def reached(self) -> int:
        """Highest level passed with every level below it passed; -1 if L0 fails."""
        r = -1
        for lv in self.levels:
            if not lv.passed:
                break
            r = lv.level
        return r

    def to_dict(self) -> dict[str, Any]:
        return {"robot": self.robot, "target": self.target, "reached": self.reached,
                "levels": [{"level": lv.level, "title": lv.title, "passed": lv.passed,
                            "findings": [f.to_dict() for f in lv.findings]} for lv in self.levels]}

    def markdown(self) -> str:
        lines = [f"# Bring-up: {self.robot}", "",
                 f"Target L{self.target}, reached **L{self.reached}**" if self.reached >= 0 else
                 f"Target L{self.target}: **L0 not passed**", "",
                 "| Level | Gate | Result |", "|---|---|---|"]
        for lv in self.levels:
            lines.append(f"| L{lv.level} | {lv.title} | {'pass' if lv.passed else 'FAIL'} |")
        for lv in self.levels:
            bad = [f for f in lv.findings if f.level != "ok"]
            if bad:
                lines += ["", f"## L{lv.level}", ""] + [f"- **{f.level}** `{f.check}`: {f.detail}" for f in bad]
        return "\n".join(lines) + "\n"


def _evidence(dir_: Path | None, name: str, required: bool) -> list[Finding]:
    if dir_ is None or not (dir_ / name).exists():
        return [Finding("error" if required else "warn", name, "not found" + ("" if required else
                        " (run it once real logs exist)"))]
    try:
        data = json.loads((dir_ / name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return [Finding("error", name, f"unreadable: {e}")]
    if data.get("passed") is True:
        return [Finding("ok", name, "passed")]
    return [Finding("error", name, "present but not passed")]


def run(desc: Descriptor, level: int, evidence_dir: str | Path | None = None) -> Bringup:
    """Evaluate every level up to ``level``. L0 and L1 run here; higher levels read evidence."""
    ev = Path(evidence_dir) if evidence_dir else None
    out: list[LevelResult] = []
    for lv in range(0, level + 1):
        title, needs = LEVELS[lv]
        fs: list[Finding] = []
        if lv == 0:
            fs = validate(desc)
            if not any(f.level == "error" for f in fs):
                with tempfile.TemporaryDirectory() as tmp:
                    try:
                        paths = generate.write_all(desc, tmp)
                        fs.append(Finding("ok", "generate", f"{len(paths)} files"))
                    except Exception as e:  # noqa: BLE001 - report any generator failure as the gate result
                        fs.append(Finding("error", "generate", f"{type(e).__name__}: {e}"))
        elif lv == 1:
            fs = mapping_round_trip(desc).findings
        for name in needs:
            fs += _evidence(ev, name, required=True)
        for name in OPTIONAL.get(lv, ()):
            fs += _evidence(ev, name, required=False)
        out.append(LevelResult(lv, title, not any(f.level == "error" for f in fs), fs))
    return Bringup(desc.name, level, out)
