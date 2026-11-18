"""Robot adaptation kit: bring a new robot into GR00T from one descriptor.

``robot.yaml`` (:mod:`.descriptor`) -> generated adapters (:mod:`.generate`, :mod:`.adapter`)
-> validators (:mod:`.validators`) -> staged bring-up L0-L5 (:mod:`.bringup`).
CLI: ``groot-robot kit init | validate | generate | roundtrip | jog | bringup``. See docs/robot-adaptation-kit.md.
"""

from pathlib import Path

from groot_eval.robot_kit.adapter import PolicyAdapter, Unpacked
from groot_eval.robot_kit.descriptor import Camera, Descriptor, Group, Joint, Safety, Timing, from_dict, load, validate

TEMPLATES_DIR = Path(__file__).parent / "templates"
TEMPLATES = sorted(p.stem for p in TEMPLATES_DIR.glob("*.yaml"))

__all__ = ["Camera", "Descriptor", "Group", "Joint", "PolicyAdapter", "Safety", "TEMPLATES", "TEMPLATES_DIR",
           "Timing", "Unpacked", "from_dict", "load", "validate"]
