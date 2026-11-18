"""Policy API adapter generated from the descriptor: robot readings in, robot commands out.

:meth:`PolicyAdapter.pack_obs` turns what the driver reports (joint readings
by name in robot units, camera frames in the camera's colour order) into the
observation dict a GR00T policy server takes. :meth:`PolicyAdapter.unpack_action`
turns the action chunk the policy returns back into robot commands, one per
control step, in robot units, clipped to the joint limits and the action-rate
limit.

Every conversion lives in one place, so the mapping round trip in
:mod:`groot_eval.robot_kit.validators` can check it end to end.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from groot_eval.robot_kit.descriptor import Descriptor, Group


@dataclass
class Unpacked:
    """One action chunk in robot space."""

    commands: list[dict[str, float]]  # one {joint name: robot value} per control step
    clipped: dict[str, int] = field(default_factory=dict)  # joint -> steps clipped at a limit
    rate_limited: int = 0  # steps where the action-rate limit cut the change


class PolicyAdapter:
    def __init__(self, desc: Descriptor):
        self.desc = desc
        self.slices = desc.slices()

    # -- joints ----------------------------------------------------------------------------------
    def _to_model(self, g: Group, robot: Mapping[str, float]) -> np.ndarray:
        missing = [j.name for j in g.joints if j.name not in robot]
        if missing:
            raise KeyError(f"group {g.name!r}: no reading for {missing}")
        if g.kind == "gripper":
            assert g.robot_range is not None
            o, c = g.robot_range
            return np.array([(float(robot[j.name]) - o) / (c - o) for j in g.joints])
        return np.array([float(robot[j.name]) / j.scale for j in g.joints])

    def _to_robot(self, g: Group, model: np.ndarray) -> dict[str, float]:
        if g.kind == "gripper":
            assert g.robot_range is not None
            o, c = g.robot_range
            return {j.name: float(o + v * (c - o)) for j, v in zip(g.joints, model)}
        return {j.name: float(v * j.scale) for j, v in zip(g.joints, model)}

    def model_state(self, robot: Mapping[str, float]) -> np.ndarray:
        """Concatenated model-space state, in descriptor order."""
        return np.concatenate([self._to_model(g, robot) for g in self.desc.groups])

    # -- observation -----------------------------------------------------------------------------
    def pack_obs(self, robot: Mapping[str, float], images: Mapping[str, np.ndarray],
                 task: str = "") -> dict[str, Any]:
        """Build a GR00T observation (batch of one): ``video.<cam>`` (1, H, W, 3) uint8 RGB,
        ``state.<group>`` (1, d) float32 in radians, and the language key."""
        obs: dict[str, Any] = {}
        for c in self.desc.cameras:
            if c.name not in images:
                raise KeyError(f"no frame for camera {c.name!r}")
            img = np.asarray(images[c.name])
            if img.shape[:2] != (c.height, c.width):
                raise ValueError(f"camera {c.name!r}: frame {img.shape[:2]}, descriptor says {(c.height, c.width)}")
            if img.dtype != np.uint8:
                if np.issubdtype(img.dtype, np.floating) and float(np.nanmax(img)) <= 1.0:
                    img = np.round(img * 255.0)
                img = np.clip(img, 0, 255).astype(np.uint8)
            if c.color_order == "bgr":
                img = img[..., ::-1]
            obs[f"video.{c.name}"] = np.ascontiguousarray(img)[None]
        for g in self.desc.groups:
            obs[f"state.{g.name}"] = self._to_model(g, robot).astype(np.float32)[None]
        obs[self.desc.language_key] = [task]
        return obs

    # -- action ----------------------------------------------------------------------------------
    def unpack_action(self, action: Mapping[str, np.ndarray], robot_now: Mapping[str, float],
                      clip: bool = True) -> Unpacked:
        """Turn ``action.<group>`` arrays of shape (T, d) or (1, T, d) into T robot commands.

        Relative groups are added to the state at the start of the chunk. With ``clip``
        each joint is held inside its limits and each step's change inside
        ``safety.max_action_rate``.
        """
        now = self.model_state(robot_now)
        chunks: dict[str, np.ndarray] = {}
        T = None
        for g in self.desc.groups:
            key = f"action.{g.name}"
            if key not in action:
                raise KeyError(f"policy returned no {key!r}")
            a = np.asarray(action[key], dtype=float)
            if a.ndim == 3:
                a = a[0]
            if a.ndim == 1:
                a = a[:, None] if g.dim == 1 else a[None]
            if a.shape[1] != g.dim:
                raise ValueError(f"{key}: width {a.shape[1]}, group has {g.dim} joints")
            if T is None:
                T = a.shape[0]
            elif a.shape[0] != T:
                raise ValueError(f"{key}: {a.shape[0]} steps, other groups have {T}")
            if g.action == "relative":
                a = a + now[self.slices[g.name]]
            chunks[g.name] = a
        assert T is not None
        model = np.concatenate([chunks[g.name] for g in self.desc.groups], axis=1)  # (T, D)
        out = Unpacked(commands=[])
        if clip:
            lo = np.array([j.limits[0] for j in self.desc.joints])
            hi = np.array([j.limits[1] for j in self.desc.joints])
            for g in self.desc.groups:  # gripper model space is [0, 1]
                if g.kind == "gripper":
                    lo[self.slices[g.name]], hi[self.slices[g.name]] = 0.0, 1.0
            over = (model < lo) | (model > hi)
            for k, n in zip(self.desc.joint_names, over.sum(axis=0)):
                if n:
                    out.clipped[k] = int(n)
            model = np.clip(model, lo, hi)
            rate = self.desc.safety.max_action_rate
            if rate:
                prev = now
                for t in range(T):
                    step = model[t] - prev
                    if np.any(np.abs(step) > rate):
                        out.rate_limited += 1
                        model[t] = prev + np.clip(step, -rate, rate)
                    prev = model[t]
        for t in range(T):
            cmd: dict[str, float] = {}
            for g in self.desc.groups:
                cmd.update(self._to_robot(g, model[t, self.slices[g.name]]))
            out.commands.append(cmd)
        return out
