"""Drive-chain stages around the wear-aware actuator core: nominal delay, dropped packets, current loop, linkage limit.

:class:`~groot_eval.actuators.core.WearAwareActuatorCore` models the motor, gearbox wear, heat, faults and the
encoder. A real drive also has things that are always there, not only when a fault fires:

1. **Transport delay**: every command reaches the joint controller some steps late (bus, DDS, scheduling).
   Each environment draws its own delay from ``[delay_steps_min, delay_steps_max]``; ``jitter_steps`` adds a
   random extra delay per step. ``drop_prob`` loses a packet, and the drive holds the last command it got.
2. **Current loop**: torque follows the controller's request with a first-order lag ``current_tau_s``.
3. **Linkage limit**: joints driven through a linkage (humanoid ankles, some knees) can deliver less torque at
   some angles. ``linkage`` gives, per joint index, a table of joint angle (rad) and peak torque (N·m);
   torque is clipped to the interpolated value.

The core still applies the joint's own PD gains, wear, friction, torque-speed curve and faults, in that order:

``command -> transport delay / drops -> core (PD, wear, limits, faults) -> current loop -> linkage limit -> joint``

:class:`DriveChain` has the same ``compute`` / ``observe`` / ``reset`` / ``telemetry`` methods as the core, so it
drops in wherever the core is used (the Isaac Lab adapter wraps the core in it when ``drive_chain`` is set).
Joint compliance (series-elastic or harmonic-drive flex) is not modelled here; it needs a motor-side state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from groot_eval.actuators.core import WearAwareActuatorCore, _to_numpy
from groot_eval.actuators.motor_models import _like


@dataclass
class DriveChainParams:
    """Always-on drive effects. Delays are in control steps; ``dt`` is the control period in seconds."""

    delay_steps_min: int = 0
    delay_steps_max: int = 0
    jitter_steps: int = 0  # extra 0..jitter_steps steps, drawn every step per env
    drop_prob: float = 0.0  # per env, per step: the packet is lost and the drive holds the last command
    current_tau_s: Any = 0.0  # current-loop time constant, s (scalar or per env, shape (E, 1)); 0 = instant
    linkage: dict[int, list[tuple[float, float]]] = field(default_factory=dict)  # joint -> [(angle rad, peak N·m)]
    dt: float = 0.005
    seed: int = 0

    def __post_init__(self) -> None:
        if self.delay_steps_min < 0 or self.delay_steps_max < self.delay_steps_min:
            raise ValueError("need 0 <= delay_steps_min <= delay_steps_max")
        if self.jitter_steps < 0 or not 0.0 <= self.drop_prob < 1.0 or np.any(np.asarray(self.current_tau_s) < 0):
            raise ValueError("jitter_steps >= 0, 0 <= drop_prob < 1 and current_tau_s >= 0 are required")
        for j, table in self.linkage.items():
            ang = [a for a, _ in table]
            if len(table) < 2 or any(b <= a for a, b in zip(ang, ang[1:])):
                raise ValueError(f"linkage table for joint {j} needs at least two points with increasing angle")
            if any(t <= 0 for _, t in table):
                raise ValueError(f"linkage table for joint {j} has a non-positive torque")

    @property
    def max_lag(self) -> int:
        return self.delay_steps_max + self.jitter_steps

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "DriveChainParams":
        d = dict(d or {})
        link = {int(k): [(float(a), float(t)) for a, t in v] for k, v in (d.pop("linkage", None) or {}).items()}
        return cls(linkage=link, **d)


class DriveChain:
    """Transport delay, drops, current loop and linkage limit around a :class:`WearAwareActuatorCore`."""

    def __init__(self, core: WearAwareActuatorCore, params: DriveChainParams) -> None:
        self.core, self.p = core, params
        self.num_envs, self.num_joints = core.num_envs, core.num_joints
        bad = [j for j in params.linkage if not 0 <= j < self.num_joints]
        if bad:
            raise ValueError(f"linkage joints {bad} are outside 0..{self.num_joints - 1}")
        self._rng = np.random.default_rng(params.seed)
        self.delay: np.ndarray = self._rng.integers(params.delay_steps_min, params.delay_steps_max + 1, self.num_envs)
        self.reset()

    # -- state ----------------------------------------------------------------------
    def reset(self, env_ids: Any = None) -> None:
        """Clear queued commands and the current loop (all envs, or ``env_ids``); the per-env delay stays."""
        e, j = self.num_envs, self.num_joints
        if env_ids is None or not hasattr(self, "_hist"):
            self._hist = np.zeros((self.p.max_lag + 1, 3, e, j))
            self._filled: np.ndarray = np.zeros(e, np.int64)
            self._held = np.zeros((3, e, j))
            self._have_held = np.zeros(e, bool)
            self._tau = np.zeros((e, j))
            self.dropped = np.zeros(e, np.int64)
            self.linkage_clipped = np.zeros((e, j), np.int64)
            self.last_lag: np.ndarray = np.zeros(e, np.int64)
            self.core.reset()
            return
        ids = np.asarray(env_ids).reshape(-1)
        self._hist[:, :, ids] = 0.0
        self._filled[ids] = 0
        self._have_held[ids] = False
        self._tau[ids] = 0.0
        self.dropped[ids] = 0
        self.linkage_clipped[ids] = 0
        self.core.reset(ids)

    # -- actuation ------------------------------------------------------------------
    def _delivered(self, cmd: np.ndarray) -> np.ndarray:
        """The command each env's drive sees this step, after delay, jitter and drops. ``cmd`` is (3, E, J)."""
        self._hist = np.roll(self._hist, 1, axis=0)
        self._hist[0] = cmd
        self._filled = np.minimum(self._filled + 1, self._hist.shape[0])
        lag = self.delay.copy()
        if self.p.jitter_steps:
            lag = lag + self._rng.integers(0, self.p.jitter_steps + 1, self.num_envs)
        lag = np.minimum(lag, self._filled - 1)  # not enough history yet: the oldest command we have
        self.last_lag = lag
        out = self._hist[lag, :, np.arange(self.num_envs)]  # (E, 3, J)
        out = np.transpose(out, (1, 0, 2))
        if self.p.drop_prob:
            drop = (self._rng.random(self.num_envs) < self.p.drop_prob) & self._have_held
            self.dropped += drop
            out = np.where(drop[None, :, None], self._held, out)
        self._held = out
        self._have_held[:] = True
        return out

    def _linkage_limit(self, q: np.ndarray) -> np.ndarray:
        lim = np.full((self.num_envs, self.num_joints), np.inf)
        for j, table in self.p.linkage.items():
            ang = np.array([a for a, _ in table])
            peak = np.array([t for _, t in table])
            lim[:, j] = np.interp(q[:, j], ang, peak)
        return lim

    def compute(self, q_target: Any, qd_target: Any, tau_ff: Any, q: Any, qd: Any) -> Any:
        """Applied effort ``(E, J)`` for this control step, in the input's array namespace."""
        qn = _to_numpy(q)
        zeros = np.zeros_like(qn)
        cmd = np.stack([_to_numpy(q_target),
                        zeros if qd_target is None else _to_numpy(qd_target),
                        zeros if tau_ff is None else _to_numpy(tau_ff)])
        got = self._delivered(cmd)
        tau = _to_numpy(self.core.compute(_like(got[0], q), _like(got[1], q), _like(got[2], q), q, qd))
        tau_c = np.asarray(self.p.current_tau_s, dtype=float)
        if np.any(tau_c > 0):
            a = np.minimum(1.0, self.p.dt / (tau_c + self.p.dt))
            self._tau = self._tau + a * (tau - self._tau)
        else:
            self._tau = tau
        out = self._tau
        if self.p.linkage:
            lim = self._linkage_limit(qn)
            clipped = np.abs(out) > lim
            self.linkage_clipped += clipped
            out = np.clip(out, -lim, lim)
        return _like(out, q)

    def observe(self, q: Any) -> Any:
        """Encoder reading, from the core (wear and sensor faults)."""
        return self.core.observe(q)

    def telemetry(self) -> dict[str, np.ndarray]:
        """Core telemetry plus this step's command lag, dropped packets and linkage-limit hits per env."""
        return {**self.core.telemetry(), "lag_steps": self.last_lag.copy(), "dropped": self.dropped.copy(),
                "linkage_clipped": self.linkage_clipped.copy()}
