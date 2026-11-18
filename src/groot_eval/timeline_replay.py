"""Replay a real episode on its own recorded clock, from the recorded state just before the failure.

:mod:`groot_eval.replay` plays recorded actions one per step from the episode's first state. That assumes the
robot sent exactly one command per control period. Real logs don't: commands arrive late, twice in one period
or not at all, and the failure you care about is usually seconds into the episode. This module replays what
the robot actually experienced:

1. **Check the log first** (:func:`check_replay_log`). Every level (policy, action, joint) present, timestamps
   increasing, the joint stream covers the actions, the streams' clocks agree to within ``max_skew_s``, and
   the version stamp (checkpoint, deployed build, firmware, gains, robot profile, twin assets) is complete.
   A log that fails is refused rather than replayed on guesses.
2. **Schedule commands by timestamp** (:func:`build_schedule`). Each logged command takes effect at the sim
   step its timestamp falls in; between commands the last one is held. Late commands, duplicates and missing
   ticks are therefore replayed as they happened, and counted.
3. **Start just before the failure** (:func:`restore_index`): the recorded state ``lead_s`` seconds before
   it, plus the commands sent just before (so an env with actuation delay can refill its queue), plus
   optional settling steps for engines that cannot restore contact state (PhysX).
4. **Three modes on one timeline** (:func:`replay_on_log_clock`): ``action`` (physics on, the recorded commands),
   ``kinematic`` (each recorded state set in turn; checks the twin can hold the real state and renders along
   the real path) and ``observation`` (the policy is fed the logged observations; checks it acts the same).
5. **Replay as a distribution** (:func:`replay_distribution`): N replays with the uncertain parameters drawn
   from their measured ranges. The result is a reproduction rate with a Wilson interval and how often the
   failure mode matched, not a single yes or no.

Envs follow :class:`groot_eval.envs.base.StatefulEnv`; states are flat vectors split into ``state.*`` keys
by ``slices``, as in :mod:`groot_eval.replay`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from groot_eval.stats import wilson_interval
from groot_eval.types import Scenario

LEVELS = ("policy", "action", "joint")
REQUIRED_STAMPS = ("checkpoint", "deployed_build", "firmware", "gains", "robot_profile", "twin_assets")


# =====================================================================================
# 1. Log checks
# =====================================================================================
@dataclass
class LogFinding:
    level: str  # error | warn
    check: str
    detail: str


@dataclass
class LogCheck:
    findings: list[LogFinding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not any(f.level == "error" for f in self.findings)

    def errors(self) -> list[str]:
        return [f"{f.check}: {f.detail}" for f in self.findings if f.level == "error"]


def check_replay_log(log: Mapping[str, Any], *, max_skew_s: float, required_stamps: Sequence[str] = REQUIRED_STAMPS,
                     levels: Sequence[str] = LEVELS) -> LogCheck:
    """Refuse logs a replay cannot trust. ``log`` = ``{"streams": {level: {"t": [...], ...}}, "clock_offsets_s":
    {stream: offset against the reference clock}, "versions": {stamp: value}}``.

    ``clock_offsets_s`` comes from the robot's clock sync (PTP offsets, or a sync pulse seen by every stream).
    Without it the skew between streams is unknown, which is an error: a replay with camera frames a few
    milliseconds off the joint states can fail for reasons the robot never had.
    """
    out = LogCheck()
    err = lambda c, d: out.findings.append(LogFinding("error", c, d))  # noqa: E731
    streams = log.get("streams") or {}
    for lv in levels:
        s = streams.get(lv)
        if not s or len(s.get("t", [])) < 2:
            err(f"{lv}.present", f"no {lv}-level stream (or fewer than 2 records)")
            continue
        t = np.asarray(s["t"], float)
        if np.any(np.diff(t) <= 0):
            err(f"{lv}.time", "timestamps are not strictly increasing")
    act, jnt = streams.get("action") or {}, streams.get("joint") or {}
    if len(act.get("t", [])) and len(jnt.get("t", [])):
        if jnt["t"][0] > act["t"][0] or jnt["t"][-1] < act["t"][-1]:
            err("cover.joint", "joint stream does not cover the whole action stream")
    offsets = log.get("clock_offsets_s")
    if not offsets:
        err("clock.sync", "no clock offsets: the skew between streams is unknown")
    else:
        missing = [k for k in streams if k not in offsets]
        if missing:
            err("clock.sync", f"no clock offset for: {', '.join(sorted(missing))}")
        vals = np.array([float(v) for v in offsets.values()])
        skew = float(vals.max() - vals.min()) if vals.size else 0.0
        if skew > max_skew_s:
            err("clock.skew", f"streams disagree by {skew * 1000:.2f} ms (limit {max_skew_s * 1000:.2f} ms)")
    versions = log.get("versions") or {}
    absent = [k for k in required_stamps if not versions.get(k)]
    if absent:
        err("versions", f"missing version stamp: {', '.join(absent)}")
    return out


# =====================================================================================
# 2. Commands scheduled by timestamp
# =====================================================================================
@dataclass
class Schedule:
    dt: float
    t0: float
    steps: np.ndarray  # (N,) sim step at which each logged command takes effect
    actions: np.ndarray  # (N, A)
    n_steps: int
    late: int = 0  # gaps longer than 1.5 nominal periods
    duplicates: int = 0  # two commands landing in one sim step (the later wins)
    missing_ticks: int = 0  # control periods with no command (the last one is held)

    def action_at(self, step: int) -> np.ndarray | None:
        """The command in force at ``step``: the last one that took effect at or before it (None before the first)."""
        i = int(np.searchsorted(self.steps, step, side="right")) - 1
        return None if i < 0 else self.actions[i]

    def stats(self) -> dict[str, int]:
        return {"commands": int(len(self.steps)), "late": self.late, "duplicates": self.duplicates,
                "missing_ticks": self.missing_ticks, "steps": self.n_steps}


def build_schedule(action_t: Sequence[float] | np.ndarray, actions: np.ndarray, dt: float, t0: float | None = None,
                   nominal_period: float | None = None, t_end: float | None = None) -> Schedule:
    """Place each command at the sim step its timestamp falls in (``floor((t - t0) / dt)``), keeping jitter.

    ``nominal_period`` is the control period the robot aimed for (default: the median command gap); it is used
    only to count late commands and missing ticks.
    """
    t = np.asarray(action_t, float)
    a = np.asarray(actions, float)
    if len(t) != len(a) or len(t) == 0:
        raise ValueError("action_t and actions must be non-empty and the same length")
    if np.any(np.diff(t) < 0):
        raise ValueError("action timestamps must not go backwards")
    start = float(t[0] if t0 is None else t0)
    steps = np.floor((t - start) / dt + 1e-9).astype(int)
    if np.any(steps < 0):
        raise ValueError("a command is timestamped before t0")
    dup = int(np.sum(np.diff(steps) == 0))
    gaps = np.diff(t)
    period = float(nominal_period if nominal_period else (np.median(gaps) if gaps.size else dt))
    late = int(np.sum(gaps > 1.5 * period))
    missing = int(np.sum(np.maximum(np.round(gaps / period) - 1, 0)))
    end = float(t[-1] if t_end is None else t_end)
    n = int(math.floor((end - start) / dt + 1e-9)) + 1
    # keep the last command of each step (a later duplicate overwrites the earlier one)
    keep = np.r_[steps[1:] != steps[:-1], True]
    return Schedule(dt=dt, t0=start, steps=steps[keep], actions=a[keep], n_steps=n, late=late, duplicates=dup,
                    missing_ticks=missing)


def restore_index(state_t: Sequence[float] | np.ndarray, t_fail: float, lead_s: float) -> int:
    """Index of the last recorded state at or before ``t_fail - lead_s`` (0 if the failure is that early)."""
    t = np.asarray(state_t, float)
    return max(0, int(np.searchsorted(t, t_fail - lead_s, side="right")) - 1)


# =====================================================================================
# 3. Replay on the log clock
# =====================================================================================
def _state_dict(vec: np.ndarray, slices: Mapping[str, slice]) -> dict[str, np.ndarray]:
    return {f"state.{k}": np.asarray(vec[s], dtype=np.float64) for k, s in slices.items()}


def _flatten(obs: Mapping[str, Any], slices: Mapping[str, slice]) -> np.ndarray:
    parts = []
    for k, s in slices.items():
        key = f"state.{k}"
        if key not in obs:
            raise ValueError(f"env observation has no {key!r}, which the log records")
        v = np.asarray(obs[key], dtype=np.float64).reshape(-1)
        if v.size != s.stop - s.start:
            raise ValueError(f"{key}: env gives {v.size} values, log has {s.stop - s.start}")
        parts.append(v)
    return np.concatenate(parts)


def _interp_states(state_t: np.ndarray, states: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(t, state_t, states[:, d], left=np.nan, right=np.nan) for d in range(states.shape[1])],
                    axis=1)


@dataclass
class TimelineResult:
    mode: str
    times: np.ndarray  # (K,) timestamps (log clock) of the compared samples
    predicted: np.ndarray  # (K, S) twin states (action, kinematic) or policy actions (observation)
    recorded: np.ndarray  # (K, S) recorded states, or logged actions
    nrmse_by_signal: dict[str, float]
    divergence_t: float | None  # first time any signal's error exceeds threshold x its spread
    ended_as: str | None = None  # the env's episode outcome, if it ended (success or failure_reason)
    ended_at_t: float | None = None
    synced: bool = True
    schedule: dict[str, int] = field(default_factory=dict)
    frames: list[dict[str, Any]] = field(default_factory=list)  # kinematic mode: video.* obs along the real path

    @property
    def nrmse(self) -> float:
        v = [x for x in self.nrmse_by_signal.values() if math.isfinite(x)]
        return float(np.mean(v)) if v else math.nan

    def to_dict(self) -> dict[str, Any]:
        return {"mode": self.mode, "nrmse": self.nrmse, "nrmse_by_signal": self.nrmse_by_signal,
                "divergence_t": self.divergence_t, "ended_as": self.ended_as, "ended_at_t": self.ended_at_t,
                "synced": self.synced, "schedule": self.schedule, "samples": int(len(self.times))}


def _score(pred: np.ndarray, rec: np.ndarray, times: np.ndarray, slices: Mapping[str, slice] | None,
           spread: np.ndarray, threshold: float) -> tuple[dict[str, float], float | None]:
    live = spread > 1e-9
    with np.errstate(invalid="ignore", divide="ignore"):
        per_dim = np.sqrt(np.nanmean((pred - rec) ** 2, axis=0)) / np.where(live, spread, np.nan)
        norm = np.abs(pred[:, live] - rec[:, live]) / spread[live]
    groups = slices or {"all": slice(0, pred.shape[1])}
    by_sig = {}
    for k, s in groups.items():
        v = per_dim[s][np.isfinite(per_dim[s])]
        by_sig[k] = float(np.sqrt(np.mean(v ** 2))) if v.size else math.nan
    div = None
    if live.any():
        bad = np.where(np.nan_to_num(norm, nan=0.0).max(axis=1) > threshold)[0]
        div = float(times[bad[0]]) if bad.size else None
    return by_sig, div


def replay_on_log_clock(env: Any, *, states: np.ndarray, state_t: Sequence[float], slices: Mapping[str, slice],
                        actions: np.ndarray, action_t: Sequence[float], dt: float, mode: str = "action",
                        start_index: int = 0, settle_steps: int = 0, scenario: Scenario | None = None,
                        policy: Callable[[dict[str, Any]], np.ndarray] | None = None,
                        observations: Sequence[dict[str, Any]] | None = None,
                        obs_t: Sequence[float] | None = None, threshold: float = 0.5) -> TimelineResult:
    """Replay one recorded episode from ``states[start_index]`` on the log's clock.

    ``mode``:

    * ``"action"``: physics on; the recorded commands are applied at their logged times (held between commands).
      The twin's states are compared with the recorded ones at the log's timestamps.
    * ``"kinematic"``: physics off in effect; each recorded state from ``start_index`` on is set with
      ``env.set_state`` and read back. A read-back error means the twin cannot even hold the real state (units,
      joint order, limits, collisions). Any ``video.*`` keys in the returned obs are kept as ``frames``.
    * ``"observation"``: ``policy`` is called on each logged observation (``observations``, at ``obs_t``) and
      its action compared with the logged command in force at that time.
    """
    st = np.asarray(state_t, float)
    X = np.asarray(states, float)
    spread = np.nanstd(X, axis=0)
    sc = scenario or Scenario(seed=0)
    if mode == "observation":
        if policy is None or observations is None or obs_t is None:
            raise ValueError("observation mode needs policy, observations and obs_t")
        sched = build_schedule(action_t, actions, dt)
        ot = np.asarray(obs_t, float)
        pred, rec, times = [], [], []
        for o, t in zip(observations, ot):
            logged = sched.action_at(int(math.floor((t - sched.t0) / dt + 1e-9)))
            if logged is None:
                continue
            pred.append(np.asarray(policy(dict(o)), float).reshape(-1))
            rec.append(logged)
            times.append(t)
        P, R = np.array(pred), np.array(rec)
        a_spread = np.nanstd(np.asarray(actions, float), axis=0)
        by, div = _score(P, R, np.array(times), None, a_spread, threshold)
        return TimelineResult("observation", np.array(times), P, R, by, div, schedule=sched.stats())

    obs = env.reset(sc)
    synced = True
    t_start = float(st[start_index])
    if mode == "kinematic":
        kin: list[np.ndarray] = []
        frames: list[dict[str, Any]] = []
        for i in range(start_index, len(X)):
            o = env.set_state(_state_dict(X[i], slices), None)
            kin.append(_flatten(o, slices))
            vid = {k: v for k, v in o.items() if str(k).startswith("video.")}
            if vid:
                frames.append({"t": float(st[i]), **vid})
        P = np.array(kin)
        kt = st[start_index:]
        by, div = _score(P, X[start_index:], kt, slices, spread, threshold)
        return TimelineResult("kinematic", kt, P, X[start_index:], by, div, frames=frames)

    if mode != "action":
        raise ValueError(f"unknown mode {mode!r} (action | kinematic | observation)")
    at = np.asarray(action_t, float)
    A = np.asarray(actions, float)
    after = at >= t_start
    if not after.any():
        raise ValueError("no recorded command after the restore point")
    sched = build_schedule(at[after].tolist(), A[after], dt, t0=t_start, t_end=float(st[-1]))
    before = A[at < t_start]
    if hasattr(env, "max_steps") and env.max_steps < sched.n_steps + settle_steps + 1:
        env.max_steps = sched.n_steps + settle_steps + 1
    try:
        obs = env.set_state(_state_dict(X[start_index], slices), before[-8:] if len(before) else None)
    except (AttributeError, NotImplementedError):
        synced = False
    held = before[-1] if len(before) else A[0]
    for _ in range(settle_steps):  # let contacts settle (PhysX cannot restore its contact state), then re-set
        obs, *_ = env.step(held)
    if settle_steps and synced:
        obs = env.set_state(_state_dict(X[start_index], slices), before[-8:] if len(before) else None)
    pred = [_flatten(obs, slices)]
    times = [t_start]
    ended_as, ended_t = None, None
    for k in range(sched.n_steps - 1):
        cmd = sched.action_at(k)
        obs, _, term, trunc, info = env.step(held if cmd is None else cmd)
        times.append(t_start + (k + 1) * dt)
        pred.append(_flatten(obs, slices))
        if term or trunc:
            ended_as = "success" if info.get("success") else str(info.get("failure_reason", "unknown"))
            ended_t = times[-1]
            break
    # compare at the logged timestamps (the dense sim trajectory is interpolated, not the sparse log)
    dense_t, dense = np.array(times), np.array(pred)
    keep = (st >= dense_t[0] - 1e-12) & (st <= dense_t[-1] + 1e-12)
    tt = st[keep]
    P = _interp_states(dense_t, dense, tt)
    R = X[keep]
    by, div = _score(P, R, tt, slices, spread, threshold)
    return TimelineResult("action", tt, P, R, by, div, ended_as=ended_as, ended_at_t=ended_t, synced=synced,
                          schedule=sched.stats())


# =====================================================================================
# 4. Replay as a distribution
# =====================================================================================
@dataclass
class DistributionResult:
    failure_mode: str
    runs: int
    reproduced: int
    rate: float
    ci: tuple[float, float]
    mode_counts: dict[str, int]
    param_effect: dict[str, float]  # mean value among reproducing draws minus among the others (normalised)
    draws: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"failure_mode": self.failure_mode, "runs": self.runs, "reproduced": self.reproduced,
                "rate": self.rate, "ci": list(self.ci), "mode_counts": self.mode_counts,
                "param_effect": self.param_effect}


def replay_distribution(make_env: Callable[[], Any], *, failure_mode: str, uncertain: Mapping[str, tuple[float, float]],
                        runs: int = 50, seed: int = 0, base_params: Mapping[str, Any] | None = None,
                        detect: Callable[[TimelineResult], str | None] | None = None,
                        **replay_kwargs: Any) -> DistributionResult:
    """Replay ``runs`` times with each uncertain parameter drawn uniformly from its measured range.

    A run reproduces the failure when its outcome equals ``failure_mode``. The outcome is the env's own
    (``failure_reason`` when it ends the episode) unless ``detect`` is given, which maps a replay result to a
    mode (or None for no failure), for failures the env does not flag itself (a slip, a drop).
    ``replay_kwargs`` go to :func:`replay_on_log_clock` (mode ``"action"``).
    """
    rng = np.random.default_rng(seed)
    names = list(uncertain)
    draws: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    hits = 0
    for i in range(runs):
        p = {k: float(rng.uniform(*uncertain[k])) for k in names}
        env = make_env()
        try:
            res = replay_on_log_clock(env, mode="action",
                                      scenario=Scenario(seed=seed + i, params={**(base_params or {}), **p}),
                                      **replay_kwargs)
        finally:
            close = getattr(env, "close", None)
            if close:
                close()
        outcome = detect(res) if detect else res.ended_as
        mode = str(outcome) if outcome else "none"
        counts[mode] = counts.get(mode, 0) + 1
        ok = mode == failure_mode
        hits += ok
        draws.append({"params": p, "outcome": mode, "reproduced": ok, "divergence_t": res.divergence_t,
                      "nrmse": res.nrmse})
    effect = {}
    for k in names:
        lo, hi = uncertain[k]
        width = (hi - lo) or 1.0
        yes = [d["params"][k] for d in draws if d["reproduced"]]
        no = [d["params"][k] for d in draws if not d["reproduced"]]
        effect[k] = float((np.mean(yes) - np.mean(no)) / width) if yes and no else math.nan
    return DistributionResult(failure_mode, runs, hits, hits / runs if runs else math.nan,
                              wilson_interval(hits, runs), counts, effect, draws)
