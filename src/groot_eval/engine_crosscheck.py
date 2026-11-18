"""Engine cross-check: when the Isaac Sim twin does not reproduce a real failure, is it Isaac or the replay?

A failure that does not reproduce in the twin has three possible causes, and they need different fixes:

* **The replay is wrong**: the twin is not started in the real state, runs are not repeatable, the commands
  are mis-timed, the units or joint order are off. Fix the replay; changing physics would only hide it.
* **The engine is the limit**: the twin's model is right but the engine cannot reproduce the effect (PhysX
  contact and friction, its non-restorable contact state, GPU non-determinism). Use another engine for this
  scenario class, or file it against the engine.
* **The twin's model is wrong or incomplete**: actuator, asset, contact parameters, or a cause outside the
  model (a loose cable, the floor). Fix the twin's parameters (system identification, the repro knob search).

The cross-check runs the same failure through the primary engine (Isaac Sim) and a reference engine (MuJoCo)
with the same replay inputs, and tells these apart:

1. **Repeatability**: the same replay twice in each engine must give the same trajectory.
2. **State restore**: setting each recorded state and reading it back must give it back.
3. **Replay**: the log-clock replay (:mod:`groot_eval.timeline_replay`) in each engine, with whether the
   failure reproduced and when the twin left the recording.

==========================  ================================================================================
verdict                     meaning and next step
==========================  ================================================================================
replay_not_repeatable       an engine gives different results for the same replay: fix seeding / stepping first
replay_restore_issue        neither engine can hold the recorded state: units, joint order, limits, state map
primary_restore_limitation  MuJoCo holds the recorded state and Isaac does not
reproduced_in_both          both engines reproduce the failure: the twin is fine for this case
primary_engine_limitation   MuJoCo reproduces, Isaac does not: an Isaac limit (contact, solver, restore)
reference_engine_gap        Isaac reproduces, MuJoCo does not: check the MuJoCo model matches (parity)
shared_model_gap            neither reproduces and both leave the recording at about the same time: the twin's
                            model or the replay inputs, not the engine (system ID, knob search)
engine_dependent_gap        neither reproduces and they diverge at very different times: engine physics differ;
                            compare contact settings before trusting either
==========================  ================================================================================

:class:`MuJoCoEnv` is a small MuJoCo env with the same interface as the harness envs (``reset`` / ``step`` /
``set_state``), exact full-state save and restore (``get_full_state`` / ``set_full_state``, including the
solver's warm start), and optional drive model (:mod:`groot_eval.actuators`) in torque mode.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from groot_eval.timeline_replay import TimelineResult, _flatten, _state_dict, replay_on_log_clock
from groot_eval.types import Scenario


# =====================================================================================
# A MuJoCo env with exact state restore
# =====================================================================================
class MuJoCoEnv:
    """MuJoCo env on an MJCF model. Obs: ``state.qpos``, ``state.qvel``; action: one value per actuator.

    ``control`` = ``"ctrl"`` writes the action to ``data.ctrl`` (position or motor actuators as the MJCF
    defines them). ``"torque"`` treats the action as joint position targets and runs them through ``drive``
    (a :class:`~groot_eval.actuators.DriveChain` or core with one env), writing the torque to ``data.ctrl``
    of motor actuators. ``param_setters`` map scenario parameters to functions ``(model, value)`` (friction,
    mass, damping, ...). ``terminate`` maps ``(model, data)`` to ``(done, info)``.
    """

    def __init__(self, xml: str, *, control_dt: float = 0.02, control: str = "ctrl", drive: Any = None,
                 param_setters: Mapping[str, Callable[[Any, float], None]] | None = None,
                 terminate: Callable[[Any, Any], tuple[bool, dict[str, Any]]] | None = None,
                 max_steps: int = 1000) -> None:
        import mujoco  # optional dependency: pip install mujoco

        self._mj = mujoco
        self._xml = xml
        self.model = (mujoco.MjModel.from_xml_path(xml) if not xml.lstrip().startswith("<")
                      else mujoco.MjModel.from_xml_string(xml))
        self.data = mujoco.MjData(self.model)
        self.n_sub = max(1, int(round(control_dt / self.model.opt.timestep)))
        self.control_dt = self.n_sub * self.model.opt.timestep
        if control not in ("ctrl", "torque"):
            raise ValueError("control is 'ctrl' or 'torque'")
        if control == "torque" and drive is None:
            raise ValueError("torque control needs a drive model")
        self.control, self.drive = control, drive
        self.param_setters = dict(param_setters or {})
        self.terminate = terminate
        self.max_steps = max_steps
        self._t = 0
        self._spec = mujoco.mjtState.mjSTATE_INTEGRATION

    # -- harness interface --------------------------------------------------------
    def reset(self, scenario: Scenario) -> dict[str, Any]:
        mj = self._mj
        if self.param_setters:  # rebuild so parameters never accumulate across episodes
            self.model = (mj.MjModel.from_xml_path(self._xml) if not self._xml.lstrip().startswith("<")
                          else mj.MjModel.from_xml_string(self._xml))
            self.data = mj.MjData(self.model)
        unknown = [k for k in scenario.params if k not in self.param_setters]
        if unknown:
            raise KeyError(f"no param setter for: {', '.join(unknown)}")
        for k, v in scenario.params.items():
            self.param_setters[k](self.model, float(v))
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        if self.drive is not None:
            self.drive.reset()
        self._t = 0
        return self._obs()

    def step(self, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        mj = self._mj
        a = np.asarray(action, float).reshape(-1)
        for _ in range(self.n_sub):
            if self.control == "torque":
                q, qd = self.data.qpos[: a.size][None, :], self.data.qvel[: a.size][None, :]
                tau = np.asarray(self.drive.compute(a[None, :], None, None, q, qd), float)[0]
                self.data.ctrl[: a.size] = tau
            else:
                self.data.ctrl[: a.size] = a
            mj.mj_step(self.model, self.data)
        self._t += 1
        done, info = False, dict[str, Any]()
        if self.terminate is not None:
            done, info = self.terminate(self.model, self.data)
        trunc = (not done) and self._t >= self.max_steps
        return self._obs(), 0.0, bool(done), bool(trunc), dict(info)

    def set_state(self, state: Mapping[str, Any], recent_actions: np.ndarray | None = None) -> dict[str, Any]:
        mj = self._mj
        if "state.qpos" in state:
            self.data.qpos[:] = np.asarray(state["state.qpos"], float)
        if "state.qvel" in state:
            self.data.qvel[:] = np.asarray(state["state.qvel"], float)
        self.data.qacc_warmstart[:] = 0.0
        mj.mj_forward(self.model, self.data)
        return self._obs()

    def close(self) -> None:
        pass

    # -- exact save / restore -----------------------------------------------------------
    def get_full_state(self) -> np.ndarray:
        """Everything that determines the next step (time, qpos, qvel, act, warm start, ...)."""
        s = np.zeros(self._mj.mj_stateSize(self.model, self._spec))
        self._mj.mj_getState(self.model, self.data, s, self._spec)
        return s

    def set_full_state(self, s: np.ndarray) -> dict[str, Any]:
        self._mj.mj_setState(self.model, self.data, np.asarray(s, float), self._spec)
        self._mj.mj_forward(self.model, self.data)
        return self._obs()

    def model_summary(self) -> dict[str, Any]:
        """Joint names, ranges and body masses: compared across engines by :func:`model_parity`."""
        mj, m = self._mj, self.model
        joints = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_JOINT, i): [float(x) for x in m.jnt_range[i]]
                  for i in range(m.njnt)}
        masses = {mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, i): float(m.body_mass[i]) for i in range(1, m.nbody)}
        return {"joints": joints, "masses": masses}

    def _obs(self) -> dict[str, Any]:
        return {"state.qpos": self.data.qpos.copy(), "state.qvel": self.data.qvel.copy()}


# =====================================================================================
# Checks
# =====================================================================================
def repeatability(make_env: Callable[[], Any], replay_kwargs: Mapping[str, Any], runs: int = 2,
                  atol: float = 0.0) -> dict[str, Any]:
    """Run the same replay ``runs`` times; return whether they match and the first differing time."""
    results = []
    for _ in range(runs):
        env = make_env()
        try:
            results.append(replay_on_log_clock(env, **dict(replay_kwargs)))
        finally:
            getattr(env, "close", lambda: None)()
    first = results[0]
    worst, first_diff = 0.0, None
    for r in results[1:]:
        n = min(len(r.predicted), len(first.predicted))
        d = np.abs(np.nan_to_num(r.predicted[:n]) - np.nan_to_num(first.predicted[:n])).max(axis=1)
        worst = max(worst, float(d.max()) if d.size else 0.0)
        bad = np.nonzero(d > atol)[0]
        if bad.size:
            t = float(first.times[bad[0]])
            first_diff = t if first_diff is None else min(first_diff, t)
        if len(r.predicted) != len(first.predicted) and first_diff is None:
            first_diff = float(first.times[n - 1])
    return {"repeatable": first_diff is None, "max_difference": worst, "first_difference_t": first_diff,
            "runs": runs, "result": first}


def restore_check(env: Any, states: np.ndarray, slices: Mapping[str, slice], indices: Sequence[int] | None = None,
                  scenario: Scenario | None = None, tol: float = 0.05) -> dict[str, Any]:
    """Set recorded states and read them back. Error is per signal, relative to the signal's spread."""
    X = np.asarray(states, float)
    idx = list(indices) if indices is not None else list(np.linspace(0, len(X) - 1, min(10, len(X))).astype(int))
    env.reset(scenario or Scenario(seed=0))
    spread = np.nanstd(X, axis=0)
    spread = np.where(spread > 1e-9, spread, 1.0)
    worst = 0.0
    for i in idx:
        try:
            obs = env.set_state(_state_dict(X[i], slices), None)
        except (AttributeError, NotImplementedError):
            return {"ok": False, "supported": False, "worst_relative_error": math.nan}
        err = np.abs(_flatten(obs, slices) - X[i]) / spread
        worst = max(worst, float(np.nanmax(err)))
    return {"ok": worst <= tol, "supported": True, "worst_relative_error": worst, "states_checked": len(idx)}


def model_parity(a: Mapping[str, Any], b: Mapping[str, Any], mass_rtol: float = 0.02,
                 range_atol: float = 1e-3) -> list[str]:
    """Differences between two engines' model summaries (joint names and ranges, body masses)."""
    out = []
    ja, jb = a.get("joints", {}), b.get("joints", {})
    for k in sorted(set(ja) ^ set(jb)):
        out.append(f"joint {k!r} only in {'primary' if k in ja else 'reference'}")
    for k in sorted(set(ja) & set(jb)):
        if np.max(np.abs(np.asarray(ja[k], float) - np.asarray(jb[k], float))) > range_atol:
            out.append(f"joint {k!r} range {ja[k]} vs {jb[k]}")
    ma, mb = a.get("masses", {}), b.get("masses", {})
    for k in sorted(set(ma) & set(mb)):
        if abs(ma[k] - mb[k]) > mass_rtol * max(abs(ma[k]), 1e-9):
            out.append(f"body {k!r} mass {ma[k]:.3f} vs {mb[k]:.3f} kg")
    return out


# =====================================================================================
# The cross-check
# =====================================================================================
@dataclass
class EngineReport:
    name: str
    repeatable: bool
    first_difference_t: float | None
    restore_ok: bool | None
    restore_error: float
    reproduced: bool
    outcome: str
    divergence_t: float | None
    nrmse: float

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class CrossCheck:
    verdict: str
    explanation: str
    engines: dict[str, EngineReport]
    parity: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "explanation": self.explanation, "parity": self.parity,
                "engines": {k: v.to_dict() for k, v in self.engines.items()}}


def _outcome(res: TimelineResult, detect: Callable[[TimelineResult], str | None] | None) -> str:
    o = detect(res) if detect else res.ended_as
    return str(o) if o else "none"


def cross_check(engines: Mapping[str, Callable[[], Any]], *, failure_mode: str, replay_kwargs: Mapping[str, Any],
                primary: str = "isaac", reference: str = "mujoco", restore_tol: float = 0.05,
                same_time_s: float | None = None, detect: Callable[[TimelineResult], str | None] | None = None,
                repeat_atol: float = 1e-9) -> CrossCheck:
    """Run the failure through the primary and reference engines and say which part is at fault.

    ``engines`` maps engine names to env factories (fresh env per call). ``replay_kwargs`` go to
    :func:`~groot_eval.timeline_replay.replay_on_log_clock` (states, state_t, slices, actions, action_t, dt,
    start_index, ...). Divergence times within ``same_time_s`` (default: 10 control steps) count as the same.
    """
    if primary not in engines or reference not in engines:
        raise ValueError(f"engines must include {primary!r} and {reference!r}")
    kw = dict(replay_kwargs)
    kw.setdefault("mode", "action")
    dt = float(kw["dt"])
    same = same_time_s if same_time_s is not None else 10 * dt
    reports: dict[str, EngineReport] = {}
    summaries: dict[str, Any] = {}
    for name in (primary, reference):
        rep = repeatability(engines[name], kw, runs=2, atol=repeat_atol)
        res: TimelineResult = rep["result"]
        env = engines[name]()
        try:
            rc = restore_check(env, kw["states"], kw["slices"], scenario=kw.get("scenario"), tol=restore_tol)
            if hasattr(env, "model_summary"):
                summaries[name] = env.model_summary()
        finally:
            getattr(env, "close", lambda: None)()
        out = _outcome(res, detect)
        reports[name] = EngineReport(name, rep["repeatable"], rep["first_difference_t"],
                                     rc["ok"] if rc["supported"] else None, rc["worst_relative_error"],
                                     out == failure_mode, out, res.divergence_t, res.nrmse)
    parity = model_parity(summaries[primary], summaries[reference]) if len(summaries) == 2 else []
    p, r = reports[primary], reports[reference]

    def done(v: str, why: str) -> CrossCheck:
        return CrossCheck(v, why, reports, parity)

    flaky = [e.name for e in (p, r) if not e.repeatable]
    if flaky:
        return done("replay_not_repeatable", f"{', '.join(flaky)} gave different results for the same replay "
                    f"(first at t={min(e.first_difference_t or math.inf for e in (p, r) if not e.repeatable):.3f} s); "
                    "fix seeding, fixed stepping or GPU determinism before comparing engines")
    if p.restore_ok is False and r.restore_ok is False:
        return done("replay_restore_issue", "neither engine holds the recorded state; check units, joint order, "
                    "joint limits and the state mapping")
    if p.restore_ok is False and r.restore_ok:
        return done("primary_restore_limitation", f"{reference} holds the recorded state and {primary} does not "
                    f"(error {p.restore_error:.3f} of spread); restore from the episode start in {primary}")
    if p.reproduced and r.reproduced:
        return done("reproduced_in_both", "both engines reproduce the failure")
    if r.reproduced and not p.reproduced:
        return done("primary_engine_limitation", f"{reference} reproduces the failure ({r.outcome}) and {primary} "
                    f"does not ({p.outcome}) with the same inputs: likely a {primary} contact, solver or restore "
                    "limit; use the reference engine for this scenario class or adjust contact settings")
    if p.reproduced and not r.reproduced:
        why = "; ".join(parity[:3]) if parity else "models agree on joints and masses; compare contact settings"
        return done("reference_engine_gap", f"{primary} reproduces and {reference} does not: {why}")
    dp, dr = p.divergence_t, r.divergence_t
    if dp is not None and dr is not None and abs(dp - dr) > same:
        return done("engine_dependent_gap", f"neither reproduces; they leave the recording at t={dp:.2f} s and "
                    f"t={dr:.2f} s, so engine physics differ here; compare contact settings before trusting either")
    return done("shared_model_gap", "neither engine reproduces and both behave alike: the twin's model or the "
                "replay inputs are missing the cause (run system identification and the repro knob search)")


def cross_check_case(case: Any, engines: Mapping[str, Callable[[], Any]], *, dt: float,
                     state_t: Sequence[float] | None = None, action_t: Sequence[float] | None = None,
                     lead_s: float | None = None, **kwargs: Any) -> CrossCheck:
    """Cross-check a :class:`~groot_eval.repro.FailureCase` (it needs a recording).

    Without timestamps the recording is taken as one state and one command per ``dt``. ``lead_s`` starts the
    replay that long before the case's ``fail_step`` instead of at the first state.
    """
    if not getattr(case, "has_recording", False):
        raise ValueError("cross-check needs a failure with recorded states and actions")
    T = len(case.states)
    st = np.asarray(state_t if state_t is not None else np.arange(T) * dt, float)
    at = np.asarray(action_t if action_t is not None else np.arange(len(case.actions)) * dt, float)
    start = 0
    if lead_s is not None and case.fail_step is not None:
        from groot_eval.timeline_replay import restore_index

        start = restore_index(st, float(st[min(case.fail_step, T - 1)]), lead_s)
    rk = {"states": case.states, "state_t": st, "slices": case.slices or {}, "actions": case.actions,
          "action_t": at, "dt": dt, "start_index": start, "scenario": case.scenario}
    return cross_check(engines, failure_mode=case.failure_reason, replay_kwargs=rk, **kwargs)
