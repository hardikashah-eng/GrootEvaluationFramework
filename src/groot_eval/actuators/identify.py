"""Identify a joint's drive from stand tests, fit the actuator model to it, and check the fit against gates.

Run each joint on a stand (robot lifted, the joint free to move against its own link) and record the commanded
position, the measured position and, if available, the measured torque:

* **Steps** give rise time, overshoot, settling time and steady-state error (:func:`step_metrics`).
* **Chirps** (frequency sweeps) give the frequency response: magnitude in dB and phase in degrees per band
  (:func:`frequency_response`), and the bandwidth (:func:`bandwidth_hz`).
* **Constant-velocity sweeps** in both directions give Coulomb and viscous friction (:func:`fit_friction`).
* **Direction reversals** give backlash (:func:`estimate_backlash`).
* Any run gives the transport delay (:func:`estimate_delay`).

:func:`fit_joint` then fits the drive model (PD gains, friction, delay, current-loop lag, backlash) to the
recorded runs by simulating a one-joint plant, ``I q'' = tau - m g l sin(q)``, through
:class:`~groot_eval.actuators.drive_chain.DriveChain`. Every candidate is one environment of a single batched
run, and a small evolution strategy refines the population; no SciPy needed.

:func:`acceptance` compares the real joint with the fitted model on the same commands: step rise time and
overshoot within ±10 %, magnitude within ±2 dB and phase within ±15 degrees up to the joint's bandwidth,
replay RMSE under a threshold, and torque saturation on the same steps. The thresholds are starting points.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from groot_eval.actuators.core import ActuatorParams, FaultTable, WearAwareActuatorCore, WearTable
from groot_eval.actuators.drive_chain import DriveChain, DriveChainParams


# =====================================================================================
# Measurements from stand tests
# =====================================================================================
def step_metrics(y: np.ndarray, target: float, start: float | None = None, dt: float = 0.005,
                 settle_band: float = 0.02) -> dict[str, float]:
    """Rise time (10-90 %), overshoot (% of the step), settling time (into ±``settle_band`` of the step), error.

    ``y`` is the measured position from the moment the step was commanded.
    """
    y = np.asarray(y, dtype=float)
    y0 = float(y[0] if start is None else start)
    span = float(target) - y0
    if abs(span) < 1e-12:
        raise ValueError("step size is zero")
    frac = (y - y0) / span
    i10 = np.argmax(frac >= 0.1) if np.any(frac >= 0.1) else None
    i90 = np.argmax(frac >= 0.9) if np.any(frac >= 0.9) else None
    rise = (i90 - i10) * dt if i10 is not None and i90 is not None else math.inf
    overshoot = max(0.0, float(frac.max()) - 1.0) * 100.0
    outside = np.nonzero(np.abs(frac - 1.0) > settle_band)[0]
    settle = math.inf if outside.size and outside[-1] == len(y) - 1 else ((outside[-1] + 1) * dt if outside.size else 0.0)
    tail = frac[-max(1, len(y) // 10):]
    return {"rise_s": float(rise), "overshoot_pct": overshoot, "settling_s": float(settle),
            "steady_state_error": float(abs(1.0 - tail.mean()) * abs(span))}


def frequency_response(u: np.ndarray, y: np.ndarray, dt: float, f_lo: float = 0.5, f_hi: float = 30.0,
                       bands: int = 16) -> dict[str, np.ndarray]:
    """Frequency response from a chirp: per log-spaced band, gain (dB), phase (deg) and coherence.

    ``u`` is the commanded position and ``y`` the measured one, both sampled every ``dt`` seconds.
    """
    u = np.asarray(u, float) - np.mean(u)
    y = np.asarray(y, float) - np.mean(y)
    U, Y = np.fft.rfft(u), np.fft.rfft(y)
    f = np.fft.rfftfreq(len(u), dt)
    edges = np.geomspace(f_lo, min(f_hi, 0.5 / dt), bands + 1)
    fc, mag, ph, coh = [], [], [], []
    for lo, hi in zip(edges, edges[1:]):
        m = (f >= lo) & (f < hi)
        if m.sum() < 2:
            continue
        pxx, pyy, pxy = np.sum(np.abs(U[m]) ** 2), np.sum(np.abs(Y[m]) ** 2), np.sum(np.conj(U[m]) * Y[m])
        if pxx <= 0:
            continue
        h = pxy / pxx
        fc.append(math.sqrt(lo * hi))
        mag.append(20 * math.log10(max(abs(h), 1e-12)))
        ph.append(math.degrees(np.angle(h)))
        coh.append(float(abs(pxy) ** 2 / (pxx * pyy)) if pyy > 0 else 0.0)
    return {"freq_hz": np.array(fc), "mag_db": np.array(mag), "phase_deg": np.array(ph), "coherence": np.array(coh)}


def bandwidth_hz(resp: dict[str, np.ndarray], drop_db: float = 3.0) -> float:
    """First band where the gain falls ``drop_db`` below the low-frequency gain (inf if it never does)."""
    mag = resp["mag_db"]
    if not mag.size:
        return math.nan
    below = np.nonzero(mag < mag[0] - drop_db)[0]
    return float(resp["freq_hz"][below[0]]) if below.size else math.inf


def onset_delay(y: np.ndarray, step_index: int, noise: float = 1e-4, state_before_command: bool = False) -> int:
    """Transport delay (steps) from a step test: steps from the commanded step until the joint first moves.

    This is the pure dead time. :func:`estimate_delay` also includes the drive's dynamic lag, so it is larger.
    ``y[k]`` is taken as the position after command ``k`` was applied (as :func:`simulate` returns it); for
    logs that record the state *before* each command (the trace convention), set ``state_before_command``.
    """
    y = np.asarray(y, float)
    base = y[step_index]
    moved = np.nonzero(np.abs(y[step_index:] - base) > noise)[0]
    if not moved.size:
        return -1
    return int(moved[0]) - (1 if state_before_command else 0)


def estimate_delay(u: np.ndarray, y: np.ndarray, max_lag: int = 50) -> int:
    """Total lag (steps) of ``y`` behind ``u``: the shift that best aligns their first differences.

    Includes dynamic lag (gains, inertia, current loop), not only dead time; use :func:`onset_delay` for that.
    """
    du, dy = np.diff(np.asarray(u, float)), np.diff(np.asarray(y, float))
    best, best_c = 0, -math.inf
    for k in range(0, min(max_lag, len(du) - 2) + 1):
        a, b = du[: len(du) - k], dy[k:]
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if c > best_c:
            best, best_c = k, c
    return best


def fit_friction(qd: np.ndarray, tau: np.ndarray, rest_velocity: float = 1e-3) -> dict[str, float]:
    """Coulomb (N·m) and viscous (N·m·s/rad) friction from constant-velocity sweeps: tau = Fc sign(qd) + b qd.

    ``tau`` is the torque needed to keep the joint moving at ``qd`` (gravity removed). Stiction is the largest
    torque seen while the joint was still at rest, if any rest samples were given.
    """
    qd, tau = np.asarray(qd, float), np.asarray(tau, float)
    moving = np.abs(qd) > rest_velocity
    if moving.sum() < 2 or len(np.unique(np.sign(qd[moving]))) < 2:
        raise ValueError("friction fit needs moving samples in both directions")
    A = np.column_stack([np.sign(qd[moving]), qd[moving]])
    (fc, b), *_ = np.linalg.lstsq(A, tau[moving], rcond=None)
    rest = ~moving
    stiction = float(np.abs(tau[rest]).max()) if rest.any() else math.nan
    return {"coulomb_friction": float(fc), "viscous_friction": float(b), "stiction_torque": stiction}


def estimate_backlash(u: np.ndarray, y: np.ndarray, move_tol: float = 1e-4) -> float:
    """Backlash (rad): how far the command travels after each reversal before the joint follows. Median over reversals."""
    u, y = np.asarray(u, float), np.asarray(y, float)
    sign = np.sign(np.diff(u))
    rev, last = [], 0.0
    for i, sg in enumerate(sign):  # a reversal may have pauses (zero steps) in between
        if sg == 0:
            continue
        if last and sg != last:
            rev.append((i, sg))
        last = sg
    dead = []
    for i, d in rev:
        for k in range(i + 1, len(y)):
            if (y[k] - y[i]) * d > move_tol:
                dead.append(abs(u[k - 1] - u[i]))
                break
    return float(np.median(dead)) if dead else 0.0


# =====================================================================================
# Fitting the model to recorded runs
# =====================================================================================
@dataclass
class JointPlant:
    """The joint on the stand: inertia (kg·m²) and gravity torque amplitude m·g·l (N·m) about the joint."""

    inertia: float = 0.05
    gravity_torque: float = 0.0


@dataclass
class DriveFit:
    stiffness: float = 100.0
    damping: float = 5.0
    coulomb_friction: float = 0.0
    viscous_friction: float = 0.0
    delay_steps: int = 0
    current_tau_s: float = 0.0
    backlash_rad: float = 0.0
    effort_limit: float = 100.0
    rmse: float = math.nan

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def drive_chain(self, dt: float) -> dict[str, Any]:
        """The ``drive_chain`` block for this joint (profile or Isaac Lab adapter config)."""
        return {"delay_steps_min": self.delay_steps, "delay_steps_max": self.delay_steps,
                "current_tau_s": self.current_tau_s, "dt": dt}


FIT_KEYS = ("stiffness", "damping", "coulomb_friction", "viscous_friction", "delay_steps", "current_tau_s",
            "backlash_rad")
LOG_KEYS = ("stiffness", "damping")  # searched on a log scale: they span orders of magnitude


def simulate(params: Sequence[DriveFit] | DriveFit, q_cmd: np.ndarray, q0: float, plant: JointPlant,
             dt: float, substeps: int = 4, tau_ff: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Run the commanded positions through the drive model and the stand plant, one env per parameter set.

    Returns ``q`` and ``tau`` of shape ``(len(params), T)`` and the step mask where torque hit its limit.
    """
    ps = [params] if isinstance(params, DriveFit) else list(params)
    E, T = len(ps), len(q_cmd)
    col = lambda k: np.array([[float(getattr(p, k))] for p in ps])  # noqa: E731 - (E, 1) per-env values
    wear = WearTable.new(E, 1)
    wear.values["backlash"][:] = col("backlash_rad")
    ap = ActuatorParams(effort_limit=col("effort_limit"), stiffness=col("stiffness"), damping=col("damping"),
                        coulomb_friction=col("coulomb_friction"), viscous_friction=col("viscous_friction"), dt=dt)
    core = WearAwareActuatorCore(ap, wear, FaultTable.none(E))
    dmax = max(int(p.delay_steps) for p in ps)
    chain = DriveChain(core, DriveChainParams(delay_steps_max=dmax, current_tau_s=col("current_tau_s"), dt=dt))
    chain.delay = np.array([int(p.delay_steps) for p in ps])
    q = np.full((E, 1), float(q0))
    qd = np.zeros((E, 1))
    qs, taus = np.zeros((E, T)), np.zeros((E, T))
    lim = col("effort_limit")
    h = dt / substeps
    for t in range(T):
        ff = np.zeros((E, 1)) if tau_ff is None else np.full((E, 1), float(tau_ff[t]))
        tau = chain.compute(np.full((E, 1), float(q_cmd[t])), np.zeros((E, 1)), ff, q, qd)
        for _ in range(substeps):
            acc = (tau - plant.gravity_torque * np.sin(q)) / plant.inertia
            qd = qd + h * acc
            q = q + h * qd
        qs[:, t], taus[:, t] = q[:, 0], tau[:, 0]
    return {"q": qs, "tau": taus, "saturated": np.abs(taus) >= 0.999 * lim}


def _search(z0: np.ndarray, cost: Any, to_fits: Any, population: int, generations: int,
            rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """One (mu, lambda) evolution-strategy run in the unit box, starting from ``z0`` plus a uniform spread."""
    z = rng.random((max(population * 2, 8), len(z0)))
    z[0] = z0
    c = cost(to_fits(z))
    order = np.argsort(c)
    best_z, best_c = z[order[0]].copy(), float(c[order[0]])
    mu = max(2, population // 4)
    w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
    w = w / w.sum()
    mean, sigma = w @ z[order[:mu]], 0.2
    for _ in range(generations):
        z = np.clip(mean + sigma * rng.standard_normal((population, len(z0))), 0.0, 1.0)
        z[0] = best_z
        c = cost(to_fits(z))
        order = np.argsort(c)
        if c[order[0]] < best_c:
            best_c, best_z = float(c[order[0]]), z[order[0]].copy()
        mean = w @ z[order[:mu]]
        sigma = max(0.005, sigma * 0.9)
    return best_z, best_c


def fit_joint(runs: Sequence[dict[str, Any]], plant: JointPlant, dt: float, *, effort_limit: float = 100.0,
              start: DriveFit | None = None, bounds: dict[str, tuple[float, float]] | None = None,
              fixed: dict[str, float] | None = None, population: int = 24, generations: int = 25,
              restarts: int = 3, seed: int = 0) -> DriveFit:
    """Fit drive parameters so the model's positions match recorded runs. Each run: ``{"q_cmd", "q", "q0"?}``.

    Minimises the RMSE of position over all runs with a (mu, lambda) evolution strategy in normalised
    parameter space; delay is searched as a continuous value and rounded. Bounds default to wide ranges.

    Gains, friction, delay and backlash trade off against each other, so a fit of all seven from position
    alone can match the behaviour with the wrong parameters. Measure delay (:func:`onset_delay`), friction
    (:func:`fit_friction`) and backlash (:func:`estimate_backlash`) from their own tests and pass them in
    ``fixed``; the fit then only searches the rest. The search restarts ``restarts`` times from different
    random spreads and keeps the best, because a single run can settle on a wrong basin (very high gains with
    a slow current loop can mimic moderate gains).
    """
    s = start or DriveFit(effort_limit=effort_limit)
    b = {"stiffness": (1.0, 2000.0), "damping": (0.0, 100.0), "coulomb_friction": (0.0, 20.0),
         "viscous_friction": (0.0, 20.0), "delay_steps": (0.0, 10.0), "current_tau_s": (0.0, 0.05),
         "backlash_rad": (0.0, 0.05), **(bounds or {})}
    for k, v in (fixed or {}).items():
        if k not in FIT_KEYS:
            raise ValueError(f"unknown parameter {k!r}; fixable: {', '.join(FIT_KEYS)}")
        b[k] = (float(v), float(v))
    logk = np.array([k in LOG_KEYS for k in FIT_KEYS])
    eps = 1e-3

    def fwd(x: np.ndarray) -> np.ndarray:  # parameter -> search coordinate
        return np.where(logk, np.log(np.maximum(x, 0.0) + eps), x)

    lo = fwd(np.array([b[k][0] for k in FIT_KEYS]))
    hi = fwd(np.array([b[k][1] for k in FIT_KEYS]))

    def to_fits(z: np.ndarray) -> list[DriveFit]:
        y = lo + np.clip(z, 0.0, 1.0) * (hi - lo)
        x = np.where(logk, np.exp(y) - eps, y)
        out = []
        for row in x:
            kw = {k: float(v) for k, v in zip(FIT_KEYS, row)}
            delay = int(round(kw.pop("delay_steps")))
            out.append(DriveFit(effort_limit=effort_limit, delay_steps=delay, **kw))
        return out

    def cost(fits: list[DriveFit]) -> np.ndarray:
        err, n = np.zeros(len(fits)), 0
        for r in runs:
            q_meas = np.asarray(r["q"], float)
            sim = simulate(fits, np.asarray(r["q_cmd"], float), float(r.get("q0", q_meas[0])), plant, dt)
            err += np.sum((sim["q"] - q_meas[None, :]) ** 2, axis=1)
            n += len(q_meas)
        return np.sqrt(err / max(n, 1))

    span = np.where(hi > lo, hi - lo, 1.0)
    z0 = np.clip((fwd(np.array([float(getattr(s, k)) for k in FIT_KEYS])) - lo) / span, 0.0, 1.0)
    best_z, best_c = z0.copy(), math.inf
    for r in range(max(1, restarts)):
        rz, rc = _search(z0, cost, to_fits, population, generations, np.random.default_rng(seed + r))
        if rc < best_c:
            best_z, best_c = rz, rc
    fit = to_fits(best_z[None, :])[0]
    fit.rmse = best_c
    return fit


# =====================================================================================
# Acceptance: does the fitted model behave like the real joint?
# =====================================================================================
@dataclass
class Gates:
    step_rel_tol: float = 0.10  # rise time and overshoot within ±10 % (overshoot: or 1 point absolute)
    step_floor_s: float = 0.01  # rise time may also differ by this much (two samples at 200 Hz)
    mag_tol_db: float = 2.0
    phase_tol_deg: float = 15.0
    replay_rmse_rad: float = 0.01
    saturation_jaccard: float = 0.7  # overlap of saturated steps, when either side saturates


@dataclass
class Acceptance:
    passed: bool
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    def failures(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v["pass"]]


def _rel_ok(sim: float, real: float, tol: float, floor: float = 0.0) -> bool:
    if math.isinf(real) or math.isinf(sim):
        return math.isinf(real) and math.isinf(sim)
    return abs(sim - real) <= max(tol * abs(real), floor)


def acceptance(*, real_step: dict[str, float] | None = None, sim_step: dict[str, float] | None = None,
               real_freq: dict[str, np.ndarray] | None = None, sim_freq: dict[str, np.ndarray] | None = None,
               real_q: np.ndarray | None = None, sim_q: np.ndarray | None = None,
               real_sat: np.ndarray | None = None, sim_sat: np.ndarray | None = None,
               gates: Gates | None = None) -> Acceptance:
    """Gate the fitted model against the real joint. Give any subset of the four comparisons."""
    g = gates or Gates()
    checks: dict[str, dict[str, Any]] = {}
    if real_step is not None and sim_step is not None:
        ok_r = _rel_ok(sim_step["rise_s"], real_step["rise_s"], g.step_rel_tol, floor=g.step_floor_s)
        ok_o = _rel_ok(sim_step["overshoot_pct"], real_step["overshoot_pct"], g.step_rel_tol, floor=1.0)
        checks["step"] = {"pass": ok_r and ok_o, "rise_s": (real_step["rise_s"], sim_step["rise_s"]),
                          "overshoot_pct": (real_step["overshoot_pct"], sim_step["overshoot_pct"])}
    if real_freq is not None and sim_freq is not None:
        bw = bandwidth_hz(real_freq)
        n = min(len(real_freq["freq_hz"]), len(sim_freq["freq_hz"]))
        band = real_freq["freq_hz"][:n] <= bw
        dmag = np.abs(real_freq["mag_db"][:n] - sim_freq["mag_db"][:n])[band]
        dph = np.abs((real_freq["phase_deg"][:n] - sim_freq["phase_deg"][:n] + 180) % 360 - 180)[band]
        worst_m = float(dmag.max()) if dmag.size else math.nan
        worst_p = float(dph.max()) if dph.size else math.nan
        checks["frequency"] = {"pass": bool(dmag.size) and worst_m <= g.mag_tol_db and worst_p <= g.phase_tol_deg,
                               "bandwidth_hz": bw, "worst_mag_db": worst_m, "worst_phase_deg": worst_p}
    if real_q is not None and sim_q is not None:
        rmse = float(np.sqrt(np.mean((np.asarray(sim_q) - np.asarray(real_q)) ** 2)))
        checks["replay"] = {"pass": rmse <= g.replay_rmse_rad, "rmse_rad": rmse}
    if real_sat is not None and sim_sat is not None:
        a, b_ = np.asarray(real_sat, bool), np.asarray(sim_sat, bool)
        union = (a | b_).sum()
        jac = 1.0 if union == 0 else float((a & b_).sum() / union)
        checks["saturation"] = {"pass": jac >= g.saturation_jaccard, "jaccard": jac,
                                "real_steps": int(a.sum()), "sim_steps": int(b_.sum())}
    return Acceptance(passed=bool(checks) and all(v["pass"] for v in checks.values()), checks=checks)


def chirp(T: int, dt: float, f0: float = 0.2, f1: float = 20.0, amp: float = 0.1, offset: float = 0.0) -> np.ndarray:
    """A linear chirp position command for frequency-response tests."""
    t = np.arange(T) * dt
    k = (f1 - f0) / max(t[-1], dt)
    return offset + amp * np.sin(2 * math.pi * (f0 * t + 0.5 * k * t ** 2))
