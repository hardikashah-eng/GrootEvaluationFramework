"""Drive-chain stages (delay, drops, current loop, linkage) and stand-test identification of a joint."""

from types import SimpleNamespace

import numpy as np
import pytest

from groot_eval.actuators.core import ActuatorParams, FaultTable, WearAwareActuatorCore, WearTable
from groot_eval.actuators.drive_chain import DriveChain, DriveChainParams
from groot_eval.actuators.identify import (DriveFit, Gates, JointPlant, acceptance, bandwidth_hz, chirp,
                                           estimate_backlash, estimate_delay, fit_friction, fit_joint,
                                           frequency_response, onset_delay, simulate, step_metrics)
from groot_eval.actuators.isaaclab_actuator import build_actuator


def core(E=2, J=1, **kw):
    return WearAwareActuatorCore(ActuatorParams(effort_limit=50.0, stiffness=10.0, damping=0.0, **kw),
                                 WearTable.new(E, J), FaultTable.none(E))


def run(chain, cmds, J=1):
    E = chain.num_envs
    z = np.zeros((E, J))
    return np.array([chain.compute(np.full((E, J), c), None, None, z, z) for c in cmds])


# -- drive chain -------------------------------------------------------------------------
def test_transport_delay_per_env():
    ch = DriveChain(core(), DriveChainParams(delay_steps_min=0, delay_steps_max=3))
    ch.delay = np.array([0, 3])
    out = run(ch, [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    assert out[:, 0, 0].tolist() == [0, 10, 20, 30, 40, 50]  # tau = K * delivered target (q = 0)
    assert out[:, 1, 0].tolist() == [0, 0, 0, 0, 10, 20]  # the first command is held until history fills


def test_dropped_packets_hold_the_last_command():
    ch = DriveChain(core(E=1), DriveChainParams(drop_prob=0.5, seed=3))
    out = run(ch, np.arange(200.0) * 0.01)[:, 0, 0] / 10.0  # stays under the torque limit
    assert ch.dropped[0] > 50
    held = np.sum(out[1:] == out[:-1])
    assert held == ch.dropped[0]  # every drop repeats the previous delivered command
    assert np.all(np.diff(out) >= 0)  # never delivers an older command than the last one


def test_current_loop_lag_is_first_order():
    dt, tau_c = 0.001, 0.01
    ch = DriveChain(core(E=1), DriveChainParams(current_tau_s=tau_c, dt=dt))
    out = run(ch, [1.0] * 100)[:, 0, 0] / 10.0
    k = int(round(tau_c / dt))
    assert out[k - 1] == pytest.approx(1 - np.exp(-1), abs=0.05)  # 63 % after one time constant
    assert out[-1] == pytest.approx(1.0, abs=1e-3)


def test_linkage_limit_depends_on_angle():
    ch = DriveChain(core(E=2), DriveChainParams(linkage={0: [(-1.0, 5.0), (0.0, 20.0), (1.0, 8.0)]}))
    q = np.array([[0.0], [0.5]])
    tau = ch.compute(np.full((2, 1), 10.0), None, None, q, np.zeros((2, 1)))  # asks for ~100 N·m
    assert tau[:, 0].tolist() == pytest.approx([20.0, 14.0])
    assert ch.telemetry()["linkage_clipped"].sum() == 2


@pytest.mark.parametrize("bad", [dict(delay_steps_min=3, delay_steps_max=1), dict(drop_prob=1.0),
                                 dict(current_tau_s=-1.0), dict(linkage={0: [(0.0, 1.0)]}),
                                 dict(linkage={0: [(0.0, 1.0), (-1.0, 2.0)]}), dict(linkage={0: [(0, 1), (1, 0)]})])
def test_drive_chain_params_are_validated(bad):
    with pytest.raises(ValueError):
        DriveChainParams(**bad)


def test_isaac_adapter_wraps_core_when_drive_chain_set():
    cfg = SimpleNamespace(wear_config=None, ages_hours=None, typical_units=False, fault_config=None, horizon_steps=10,
                          dt=0.005, fault_seed=0, coulomb_friction=0.0, viscous_friction=0.0, motor_model=None,
                          saturation_effort=None, velocity_limit_dc=float("inf"), drive_chain=None)
    assert isinstance(build_actuator(cfg, 4, 2, 50.0, 10.0, 1.0), WearAwareActuatorCore)
    cfg.drive_chain = {"delay_steps_min": 1, "delay_steps_max": 2, "linkage": {"1": [[-1, 5], [1, 5]]}}
    act = build_actuator(cfg, 4, 2, 50.0, 10.0, 1.0)
    assert isinstance(act, DriveChain) and set(act.delay) <= {1, 2} and 1 in act.p.linkage


# -- identification -----------------------------------------------------------------------
DT = 0.005
PLANT = JointPlant(inertia=0.02, gravity_torque=0.5)
TRUE = DriveFit(stiffness=60, damping=2.0, coulomb_friction=0.3, viscous_friction=0.2, delay_steps=3,
                current_tau_s=0.004, backlash_rad=0.004, effort_limit=20)
STEP = np.r_[np.zeros(20), np.full(200, 0.3)]
CHIRP = chirp(600, DT, 0.2, 12.0, 0.15)


def measured(fit, cmd):
    return simulate(fit, cmd, 0.0, PLANT, DT)["q"][0]


def test_step_metrics_on_first_order_response():
    t = np.arange(400) * DT
    y = 1 - np.exp(-t / 0.1)
    m = step_metrics(y, target=1.0, start=0.0, dt=DT)
    assert m["rise_s"] == pytest.approx(0.1 * np.log(9), abs=2 * DT)  # 10-90 % of a first-order lag
    assert m["overshoot_pct"] == 0.0
    y2 = 1 - np.exp(-t / 0.05) * np.cos(30 * t)
    assert step_metrics(y2, 1.0, 0.0, DT)["overshoot_pct"] > 10


def test_frequency_response_of_a_known_gain_and_delay():
    u = chirp(4000, DT, 0.2, 20.0, 1.0)
    y = 0.5 * np.r_[np.zeros(4), u[:-4]]  # gain 0.5 (-6 dB), 4-step delay
    r = frequency_response(u, y, DT, 0.5, 15.0, 8)
    assert np.allclose(r["mag_db"], 20 * np.log10(0.5), atol=0.3)
    expect = np.degrees(np.angle(np.exp(-2j * np.pi * r["freq_hz"] * 4 * DT)))
    assert np.allclose((r["phase_deg"] - expect + 180) % 360 - 180, 0, atol=6)
    assert bandwidth_hz(r) == np.inf


def test_delay_friction_and_backlash_estimates():
    y = measured(TRUE, STEP)
    assert onset_delay(y, 20) == 3  # pure dead time
    assert estimate_delay(CHIRP, measured(TRUE, CHIRP)) > 3  # total lag includes the drive's dynamics
    qd = np.r_[np.linspace(-1, -0.1, 20), np.zeros(5), np.linspace(0.1, 1, 20)]
    tau = 0.3 * np.sign(qd) + 0.2 * qd
    tau[20:25] = [0.1, 0.2, 0.25, 0.28, 0.29]
    f = fit_friction(qd, tau)
    assert f["coulomb_friction"] == pytest.approx(0.3) and f["viscous_friction"] == pytest.approx(0.2)
    assert f["stiction_torque"] == pytest.approx(0.29)
    with pytest.raises(ValueError):
        fit_friction(np.linspace(0.1, 1, 10), np.ones(10))
    u = np.r_[np.linspace(0, 0.1, 51), np.linspace(0.1, 0, 51)[1:], np.linspace(0, 0.1, 51)[1:]]
    out, pos = [], 0.0
    for x in u:  # a gear with 0.01 rad of play
        pos = x - 0.005 if x - pos > 0.005 else (x + 0.005 if pos - x > 0.005 else pos)
        out.append(pos)
    assert estimate_backlash(u, np.array(out)) == pytest.approx(0.01, abs=0.0025)  # within one command increment


def test_fit_recovers_drive_parameters_when_delay_friction_backlash_are_measured():
    runs = [{"q_cmd": c, "q": measured(TRUE, c), "q0": 0.0} for c in (STEP, CHIRP)]
    fixed = dict(delay_steps=onset_delay(runs[0]["q"], 20), coulomb_friction=0.3, viscous_friction=0.2,
                 backlash_rad=0.004)
    fit = fit_joint(runs, PLANT, DT, effort_limit=20, fixed=fixed, population=16, generations=20, restarts=2)
    assert fit.stiffness == pytest.approx(60, rel=0.15)
    assert fit.damping == pytest.approx(2.0, rel=0.2)
    assert fit.current_tau_s == pytest.approx(0.004, abs=0.0015)
    assert fit.rmse < 0.003


def test_fit_validates_fixed_names():
    with pytest.raises(ValueError, match="unknown parameter"):
        fit_joint([{"q_cmd": STEP, "q": STEP}], PLANT, DT, fixed={"stifness": 1.0})


def test_acceptance_passes_the_true_model_and_fails_a_wrong_one():
    real_step = step_metrics(measured(TRUE, STEP)[20:], 0.3, 0.0, DT)
    real_freq = frequency_response(CHIRP, measured(TRUE, CHIRP), DT, 0.3, 10.0, 10)
    good = TRUE
    bad = DriveFit(stiffness=20, damping=0.5, delay_steps=8, effort_limit=20)
    for model, expect in ((good, True), (bad, False)):
        sim_c = simulate(model, CHIRP, 0.0, PLANT, DT)
        a = acceptance(real_step=real_step, sim_step=step_metrics(measured(model, STEP)[20:], 0.3, 0.0, DT),
                       real_freq=real_freq, sim_freq=frequency_response(CHIRP, sim_c["q"][0], DT, 0.3, 10.0, 10),
                       real_q=measured(TRUE, CHIRP), sim_q=sim_c["q"][0],
                       real_sat=simulate(TRUE, CHIRP, 0.0, PLANT, DT)["saturated"][0], sim_sat=sim_c["saturated"][0])
        assert a.passed is expect, a.checks
    assert not acceptance().passed  # nothing compared: not a pass
    assert Gates().mag_tol_db == 2.0
