"""Engine cross-check: tell a replay problem from an Isaac (primary engine) limit from a shared twin-model gap.

The 'real robot' and both engines are MuJoCo pendulums here; the primary engine is given the kind of defect
each verdict is meant to catch (extra solver damping, non-repeatable steps, a lossy state restore).
"""

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from groot_eval.actuators.core import ActuatorParams, FaultTable, WearAwareActuatorCore, WearTable  # noqa: E402
from groot_eval.actuators.drive_chain import DriveChain, DriveChainParams  # noqa: E402
from groot_eval.engine_crosscheck import (MuJoCoEnv, cross_check, cross_check_case, model_parity,  # noqa: E402
                                          repeatability, restore_check)
from groot_eval.types import Scenario  # noqa: E402

DT = 0.01
SLICES = {"qpos": slice(0, 1), "qvel": slice(1, 2)}


def pendulum(damping=0.05, mass=1.0, motor=True):
    act = '<motor joint="j" gear="1" ctrlrange="-50 50"/>' if motor else ""
    return f"""<mujoco><compiler angle="radian"/><option timestep="0.002" gravity="0 0 -9.81"/>
    <worldbody><body name="link"><joint name="j" type="hinge" axis="0 1 0" damping="{damping}" range="-3 3"/>
    <geom type="capsule" fromto="0 0 0 0 0 -0.4" size="0.03" mass="{mass}"/></body></worldbody>
    <actuator>{act}</actuator></mujoco>"""


def fell(model, data):
    return (abs(data.qpos[0]) > 1.2, {"success": False, "failure_reason": "fell"} if abs(data.qpos[0]) > 1.2 else {})


def make(damping=0.05, mass=1.0, **kw):
    return lambda: MuJoCoEnv(pendulum(damping, mass), control_dt=DT, terminate=fell, **kw)


def recording(damping=0.05, T=100):
    """The robot: a 0.4 s torque push that swings the pendulum past 1.2 rad ('fell') at about 0.39 s."""
    env = MuJoCoEnv(pendulum(damping), control_dt=DT)
    o = env.reset(Scenario(seed=0))
    acts = np.r_[np.full(40, 1.5), np.zeros(T - 40)][:, None]
    states = [np.r_[o["state.qpos"], o["state.qvel"]]]  # states[k] is what the robot saw before acts[k]
    for a in acts[:-1]:
        o = env.step(a)[0]
        states.append(np.r_[o["state.qpos"], o["state.qvel"]])
    t = np.arange(T) * DT
    return np.array(states), t, acts, t.copy()


class Noisy(MuJoCoEnv):
    """An engine whose steps are not repeatable (stands in for GPU non-determinism)."""

    def step(self, action):
        self.data.qvel[:] += np.random.normal(0, 1e-6, self.data.qvel.shape)
        return super().step(action)


class LossyRestore(MuJoCoEnv):
    """An engine that cannot restore velocity (stands in for state PhysX cannot set)."""

    def set_state(self, state, recent_actions=None):
        state = {k: v for k, v in state.items() if k != "state.qvel"}
        self.data.qvel[:] = 0.0
        return super().set_state(state, recent_actions)


@pytest.fixture(scope="module")
def rec():
    states, st, acts, at = recording()
    assert abs(states[:, 0]).max() > 1.2  # the robot did fall
    return dict(states=states, state_t=st, slices=SLICES, actions=acts, action_t=at, dt=DT)


def test_full_state_restore_is_exact():
    env = make()()
    env.reset(Scenario(seed=0))
    for _ in range(20):
        env.step([2.0])
    s = env.get_full_state()
    a = [env.step([0.5])[0]["state.qpos"].copy() for _ in range(30)]
    env.set_full_state(s)
    b = [env.step([0.5])[0]["state.qpos"].copy() for _ in range(30)]
    assert np.array_equal(np.array(a), np.array(b))


def test_checks_on_their_own(rec):
    r = repeatability(make(), rec)
    assert r["repeatable"] and r["max_difference"] == 0.0
    assert not repeatability(lambda: Noisy(pendulum(), control_dt=DT, terminate=fell), rec, atol=0.0)["repeatable"]
    assert restore_check(make()(), rec["states"], SLICES)["ok"]
    assert not restore_check(LossyRestore(pendulum(), control_dt=DT), rec["states"], SLICES)["ok"]
    assert model_parity(make()().model_summary(), make(mass=1.5)().model_summary()) == ["body 'link' mass 1.000 vs 1.500 kg"]


@pytest.mark.parametrize("primary, reference, verdict", [
    (make(), make(), "reproduced_in_both"),
    (make(damping=0.8), make(), "primary_engine_limitation"),
    (lambda: Noisy(pendulum(), control_dt=DT, terminate=fell), make(), "replay_not_repeatable"),
    (lambda: LossyRestore(pendulum(), control_dt=DT, terminate=fell), make(), "primary_restore_limitation"),
    (make(), make(mass=2.5), "reference_engine_gap"),
    (make(damping=0.6), make(damping=0.65), "shared_model_gap"),
    (make(damping=5.0), make(damping=0.3), "engine_dependent_gap"),
])
def test_verdicts(rec, primary, reference, verdict):
    cc = cross_check({"isaac": primary, "mujoco": reference}, failure_mode="fell", replay_kwargs=rec)
    assert cc.verdict == verdict, (cc.verdict, cc.explanation, cc.to_dict()["engines"])
    if verdict == "reference_engine_gap":
        assert any("mass" in p for p in cc.parity)


def test_cross_check_case_from_a_failure_record(rec):
    from groot_eval.repro import FailureCase

    fail = int(np.argmax(np.abs(rec["states"][:, 0]) > 1.2))
    case = FailureCase("c1", Scenario(seed=0), "fell", "trace", "real", states=rec["states"], actions=rec["actions"],
                       slices=SLICES, fail_step=fail)
    cc = cross_check_case(case, {"isaac": make(damping=0.8), "mujoco": make()}, dt=DT, lead_s=0.2)
    assert cc.verdict == "primary_engine_limitation"
    with pytest.raises(ValueError):
        cross_check_case(FailureCase("c2", Scenario(seed=0), "fell", "record"), {"isaac": make(), "mujoco": make()},
                         dt=DT)


def test_torque_mode_runs_the_drive_model():
    core = WearAwareActuatorCore(ActuatorParams(effort_limit=30.0, stiffness=40.0, damping=2.0, dt=0.002),
                                 WearTable.new(1, 1), FaultTable.none(1))
    drive = DriveChain(core, DriveChainParams(delay_steps_min=2, delay_steps_max=2, dt=0.002))
    env = MuJoCoEnv(pendulum(), control_dt=DT, control="torque", drive=drive)
    env.reset(Scenario(seed=0))
    for _ in range(100):
        obs = env.step([0.5])[0]
    assert obs["state.qpos"][0] == pytest.approx(0.5, abs=0.15)  # holds near the target against gravity
    with pytest.raises(ValueError):
        MuJoCoEnv(pendulum(), control="torque")


def test_param_setters_apply_per_episode_and_reject_unknown():
    def set_damping(m, v):
        m.dof_damping[0] = v

    env = MuJoCoEnv(pendulum(), control_dt=DT, param_setters={"damping": set_damping})
    env.reset(Scenario(seed=0, params={"damping": 3.0}))
    assert env.model.dof_damping[0] == 3.0
    env.reset(Scenario(seed=0))
    assert env.model.dof_damping[0] == pytest.approx(0.05)  # rebuilt: no carry-over
    with pytest.raises(KeyError):
        env.reset(Scenario(seed=0, params={"friction": 1.0}))
