"""Log-clock replay: log checks, timestamp scheduling, restore point, three modes, replay as a distribution."""

import numpy as np
import pytest

from groot_eval.timeline_replay import (build_schedule, check_replay_log, replay_distribution, replay_on_log_clock,
                                        restore_index)
from groot_eval.types import Scenario

PHYS_DT = 0.001
SLICES = {"pos": slice(0, 1), "vel": slice(1, 2)}


class Mass1D:
    """A unit mass pushed by a force; fails ("overshot") when it passes x = limit. Supports set_state."""

    def __init__(self, dt=PHYS_DT, limit=0.5):
        self.dt, self.limit, self.max_steps = dt, limit, 100000
        self.mass, self.friction = 1.0, 0.0

    def reset(self, scenario):
        self.mass = float(scenario.params.get("mass", 1.0))
        self.friction = float(scenario.params.get("friction", 0.0))
        self.x = np.zeros(2)
        return self._obs()

    def set_state(self, state, recent_actions=None):
        self.x = np.array([float(state["state.pos"][0]), float(state["state.vel"][0])])
        return self._obs()

    def step(self, action):
        f = float(np.asarray(action).reshape(-1)[0]) - self.friction * self.x[1]
        self.x[1] += self.dt * f / self.mass
        self.x[0] += self.dt * self.x[1]
        done = self.x[0] > self.limit
        return self._obs(), 0.0, done, False, ({"success": False, "failure_reason": "overshot"} if done else {})

    def _obs(self):
        return {"state.pos": self.x[:1].copy(), "state.vel": self.x[1:].copy(), "video.cam": np.full((2, 2), self.x[0])}


def real_run(seed=0, T=1.0, period=0.02, mass=1.0, limit=10.0):
    """The 'robot': commands every ~20 ms with jitter, one late command and one duplicate; states logged at 100 Hz."""
    rng = np.random.default_rng(seed)
    at = np.cumsum(np.r_[0.0, period + rng.normal(0, 0.003, int(T / period) - 1)])
    at[20] += 0.03  # a late command
    at = np.sort(np.r_[at, at[30] + 0.0004])  # a duplicate in the same physics step
    acts = (np.sin(np.arange(len(at)) * 0.3) * 2.0)[:, None]
    env = Mass1D(limit=limit)
    env.reset(Scenario(seed=0, params={"mass": mass}))
    sched = build_schedule(at, acts, PHYS_DT, t0=0.0, t_end=at[-1] + 0.05)
    st, states = [], []
    for k in range(sched.n_steps):
        if k % 10 == 0:
            st.append(k * PHYS_DT)
            states.append(env.x.copy())
        cmd = sched.action_at(k)
        env.step(acts[0] if cmd is None else cmd)
    return np.array(states), np.array(st), acts, at


def good_log(**over):
    log = {"streams": {"policy": {"t": [0, 1]}, "action": {"t": [0.1, 0.9]}, "joint": {"t": [0.0, 1.0]}},
           "clock_offsets_s": {"policy": 0.0, "action": 0.0002, "joint": -0.0001},
           "versions": {k: "x" for k in ("checkpoint", "deployed_build", "firmware", "gains", "robot_profile",
                                         "twin_assets")}}
    log.update(over)
    return log


# -- log checks ----------------------------------------------------------------------------
def test_good_log_passes():
    assert check_replay_log(good_log(), max_skew_s=0.001).ok


@pytest.mark.parametrize("change, expect", [
    (lambda log: log["streams"].pop("joint"), "joint.present"),
    (lambda log: log["streams"]["action"].update(t=[0.5, 0.4]), "action.time"),
    (lambda log: log["streams"]["joint"].update(t=[0.2, 0.5]), "cover.joint"),
    (lambda log: log.pop("clock_offsets_s"), "clock.sync"),
    (lambda log: log["clock_offsets_s"].update(action=0.004), "clock.skew"),
    (lambda log: log["clock_offsets_s"].pop("joint"), "clock.sync"),
    (lambda log: log["versions"].pop("firmware"), "versions"),
])
def test_bad_logs_are_refused(change, expect):
    log = good_log()
    change(log)
    c = check_replay_log(log, max_skew_s=0.001)
    assert not c.ok and any(e.startswith(expect) for e in c.errors()), c.errors()


# -- scheduling ------------------------------------------------------------------------------
def test_schedule_keeps_real_timing():
    t = [0.0, 0.020, 0.041, 0.0412, 0.100]  # jitter, a duplicate in one step, a 59 ms gap
    a = np.arange(5.0)[:, None]
    s = build_schedule(t, a, dt=0.01, nominal_period=0.02)
    assert s.steps.tolist() == [0, 2, 4, 10]  # the duplicate's later command wins in step 4
    assert s.action_at(3)[0] == 1.0 and s.action_at(4)[0] == 3.0 and s.action_at(9)[0] == 3.0
    assert s.stats() == {"commands": 4, "late": 1, "duplicates": 1, "missing_ticks": 2, "steps": 11}
    assert build_schedule([0.5], [[1.0]], 0.01, t0=0.0).action_at(10) is None
    with pytest.raises(ValueError):
        build_schedule([0.1, 0.0], [[0], [1]], 0.01)


def test_restore_index():
    t = np.arange(0, 2.0, 0.1)
    assert restore_index(t, t_fail=1.55, lead_s=0.5) == 10
    assert restore_index(t, t_fail=0.2, lead_s=1.0) == 0


# -- replay modes ------------------------------------------------------------------------------
def test_action_replay_on_log_clock_matches_and_beats_one_command_per_step():
    states, st, acts, at = real_run()
    kw = dict(states=states, state_t=st, slices=SLICES, actions=acts, action_t=at, dt=PHYS_DT)
    res = replay_on_log_clock(Mass1D(limit=10.0), **kw)
    assert res.nrmse < 1e-9 and res.divergence_t is None and res.synced
    assert res.schedule["late"] >= 1 and res.schedule["duplicates"] == 1
    # the naive replay: commands spaced evenly at the nominal period, ignoring the log's timestamps
    even = np.arange(len(at)) * 0.02
    naive = replay_on_log_clock(Mass1D(limit=10.0), **{**kw, "action_t": even})
    assert naive.nrmse > 100 * max(res.nrmse, 1e-12)


def test_replay_from_restore_point_before_failure():
    states, st, acts, at = real_run(limit=10.0)
    i = restore_index(st, t_fail=0.8, lead_s=0.3)
    res = replay_on_log_clock(Mass1D(limit=10.0), states=states, state_t=st, slices=SLICES, actions=acts,
                              action_t=at, dt=PHYS_DT, start_index=i)
    assert res.times[0] == pytest.approx(st[i]) and res.nrmse < 1e-6


def test_failure_outcome_and_settling():
    states, st, acts, at = real_run()
    lim = float(np.nanmax(states[:, 0])) * 0.9  # the twin's limit sits inside the real motion: it ends early
    res = replay_on_log_clock(Mass1D(limit=lim), states=states, state_t=st, slices=SLICES, actions=acts,
                              action_t=at, dt=PHYS_DT, settle_steps=5)
    assert res.ended_as == "overshot" and res.ended_at_t is not None


def test_kinematic_mode_reads_back_states_and_keeps_frames():
    states, st, acts, at = real_run()
    res = replay_on_log_clock(Mass1D(limit=10.0), states=states, state_t=st, slices=SLICES, actions=acts,
                              action_t=at, dt=PHYS_DT, mode="kinematic", start_index=5)
    assert res.nrmse == pytest.approx(0.0, abs=1e-12) and len(res.frames) == len(st) - 5
    assert res.frames[0]["video.cam"][0, 0] == pytest.approx(states[5, 0])


def test_observation_mode_compares_policy_with_logged_commands():
    states, st, acts, at = real_run()
    obs = [{"state.pos": s[:1], "i": i} for i, s in enumerate(states)]
    sched = build_schedule(at, acts, PHYS_DT)
    perfect = lambda o: sched.action_at(int(np.floor(st[o["i"]] / PHYS_DT + 1e-9)))  # noqa: E731
    res = replay_on_log_clock(None, states=states, state_t=st, slices=SLICES, actions=acts, action_t=at,
                              dt=PHYS_DT, mode="observation", policy=perfect, observations=obs, obs_t=st)
    assert res.nrmse == pytest.approx(0.0, abs=1e-12)
    off = replay_on_log_clock(None, states=states, state_t=st, slices=SLICES, actions=acts, action_t=at,
                              dt=PHYS_DT, mode="observation", policy=lambda o: perfect(o) + 1.0,
                              observations=obs, obs_t=st)
    assert off.nrmse > 0.5
    with pytest.raises(ValueError):
        replay_on_log_clock(None, states=states, state_t=st, slices=SLICES, actions=acts, action_t=at,
                            dt=PHYS_DT, mode="observation")


# -- distribution -------------------------------------------------------------------------------
def test_replay_distribution_rate_and_which_parameter_matters():
    states, st, acts, at = real_run(mass=1.0)
    peak = float(np.nanmax(states[:, 0]))
    # the real robot failed; the twin fails only when the mass is light enough to travel past `peak`
    res = replay_distribution(lambda: Mass1D(limit=peak * 0.999), failure_mode="overshot",
                              uncertain={"mass": (0.8, 1.2), "friction": (0.0, 0.01)}, runs=40, seed=1,
                              states=states, state_t=st, slices=SLICES, actions=acts, action_t=at, dt=PHYS_DT)
    assert 0.2 < res.rate < 0.8
    assert res.ci[0] < res.rate < res.ci[1]
    assert res.param_effect["mass"] < -0.3  # lighter twins reproduce it
    assert sum(res.mode_counts.values()) == 40 and set(res.mode_counts) <= {"overshot", "none"}
    assert res.to_dict()["runs"] == 40
