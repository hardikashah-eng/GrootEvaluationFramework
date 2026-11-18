"""Real logs to the sim's replay format: which conversions are lossless, which round, which lose information."""

import math

import numpy as np
import pytest

from groot_eval.real_format import convert, plan_lossless, reorder_quat, round_trip, time_resolution


def streams(t0=0.0, n=500, rate=200.0):
    t = t0 + np.arange(n) / rate
    rng = np.random.default_rng(0)
    joints = np.degrees(np.c_[np.sin(t), np.cos(2 * t), 0.3 * np.sin(5 * t)]) + rng.normal(0, 1e-3, (n, 3))
    yaw = 0.5 * np.sin(t)
    quat_wxyz = np.c_[np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)]
    base = np.c_[np.zeros((n, 3)), quat_wxyz]
    return {
        "joint_pos": {"t": t, "values": joints, "names": ["hip", "knee", "ankle"], "unit": "deg"},
        "base_pose": {"t": t, "values": base, "names": ["x", "y", "z", "qw", "qx", "qy", "qz"], "unit": "m",
                      "quat_order": "wxyz", "quat_cols": [3]},
    }


def test_float64_native_timestamps_is_lossless_or_rounding_only():
    src = streams()
    out, rep = convert(src, {**plan_lossless(src), "units": {"joint_pos": "rad"}})
    assert rep.verdict in ("lossless", "rounding")
    assert rep.streams["base_pose"].verdict == "lossless"  # reorder + float64: bit-exact
    units = next(s for s in rep.streams["joint_pos"].steps if s.step == "units")
    assert units.relative_error < 1e-14  # deg -> rad only rounds
    before = src["base_pose"]["values"].copy()
    errs = round_trip(out, src)
    assert np.array_equal(src["base_pose"]["values"], before)
    assert errs["base_pose"] == 0.0 and errs["joint_pos"] < 1e-12
    assert out["base_pose"]["names"][3:] == ["qx", "qy", "qz", "qw"]  # names move with their values
    assert src["base_pose"]["names"][3:] == ["qw", "qx", "qy", "qz"]  # the source log is untouched
    q = out["base_pose"]["values"][:, 3:]
    assert np.allclose(q[:, 3], src["base_pose"]["values"][:, 3])  # w moved to the end (xyzw)


def test_float32_values_round_within_tolerance_but_float32_epoch_time_is_lossy():
    src = streams(t0=1.7e9)  # Unix epoch timestamps
    out, rep = convert(src, {"dtype": "float32", "time_dtype": "float32", "units": {"joint_pos": "rad"},
                             "quat_order": "xyzw"}, tolerance=1e-5)
    steps = {s.step: s for s in rep.streams["joint_pos"].steps}
    assert not steps["value_dtype"].lossy and steps["value_dtype"].relative_error < 1e-5
    assert steps["time_dtype"].lossy and steps["time_dtype"].max_abs_error > 1.0  # seconds lost per sample
    assert rep.verdict == "lossy"
    assert time_resolution(1.7e9) == 128.0
    assert time_resolution(3600.0) == pytest.approx(2.44e-4, rel=0.01)  # an hour-long log in float32


def test_resampling_is_reported_as_lossy_with_its_error():
    src = streams()
    out, rep = convert(src, {"rate_hz": 50.0})
    step = next(s for s in rep.streams["joint_pos"].steps if s.step == "resampling")
    assert step.lossy and step.max_abs_error > 0 and "200.0 Hz -> 50.0 Hz" in step.detail
    assert len(out["joint_pos"]["t"]) < len(src["joint_pos"]["t"])
    assert math.isnan(round_trip(out, src)["joint_pos"])  # samples no longer line up


def test_dropping_columns_is_lossy_and_missing_columns_are_refused():
    src = streams()
    _, rep = convert(src, {"columns": {"joint_pos": ["hip", "knee"]}})
    assert rep.streams["joint_pos"].verdict == "lossy" and rep.streams["joint_pos"].dropped_columns == ["ankle"]
    with pytest.raises(ValueError, match="does not have"):
        convert(src, {"columns": {"joint_pos": ["hip", "wrist"]}})
    plan = plan_lossless(src, {"joint_pos": ["hip", "knee"]})
    assert plan["add_sim_columns"] == {"joint_pos": ["ankle"]} and plan["rate_hz"] is None


def test_units_and_quaternions():
    q = np.array([[1.0, 0.1, 0.2, 0.3]])
    assert np.array_equal(reorder_quat(reorder_quat(q, "wxyz", "xyzw"), "xyzw", "wxyz"), q)
    with pytest.raises(ValueError):
        reorder_quat(q, "wxyz", "zyxw")
    with pytest.raises(ValueError, match="cannot convert"):
        convert({"a": {"t": [0, 1], "values": [[1], [2]], "unit": "deg"}}, {"units": {"a": "m"}})
    out, rep = convert({"a": {"t": [0, 1], "values": [[1000.0], [2000.0]], "unit": "mm"}}, {"units": {"a": "m"}})
    assert out["a"]["values"][:, 0].tolist() == [1.0, 2.0] and rep.verdict in ("lossless", "rounding")
    assert "Overall" in rep.markdown() and rep.to_dict()["verdict"] == rep.verdict
