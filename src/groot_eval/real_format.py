"""Real robot logs to the sim's replay format, with a report of exactly what was lost.

Numeric robot data can usually reach the sim without losing information, but only if each step of the
conversion is chosen to be reversible. This module converts log streams and measures every step:

======================  ==========================================================  =================================
step                    what can be lost                                            lossless when
======================  ==========================================================  =================================
joint / column mapping  columns the sim has no slot for are dropped                 every source column is mapped
unit conversion         nothing but rounding (deg -> rad, mm -> m)                  always, to ~1e-16 relative
quaternion order        nothing (w-first <-> w-last is a reorder)                   always; the sign flip q -> -q
                                                                                    is the same rotation
value precision         float64 -> float32 keeps ~7 significant digits              float64 kept, or the float32
                                                                                    error is far below sensor noise
timestamps              float32 seconds resolve ~0.24 ms at one hour and ~2 min      timestamps kept as float64
                        for Unix epoch time
resampling              any change of rate interpolates, smooths the fast parts      native timestamps kept (no
                        and aliases anything above the new Nyquist frequency         resampling); replay on the log
                                                                                    clock instead
images                  codec loss, colour order, resizing                          raw or lossless codec; checked by
                                                                                    :mod:`groot_eval.conversion_fidelity`
======================  ==========================================================  =================================

Hidden state that the robot never logged (contact forces, motor temperature, how an object sits in the
gripper) is not in the data at all, so no conversion can recover it.

:func:`convert` returns the converted streams and a :class:`LossReport` with a verdict per stream:
``lossless`` (exact round trip), ``rounding`` (only float rounding, below ``tolerance`` of the signal's
spread) or ``lossy`` (information lost; the report says which step). :func:`round_trip` converts back.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

UNIT_SCALE = {  # unit -> (base unit, factor to base)
    "rad": ("rad", 1.0), "deg": ("rad", math.pi / 180.0),
    "m": ("m", 1.0), "mm": ("m", 1e-3), "cm": ("m", 1e-2),
    "s": ("s", 1.0), "ms": ("s", 1e-3), "us": ("s", 1e-6), "ns": ("s", 1e-9),
    "rad/s": ("rad/s", 1.0), "deg/s": ("rad/s", math.pi / 180.0),
    "m/s": ("m/s", 1.0), "mm/s": ("m/s", 1e-3),
    "N*m": ("N*m", 1.0), "Nm": ("N*m", 1.0), "mN*m": ("N*m", 1e-3),
    "N": ("N", 1.0), "kg": ("kg", 1.0), "g": ("kg", 1e-3),
    "1": ("1", 1.0),
}


@dataclass
class StepLoss:
    step: str
    lossy: bool
    max_abs_error: float = 0.0
    relative_error: float = 0.0  # max error / signal spread
    detail: str = ""


@dataclass
class StreamReport:
    name: str
    verdict: str  # lossless | rounding | lossy
    steps: list[StepLoss] = field(default_factory=list)
    dropped_columns: list[str] = field(default_factory=list)


@dataclass
class LossReport:
    streams: dict[str, StreamReport]

    @property
    def verdict(self) -> str:
        order = ["lossless", "rounding", "lossy"]
        return max((s.verdict for s in self.streams.values()), key=order.index, default="lossless")

    def to_dict(self) -> dict[str, Any]:
        return {"verdict": self.verdict, "streams": {
            k: {"verdict": v.verdict, "dropped_columns": v.dropped_columns,
                "steps": [s.__dict__ for s in v.steps]} for k, v in self.streams.items()}}

    def markdown(self) -> str:
        lines = [f"Overall: **{self.verdict}**", "", "| Stream | Verdict | Step | Max error | Of spread | Note |",
                 "|---|---|---|---|---|---|"]
        for k, v in self.streams.items():
            for s in v.steps:
                lines.append(f"| {k} | {v.verdict} | {s.step} | {s.max_abs_error:.3g} | {s.relative_error:.2e} "
                             f"| {s.detail} |")
        return "\n".join(lines) + "\n"


def _spread(x: np.ndarray) -> float:
    s = float(np.nanstd(x)) if x.size else 0.0
    return s if s > 1e-12 else 1.0


def _unit_factor(src: str, dst: str) -> float:
    if src not in UNIT_SCALE or dst not in UNIT_SCALE:
        raise ValueError(f"unknown unit {src!r} or {dst!r}")
    (bs, fs), (bd, fd) = UNIT_SCALE[src], UNIT_SCALE[dst]
    if bs != bd:
        raise ValueError(f"cannot convert {src} to {dst}")
    return fs / fd


def reorder_quat(q: np.ndarray, src: str, dst: str) -> np.ndarray:
    """Reorder quaternions between ``wxyz`` and ``xyzw`` (last axis of length 4)."""
    if src == dst:
        return q
    if {src, dst} != {"wxyz", "xyzw"}:
        raise ValueError("quaternion order is 'wxyz' or 'xyzw'")
    return np.roll(q, -1 if src == "wxyz" else 1, axis=-1)


def convert(streams: Mapping[str, Mapping[str, Any]], target: Mapping[str, Any], tolerance: float = 1e-6
            ) -> tuple[dict[str, dict[str, Any]], LossReport]:
    """Convert log streams to the sim's format and measure what each step lost.

    ``streams[name]`` = ``{"t": (T,), "values": (T, D), "names": [D column names], "unit": "deg",
    "time_unit": "s", "quat_order": "wxyz" | None, "quat_cols": [start of each quaternion]}``.
    ``target`` = ``{"dtype": "float32" | "float64", "time_dtype": ..., "rate_hz": None or Hz,
    "units": {stream: unit}, "quat_order": "xyzw", "columns": {stream: [names in sim order]}}``.
    """
    out: dict[str, dict[str, Any]] = {}
    reports: dict[str, StreamReport] = {}
    vdt = np.dtype(target.get("dtype", "float64"))
    tdt = np.dtype(target.get("time_dtype", "float64"))
    rate = target.get("rate_hz")
    for name, s in streams.items():
        steps: list[StepLoss] = []
        t = np.asarray(s["t"], np.float64) * _unit_factor(s.get("time_unit", "s"), "s")
        x = np.atleast_2d(np.array(s["values"], dtype=np.float64, copy=True))  # never modify the caller's log
        if x.shape[0] != len(t):
            x = x.T
        names = list(s.get("names") or [f"c{i}" for i in range(x.shape[1])])
        spread = np.array([_spread(x[:, i]) for i in range(x.shape[1])])
        dropped: list[str] = []
        # 1. column mapping
        want = (target.get("columns") or {}).get(name)
        if want is not None:
            missing = [c for c in want if c not in names]
            if missing:
                raise ValueError(f"{name}: sim expects columns the log does not have: {missing}")
            dropped = [c for c in names if c not in want]
            idx = [names.index(c) for c in want]
            x, spread, names = x[:, idx], spread[idx], list(want)
            steps.append(StepLoss("mapping", bool(dropped), detail=f"dropped {dropped}" if dropped else "all mapped"))
        # 2. units
        dst_unit = (target.get("units") or {}).get(name, s.get("unit", "1"))
        f = _unit_factor(s.get("unit", "1"), dst_unit)
        if f != 1.0:
            y = x * f
            back = y / f
            err = float(np.nanmax(np.abs(back - x))) if x.size else 0.0
            steps.append(StepLoss("units", False, err, float(np.nanmax(np.abs(back - x) / spread)) if x.size else 0.0,
                                  f"{s.get('unit')} -> {dst_unit}"))
            x = y
            spread = spread * abs(f)
        # 3. quaternion order
        qo, qdst = s.get("quat_order"), target.get("quat_order")
        if qo and qdst and qo != qdst:
            for c in s.get("quat_cols", []):
                x[:, c:c + 4] = reorder_quat(x[:, c:c + 4], qo, qdst)
                names[c:c + 4] = list(reorder_quat(np.array(names[c:c + 4], dtype=object), qo, qdst))
            steps.append(StepLoss("quaternion_order", False, detail=f"{qo} -> {qdst} (exact reorder)"))
        # 4. resampling
        if rate:
            grid = np.arange(t[0], t[-1] + 1e-12, 1.0 / float(rate))
            y = np.stack([np.interp(grid, t, x[:, i]) for i in range(x.shape[1])], axis=1)
            back = np.stack([np.interp(t, grid, y[:, i]) for i in range(x.shape[1])], axis=1)
            rs_err = np.abs(back - x)
            native = 1.0 / float(np.median(np.diff(t))) if len(t) > 1 else float(rate)
            steps.append(StepLoss("resampling", True, float(np.nanmax(rs_err)), float(np.nanmax(rs_err / spread)),
                                  f"{native:.1f} Hz -> {float(rate):.1f} Hz (interpolated; replay on the log clock "
                                  "keeps native timestamps)"))
            t, x = grid, y
        # 5. value precision
        xv = x.astype(vdt)
        dt_err = np.abs(xv.astype(np.float64) - x)
        rel = float(np.nanmax(dt_err / spread)) if x.size else 0.0
        steps.append(StepLoss("value_dtype", rel > tolerance, float(np.nanmax(dt_err)) if x.size else 0.0, rel,
                              f"float64 -> {vdt.name}"))
        # 6. timestamp precision
        tv = t.astype(tdt)
        terr = float(np.max(np.abs(tv.astype(np.float64) - t))) if len(t) else 0.0
        period = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
        tlossy = terr > 0.01 * period
        steps.append(StepLoss("time_dtype", tlossy, terr, terr / period,
                              f"float64 -> {tdt.name}; worst error {terr * 1e3:.4f} ms vs period {period * 1e3:.3f} ms"))
        lossy = any(st.lossy for st in steps)
        exact = all(st.max_abs_error == 0.0 for st in steps)
        verdict = "lossy" if lossy else ("lossless" if exact else "rounding")
        reports[name] = StreamReport(name, verdict, steps, dropped)
        out[name] = {"t": tv, "values": xv, "names": names, "unit": dst_unit,
                     "quat_order": qdst if qo else None, "quat_cols": list(s.get("quat_cols", []))}
    return out, LossReport(reports)


def round_trip(converted: Mapping[str, Mapping[str, Any]], original: Mapping[str, Mapping[str, Any]]
               ) -> dict[str, float]:
    """Convert back to each original stream's units and quaternion order; max abs error per stream (NaN if the
    columns or samples no longer line up, e.g. after dropping columns or resampling)."""
    errs: dict[str, float] = {}
    for name, c in converted.items():
        o = original[name]
        ox = np.atleast_2d(np.asarray(o["values"], np.float64))
        onames = list(o.get("names") or [f"c{i}" for i in range(ox.shape[1])])
        x = np.asarray(c["values"], np.float64).copy()
        names = list(c["names"])
        if c.get("quat_order") and o.get("quat_order") and c["quat_order"] != o["quat_order"]:
            for q in c.get("quat_cols", []):
                x[:, q:q + 4] = reorder_quat(x[:, q:q + 4], c["quat_order"], o["quat_order"])
                names[q:q + 4] = list(reorder_quat(np.array(names[q:q + 4], dtype=object), c["quat_order"],
                                                   o["quat_order"]))
        if names != onames or x.shape != ox.shape:
            errs[name] = math.nan
            continue
        x = x * _unit_factor(c["unit"], o.get("unit", "1"))
        errs[name] = float(np.max(np.abs(x - ox))) if x.size else 0.0
    return errs


def time_resolution(t_max_s: float, dtype: str = "float32") -> float:
    """Spacing of representable timestamps near ``t_max_s`` seconds: how coarse a time column becomes."""
    return float(np.spacing(np.array(t_max_s, dtype=np.dtype(dtype))))


def plan_lossless(streams: Mapping[str, Mapping[str, Any]], sim_columns: Mapping[str, Sequence[str]] | None = None
                  ) -> dict[str, Any]:
    """The target settings that convert these streams without loss: float64 values and times, native timestamps,
    every column mapped (lists the columns the sim would need to add)."""
    unmapped = {}
    for name, s in streams.items():
        names = list(s.get("names") or [])
        want = (sim_columns or {}).get(name)
        if want is not None:
            extra = [c for c in names if c not in want]
            if extra:
                unmapped[name] = extra
    return {"dtype": "float64", "time_dtype": "float64", "rate_hz": None, "quat_order": "xyzw",
            "add_sim_columns": unmapped}
