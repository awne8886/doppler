"""
Trajectory conditioning for real ADS-B tracks.

Real tracks are irregularly sampled and jittery (CPR position quantisation,
25 ft altitude steps, dropped messages), and differentiating raw positions
gives noisy velocities that feed straight into the Doppler model.  This module
resamples each track to a regular 1 s grid with **Gaussian-kernel local
quadratic regression**, which returns a smoothed position *and* its time
derivative at every output time.  A quadratic (rather than linear) local model
keeps the bias in turns small: the leading bias term of a local-linear fit is
``h^2/2 * acceleration`` (about 9 m for a 1.5 deg/s turn at h = 2 s), while the
quadratic fit cancels it.

Reported kinematics are fused where present: ground speed + track give a
horizontal velocity vector and vertical rate gives the vertical component.
They are kernel-smoothed onto the same grid and blended with the
position-derived velocity (``reported_weight``, default 0.7, because ADS-B
velocity reports are less noisy than differentiated positions).

Tracks are split at gaps longer than ``max_gap_s``; no data is invented inside
a gap, so the matcher's existing ``max_gap_s`` support check still applies.

The algorithm is deliberately simple so the browser implementation
(``doppler_analysis.js``: ``conditionTrajectory``) can mirror it exactly.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

EARTH_R = 6371000.0


def _bandwidth(dt_med: float, bandwidth_s: Optional[float]) -> float:
    if bandwidth_s is not None:
        return float(bandwidth_s)
    # 2 s at 1 Hz: ~1 m/s velocity noise from 6 m position jitter, <1 m/s bias in a 3 deg/s turn.
    return float(min(30.0, max(2.0, 2.0 * dt_med)))


def local_quadratic(t: np.ndarray, y: np.ndarray, tq: np.ndarray, h: float):
    """
    Gaussian-weighted local quadratic fit of samples (t, y) evaluated at tq.
    Returns (value, derivative).  Falls back to local linear / constant when
    fewer than three samples carry weight.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    tq = np.asarray(tq, dtype=np.float64)
    val = np.empty(tq.size)
    der = np.empty(tq.size)
    reach = 4.0 * h
    chunk = 256
    for c0 in range(0, tq.size, chunk):
        q = tq[c0:c0 + chunk]
        lo = int(np.searchsorted(t, q[0] - reach, side="left"))
        hi = int(np.searchsorted(t, q[-1] + reach, side="right"))
        d = t[None, lo:hi] - q[:, None]  # (Q, N)
        w = np.exp(-0.5 * (d / h) ** 2) * (np.abs(d) <= reach)
        yy = y[None, lo:hi]
        s0, s1, s2 = w.sum(1), (w * d).sum(1), (w * d * d).sum(1)
        s3, s4 = (w * d ** 3).sum(1), (w * d ** 4).sum(1)
        r0, r1, r2 = (w * yy).sum(1), (w * d * yy).sum(1), (w * d * d * yy).sum(1)
        a, b = _solve_quadratic(s0, s1, s2, s3, s4, r0, r1, r2)
        val[c0:c0 + q.size] = a
        der[c0:c0 + q.size] = b
    return val, der


def _solve_quadratic(s0, s1, s2, s3, s4, r0, r1, r2):
    """Solve [[s0,s1,s2],[s1,s2,s3],[s2,s3,s4]] @ [a,b,c] = [r0,r1,r2] per row (Cramer's rule), with fallbacks."""
    det = s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s3 * s2) + s2 * (s1 * s3 - s2 * s2)
    scale = np.maximum(np.abs(s0 * s2 * s4), 1e-300)
    ok3 = np.abs(det) > 1e-9 * scale
    safe = np.where(ok3, det, 1.0)
    a3 = (r0 * (s2 * s4 - s3 * s3) - s1 * (r1 * s4 - s3 * r2) + s2 * (r1 * s3 - s2 * r2)) / safe
    b3 = (s0 * (r1 * s4 - s3 * r2) - r0 * (s1 * s4 - s3 * s2) + s2 * (s1 * r2 - r1 * s2)) / safe
    det2 = s0 * s2 - s1 * s1
    ok2 = np.abs(det2) > 1e-9 * np.maximum(np.abs(s0 * s2), 1e-300)
    safe2 = np.where(ok2, det2, 1.0)
    a2 = (s2 * r0 - s1 * r1) / safe2
    b2 = (s0 * r1 - s1 * r0) / safe2
    a1 = r0 / np.maximum(s0, 1e-300)
    a = np.where(ok3, a3, np.where(ok2, a2, a1))
    b = np.where(ok3, b3, np.where(ok2, b2, 0.0))
    return a, b


def _segments(t: np.ndarray, max_gap_s: float) -> List[np.ndarray]:
    breaks = np.where(np.diff(t) > max_gap_s)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [t.size]])
    return [np.arange(a, b) for a, b in zip(starts, ends)]


def condition_trajectory(traj: Dict[str, Any], max_gap_s: float = 20.0, dt: float = 1.0,
                         bandwidth_s: Optional[float] = None, reported_weight: float = 0.7) -> Dict[str, Any]:
    """
    Return a conditioned copy of ``traj``: regular ``dt`` sampling inside each
    gap-free segment, smoothed positions, fused velocities in
    ``velocity_east/north/up`` [m/s] and derived ``speeds``/``headings``/
    ``vertical_rates``.  Provenance fields are preserved and a ``conditioning``
    summary is attached.
    """
    t_raw = np.asarray(traj["timestamps"], dtype=np.float64)
    order = np.argsort(t_raw, kind="stable")
    t = t_raw[order]
    keep = np.concatenate([[True], np.diff(t) > 1e-6])
    idx = order[keep]
    t = t[keep]
    if t.size < 2:
        return dict(traj)
    lat = np.asarray(traj["latitudes"], dtype=np.float64)[idx]
    lon = np.asarray(traj["longitudes"], dtype=np.float64)[idx]
    alt = np.asarray(traj["altitudes"], dtype=np.float64)[idx]
    lat0, lon0 = float(lat[0]), float(lon[0])
    x = EARTH_R * np.radians(lon - lon0) * np.cos(np.radians(0.5 * (lat + lat0)))
    y = EARTH_R * np.radians(lat - lat0)

    def reported(key):
        arr = traj.get(key) or []
        if len(arr) != t_raw.size or any(v is None for v in arr):
            return None
        return np.asarray(arr, dtype=np.float64)[idx]

    gs, trk, vr = reported("speeds"), reported("headings"), reported("vertical_rates")
    has_h = gs is not None and trk is not None
    if has_h:
        ve_rep = gs * np.sin(np.radians(trk))
        vn_rep = gs * np.cos(np.radians(trk))

    out_t, out_x, out_y, out_z, out_ve, out_vn, out_vu = ([] for _ in range(7))
    n_seg = 0
    hs = []
    for seg in _segments(t, max_gap_s):
        ts = t[seg]
        if ts.size < 2:
            continue
        n_seg += 1
        n_out = int(math.floor((ts[-1] - ts[0]) / dt + 1e-9)) + 1
        tg = ts[0] + dt * np.arange(n_out)
        if tg[-1] < ts[-1] - 1e-6:
            tg = np.append(tg, ts[-1])
        h = _bandwidth(float(np.median(np.diff(ts))), bandwidth_s)
        hs.append(h)
        px, dx = local_quadratic(ts, x[seg], tg, h)
        py, dy = local_quadratic(ts, y[seg], tg, h)
        pz, dz = local_quadratic(ts, alt[seg], tg, h)
        if has_h:
            rve, _ = local_quadratic(ts, ve_rep[seg], tg, h)
            rvn, _ = local_quadratic(ts, vn_rep[seg], tg, h)
            dx = reported_weight * rve + (1.0 - reported_weight) * dx
            dy = reported_weight * rvn + (1.0 - reported_weight) * dy
        if vr is not None:
            rvz, _ = local_quadratic(ts, vr[seg], tg, h)
            dz = reported_weight * rvz + (1.0 - reported_weight) * dz
        for acc, v in ((out_t, tg), (out_x, px), (out_y, py), (out_z, pz), (out_ve, dx), (out_vn, dy), (out_vu, dz)):
            acc.append(v)
    if not out_t:
        return dict(traj)
    T, X, Y, Z = (np.concatenate(a) for a in (out_t, out_x, out_y, out_z))
    VE, VN, VU = (np.concatenate(a) for a in (out_ve, out_vn, out_vu))
    LAT = lat0 + np.degrees(Y / EARTH_R)
    LON = lon0 + np.degrees(X / (EARTH_R * np.cos(np.radians(0.5 * (LAT + lat0)))))
    out = {k: v for k, v in traj.items() if k not in ("timestamps", "latitudes", "longitudes", "altitudes", "speeds",
                                                     "headings", "vertical_rates")}
    out.update({
        "timestamps": [round(float(v), 3) for v in T],
        "latitudes": [round(float(v), 7) for v in LAT],
        "longitudes": [round(float(v), 7) for v in LON],
        "altitudes": [round(float(v), 2) for v in Z],
        "velocity_east": [round(float(v), 3) for v in VE],
        "velocity_north": [round(float(v), 3) for v in VN],
        "velocity_up": [round(float(v), 3) for v in VU],
        "speeds": [round(float(v), 2) for v in np.hypot(VE, VN)],
        "headings": [round(float(v) % 360.0, 2) for v in np.degrees(np.arctan2(VE, VN))],
        "vertical_rates": [round(float(v), 3) for v in VU],
        "conditioning": {
            "method": "gaussian_local_quadratic",
            "bandwidth_s": round(float(np.median(hs)), 3) if hs else None,
            "resample_dt_s": dt,
            "segments": n_seg,
            "max_gap_s": max_gap_s,
            "fused_reported_velocity": bool(has_h),
            "fused_reported_vertical_rate": vr is not None,
            "reported_weight": reported_weight,
            "input_points": int(t_raw.size),
            "output_points": int(T.size),
        },
    })
    return out
