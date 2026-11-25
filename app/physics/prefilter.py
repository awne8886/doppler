"""
Kinematic and time-window pre-filter.

Before the expensive grid search, discard candidate flights that *cannot*
have produced any of the recorded Doppler events, and report why.  A flight is
discarded only when it is excluded for **every** event group (it might still
explain one group even if it cannot explain another).  Per group a flight is
excluded when either

* ``no_time_overlap``: the recording clock is trusted (``clock_window``) and
  the group's absolute receive times cannot fall inside the flight's possible
  arrival interval ``[first emission, last emission + max_delay_s]``; or
* ``speed_incompatible`` / ``height_incompatible``: the group's constant-
  velocity flyover fit is *reliable* and its fitted speed / CPA slant range is
  inconsistent with the flight (generous tolerances, see below).

Reliability of a fit (all must hold): at least 8 samples, RMSE <= 4 Hz,
speed and slant not at their search bounds.  Unreliable fits never exclude.

Tolerances are deliberately wide because the fit assumes a straight, constant
speed pass in still air while ADS-B reports ground speed:

* speed: ``max(30 m/s, 25 % of fitted speed)``;
* height: the flight's lowest height above the observer must not exceed the
  fitted CPA slant range + ``max(800 m, 60 % of fitted slant)``.

These values are fixed (they do not use fit covariances) so the browser
implementation makes identical decisions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

SPEED_TOL_MIN = 30.0
SPEED_TOL_FRAC = 0.25
SLANT_TOL_MIN = 800.0
SLANT_TOL_FRAC = 0.6
MAX_DELAY_S = 200.0
FIT_MAX_RMSE_HZ = 4.0
FIT_MIN_SAMPLES = 8


def fit_is_reliable(fit: Optional[Dict[str, Any]]) -> bool:
    if not fit:
        return False
    at = fit.get("parameter_at_bound") or {}
    v, d = float(fit.get("speed_mps", 0.0)), float(fit.get("cpa_slant_m", 0.0))
    at_bound = bool(at.get("speed") or at.get("distance")) or not (15.5 < v < 329.5) or not (31.0 < d < 39990.0)
    return (int(fit.get("n_samples", 0)) >= FIT_MIN_SAMPLES and float(fit.get("rmse_hz", 1e9)) <= FIT_MAX_RMSE_HZ
            and not at_bound)


def flight_ground_speed(traj: Dict[str, Any]) -> Optional[float]:
    speeds = [s for s in (traj.get("speeds") or []) if s is not None]
    if speeds:
        return float(np.median(speeds))
    t = np.asarray(traj.get("timestamps") or [], dtype=np.float64)
    if t.size < 2:
        return None
    lat = np.radians(np.asarray(traj["latitudes"], dtype=np.float64))
    lon = np.radians(np.asarray(traj["longitudes"], dtype=np.float64))
    dx = 6371000.0 * np.diff(lon) * np.cos(0.5 * (lat[1:] + lat[:-1]))
    dy = 6371000.0 * np.diff(lat)
    dt = np.diff(t)
    ok = dt > 0
    if not np.any(ok):
        return None
    return float(np.median(np.hypot(dx[ok], dy[ok]) / dt[ok]))


def _group_spans(raw_events: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[float, float, List[str]]]:
    spans: Dict[str, Tuple[float, float, List[str]]] = {}
    for i, ev in enumerate(raw_events):
        pts = ev.get("traced_curve") or ev.get("points") or []
        if not pts:
            continue
        eid = str(ev.get("id") or f"E{i + 1}")
        gid = str(ev.get("group_id") or eid)
        lo, hi = min(p["time"] for p in pts), max(p["time"] for p in pts)
        if gid in spans:
            a, b, ids = spans[gid]
            spans[gid] = (min(a, lo), max(b, hi), ids + [eid])
        else:
            spans[gid] = (lo, hi, [eid])
    return spans


def prefilter_flights(flights: Sequence[Dict[str, Any]], raw_events: Sequence[Dict[str, Any]],
                      flyover_fits: Dict[str, Dict[str, Any]], clock_window: Optional[Tuple[float, float]] = None,
                      observer_alt_m: float = 0.0) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(kept, discarded)``; each discarded entry lists per-group reasons."""
    spans = _group_spans(raw_events)
    # The most informative fit per group: the reliable one with most samples.
    group_fit: Dict[str, Dict[str, Any]] = {}
    for gid, (_, _, ids) in spans.items():
        fits = [flyover_fits[e] for e in ids if fit_is_reliable(flyover_fits.get(e))]
        if fits:
            group_fit[gid] = max(fits, key=lambda f: f.get("n_samples", 0))
    kept, discarded = [], []
    for f in flights:
        reasons = []
        t = f.get("timestamps") or []
        vg = flight_ground_speed(f)
        alts = [a for a in (f.get("altitudes") or []) if a is not None]
        min_h = (min(alts) - observer_alt_m) if alts else None
        excluded_all = bool(spans)
        for gid, (lo, hi, _) in spans.items():
            why = []
            if clock_window is not None and t:
                arr_lo, arr_hi = float(t[0]), float(t[-1]) + MAX_DELAY_S
                if clock_window[1] + hi < arr_lo or clock_window[0] + lo > arr_hi:
                    why.append({"reason": "no_time_overlap",
                                "event_window": [round(clock_window[0] + lo, 1), round(clock_window[1] + hi, 1)],
                                "flight_arrival_window": [round(arr_lo, 1), round(arr_hi, 1)]})
            fit = group_fit.get(gid)
            if fit is not None:
                v_fit, d_fit = float(fit["speed_mps"]), float(fit["cpa_slant_m"])
                v_tol = max(SPEED_TOL_MIN, SPEED_TOL_FRAC * v_fit)
                d_tol = max(SLANT_TOL_MIN, SLANT_TOL_FRAC * d_fit)
                if vg is not None and abs(vg - v_fit) > v_tol:
                    why.append({"reason": "speed_incompatible", "flight_speed_mps": round(vg, 1),
                                "fitted_speed_mps": round(v_fit, 1), "tolerance_mps": round(v_tol, 1)})
                if min_h is not None and min_h > d_fit + d_tol:
                    why.append({"reason": "height_incompatible", "min_height_above_observer_m": round(min_h, 1),
                                "fitted_cpa_slant_m": round(d_fit, 1), "tolerance_m": round(d_tol, 1)})
            if why:
                reasons.append({"group_id": gid, "reasons": why})
            else:
                excluded_all = False
        if excluded_all:
            discarded.append({"flight_id": f.get("flight_id"), "callsign": f.get("callsign"),
                              "data_source": f.get("data_source"), "groups": reasons})
        else:
            kept.append(f)
    return kept, discarded
