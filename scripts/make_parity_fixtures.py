#!/usr/bin/env python3
"""
Generate golden fixtures from the Python engine for the Node parity tests
(tests/js/parity.test.mjs), which check the browser engine against them.

    python scripts/make_parity_fixtures.py
"""

from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.adsb.conditioning import condition_trajectory  # noqa: E402
from app.adsb.sample_flights import get_curated_scenarios  # noqa: E402
from app.adsb.track_import import parse_tracks  # noqa: E402
from app.audio.generator import simulate_events  # noqa: E402
from app.physics.confidence import DecisionPolicy, decide_location  # noqa: E402
from app.physics.doppler_engine import DopplerEngine  # noqa: E402
from app.physics.flyover_fit import fit_flyover, flyover_frequency  # noqa: E402
from app.physics.matching import JointMatcher, MatchOptions, TrajectoryModel, build_events, fit_event, propagation  # noqa: E402
from app.physics.prefilter import prefilter_flights  # noqa: E402

OUT = os.path.join(ROOT, "tests", "fixtures", "parity")


def _f(x):
    return [float(v) for v in x]


def main():
    os.makedirs(OUT, exist_ok=True)
    sc = get_curated_scenarios()["sfo_dual"]
    c = sc["search_center"]
    rng = np.random.default_rng(42)
    data = {}

    # 1. Flyover fit on a fixed noisy curve.
    t = np.linspace(0, 30, 90)
    f = flyover_frequency(t, 750.0, 150.0, 1800.0, 12.0) + rng.normal(0, 0.8, t.size)
    fit = fit_flyover(t, f)
    data["flyover_fit"] = {"times": _f(t), "freqs": _f(f), "expected": {k: fit[k] for k in (
        "f0_hz", "speed_mps", "cpa_slant_m", "cpa_emission_time_s", "cpa_arrival_time_s", "rmse_hz")}}

    # 2. fit_event on fixed inputs (with and without wind, fitted and fixed f0).
    events = simulate_events(sc["flights"], (c["lat"], c["lon"]), sc["recording_start_epoch"], sc["audio_duration_sec"])
    cases = []
    for wind, fit_f0, drift in (((0.0, 0.0), True, 0.02), ((6.0, -4.0), True, 0.005), ((0.0, 0.0), False, 0.02)):
        opts = MatchOptions(fit_f0=fit_f0, drift_fraction=drift, wind_e=wind[0], wind_n=wind[1])
        raw = [dict(e, f0=820.0) for e in events]
        evs = build_events(raw, opts)
        m = TrajectoryModel(sc["flights"][0], c["lat"], c["lon"], DopplerEngine())
        obs = np.array([350.0, -420.0, 0.0])
        curve = m.arrival(obs, opts.speed_of_sound, wind)
        ev = evs[0]
        starts = curve["t_cpa"] - ev.cpa_time_s + np.linspace(-3, 3, 13)
        r = fit_event(curve, ev, starts, opts)
        cases.append({
            "opts": {"fit_f0": fit_f0, "drift_fraction": drift, "wind_e": wind[0], "wind_n": wind[1]},
            "flight": sc["flights"][0], "observer": _f(obs), "event": raw[0], "event_cpa_time_s": ev.cpa_time_s,
            "starts": _f(starts), "loss": [float(v) if np.isfinite(v) else None for v in r["loss"]],
            "f0": _f(r["f0"]), "drift": _f(r["drift"]), "rmse": [float(v) if np.isfinite(v) else None for v in r["rmse"]],
            "curve_t": _f(curve["t"]), "curve_g": _f(curve["g"]),
        })
    data["fit_event"] = cases

    # 3. Propagation and speed of sound.
    R = rng.normal(0, 3000, (20, 3))
    V = rng.normal(0, 120, (20, 3))
    T, g = propagation(R, V, 340.0, (7.5, -11.0))
    data["propagation"] = {"R": R.tolist(), "V": V.tolist(), "c": 340.0, "wind": [7.5, -11.0], "T": _f(T), "g": _f(g)}
    data["speed_of_sound"] = [{"t": t_, "rh": rh, "c": DopplerEngine.calculate_speed_of_sound(t_, rh)}
                              for t_ in (-10.0, 0.0, 20.0, 35.0) for rh in (0.0, 0.5, 1.0)]

    # 4. Trajectory conditioning (irregular, noisy, with a gap and reported velocity).
    ref = sc["flights"][0]
    keep = [i for i in range(len(ref["timestamps"])) if not (40 <= i < 70) and rng.uniform() > 0.15]
    noisy = {k: [ref[k][i] for i in keep] for k in ("timestamps", "latitudes", "longitudes", "altitudes", "speeds",
                                                    "headings", "vertical_rates")}
    noisy["latitudes"] = _f(np.array(noisy["latitudes"]) + rng.normal(0, 5e-5, len(keep)))
    noisy["longitudes"] = _f(np.array(noisy["longitudes"]) + rng.normal(0, 6e-5, len(keep)))
    noisy.update(flight_id="NOISY", data_source="uploaded")
    cond = condition_trajectory(noisy)
    data["conditioning"] = {"input": noisy, "expected": {k: cond[k] for k in (
        "timestamps", "latitudes", "longitudes", "altitudes", "velocity_east", "velocity_north", "velocity_up")}}

    # 5. Pre-filter (Python flyover fits fed to both implementations).
    fits = {}
    for ev in events:
        pts = ev["traced_curve"]
        fit = fit_flyover([p["time"] for p in pts], [p["freq"] for p in pts])
        fit.pop("model_curve", None)
        fits[ev["id"]] = fit
    late = dict(copy.deepcopy(sc["flights"][0]), flight_id="LATE", timestamps=[x + 3600 for x in sc["flights"][0]["timestamps"]])
    flights = sc["flights"] + [late]
    clock = (sc["recording_start_epoch"] - 2, sc["recording_start_epoch"] + 2)
    kept, discarded = prefilter_flights(flights, events, fits, clock)
    data["prefilter"] = {"flights": flights, "events": events, "fits": fits, "clock": list(clock),
                         "kept": [k["flight_id"] for k in kept], "discarded": discarded}

    # 6. decide_location on hand-built results covering each gate.
    base = {"status": "ok", "robust_rmse_hz": 0.6, "boundary_limited": False,
            "diagnostics": [
                {"event_id": "E1", "group_id": "G1", "flight_id": "A", "matched": True, "support_fraction": 1.0,
                 "rmse_hz": 0.8, "robust_rmse_hz": 0.7, "outlier_fraction": 0.0, "drift_at_bound": False},
                {"event_id": "E2", "group_id": "G2", "flight_id": "B", "matched": True, "support_fraction": 0.99,
                 "rmse_hz": 1.2, "robust_rmse_hz": 1.0, "outlier_fraction": 0.02, "drift_at_bound": False}],
            "alternatives": [{"lat": 1.0, "lon": 2.0, "robust_rmse_hz": 1.5, "distance_from_best_m": 3000.0}],
            "sensitivity": {"complete": True, "radius_m": 300.0}}
    variants = {
        "supported": base,
        "near_tie": {**base, "alternatives": [{"lat": 1.0, "lon": 2.0, "robust_rmse_hz": 0.9, "distance_from_best_m": 3000.0}]},
        "single_flyover": {**base, "diagnostics": [base["diagnostics"][0]]},
        "residuals": {**base, "diagnostics": [base["diagnostics"][0], {**base["diagnostics"][1], "rmse_hz": 3.5}]},
        "support": {**base, "diagnostics": [base["diagnostics"][0], {**base["diagnostics"][1], "support_fraction": 0.9}]},
        "drift": {**base, "diagnostics": [base["diagnostics"][0], {**base["diagnostics"][1], "drift_at_bound": True}]},
        "boundary": {**base, "boundary_limited": True},
        "radius": {**base, "sensitivity": {"complete": True, "radius_m": 1500.0}},
        "unmatched": {**base, "diagnostics": [base["diagnostics"][0], {"event_id": "E2", "group_id": "G2", "matched": False}]},
        "no_match": {"status": "no_match"},
        "no_adsb": {"status": "no_adsb_data"},
    }
    data["decide"] = []
    for name, res in variants.items():
        for pol in ({}, {"residual_metric": "robust", "max_rmse_hz": 3.0}, {"min_independent_flyovers": 1}):
            d = decide_location(res, DecisionPolicy(**pol), n_events=2)
            data["decide"].append({"name": name, "policy": pol, "result": res, "status": d["status"],
                                   "reason_code": d["reason_code"],
                                   "checks": [[ch["check"], ch["passed"]] for ch in d["checks"]]})

    # 7. Track import (generic CSV, FR24, GeoJSON, readsb trace).
    data["track_import"] = []
    tracks_dir = os.path.join(ROOT, "tests", "fixtures", "tracks")
    for name in ("generic.csv", "fr24.csv", "tracks.geojson", "trace_full_a1b2c3.json"):
        with open(os.path.join(tracks_dir, name), "rb") as fh:
            res = parse_tracks(fh.read(), name)
        data["track_import"].append({"file": name, "format": res["format"], "flights": [
            {k: f[k] for k in ("flight_id", "callsign", "altitude_basis", "timestamps", "latitudes", "longitudes",
                               "altitudes", "speeds")} for f in res["flights"]]})

    # 8. End-to-end match (grid 11, no sensitivity) on noise-free sfo_dual events.
    m = JointMatcher(sc["flights"], events, c["lat"], c["lon"], MatchOptions())
    res = m.run(8.0, 11, sensitivity=False)
    lat, lon = m.to_geo(res["best"]["x"], res["best"]["y"])
    dec = decide_location(res)
    data["end_to_end"] = {
        "center": c, "flights": sc["flights"], "events": events, "radius_km": 8.0, "grid": 11,
        "best": {"lat": lat, "lon": lon, "start": res["best"]["start"], "robust_rmse_hz": res["robust_rmse_hz"]},
        "matched": sorted({d["flight_id"] for d in res["diagnostics"] if d["matched"]}),
        "status": dec["status"], "reason_code": dec["reason_code"],
        "grid_loss": [[float(v) if np.isfinite(v) else None for v in row] for row in m.grid["loss"]],
    }

    with open(os.path.join(OUT, "golden.json"), "w") as fh:
        json.dump(data, fh)
    print("wrote", os.path.relpath(os.path.join(OUT, "golden.json"), ROOT), os.path.getsize(os.path.join(OUT, "golden.json")), "bytes")


if __name__ == "__main__":
    main()
