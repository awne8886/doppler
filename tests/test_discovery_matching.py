"""
Tests for automatic ridge discovery, flyover fitting, joint shared-clock
matching and the supported / inconclusive decision layer.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient

from app.adsb.adsblol_client import AdsbLolClient
from app.adsb.sample_flights import get_curated_scenarios
from app.audio.discovery import discover_file, discover_ridges
from app.main import app
from app.physics.confidence import decide_location
from app.physics.doppler_engine import DopplerEngine
from app.physics.flyover_fit import fit_flyover, flyover_frequency
from app.physics.matching import JointMatcher, TrajectoryModel

SAMPLES = os.path.join(os.path.dirname(__file__), "..", "app", "static", "samples")
client = TestClient(app)


def _haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _model_events(scenario, observer_xy=(0.0, 0.0), shift_s=None, step=0.25):
    """Exact emission-domain curves for every flight audible during the recording."""
    c = scenario["search_center"]
    engine = DopplerEngine()
    start = scenario["recording_start_epoch"]
    dur = scenario["audio_duration_sec"]
    events = []
    for i, tr in enumerate(scenario["flights"]):
        m = TrajectoryModel(tr, c["lat"], c["lon"], engine)
        a = m.arrival(np.array([observer_xy[0], observer_xy[1], 0.0]), 343.2)
        lo, hi = max(a["t"][0], start + 1.0), min(a["t"][-1], start + dur - 1.0)
        if hi - lo < 8.0:
            continue
        t_abs = np.arange(lo, hi, step)
        f = np.interp(t_abs, a["t"], a["g"]) * tr["f0_hz"]
        t_rel = t_abs - start
        if shift_s and i == 1:
            t_rel = t_rel + shift_s
        events.append({"id": tr["flight_id"], "points": [{"time": float(t), "freq": float(x)} for t, x in zip(t_rel, f)]})
    return events


# ---------------------------------------------------------------- flyover fit

def test_analytic_flyover_matches_numerical_engine():
    engine = DopplerEngine()
    te = np.linspace(-60, 60, 4001)
    v, D = 170.0, 1200.0
    pos = np.stack([v * te, np.full_like(te, 0.6 * D), np.full_like(te, 0.8 * D)], axis=1)
    vel = np.tile([v, 0.0, 0.0], (te.size, 1))
    t_obs, f_obs = engine.compute_doppler_curve(te, pos, vel, np.zeros(3), 800.0)
    t = np.linspace(-15, 20, 80) + D / 343.2
    assert np.max(np.abs(flyover_frequency(t, 800.0, v, D, 0.0) - np.interp(t, t_obs, f_obs))) < 0.05


def test_flyover_fit_recovers_parameters():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 30, 90)
    f = flyover_frequency(t, 750.0, 150.0, 1800.0, 12.0) + rng.normal(0, 0.8, t.size)
    fit = fit_flyover(t, f)
    assert abs(fit["f0_hz"] - 750.0) < 2.0
    assert abs(fit["speed_mps"] - 150.0) < 5.0
    assert abs(fit["cpa_slant_m"] - 1800.0) < 120.0
    assert abs(fit["cpa_emission_time_s"] - 12.0) < 0.5


# ------------------------------------------------------------------ discovery

def test_discovery_finds_harmonic_group_in_sample():
    res = discover_file(os.path.join(SAMPLES, "sfo_ual421.wav"))
    assert res["n_groups"] == 1
    ridges = res["ridges"]
    assert {r["harmonic_ratio"] for r in ridges} >= {"1", "2"}
    fundamental = next(r for r in ridges if r["harmonic_ratio"] == "1")
    assert fundamental["doppler_like"]
    assert abs(fundamental["flyover_fit"]["f0_hz"] - 840.0) < 3.0
    assert abs(fundamental["flyover_fit"]["speed_mps"] - 168.0) < 8.0


def test_discovery_rejects_noise_only_audio():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.1, 8000 * 30).astype(np.float32)
    assert discover_ridges(x, 8000)["n_ridges"] == 0


def test_discovery_separates_two_flyovers():
    res = discover_file(os.path.join(SAMPLES, "sfo_dual.wav"))
    dop_groups = [g for g in res["harmonic_groups"] if g["doppler_like"]]
    assert len(dop_groups) == 2
    f0s = sorted(
        r["flyover_fit"]["f0_hz"] for r in res["ridges"]
        if r["doppler_like"] and r.get("harmonic_ratio") == "1"
    )
    assert abs(f0s[0] - 590.0) < 3.0 and abs(f0s[-1] - 820.0) < 3.0


# ------------------------------------------------------------------- matching

def test_single_flyover_is_reported_inconclusive():
    sc = get_curated_scenarios()["sfo_ual421"]
    c = sc["search_center"]
    events = _model_events(sc)
    m = JointMatcher(sc["flights"], events, c["lat"], c["lon"])
    res = m.run(10.0, 15, sensitivity=False)
    assert res["status"] == "ok"
    decision = decide_location(res)
    assert decision["status"] == "inconclusive"
    assert decision["reason_code"] in {"ambiguous_locations", "insufficient_independent_events"}


def test_two_flyovers_with_shared_clock_are_supported():
    sc = get_curated_scenarios()["sfo_dual"]
    c = sc["search_center"]
    events = _model_events(sc)
    assert len(events) == 2
    m = JointMatcher(sc["flights"], events, c["lat"], c["lon"])
    res = m.run(8.0, 15, sensitivity=True)
    lat, lon = m.to_geo(res["best"]["x"], res["best"]["y"])
    assert _haversine_m(lat, lon, c["lat"], c["lon"]) < 100.0
    assert abs(res["best"]["start"] - sc["recording_start_epoch"]) < 0.2
    assert {d["flight_id"] for d in res["diagnostics"]} == {"UAL1714", "SWA2286"}  # decoy ASA331 unused
    decision = decide_location(res)
    assert decision["status"] == "supported", decision


def test_second_event_cannot_get_its_own_time_shift():
    """A 6 s clock error on one event must not be absorbed by an independent shift."""
    sc = get_curated_scenarios()["sfo_dual"]
    c = sc["search_center"]
    events = _model_events(sc, shift_s=6.0)
    m = JointMatcher(sc["flights"], events, c["lat"], c["lon"])
    res = m.run(8.0, 15, sensitivity=False)
    decision = decide_location(res)
    assert decision["status"] == "inconclusive"


def test_events_outside_trajectory_support_do_not_match():
    sc = get_curated_scenarios()["sfo_ual421"]
    c = sc["search_center"]
    t = np.arange(0, 400, 1.0)  # far longer than the 60 s trajectory can explain
    events = [{"id": "E1", "points": [{"time": float(x), "freq": 800.0 - 0.1 * float(x)} for x in t]}]
    m = JointMatcher(sc["flights"], events, c["lat"], c["lon"])
    res = m.run(5.0, 11, sensitivity=False)
    assert decide_location(res)["status"] == "inconclusive"


def test_no_events_decision():
    assert decide_location({"status": "no_match"}, n_events=0)["reason_code"] == "no_doppler_events"


# ------------------------------------------------------------------------ API

def test_api_discover_sample():
    resp = client.post("/api/audio/discover", data={"sample_id": "sfo_ual421"})
    assert resp.status_code == 200
    assert resp.json()["n_ridges"] >= 1


def test_api_discover_rejects_path_traversal():
    resp = client.post("/api/audio/discover", data={"saved_as": "../../main.py"})
    assert resp.status_code in (400, 404)


def test_api_flyover_fit():
    t = np.linspace(0, 30, 60)
    f = flyover_frequency(t, 700.0, 140.0, 1500.0, 14.0)
    resp = client.post("/api/flyover/fit", json={"traced_curve": [{"time": float(a), "freq": float(b)} for a, b in zip(t, f)]})
    assert resp.status_code == 200
    assert abs(resp.json()["f0_hz"] - 700.0) < 1.0


def test_api_geolocate_multi_event_from_discovery():
    disc = client.post("/api/audio/discover", data={"sample_id": "sfo_dual"}).json()
    events = [
        {"id": r["id"], "group_id": r["harmonic_group_id"], "traced_curve": r["points"]}
        for r in disc["ridges"] if r["doppler_like"]
    ]
    sc = get_curated_scenarios()["sfo_dual"]
    resp = client.post("/api/geolocate", json={
        "center_lat": sc["search_center"]["lat"], "center_lon": sc["search_center"]["lon"],
        "radius_km": 8.0, "events": events, "scenario_id": "sfo_dual", "grid_resolution": 15,
    })
    assert resp.status_code == 200
    data = resp.json()
    best = data["best_location"]
    assert _haversine_m(best["lat"], best["lon"], sc["search_center"]["lat"], sc["search_center"]["lon"]) < 150.0
    assert data["decision"]["status"] == "supported"
    assert data["sensitivity"]["complete"]
    assert len(data["heatmap"]) > 50


def test_adsblol_parser_converts_units():
    states = AdsbLolClient.parse({"now": 1700000000000, "ac": [
        {"hex": "abc123", "flight": "UAL1 ", "lat": 37.5, "lon": -122.3, "alt_baro": 3000, "alt_geom": 3100,
         "gs": 200, "track": 90, "geom_rate": -600, "seen_pos": 1.5, "t": "B738"},
        {"hex": "def456", "lat": 37.6, "lon": -122.4, "alt_baro": "ground"},
    ]})
    assert len(states) == 1
    s = states[0]
    assert s["altitude_basis"] == "geometric" and abs(s["altitude"] - 944.88) < 0.01
    assert abs(s["velocity"] - 102.89) < 0.01 and abs(s["vertical_rate"] + 3.048) < 0.001
    assert abs(s["time_position"] - 1699999998.5) < 1e-6


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
