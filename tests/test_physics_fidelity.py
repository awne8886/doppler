"""
Phase 3 regression tests: uniform wind, barometric altitude datum (QNH) and
observer ground elevation.  For each effect a model that ignores it must show
a measurable bias on data generated *with* it, and the aware model must remove
the bias.  Test data come from the first-principles forward model in
``app.audio.generator`` (fixed-point travel time, frequency = f0 * d tau / d t),
which does not use the matcher's Doppler formula.
"""

import math
import os

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.adsb.sample_flights import get_curated_scenarios
from app.audio.discovery import discover_file
from app.audio.generator import emission_arrivals, simulate_events, synthesize_emission_phase_wav
from app.physics.atmosphere import EXP, L, P0, T0, apply_qnh, qnh_altitude
from app.physics.confidence import decide_location
from app.physics.doppler_engine import DopplerEngine
from app.physics.elevation import LatLonElevation, OpenTopoDataProvider, dem_grid
from app.physics.matching import JointMatcher, MatchOptions, TrajectoryModel, wind_vector

SC = get_curated_scenarios()["sfo_dual"]
C = SC["search_center"]
OBS = (C["lat"], C["lon"])
client = TestClient(main.app)


def _err_m(m, res):
    lat, lon = m.to_geo(res["best"]["x"], res["best"]["y"])
    p1, p2 = math.radians(lat), math.radians(C["lat"])
    dp, dl = p2 - p1, math.radians(C["lon"] - lon)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


def _run(flights, events, opts, grid=15):
    m = JointMatcher(flights, events, C["lat"], C["lon"], opts)
    return m, m.run(8.0, grid, sensitivity=False)


# ---------------------------------------------------------------------- wind

def test_wind_vector_convention():
    e, n = wind_vector(10.0, 270.0)  # wind FROM the west blows TOWARD the east
    assert e == pytest.approx(10.0) and n == pytest.approx(0.0, abs=1e-9)
    e, n = wind_vector(10.0, 0.0)  # from the north -> toward the south
    assert n == pytest.approx(-10.0) and abs(e) < 1e-9


def test_wind_model_matches_first_principles_forward_model():
    W = wind_vector(15.0, 250.0)
    f = SC["flights"][0]
    m = TrajectoryModel(f, C["lat"], C["lon"], DopplerEngine())
    a = m.arrival(np.zeros(3), 343.2, W)
    ref = emission_arrivals(f, OBS, W, 343.2)
    tt = np.linspace(a["t"][5], a["t"][-5], 400)
    err_exact = np.max(np.abs(np.interp(tt, a["t"], a["g"]) - np.interp(tt, ref["t"], ref["factor"]))) * 820.0
    calm = m.arrival(np.zeros(3), 343.2, (0.0, 0.0))
    ref0 = emission_arrivals(f, OBS, (0.0, 0.0), 343.2)
    err_calm = np.max(np.abs(np.interp(tt, calm["t"], calm["g"]) - np.interp(tt, ref0["t"], ref0["factor"]))) * 820.0
    # The wind model adds no error beyond the 1 s trajectory-sampling floor of the calm model.
    assert err_exact < err_calm + 0.2
    # Arrival times agree with the iterative travel-time solution.
    assert np.max(np.abs(np.interp(a["tau"], ref["tau"], ref["t"]) - a["t"])) < 1e-3
    assert np.max(np.abs(np.interp(calm["tau"], ref0["tau"], ref0["t"]) - calm["t"])) < 1e-3

    # The simplified c_eff/(c_eff + u_r) form (ground ray instead of wave normal) is far off.
    d = np.linalg.norm(m.pos, axis=1)
    n_hat = -m.pos / d[:, None]
    c_eff = 343.2 + n_hat[:, :2] @ np.array(W)
    u_r = np.sum(m.pos * m.vel, axis=1) / d
    t_s, g_s = m.t_emit + d / c_eff, c_eff / (c_eff + u_r)
    o = np.argsort(t_s)
    err_simple = np.max(np.abs(np.interp(tt, t_s[o], g_s[o]) - np.interp(tt, ref["t"], ref["factor"]))) * 820.0
    assert err_simple > 10.0


def test_ignoring_wind_biases_location_and_modelling_it_removes_the_bias():
    W = wind_vector(15.0, 180.0)
    events = simulate_events(SC["flights"], OBS, SC["recording_start_epoch"], SC["audio_duration_sec"], W)
    m0, r0 = _run(SC["flights"], events, MatchOptions())
    m1, r1 = _run(SC["flights"], events, MatchOptions(wind_e=W[0], wind_n=W[1]))
    assert _err_m(m0, r0) > 40.0
    assert _err_m(m1, r1) < 15.0
    assert r1["robust_rmse_hz"] < r0["robust_rmse_hz"]


def test_wind_audio_end_to_end(tmp_path):
    """Emission-phase audio with wind -> discovery -> wind-aware matching."""
    W = wind_vector(12.0, 200.0)
    wav = str(tmp_path / "wind.wav")
    # Fundamental only: this test is about propagation physics, not harmonic grouping.
    synthesize_emission_phase_wav(SC["flights"], OBS, wav, SC["recording_start_epoch"], SC["audio_duration_sec"], W,
                                  harmonics=(1.0,))
    disc = discover_file(wav)
    events = [{"id": r["id"], "group_id": r["harmonic_group_id"], "traced_curve": r["points"]}
              for r in disc["ridges"] if r["doppler_like"]]
    assert len({e["group_id"] for e in events}) >= 2  # crossing ridges may be split into fragments
    m_calm, r_calm = _run(SC["flights"], events, MatchOptions())
    m_wind, r_wind = _run(SC["flights"], events, MatchOptions(wind_e=W[0], wind_n=W[1]))
    assert _err_m(m_wind, r_wind) < 20.0
    assert _err_m(m_calm, r_calm) > _err_m(m_wind, r_wind) + 10.0
    assert decide_location(r_wind)["status"] == "supported"


def test_sensitivity_includes_wind_scenarios():
    events = simulate_events(SC["flights"], OBS, SC["recording_start_epoch"], SC["audio_duration_sec"])
    m = JointMatcher(SC["flights"][:2], events, C["lat"], C["lon"], MatchOptions(wind_uncertainty_mps=4.0))
    res = m.run(8.0, 11, sensitivity=True)
    names = [s["scenario"] for s in res["sensitivity"]["scenarios"]]
    assert {"wind_east_4mps", "wind_west_4mps", "wind_north_4mps", "wind_south_4mps"} <= set(names)
    assert "wind" in res["sensitivity"]["sources"]


# ------------------------------------------------------------ altitude datum

def _pressure_altitude(h_msl, qnh):
    p = qnh * (1.0 - L * np.asarray(h_msl) / T0) ** EXP
    return (T0 / L) * (1.0 - (p / P0) ** (1.0 / EXP))


def test_qnh_roundtrip():
    h = np.array([0.0, 500.0, 2500.0])
    assert np.allclose(qnh_altitude(_pressure_altitude(h, 1030.0), 1030.0), h, atol=1e-6)
    assert qnh_altitude(0.0, 1033.25) == pytest.approx(164.6, abs=0.5)  # ~8.2 m/hPa near sea level


def test_uncorrected_barometric_altitude_biases_and_qnh_removes_it():
    qnh = 1040.0  # strong high: pressure altitude reads ~220 m below true height
    events = simulate_events(SC["flights"], OBS, SC["recording_start_epoch"], SC["audio_duration_sec"])
    baro = [dict(f, altitudes=[float(a) for a in _pressure_altitude(f["altitudes"], qnh)],
                 altitude_basis="barometric") for f in SC["flights"][:2]]
    m0, r0 = _run(baro, events, MatchOptions())
    m1, r1 = _run([apply_qnh(f, qnh) for f in baro], events, MatchOptions())
    assert _err_m(m0, r0) > 40.0
    assert _err_m(m1, r1) < 15.0


def test_altitude_perturbation_widens_for_barometric_tracks():
    events = simulate_events(SC["flights"], OBS, SC["recording_start_epoch"], SC["audio_duration_sec"])

    def delta(flights):
        return JointMatcher(flights, events, C["lat"], C["lon"]).altitude_perturbation_m()

    geo = [dict(f, altitude_basis="geometric") for f in SC["flights"][:2]]
    baro = [dict(f, altitude_basis="barometric") for f in SC["flights"][:2]]
    assert delta(geo) == 150.0
    assert delta(baro) == 300.0
    assert delta([apply_qnh(f, 1013.25) for f in baro]) == 200.0
    assert delta([dict(f, altitude_basis="unknown") for f in SC["flights"][:2]]) == 300.0


# ---------------------------------------------------------- ground elevation

def test_ground_elevation_grid_removes_height_bias():
    elev = 450.0
    events = simulate_events(SC["flights"], OBS, SC["recording_start_epoch"], SC["audio_duration_sec"],
                             observer_alt_m=elev)
    # Sloping terrain: 450 m at the microphone, varying across the search area.
    lats = np.linspace(C["lat"] - 0.08, C["lat"] + 0.08, 5)
    lons = np.linspace(C["lon"] - 0.1, C["lon"] + 0.1, 5)
    values = [[elev + 2000.0 * (la - C["lat"]) - 1500.0 * (lo - C["lon"]) for lo in lons] for la in lats]
    grid = {"lat_min": lats[0], "lat_max": lats[-1], "lon_min": lons[0], "lon_max": lons[-1], "values": values}
    model = LatLonElevation(grid, C["lat"], C["lon"])
    assert model(0.0, 0.0) == pytest.approx(elev, abs=0.5)
    m0, r0 = _run(SC["flights"][:2], events, MatchOptions())
    m1, r1 = _run(SC["flights"][:2], events, MatchOptions(elevation=model))
    assert _err_m(m0, r0) > 40.0
    assert _err_m(m1, r1) < 15.0


class _DemSession:
    def __init__(self):
        self.calls = 0

    def get(self, url, params=None, **kw):
        self.calls += 1
        locs = params["locations"].split("|")

        class R:
            status_code = 200

            def json(self_inner):
                return {"results": [{"elevation": 100.0 + float(p.split(",")[0])} for p in locs]}
        return R()


def test_dem_grid_is_batched_and_cached(tmp_path):
    from app.adsb.providers import DiskCache

    sess = _DemSession()
    prov = OpenTopoDataProvider("https://dem.example/v1/test", DiskCache(str(tmp_path)), session=sess)
    g = dem_grid(prov, C["lat"], C["lon"], 8.0, n=11)  # 121 points -> 2 requests
    assert sess.calls == 2 and len(g["values"]) == 11
    dem_grid(prov, C["lat"], C["lon"], 8.0, n=11)
    assert sess.calls == 2  # cached


# ----------------------------------------------------------------------- API

def test_api_accepts_wind_qnh_and_falls_back_without_dem():
    events = simulate_events(SC["flights"], OBS, SC["recording_start_epoch"], SC["audio_duration_sec"],
                             wind_vector(10.0, 90.0))
    resp = client.post("/api/geolocate", json={
        "center_lat": C["lat"], "center_lon": C["lon"], "radius_km": 8.0, "events": events, "scenario_id": "sfo_dual",
        "grid_resolution": 11, "sensitivity": False, "wind_speed_mps": 10.0, "wind_from_deg": 90.0,
        "qnh_hpa": 1020.0, "dem_lookup": True,
    })
    assert resp.status_code == 200
    atm = resp.json()["atmosphere"]
    assert atm["wind_east_mps"] == pytest.approx(-10.0) and atm["qnh_hpa"] == 1020.0
    assert atm["elevation"] == "dem_unavailable_flat_ground"  # network is disabled in tests


def test_api_rejects_bad_elevation_grid():
    events = simulate_events(SC["flights"], OBS, SC["recording_start_epoch"], SC["audio_duration_sec"])
    resp = client.post("/api/geolocate", json={
        "center_lat": C["lat"], "center_lon": C["lon"], "radius_km": 8.0, "events": events, "scenario_id": "sfo_dual",
        "grid_resolution": 11, "observer_elevation_grid": {"lat_min": 1, "lat_max": 0, "lon_min": 0, "lon_max": 1,
                                                           "values": [[0, 0], [0, 0]]},
    })
    assert resp.status_code == 400


if os.environ.get("DOPPLER_SLOW_TESTS") == "0":  # pragma: no cover
    pytestmark = pytest.mark.skip("slow physics tests disabled")
