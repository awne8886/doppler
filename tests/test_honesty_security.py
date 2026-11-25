"""
Phase 1 regression tests: no silent fabrication of flights, provenance tags,
ADS-B parsing rules and server hygiene (uploads, CORS, error leaks).
"""

import inspect
import os
import time

import numpy as np
from fastapi.testclient import TestClient

import app.main as main
from app.adsb.adsblol_client import AdsbLolClient
from app.adsb.flight_manager import FlightManager
from app.adsb.opensky_client import OpenSkyClient
from app.adsb.sample_flights import get_curated_scenarios
from app.physics.flyover_fit import flyover_frequency

client = TestClient(main.app)
SFO = {"center_lat": 37.5585, "center_lon": -122.2711, "radius_km": 10.0}


def _events():
    t = np.linspace(0, 20, 40)
    f = flyover_frequency(t, 800.0, 160.0, 1500.0, 10.0)
    return [{"id": "E1", "traced_curve": [{"time": float(a), "freq": float(b)} for a, b in zip(t, f)]}]


# ----------------------------------------------------------- opt-in demo data

def test_search_near_curated_scenario_without_opt_in_returns_no_data():
    resp = client.post("/api/flights/search", json=SFO)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "no_adsb_data"
    assert data["count"] == 0 and data["flights"] == []
    assert data["data_source"] is None


def test_search_demo_opt_in_is_tagged():
    near = client.post("/api/flights/search", json={**SFO, "demo": True}).json()
    assert near["status"] == "ok" and near["data_source"] == "curated_demo"
    assert all(f["data_source"] == "curated_demo" for f in near["flights"])
    far = client.post("/api/flights/search", json={"center_lat": 10.0, "center_lon": 10.0, "radius_km": 10, "demo": True}).json()
    assert far["data_source"] == "synthetic"
    assert all(f["data_source"] == "synthetic" and "ground_truth_mic" not in f for f in far["flights"])


def test_explicit_scenario_is_curated_demo():
    data = client.post("/api/flights/search", json={**SFO, "scenario_id": "sfo_dual"}).json()
    assert data["data_source"] == "curated_demo"
    assert {f["data_source"] for f in data["flights"]} == {"curated_demo"}


def test_unknown_scenario_is_rejected():
    resp = client.post("/api/flights/search", json={**SFO, "scenario_id": "nope"})
    assert resp.status_code == 400


def test_geolocate_without_real_data_is_no_adsb_data():
    resp = client.post("/api/geolocate", json={**SFO, "events": _events(), "grid_resolution": 11})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "no_adsb_data"
    assert data["decision"]["status"] == "inconclusive"
    assert data["decision"]["reason_code"] == "no_adsb_data"


def test_geolocate_success_reports_data_source():
    sc = get_curated_scenarios()["sfo_ual421"]
    data = client.post("/api/geolocate", json={**SFO, "events": _events(), "scenario_id": "sfo_ual421",
                                              "grid_resolution": 11, "sensitivity": False}).json()
    assert data["data_source"] == "curated_demo"
    assert data["best_flight"]["data_source"] == "curated_demo"
    assert all(f["data_source"] == "curated_demo" for f in data["evaluated_flights"])
    assert sc["trajectory"]["flight_id"] in {f["flight_id"] for f in data["evaluated_flights"]}


# ------------------------------------------------------------- live snapshot

class _FakeOpenSky:
    def __init__(self, states):
        self.states = states

    def get_live_states(self, *a, **k):
        return self.states


class _NoAdsbLol:
    def get_live_states(self, *a, **k):
        return None


def _state(icao, lat, lon, t=1_800_000_000.0, **kw):
    s = {"icao24": icao, "callsign": icao.upper(), "time_position": t, "lat": lat, "lon": lon, "altitude": 1500.0,
         "altitude_basis": "geometric", "velocity": 150.0, "heading": 90.0, "vertical_rate": 0.0, "on_ground": False}
    s.update(kw)
    return s


def test_live_projection_is_centred_on_state_time_and_tagged():
    states = [_state("aaa111", 37.56, -122.27), _state("bbb222", 37.70, -122.27), _state("ccc333", 37.56, -122.26),
              _state("ddd444", 37.56, -122.27, time_position=None)]
    fm = FlightManager(_FakeOpenSky(states), _NoAdsbLol(), max_live_aircraft=2, clock=lambda: 1_800_000_010.0)
    res = fm.find_flights(37.5585, -122.2711, 10.0)
    assert res["status"] == "ok" and res["data_source"] == "opensky_live"
    ids = [f["flight_id"] for f in res["flights"]]
    assert ids == ["AAA111", "CCC333"]  # nearest two; the far one is capped, the one without a timestamp skipped
    f = res["flights"][0]
    assert "ground_truth_mic" not in f and "f0_hz" not in f
    i = int(np.argmin(np.abs(np.array(f["timestamps"]) - 1_800_000_000.0)))
    assert abs(f["timestamps"][i] - 1_800_000_000.0) < 1e-6
    assert abs(f["latitudes"][i] - 37.56) < 1e-4 and abs(f["longitudes"][i] + 122.27) < 1e-4
    assert abs(f["altitudes"][i] - 1500.0) < 0.5
    assert res["limitations"]


def test_live_snapshot_not_used_for_old_time_window():
    fm = FlightManager(_FakeOpenSky([_state("aaa111", 37.56, -122.27)]), _NoAdsbLol(), clock=lambda: 1_800_000_000.0)
    res = fm.find_flights(37.5585, -122.2711, 10.0, time_window_start=1_799_000_000.0, time_window_end=1_799_000_100.0)
    assert res["status"] == "no_adsb_data"
    assert any("not recent" in n for n in res["notes"])


# ------------------------------------------------------------------- parsing

def test_opensky_parser_prefers_geometric_and_skips_incomplete():
    row = ["abc123", "UAL1    ", "US", 1700000000, 1700000001, -122.3, 37.5, 3000.0, False, 120.0, 90.0, -2.0, None, 3100.0]
    payload = {"states": [
        row,
        row[:7] + [3000.0, True] + row[9:],                     # on ground
        row[:9] + [None] + row[10:],                            # no velocity
        row[:10] + [None] + row[11:],                           # no track
        row[:7] + [None] + row[8:13] + [None],                  # no altitude at all
        row[:7] + [2900.0] + row[8:13] + [None],                # baro only
        row[:11] + [None] + row[12:],                           # no vertical rate
    ]}
    states = OpenSkyClient.parse_states(payload)
    assert len(states) == 3
    assert states[0]["altitude"] == 3100.0 and states[0]["altitude_basis"] == "geometric"
    assert states[1]["altitude"] == 2900.0 and states[1]["altitude_basis"] == "barometric"
    assert states[2]["vertical_rate"] == 0.0 and states[2]["vertical_rate_reported"] is False


def test_adsblol_parser_skips_states_without_speed_or_track():
    states = AdsbLolClient.parse({"now": 1700000000000, "ac": [
        {"hex": "a1", "lat": 37.5, "lon": -122.3, "alt_geom": 3000, "track": 90},
        {"hex": "a2", "lat": 37.5, "lon": -122.3, "alt_geom": 3000, "gs": 200},
        {"hex": "a3", "lat": 37.5, "lon": -122.3, "alt_geom": 3000, "gs": 200, "track": 90, "ground": True},
        {"hex": "a4", "lat": 37.5, "lon": -122.3, "alt_geom": 3000, "gs": 200, "track": 90},
    ]})
    assert [s["icao24"] for s in states] == ["a4"]
    assert states[0]["vertical_rate_reported"] is False


# ----------------------------------------------------------- server hygiene

def test_discover_endpoint_runs_off_the_event_loop():
    assert not inspect.iscoroutinefunction(main.discover_audio)
    assert not inspect.iscoroutinefunction(main.upload_audio)


def test_upload_is_stored_outside_static_and_capped(monkeypatch):
    resp = client.post("/api/audio/upload", files={"file": ("a.wav", b"RIFF0000WAVE", "audio/wav")})
    assert resp.status_code == 200
    data = resp.json()
    path = os.path.join(main.settings.upload_dir, data["saved_as"])
    assert os.path.isfile(path)
    assert not os.path.realpath(path).startswith(os.path.realpath(main.STATIC_DIR))
    assert client.get(data["url"]).content == b"RIFF0000WAVE"
    assert client.get(f"/static/uploads/{data['saved_as']}").status_code == 404

    monkeypatch.setattr(main.settings, "max_upload_bytes", 10)
    before = set(os.listdir(main.settings.upload_dir))
    big = client.post("/api/audio/upload", files={"file": ("b.wav", b"x" * 100, "audio/wav")})
    assert big.status_code == 413
    assert set(os.listdir(main.settings.upload_dir)) == before  # partial file removed


def test_upload_ttl_cleanup():
    resp = client.post("/api/audio/upload", files={"file": ("old.wav", b"RIFF", "audio/wav")}).json()
    path = os.path.join(main.settings.upload_dir, resp["saved_as"])
    old = time.time() - main.settings.upload_ttl_s - 10
    os.utime(path, (old, old))
    assert main.cleanup_uploads() >= 1
    assert not os.path.exists(path)


def test_uploaded_file_lookup_rejects_traversal():
    assert client.get("/api/audio/uploads/..%2F..%2Fmain.py").status_code in (400, 404)


def test_cors_does_not_allow_credentials():
    resp = client.options("/api/health", headers={"Origin": "https://example.org", "Access-Control-Request-Method": "GET"})
    assert resp.headers.get("access-control-allow-credentials") is None


def test_internal_errors_do_not_leak_details(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("secret-internal-detail")

    monkeypatch.setattr(main.flight_manager, "find_flights", boom)
    resp = client.post("/api/flights/search", json=SFO)
    assert resp.status_code == 500
    assert "secret" not in resp.text
