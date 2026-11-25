"""
Phase 2 tests: track import, trajectory conditioning, historical provider
(mocked HTTP), recording start from metadata and the kinematic pre-filter.
No test touches the network.
"""

import gzip
import io
import json
import math
import os
import shutil
import struct
import subprocess
import wave

import numpy as np
import pytest
from conftest import FIXTURES
from fastapi.testclient import TestClient

import app.main as main
from app.adsb.conditioning import condition_trajectory
from app.adsb.flight_manager import FlightManager
from app.adsb.providers import DiskCache, OpenSkyHistoricalProvider, OpenSkyTokenManager
from app.adsb.sample_flights import get_curated_scenarios
from app.adsb.track_import import ImportOptions, TrackImportError, parse_tracks, validate_uploaded_trajectory
from app.audio.metadata import parse_ffprobe_json, read_recording_start
from app.physics.doppler_engine import DopplerEngine
from app.physics.flyover_fit import fit_flyover
from app.physics.matching import TrajectoryModel
from app.physics.prefilter import prefilter_flights

TRACKS = os.path.join(FIXTURES, "tracks")
client = TestClient(main.app)
SC = get_curated_scenarios()["sfo_dual"]
CENTER = SC["search_center"]


def _read(name):
    with open(os.path.join(TRACKS, name), "rb") as fh:
        return fh.read()


def _haversine_m(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000.0 * math.asin(math.sqrt(a))


# ---------------------------------------------------------------- track import

@pytest.mark.parametrize("name,fmt,n_flights,basis", [
    ("generic.csv", "csv", 3, "unknown"),
    ("fr24.csv", "fr24_csv", 1, "barometric"),
    ("track.kml", "kml", 1, "unknown"),
    ("tracks.geojson", "geojson", 3, "geometric"),
    ("trace_full_a1b2c3.json", "readsb", 1, "geometric"),
])
def test_fixture_formats_parse_with_correct_units(name, fmt, n_flights, basis):
    res = parse_tracks(_read(name), name)
    assert res["format"] == fmt
    assert len(res["flights"]) == n_flights
    truth = {f["callsign"]: f for f in SC["flights"]}
    for f in res["flights"]:
        assert f["data_source"] == "uploaded" and f["altitude_basis"] == basis
        ref = truth[f["callsign"]]
        i = ref["timestamps"].index(f["timestamps"][0])
        assert abs(f["altitudes"][0] - ref["altitudes"][i]) < 1.0  # feet -> metres handled
        assert abs(f["latitudes"][0] - ref["latitudes"][i]) < 1e-4
        if f["speeds"]:
            assert abs(f["speeds"][0] - ref["speeds"][i]) < 0.5  # knots -> m/s handled


def test_import_endpoint_and_gzip():
    resp = client.post("/api/flights/import", files={"file": ("trace.json.gz", gzip.compress(_read("trace_full_a1b2c3.json")))})
    assert resp.status_code == 200
    body = resp.json()
    assert body["format"] == "readsb" and body["count"] == 1 and body["data_source"] == "uploaded"
    assert body["flights"][0]["callsign"] == "UAL1714"


def test_naive_iso_times_need_a_timezone():
    csv_text = "time,lat,lon,alt_m\n2024-05-01T12:00:00,37.5,-122.3,1000\n2024-05-01T12:00:01,37.5005,-122.3,1000\n" \
               "2024-05-01T12:00:02,37.501,-122.3,1000\n"
    with pytest.raises(TrackImportError, match="time-zone"):
        parse_tracks(csv_text.encode(), "a.csv")
    utc = parse_tracks(csv_text.encode(), "a.csv", options=ImportOptions(timezone="UTC"))["flights"][0]
    plus2 = parse_tracks(csv_text.encode(), "a.csv", options=ImportOptions(timezone="+02:00"))["flights"][0]
    assert utc["timestamps"][0] - plus2["timestamps"][0] == pytest.approx(7200.0)
    resp = client.post("/api/flights/import", files={"file": ("a.csv", csv_text.encode())})
    assert resp.status_code == 400 and "time-zone" in resp.json()["detail"]


def test_altitude_unit_must_be_known_and_plausible():
    rows = "time,lat,lon,altitude\n1700000000,37.5,-122.3,35000\n1700000001,37.5005,-122.3,35000\n1700000002,37.501,-122.3,35000\n"
    with pytest.raises(TrackImportError, match="does not state its unit"):
        parse_tracks(rows.encode(), "a.csv")
    with pytest.raises(TrackImportError, match="feet vs metres"):
        parse_tracks(rows.encode(), "a.csv", options=ImportOptions(altitude_unit="m"))
    ok = parse_tracks(rows.encode(), "a.csv", options=ImportOptions(altitude_unit="ft"))["flights"][0]
    assert ok["altitudes"][0] == pytest.approx(35000 * 0.3048)


def test_relative_times_and_bad_coordinates_rejected():
    rel = "time,lat,lon,alt_m\n0,37.5,-122.3,1000\n1,37.5005,-122.3,1000\n2,37.501,-122.3,1000\n"
    with pytest.raises(TrackImportError, match="relative"):
        parse_tracks(rel.encode(), "a.csv")
    bad = "time,lat,lon,alt_m\n1700000000,97.5,-122.3,1000\n1700000001,37.5005,-122.3,1000\n1700000002,37.501,-122.3,1000\n"
    with pytest.raises(TrackImportError, match="out of range"):
        parse_tracks(bad.encode(), "a.csv")


def test_speed_unit_mismatch_is_flagged():
    lines = ["time,lat,lon,alt_m,speed_mps"]
    for i in range(10):  # ~150 m/s northbound, but speed column holds knots
        lines.append(f"{1700000000 + i},{37.5 + i * 150 / 111000:.6f},-122.3,1000,{150 / 0.514444:.1f}")
    res = parse_tracks("\n".join(lines).encode(), "a.csv")
    assert any("speed unit" in w for w in res["warnings"])


def test_readsb_skips_ground_and_prefers_geometric():
    trace = {"icao": "abc123", "timestamp": 1700000000, "trace": [
        [0, 37.5, -122.3, "ground", 10, 90, 0, 0, None, "adsb_icao", None],
        [1, 37.5, -122.3, 3000, 250, 90, 0, 0, {"flight": "TST1"}, "adsb_icao", 3100],
        [2, 37.5, -122.299, 3000, 250, 90, 0, 0, None, "adsb_icao", 3100],
        [3, 37.5, -122.298, 3000, 250, 90, 0, 0, None, "adsb_icao", 3100],
    ]}
    res = parse_tracks(json.dumps(trace).encode(), "t.json")
    f = res["flights"][0]
    assert len(f["timestamps"]) == 3 and f["altitude_basis"] == "geometric"
    assert f["altitudes"][0] == pytest.approx(3100 * 0.3048)
    assert any("ground" in w for w in res["warnings"])


def test_client_trajectories_are_revalidated_and_retagged():
    f = dict(parse_tracks(_read("tracks.geojson"), "x")["flights"][0], data_source="opensky_live")
    assert validate_uploaded_trajectory(f)["data_source"] == "uploaded"
    with pytest.raises(TrackImportError):
        validate_uploaded_trajectory(dict(f, latitudes=f["latitudes"][:-1]))


def _discovered_events():
    disc = client.post("/api/audio/discover", data={"sample_id": "sfo_dual"}).json()
    return [{"id": r["id"], "group_id": r["harmonic_group_id"], "traced_curve": r["points"]}
            for r in disc["ridges"] if r["doppler_like"]]


def test_geolocate_with_imported_tracks_end_to_end():
    flights = client.post("/api/flights/import", files={"file": ("generic.csv", _read("generic.csv"))}).json()["flights"]
    # Add realistic jitter (CPR-like position noise) to the uploaded tracks.
    rng = np.random.default_rng(5)
    for f in flights:
        n = len(f["latitudes"])
        f["latitudes"] = list(np.array(f["latitudes"]) + rng.normal(0, 5 / 111000, n))
        f["longitudes"] = list(np.array(f["longitudes"]) + rng.normal(0, 5 / 88000, n))
    resp = client.post("/api/geolocate", json={
        "center_lat": CENTER["lat"], "center_lon": CENTER["lon"], "radius_km": 8.0, "events": _discovered_events(),
        "trajectories": flights, "grid_resolution": 15,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success" and data["data_source"] == "uploaded"
    best = data["best_location"]
    assert _haversine_m(best["lat"], best["lon"], CENTER["lat"], CENTER["lon"]) < 150.0
    assert data["decision"]["status"] == "supported"
    used = [f for f in data["evaluated_flights"] if not f["discarded"]]
    assert all(f["conditioning"]["method"] == "gaussian_local_quadratic" for f in used)
    assert {d["flight_id"] for d in data["discarded_flights"]} == {"ASA331"}


# ---------------------------------------------------------------- conditioning

def test_conditioning_reduces_velocity_noise_and_keeps_gaps():
    ref = SC["flights"][0]
    rng = np.random.default_rng(0)
    n = len(ref["timestamps"])
    keep = [i for i in range(n) if not (40 <= i < 70)]
    noisy = {k: [ref[k][i] for i in keep] for k in ("timestamps", "latitudes", "longitudes", "altitudes")}
    noisy["latitudes"] = list(np.array(noisy["latitudes"]) + rng.normal(0, 6 / 111000, len(keep)))
    noisy["longitudes"] = list(np.array(noisy["longitudes"]) + rng.normal(0, 6 / 88000, len(keep)))
    noisy.update(flight_id="X", data_source="uploaded")
    true_v = np.array([165 * math.sin(math.radians(305)), 165 * math.cos(math.radians(305))])

    eng = DopplerEngine()
    raw_model = TrajectoryModel(noisy, CENTER["lat"], CENTER["lon"], eng)
    cond = condition_trajectory(noisy)
    cond_model = TrajectoryModel(cond, CENTER["lat"], CENTER["lon"], eng)
    raw_err = np.median(np.hypot(*(raw_model.vel[:, :2] - true_v).T))
    cond_err = np.median(np.hypot(*(cond_model.vel[:, :2] - true_v).T))
    assert cond_err < 0.6 * raw_err
    assert cond["conditioning"]["segments"] == 2
    assert len(cond_model.gap_after) == 1  # the 30 s hole is still a gap for support checks
    assert np.all(np.diff(cond["timestamps"])[np.diff(cond["timestamps"]) < 20] == pytest.approx(1.0))

    with_rep = dict(noisy, speeds=[165.0] * len(keep), headings=[305.0] * len(keep))
    fused = TrajectoryModel(condition_trajectory(with_rep), CENTER["lat"], CENTER["lon"], eng)
    assert np.median(np.hypot(*(fused.vel[:, :2] - true_v).T)) < 0.5 * cond_err


# -------------------------------------------------------- historical provider

class _Resp:
    def __init__(self, status, body=None, headers=None):
        self.status_code, self._body, self.headers = status, body, headers or {}

    def json(self):
        return self._body


class _FakeSession:
    def __init__(self, states_by_time=None, tracks=None, status=200):
        self.states_by_time = states_by_time or {}
        self.tracks = tracks or {}
        self.status = status
        self.gets, self.posts = [], []

    def post(self, url, data=None, **kw):
        self.posts.append((url, data))
        assert data["grant_type"] == "client_credentials" and data["client_id"] == "id"
        return _Resp(200, {"access_token": f"tok{len(self.posts)}", "expires_in": 1800})

    def get(self, url, params=None, headers=None, **kw):
        assert headers["Authorization"].startswith("Bearer tok")
        self.gets.append((url, dict(params)))
        if self.status != 200:
            return _Resp(self.status, {}, {"X-Rate-Limit-Retry-After-Seconds": "60"})
        if url.endswith("/states/all"):
            return _Resp(200, {"time": params["time"], "states": self.states_by_time.get(params["time"], [])})
        return _Resp(200, self.tracks.get(params["icao24"])) if params["icao24"] in self.tracks else _Resp(404)


def _states_for(flight):
    out = {}
    for i, t in enumerate(flight["timestamps"]):
        if int(t) % 10:
            continue
        out[int(t)] = [[flight["icao24"], flight["callsign"], "US", int(t), int(t), flight["longitudes"][i],
                        flight["latitudes"][i], flight["altitudes"][i] + 40, False, flight["speeds"][i],
                        flight["headings"][i], flight["vertical_rates"][i], None, flight["altitudes"][i]]]
    return out


def test_opensky_token_manager_caches_and_refreshes():
    now = [1000.0]
    sess = _FakeSession()
    tm = OpenSkyTokenManager("id", "secret", session=sess, clock=lambda: now[0])
    assert tm.token() == "tok1" and tm.token() == "tok1"
    now[0] += 1800.0
    assert tm.token() == "tok2"
    assert not OpenSkyTokenManager(None, None).configured


def test_opensky_states_provider_assembles_tracks_and_caches(tmp_path):
    f = SC["flights"][0]
    t0, t1 = f["timestamps"][20], f["timestamps"][80]
    now = t1 + 1200.0  # within the last hour, older than the cache threshold
    sess = _FakeSession(_states_for(f))
    prov = OpenSkyHistoricalProvider(OpenSkyTokenManager("id", "secret", session=sess, clock=lambda: now),
                                     DiskCache(str(tmp_path)), session=sess, clock=lambda: now)
    res = prov.get_tracks(CENTER["lat"], CENTER["lon"], 8.0, t0, t1)
    assert len(res["flights"]) == 1
    tr = res["flights"][0]
    assert tr["data_source"] == "historical" and tr["altitude_basis"] == "geometric"
    assert tr["altitudes"][0] == pytest.approx(f["altitudes"][f["timestamps"].index(tr["timestamps"][0])])
    n_calls = len(sess.gets)
    assert n_calls > 3 and all(p["time"] % 10 == 0 for _, p in sess.gets)
    prov.get_tracks(CENTER["lat"], CENTER["lon"], 8.0, t0, t1)
    assert len(sess.gets) == n_calls  # served from the disk cache


def test_opensky_old_window_needs_hints_and_uses_tracks():
    f = SC["flights"][0]
    now = f["timestamps"][0] + 5 * 86400.0
    path = [[int(f["timestamps"][i]), f["latitudes"][i], f["longitudes"][i], f["altitudes"][i], 305.0, False]
            for i in (0, 60, 120)]
    sess = _FakeSession(tracks={f["icao24"]: {"icao24": f["icao24"], "calllsign": "UAL1714 ", "path": path}})
    prov = OpenSkyHistoricalProvider(OpenSkyTokenManager("id", "secret", session=sess, clock=lambda: now),
                                     None, session=sess, clock=lambda: now)
    t0, t1 = f["timestamps"][30], f["timestamps"][90]
    res = prov.get_tracks(CENTER["lat"], CENTER["lon"], 8.0, t0, t1)
    assert res["flights"] == [] and any("icao24" in n for n in res["notes"])
    res = prov.get_tracks(CENTER["lat"], CENTER["lon"], 8.0, t0, t1, icao24_hints=[f["icao24"]])
    tr = res["flights"][0]
    assert tr["callsign"] == "UAL1714" and tr["altitude_basis"] == "barometric"
    assert np.allclose(np.diff(tr["timestamps"]), 1.0)


def test_opensky_rate_limit_becomes_a_note():
    f = SC["flights"][0]
    now = f["timestamps"][-1] + 100.0
    sess = _FakeSession(status=429)
    prov = OpenSkyHistoricalProvider(OpenSkyTokenManager("id", "secret", session=sess, clock=lambda: now),
                                     None, session=sess, clock=lambda: now)
    res = prov.get_tracks(CENTER["lat"], CENTER["lon"], 8.0, f["timestamps"][10], f["timestamps"][50])
    assert res["flights"] == [] and any("429" in n for n in res["notes"])


def test_flight_manager_prefers_historical_provider():
    f = SC["flights"][0]
    now = f["timestamps"][-1] + 1200.0
    sess = _FakeSession(_states_for(f))
    prov = OpenSkyHistoricalProvider(OpenSkyTokenManager("id", "secret", session=sess, clock=lambda: now),
                                     None, session=sess, clock=lambda: now)

    class NoLive:
        def get_live_states(self, *a, **k):
            raise AssertionError("live data must not be used for a historical window")

    fm = FlightManager(NoLive(), NoLive(), historical_providers=[prov], clock=lambda: now)
    res = fm.find_flights(CENTER["lat"], CENTER["lon"], 8.0, f["timestamps"][10], f["timestamps"][60])
    assert res["data_source"] == "historical" and res["flights"][0]["data_source"] == "historical"


# ------------------------------------------------------------------- metadata

def _wav_with_chunks(path, extra_chunks, sr=8000, seconds=1.0):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(b"\x00\x00" * int(sr * seconds))
    raw = buf.getvalue()
    body = raw[12:]
    for cid, payload in extra_chunks:
        pad = b"\x00" if len(payload) % 2 else b""
        body = cid + struct.pack("<I", len(payload)) + payload + pad + body
    with open(path, "wb") as fh:
        fh.write(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)


def _bext(date, tod, time_ref):
    p = bytearray(602)
    p[320:330] = date.encode()
    p[330:338] = tod.encode()
    p[338:346] = struct.pack("<II", time_ref & 0xFFFFFFFF, time_ref >> 32)
    return bytes(p)


def test_bwf_time_reference_gives_start(tmp_path):
    path = str(tmp_path / "rec.wav")
    _wav_with_chunks(path, [(b"bext", _bext("2024-05-01", "12:00:09", (12 * 3600 + 5) * 8000 + 4000))])
    meta = read_recording_start(path)
    assert meta["source"] == "bwf_bext" and meta["field"] == "time_reference"
    assert meta["timezone_known"] is False and meta["requires_confirmation"] is True
    assert meta["start_epoch"] == pytest.approx(1714564805.5)  # 2024-05-01T12:00:05.5 "as UTC"


def test_bwf_origination_time_and_ixml(tmp_path):
    p1 = str(tmp_path / "a.wav")
    _wav_with_chunks(p1, [(b"bext", _bext("2024-05-01", "12:00:09", 0))])
    assert read_recording_start(p1)["start_epoch"] == pytest.approx(1714564809.0)
    p2 = str(tmp_path / "b.wav")
    ixml = (b"<BWFXML><BWF_ORIGINATION_DATE>2024-05-01</BWF_ORIGINATION_DATE>"
            b"<BWF_ORIGINATION_TIME>12:00:09</BWF_ORIGINATION_TIME>"
            b"<BWF_TIME_REFERENCE_LOW>345600000</BWF_TIME_REFERENCE_LOW>"
            b"<BWF_TIME_REFERENCE_HIGH>0</BWF_TIME_REFERENCE_HIGH></BWFXML>")
    _wav_with_chunks(p2, [(b"iXML", ixml)])
    m = read_recording_start(p2)
    assert m["source"] == "ixml" and m["start_epoch"] == pytest.approx(1714564800.0)  # 12:00:00 at 8 kHz


def test_ffprobe_creation_time_parsing():
    m = parse_ffprobe_json({"format": {"duration": "30.5", "tags": {"creation_time": "2024-05-01T12:00:00.000000Z"}}})
    assert m["timezone_known"] and m["start_epoch"] == pytest.approx(1714564800.0)
    assert m["alternatives"][0]["start_epoch"] == pytest.approx(1714564800.0 - 30.5)
    assert parse_ffprobe_json({"format": {"tags": {}}}) is None


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")
def test_ffprobe_integration(tmp_path):
    out = str(tmp_path / "a.m4a")
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", "1",
                    "-metadata", "creation_time=2024-05-01T12:00:00Z", out], check=True)
    m = read_recording_start(out)
    assert m["source"] == "ffprobe_creation_time" and m["start_epoch"] == pytest.approx(1714564800.0)


def test_metadata_endpoint_and_upload_report(tmp_path):
    path = str(tmp_path / "rec.wav")
    _wav_with_chunks(path, [(b"bext", _bext("2024-05-01", "12:00:09", 0))])
    with open(path, "rb") as fh:
        data = fh.read()
    meta = client.post("/api/audio/metadata", files={"file": ("rec.wav", data)}).json()
    assert meta["start_epoch"] == pytest.approx(1714564809.0)
    up = client.post("/api/audio/upload", files={"file": ("rec.wav", data)}).json()
    assert up["recording_metadata"]["source"] == "bwf_bext"
    none = client.post("/api/audio/metadata", data={"sample_id": "sfo_dual"}).json()
    assert none["start_epoch"] is None


# ------------------------------------------------------------------ prefilter

def _fits(events):
    out = {}
    for ev in events:
        pts = ev["traced_curve"]
        out[ev["id"]] = fit_flyover([p["time"] for p in pts], [p["freq"] for p in pts])
    return out


def test_prefilter_discards_impossible_decoy_only():
    events = _discovered_events()
    kept, discarded = prefilter_flights(SC["flights"], events, _fits(events))
    assert {f["flight_id"] for f in kept} == {"UAL1714", "SWA2286"}
    assert [d["flight_id"] for d in discarded] == ["ASA331"]
    reasons = {r["reason"] for g in discarded[0]["groups"] for r in g["reasons"]}
    assert reasons & {"speed_incompatible", "height_incompatible"}


def test_prefilter_time_overlap_with_trusted_clock():
    events = _discovered_events()
    start = SC["recording_start_epoch"]
    shifted = dict(SC["flights"][0], flight_id="LATE", timestamps=[t + 3600 for t in SC["flights"][0]["timestamps"]])
    kept, discarded = prefilter_flights(SC["flights"][:2] + [shifted], events, {}, clock_window=(start - 2, start + 2))
    assert [d["flight_id"] for d in discarded] == ["LATE"]
    assert discarded[0]["groups"][0]["reasons"][0]["reason"] == "no_time_overlap"
    # Without a trusted clock time cannot exclude anything, and without reliable fits kinematics cannot either.
    kept, discarded = prefilter_flights(SC["flights"][:2] + [shifted], events, {})
    assert discarded == []


def test_prefilter_ignores_unreliable_fits():
    events = [{"id": "E1", "traced_curve": [{"time": float(t), "freq": 800.0 - t} for t in range(6)]}]
    fits = {"E1": {"speed_mps": 50.0, "cpa_slant_m": 100.0, "rmse_hz": 0.1, "n_samples": 6, "parameter_at_bound": {}}}
    kept, discarded = prefilter_flights(SC["flights"], events, fits)
    assert discarded == [] and len(kept) == 3
