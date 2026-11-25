"""
Phase 4 tests: the batched kernel and evaluation are equivalent to the
reference per-event implementation, parallel and coarse-to-fine runs give the
same answer, and the async jobs API (polling, SSE, cancellation) works.
"""

import time

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.adsb.sample_flights import get_curated_scenarios
from app.audio.generator import simulate_events
from app.jobs import JobStore
from app.physics.matching import (JointMatcher, MatchOptions, _CurvePack, _EventPack, fit_event, fit_rows,
                                  wind_vector)

SC = get_curated_scenarios()["sfo_dual"]
C = SC["search_center"]
client = TestClient(main.app)


def _events(wind=(0.0, 0.0)):
    return simulate_events(SC["flights"], (C["lat"], C["lon"]), SC["recording_start_epoch"],
                           SC["audio_duration_sec"], wind)


def _reference_point(m, obs, starts=None, opts=None, events=None):
    """The pre-batching evaluate_point, built on the per-event reference fit."""
    opts = opts or m.opts
    curves = [mm.arrival(obs, opts.speed_of_sound, m.wind(opts)) for mm in m.models]
    starts = m._candidate_starts(curves) if starts is None else starts
    total, choice = m._joint(curves, starts, opts, events)
    if not np.any(np.isfinite(total)):
        return float("inf"), None, None
    k = int(np.argmin(total))
    fine = m._clip_clock(starts[k] + np.arange(-opts.fine_halfwidth_s, opts.fine_halfwidth_s + 1e-9, opts.fine_step_s))
    tf, cf = m._joint(curves, fine, opts, events)
    if np.any(np.isfinite(tf)) and np.min(tf) <= total[k]:
        kf = int(np.argmin(tf))
        return float(tf[kf]), float(fine[kf]), {g: int(cf[g][kf]) for g in m.group_ids}
    return float(total[k]), float(starts[k]), {g: int(choice[g][k]) for g in m.group_ids}


def test_fit_rows_matches_reference_fit_event():
    rng = np.random.default_rng(3)
    events = _events(wind_vector(8.0, 120.0))
    for fit_f0, drift in ((True, 0.02), (True, 0.001), (False, 0.02)):
        raw = [dict(e, f0=820.0) for e in events]
        m = JointMatcher(SC["flights"], raw, C["lat"], C["lon"],
                         MatchOptions(fit_f0=fit_f0, drift_fraction=drift, wind_e=3.0, wind_n=-2.0))
        obs_list = [m.observer(float(x), float(y), m.opts) for x, y in rng.uniform(-3000, 3000, (3, 2))]
        curves = [mm.arrival(o, m.opts.speed_of_sound, m.wind(m.opts)) for o in obs_list for mm in m.models]
        cp, ep = _CurvePack(curves), _EventPack(m.events)
        for ci, curve in enumerate(curves):
            for ei, ev in enumerate(m.events):
                starts = curve["t_cpa"] - ev.cpa_time_s + np.linspace(-8, 8, 33)
                ref = fit_event(curve, ev, starts, m.opts)["loss"]
                got = fit_rows(cp, ep, np.full(starts.size, ci), np.full(starts.size, ei), starts, m.opts)
                assert np.array_equal(np.isfinite(ref), np.isfinite(got))
                fin = np.isfinite(ref)
                assert np.allclose(got[fin], ref[fin], rtol=1e-9, atol=1e-9)


def test_batched_evaluation_matches_reference_joint():
    m = JointMatcher(SC["flights"], _events(), C["lat"], C["lon"], MatchOptions())
    obs = [m.observer(x, y, m.opts) for x, y in ((0.0, 0.0), (1500.0, -800.0), (-2500.0, 2000.0), (6000.0, 6000.0))]
    for ob, got in zip(obs, m.evaluate_cells(obs, events=m.events_coarse)):
        ref = _reference_point(m, ob, events=m.events_coarse)
        if not np.isfinite(ref[0]):
            assert not np.isfinite(got[0])
            continue
        assert got[0] == pytest.approx(ref[0], rel=1e-9, abs=1e-12)
        assert got[1] == pytest.approx(ref[1]) and got[2] == ref[2]


def _summary(res, m):
    lat, lon = m.to_geo(res["best"]["x"], res["best"]["y"])
    return lat, lon, res["robust_rmse_hz"], [round(a["distance_from_best_m"], 3) for a in res["alternatives"]], \
        (res["sensitivity"] or {}).get("radius_m")


def test_parallel_run_is_identical_to_serial():
    ev = _events()
    m1 = JointMatcher(SC["flights"], ev, C["lat"], C["lon"], MatchOptions(workers=1))
    m2 = JointMatcher(SC["flights"], ev, C["lat"], C["lon"], MatchOptions(workers=2))
    r1, r2 = m1.run(8.0, 13, sensitivity=True), m2.run(8.0, 13, sensitivity=True)
    s1, s2 = _summary(r1, m1), _summary(r2, m2)
    assert s1[:2] == s2[:2] and s1[2] == pytest.approx(s2[2], rel=1e-12) and s1[3:] == s2[3:]


def test_coarse_to_fine_finds_the_same_optimum():
    ev = _events()
    full = JointMatcher(SC["flights"], ev, C["lat"], C["lon"], MatchOptions(grid_strategy="full"))
    c2f = JointMatcher(SC["flights"], ev, C["lat"], C["lon"], MatchOptions(grid_strategy="coarse_to_fine"))
    rf, rc = full.run(8.0, 21, sensitivity=False), c2f.run(8.0, 21, sensitivity=False)
    assert c2f.grid["strategy"] == "coarse_to_fine" and c2f.grid["evaluated"].sum() < 0.7 * full.grid["inside"].sum()
    assert abs(rf["best"]["x"] - rc["best"]["x"]) < 1.0 and abs(rf["best"]["y"] - rc["best"]["y"]) < 1.0
    # Evaluated cells carry the exact loss; interpolated cells never become local minima.
    ev_mask = c2f.grid["evaluated"]
    assert np.allclose(c2f.grid["loss"][ev_mask], full.grid["loss"][ev_mask])
    assert all(ev_mask[p] for p in c2f.local_minima())


def test_progress_callback_reports_stages():
    seen = []
    m = JointMatcher(SC["flights"][:2], _events(), C["lat"], C["lon"], MatchOptions())
    m.run(8.0, 11, sensitivity=True, progress=lambda stage, f: seen.append((stage, f)))
    stages = [s for s, _ in seen]
    assert stages[0] == "grid" and "refine" in stages and stages[-1] == "sensitivity"
    assert seen[-1][1] == pytest.approx(1.0)


# ------------------------------------------------------------------ jobs API

def _payload(**kw):
    p = {"center_lat": C["lat"], "center_lon": C["lon"], "radius_km": 8.0, "events": _events(),
         "scenario_id": "sfo_dual", "grid_resolution": 11, "sensitivity": False}
    p.update(kw)
    return p


def _wait(job_id, timeout=120.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("done", "error", "cancelled"):
            return job
        time.sleep(0.1)
    raise AssertionError("job did not finish")


def test_job_returns_same_result_as_sync_endpoint():
    sub = client.post("/api/jobs/geolocate", json=_payload())
    assert sub.status_code == 202
    job = _wait(sub.json()["job_id"])
    assert job["status"] == "done" and job["progress"] == 1.0
    sync = client.post("/api/geolocate", json=_payload()).json()
    res = job["result"]
    assert set(res) == set(sync)
    assert res["best_location"]["lat"] == sync["best_location"]["lat"]
    assert res["decision"]["reason_code"] == sync["decision"]["reason_code"]


def test_job_events_stream_ends_with_done():
    jid = client.post("/api/jobs/geolocate", json=_payload()).json()["job_id"]
    with client.stream("GET", f"/api/jobs/{jid}/events") as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        kinds = [line.split(": ", 1)[1] for line in resp.iter_lines() if line.startswith("event: ")]
    assert kinds[-1] == "done"


def test_job_errors_and_unknown_ids():
    assert client.post("/api/jobs/geolocate", json=_payload(events=[])).status_code == 400
    assert client.get("/api/jobs/does-not-exist").status_code == 404
    jid = client.post("/api/jobs/geolocate", json=_payload(scenario_id="nope")).json()["job_id"]
    job = _wait(jid)
    assert job["status"] == "error" and job["error"]["status_code"] == 400


def test_job_store_cancel_and_ttl():
    now = [1000.0]
    store = JobStore(max_workers=1, ttl_s=10.0, clock=lambda: now[0])
    gate = {"go": False}

    def slow(progress):
        while not gate["go"]:
            progress("grid", 0.1)
            time.sleep(0.01)
        return {"ok": True}

    a = store.submit(slow)
    b = store.submit(lambda progress: {"ok": True})  # queued behind a
    assert store.cancel(b) and store.get(b)["status"] == "cancelled"
    time.sleep(0.05)
    assert store.cancel(a)
    for _ in range(200):
        if store.get(a)["status"] == "cancelled":
            break
        time.sleep(0.01)
    assert store.get(a)["status"] == "cancelled"
    now[0] += 11.0
    assert store.get(a) is None and store.get(b) is None
