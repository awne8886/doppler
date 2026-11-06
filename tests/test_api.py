"""
API Integration tests for Doppler Acoustic Aircraft Geolocator.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"


def test_list_scenarios():
    resp = client.get("/api/scenarios")
    assert resp.status_code == 200
    scenarios = resp.json()
    assert len(scenarios) >= 3
    ids = [s["id"] for s in scenarios]
    assert "sfo_ual421" in ids
    assert "lhr_baw282" in ids
    assert "jfk_dal410" in ids


def test_search_flights():
    resp = client.post("/api/flights/search", json={
        "center_lat": 37.5585,
        "center_lon": -122.2711,
        "radius_km": 10.0,
        "scenario_id": "sfo_ual421",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] >= 1
    flight = data["flights"][0]
    assert flight["callsign"] == "UAL421"
    assert len(flight["latitudes"]) > 10


def test_geolocate_endpoint():
    # Use ground truth Doppler curve from the curated scenario
    from app.adsb.sample_flights import get_curated_scenarios
    from app.physics.doppler_engine import DopplerEngine

    scenario = get_curated_scenarios()["sfo_ual421"]
    traj = scenario["trajectory"]
    gt = traj["ground_truth_mic"]
    f0 = scenario["f0_target"]

    engine = DopplerEngine()
    t_emit = np.array(traj["timestamps"]) - traj["timestamps"][0]
    px, py, pz = engine.geodetic_to_cartesian(
        np.array(traj["latitudes"]), np.array(traj["longitudes"]), np.array(traj["altitudes"]),
        gt["lat"], gt["lon"], 0.0
    )
    plane_pos = np.stack([px, py, pz], axis=1)
    vx = np.gradient(px, t_emit)
    vy = np.gradient(py, t_emit)
    vz = np.gradient(pz, t_emit)
    plane_vel = np.stack([vx, vy, vz], axis=1)

    t_obs, f_obs = engine.compute_doppler_curve(t_emit, plane_pos, plane_vel, np.array([0., 0., 0.]), f0)
    t_samples = np.linspace(t_obs[10], t_obs[-10], 20)
    f_samples = np.interp(t_samples, t_obs, f_obs)
    traced_curve = [{"time": float(t - t_samples[0]), "freq": float(f)} for t, f in zip(t_samples, f_samples)]

    payload = {
        "center_lat": 37.5585,
        "center_lon": -122.2711,
        "radius_km": 10.0,
        "f0": f0,
        "traced_curve": traced_curve,
        "scenario_id": "sfo_ual421",
        "grid_resolution": 15,
    }

    resp = client.post("/api/geolocate", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert "best_location" in data
    assert "heatmap" in data
    assert len(data["heatmap"]) > 50
    assert data["best_location"]["score"] > 95.0
    assert data["best_location"]["rmse_hz"] < 5.0
    assert "curve_comparison" in data
    assert len(data["curve_comparison"]) == len(traced_curve)


def test_serve_frontend_index():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "DOPPLER" in resp.text
    assert "spectrogramCanvas" in resp.text


if __name__ == "__main__":
    test_health_check()
    test_list_scenarios()
    test_search_flights()
    test_geolocate_endpoint()
    test_serve_frontend_index()
    print("All API integration tests passed successfully!")
