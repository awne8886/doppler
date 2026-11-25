"""
Unit tests for the acoustic Doppler physics engine and the joint matcher's
single-flyover recovery (ported from the removed legacy GridScorer test).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.physics.doppler_engine import DopplerEngine
from app.physics.matching import JointMatcher, MatchOptions
from app.adsb.sample_flights import generate_linear_flight_trajectory, trajectory_start_for_cpa


def test_doppler_equation_and_radial_velocity():
    engine = DopplerEngine(speed_of_sound=340.0)
    obs = np.array([0.0, 0.0, 0.0])
    f0 = 1000.0

    # Plane 1: Directly approaching observer along x-axis at 170 m/s (Mach 0.5)
    plane_pos_approach = np.array([[-1000.0, 0.0, 0.0]])
    plane_vel_approach = np.array([[170.0, 0.0, 0.0]])
    d, v_r = engine.compute_radial_velocity(plane_pos_approach, plane_vel_approach, obs)
    assert np.isclose(d[0], 1000.0)
    assert np.isclose(v_r[0], 170.0)

    t_obs, f_obs = engine.compute_doppler_curve(
        np.array([0.0]), plane_pos_approach, plane_vel_approach, obs, f0
    )
    # Doppler formula: f = f0 * (c / (c - v_r)) = 1000 * (340 / (340 - 170)) = 2000.0 Hz
    assert np.isclose(f_obs[0], 2000.0)
    # Sound delay: t_obs = t_emit + d / c = 0 + 1000 / 340
    assert np.isclose(t_obs[0], 1000.0 / 340.0)

    # Plane 2: Receding directly away from observer along x-axis at 170 m/s
    plane_pos_recede = np.array([[1000.0, 0.0, 0.0]])
    plane_vel_recede = np.array([[170.0, 0.0, 0.0]])
    d_rec, v_r_rec = engine.compute_radial_velocity(plane_pos_recede, plane_vel_recede, obs)
    assert np.isclose(v_r_rec[0], -170.0)

    _, f_obs_rec = engine.compute_doppler_curve(
        np.array([0.0]), plane_pos_recede, plane_vel_recede, obs, f0
    )
    # Doppler formula: f = f0 * (c / (c - (-170))) = 1000 * (340 / 510) = 666.67 Hz
    assert np.isclose(f_obs_rec[0], 1000.0 * (340.0 / 510.0))

    # Plane 3: At closest point of approach (CPA), velocity perpendicular to position vector
    plane_pos_cpa = np.array([[0.0, 1000.0, 0.0]])
    plane_vel_cpa = np.array([[170.0, 0.0, 0.0]])
    d_cpa, v_r_cpa = engine.compute_radial_velocity(plane_pos_cpa, plane_vel_cpa, obs)
    assert np.isclose(v_r_cpa[0], 0.0)
    _, f_obs_cpa = engine.compute_doppler_curve(
        np.array([0.0]), plane_pos_cpa, plane_vel_cpa, obs, f0
    )
    # At CPA, v_r = 0, so f_obs = f0 exactly
    assert np.isclose(f_obs_cpa[0], f0)


def test_geodetic_cartesian_roundtrip():
    engine = DopplerEngine()
    ref_lat, ref_lon = 37.5585, -122.2711

    # Test point offset by ~2km
    test_lat, test_lon, test_alt = 37.5750, -122.2500, 1500.0
    x, y, z = engine.geodetic_to_cartesian(
        np.array([test_lat]), np.array([test_lon]), np.array([test_alt]),
        ref_lat, ref_lon, 0.0
    )
    lat_back, lon_back, alt_back = engine.cartesian_to_geodetic(x, y, z, ref_lat, ref_lon, 0.0)

    assert np.isclose(lat_back[0], test_lat, atol=1e-5)
    assert np.isclose(lon_back[0], test_lon, atol=1e-5)
    assert np.isclose(alt_back[0], test_alt, atol=1e-2)


def _mirror_about_track(traj, ref_lat, ref_lon):
    """Mirror image of the reference point (local origin) about the straight ground track."""
    engine = DopplerEngine()
    x, y, _ = engine.geodetic_to_cartesian(np.array(traj["latitudes"]), np.array(traj["longitudes"]),
                                           np.zeros(len(traj["latitudes"])), ref_lat, ref_lon, 0.0)
    p0 = np.array([x[0], y[0]])
    u = np.array([x[-1] - x[0], y[-1] - y[0]])
    u /= np.linalg.norm(u)
    foot = p0 + u * np.dot(-p0, u)  # projection of the origin onto the track
    return 2.0 * foot  # reflection of the origin


def test_joint_matcher_single_flyover_recovery():
    """A clean single flyover is recovered up to its mirror ambiguity, with a near-perfect fit."""
    engine = DopplerEngine()
    mic = (37.5585, -122.2711)
    f0 = 800.0
    start = trajectory_start_for_cpa(mic, 315.0, 170.0, 900.0, 30.0)
    traj = generate_linear_flight_trajectory(
        flight_id="TEST01", callsign="TEST01", aircraft_type="Test Jet",
        start_lat=start[0], start_lon=start[1], start_alt=1200.0, heading_deg=315.0,
        speed_mps=170.0, vertical_rate_mps=0.0, duration_sec=60.0, start_epoch=1700000000.0,
        f0_target=f0, ground_truth_obs=mic,
    )

    t_emit = np.array(traj["timestamps"]) - traj["timestamps"][0]
    px, py, pz = engine.geodetic_to_cartesian(
        np.array(traj["latitudes"]), np.array(traj["longitudes"]), np.array(traj["altitudes"]), mic[0], mic[1], 0.0
    )
    plane_pos = np.stack([px, py, pz], axis=1)
    plane_vel = np.stack([np.gradient(px, t_emit), np.gradient(py, t_emit), np.gradient(pz, t_emit)], axis=1)
    t_obs, f_obs = engine.compute_doppler_curve(t_emit, plane_pos, plane_vel, np.array([0.0, 0.0, 0.0]), f0)
    t_samples = np.linspace(t_obs[10], t_obs[-10], 20)
    f_samples = np.interp(t_samples, t_obs, f_obs)
    events = [{"id": "E1", "traced_curve": [{"time": float(t - t_samples[0]), "freq": float(f)}
                                            for t, f in zip(t_samples, f_samples)]}]

    m = JointMatcher([traj], events, mic[0], mic[1], MatchOptions())
    res = m.run(10.0, 21, sensitivity=False)
    assert res["status"] == "ok"
    assert res["robust_rmse_hz"] < 0.5
    assert int(np.sum(m.grid["inside"])) > 100
    d = res["diagnostics"][0]
    assert d["matched"] and d["rmse_hz"] < 1.0 and abs(d["f0_hz"] - f0) < 2.0
    assert len(d["times"]) == len(events[0]["traced_curve"])
    best = np.array([res["best"]["x"], res["best"]["y"]])
    mirror = _mirror_about_track(traj, mic[0], mic[1])
    # Single flyover: either the true position or its mirror image about the track.
    assert min(np.linalg.norm(best), np.linalg.norm(best - mirror)) < 150.0


def test_speed_of_sound_humidity():
    c = DopplerEngine.calculate_speed_of_sound
    assert abs(c(0.0, 0.0) - 331.45) < 0.05  # dry air at 0 degC
    assert abs(c(20.0, 0.0) - 343.36) < 0.05
    # Humidity raises the speed of sound; the effect grows with temperature.
    assert 0.5 < c(20.0, 0.5) - c(20.0, 0.0) < 0.8
    assert c(30.0, 1.0) - c(30.0, 0.0) > c(10.0, 1.0) - c(10.0, 0.0) > 0.0
    assert abs(c(20.0, 50.0) - c(20.0, 0.5)) < 1e-9  # percent accepted


if __name__ == "__main__":
    test_doppler_equation_and_radial_velocity()
    test_geodetic_cartesian_roundtrip()
    test_joint_matcher_single_flyover_recovery()
    test_speed_of_sound_humidity()
    print("All physics unit tests passed!")
