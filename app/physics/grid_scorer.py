"""
Grid scorer and Least-Squares matching engine for Doppler acoustic geolocation.
Evaluates candidate microphone locations over a spatial grid and computes
heat map score distributions against ADS-B trajectories.
"""

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from app.physics.doppler_engine import DopplerEngine


class GridScorer:
    def __init__(self, doppler_engine: Optional[DopplerEngine] = None):
        self.engine = doppler_engine or DopplerEngine()

    def generate_candidate_grid(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        grid_resolution: int = 25,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate a 2D spatial grid of candidate observer points.

        :param center_lat: Center latitude in degrees
        :param center_lon: Center longitude in degrees
        :param radius_km: Search radius in kilometers
        :param grid_resolution: Number of grid steps along each axis (e.g. 25x25)
        :return: (grid_lats, grid_lons, grid_x, grid_y)
        """
        radius_m = radius_km * 1000.0
        x_lin = np.linspace(-radius_m, radius_m, grid_resolution)
        y_lin = np.linspace(-radius_m, radius_m, grid_resolution)
        x_mesh, y_mesh = np.meshgrid(x_lin, y_lin)

        grid_x = x_mesh.flatten()
        grid_y = y_mesh.flatten()
        grid_z = np.zeros_like(grid_x)

        grid_lats, grid_lons, _ = self.engine.cartesian_to_geodetic(
            grid_x, grid_y, grid_z, center_lat, center_lon, 0.0
        )

        return (
            np.asarray(grid_lats),
            np.asarray(grid_lons),
            grid_x,
            grid_y,
        )

    def score_single_observer(
        self,
        obs_cartesian: np.ndarray,
        t_emit: np.ndarray,
        plane_pos: np.ndarray,
        plane_vel: np.ndarray,
        traced_times: np.ndarray,
        traced_freqs: np.ndarray,
        f0: float,
        time_offsets_to_try: Optional[np.ndarray] = None,
    ) -> Tuple[float, float, float, np.ndarray]:
        """
        Compute Least Squares match score for a single observer location.

        :param obs_cartesian: (3,) [x, y, z] in meters
        :param t_emit: (N,) timestamps of plane trajectory
        :param plane_pos: (N, 3) plane positions in meters
        :param plane_vel: (N, 3) plane velocities in m/s
        :param traced_times: (M,) user trace times
        :param traced_freqs: (M,) user trace frequencies in Hz
        :param f0: Fundamental frequency in Hz
        :param time_offsets_to_try: optional candidate time shifts
        :return: (best_score, best_rmse, best_time_offset, modeled_freqs)
        """
        # Precompute theoretical curve at this observer position
        t_obs, f_obs = self.engine.compute_doppler_curve(
            t_emit, plane_pos, plane_vel, obs_cartesian, f0
        )
        sort_idx = np.argsort(t_obs)
        t_obs_sorted = t_obs[sort_idx]
        f_obs_sorted = f_obs[sort_idx]

        min_t = t_obs_sorted[0]
        max_t = t_obs_sorted[-1]

        # Determine time offsets to search
        if time_offsets_to_try is None or len(time_offsets_to_try) == 0:
            # Automatic CPA inflection alignment
            # Find approximate CPA in user trace (point closest to f0)
            idx_cpa_trace = np.argmin(np.abs(traced_freqs - f0))
            t_cpa_trace = traced_times[idx_cpa_trace]

            # Find CPA in modeled observer curve
            idx_cpa_obs = np.argmin(np.abs(f_obs_sorted - f0))
            t_cpa_obs = t_obs_sorted[idx_cpa_obs]

            nominal_offset = t_cpa_obs - t_cpa_trace
            # Search within a +/- 5s window around nominal offset
            time_offsets_to_try = nominal_offset + np.linspace(-4.0, 4.0, 17)

        best_mse = float("inf")
        best_offset = 0.0
        best_modeled = np.zeros_like(traced_freqs)

        for offset in time_offsets_to_try:
            t_query = traced_times + offset
            # Only evaluate if queries lie mostly within observed bounds
            if t_query[0] < min_t or t_query[-1] > max_t:
                continue

            f_model = np.interp(t_query, t_obs_sorted, f_obs_sorted)
            mse = float(np.mean((traced_freqs - f_model) ** 2))

            if mse < best_mse:
                best_mse = mse
                best_offset = float(offset)
                best_modeled = f_model

        if not math.isinf(best_mse):
            # Fine sub-second refinement around best coarse offset
            fine_offsets = best_offset + np.linspace(-0.5, 0.5, 11)
            for fine_off in fine_offsets:
                t_query = traced_times + fine_off
                if t_query[0] < min_t or t_query[-1] > max_t:
                    continue
                f_model = np.interp(t_query, t_obs_sorted, f_obs_sorted)
                mse = float(np.mean((traced_freqs - f_model) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_offset = float(fine_off)
                    best_modeled = f_model

        if math.isinf(best_mse):
            # Fallback if trajectory range did not overlap
            t_query = np.clip(traced_times + (min_t - traced_times[0]), min_t, max_t)
            best_modeled = np.interp(t_query, t_obs_sorted, f_obs_sorted)
            best_mse = float(np.mean((traced_freqs - best_modeled) ** 2))
            best_offset = min_t - traced_times[0]

        rmse = math.sqrt(max(0.0, best_mse))

        # Match score normalized to [0, 100] using Gaussian likelihood with sigma=18 Hz
        sigma = 18.0
        score = 100.0 * math.exp(-0.5 * (rmse / sigma) ** 2)

        return score, rmse, best_offset, best_modeled

    def evaluate_grid(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        flight_trajectory: Dict[str, Any],
        traced_curve: List[Dict[str, float]],
        f0: float,
        grid_resolution: int = 25,
    ) -> Dict[str, Any]:
        """
        Evaluate candidate observer grid against a flight trajectory.

        :param center_lat: Center latitude
        :param center_lon: Center longitude
        :param radius_km: Geographic search radius in km
        :param flight_trajectory: trajectory dict containing timestamps, lats, lons, altitudes, speeds, headings
        :param traced_curve: list of {"time": float, "freq": float}
        :param f0: fundamental frequency in Hz
        :param grid_resolution: grid dimensions (e.g. 25 -> 625 points)
        """
        if not traced_curve or len(traced_curve) < 3:
            raise ValueError("Traced Doppler curve must have at least 3 points.")

        traced_times = np.array([p["time"] for p in traced_curve], dtype=np.float64)
        traced_freqs = np.array([p["freq"] for p in traced_curve], dtype=np.float64)

        # Normalize relative time so it begins at 0.0 if not already
        t_base = traced_times[0]
        t_rel = traced_times - t_base

        # Extract trajectory arrays
        t_emit = np.array(flight_trajectory["timestamps"], dtype=np.float64)
        plane_lats = np.array(flight_trajectory["latitudes"], dtype=np.float64)
        plane_lons = np.array(flight_trajectory["longitudes"], dtype=np.float64)
        plane_alts = np.array(flight_trajectory["altitudes"], dtype=np.float64)

        # Convert plane positions to local Cartesian meters relative to search center
        plane_x, plane_y, plane_z = self.engine.geodetic_to_cartesian(
            plane_lats, plane_lons, plane_alts, center_lat, center_lon, 0.0
        )
        plane_pos = np.stack([plane_x, plane_y, plane_z], axis=1)

        # Compute velocities by gradient
        vx = np.gradient(plane_x, t_emit)
        vy = np.gradient(plane_y, t_emit)
        vz = np.gradient(plane_z, t_emit)
        plane_vel = np.stack([vx, vy, vz], axis=1)

        # Generate candidate observer grid
        grid_lats, grid_lons, grid_x, grid_y = self.generate_candidate_grid(
            center_lat, center_lon, radius_km, grid_resolution
        )

        n_pts = len(grid_x)
        scores = np.zeros(n_pts, dtype=np.float64)
        rmses = np.zeros(n_pts, dtype=np.float64)
        best_overall_rmse = float("inf")
        best_overall_idx = 0
        best_modeled_curve = None
        best_offset = 0.0

        for i in range(n_pts):
            obs_cart = np.array([grid_x[i], grid_y[i], 0.0])
            score, rmse, offset, modeled = self.score_single_observer(
                obs_cart, t_emit, plane_pos, plane_vel, t_rel, traced_freqs, f0
            )
            scores[i] = score
            rmses[i] = rmse

            if rmse < best_overall_rmse:
                best_overall_rmse = rmse
                best_overall_idx = i
                best_modeled_curve = modeled
                best_offset = offset

        # Prepare heatmap data points [lat, lon, score, rmse]
        # Filter circular search area and format for Leaflet heatmap
        heatmap_points = []
        radius_m = radius_km * 1000.0
        for i in range(n_pts):
            dist_from_center = math.hypot(grid_x[i], grid_y[i])
            if dist_from_center <= radius_m * 1.05:  # within search circle
                heatmap_points.append({
                    "lat": round(float(grid_lats[i]), 6),
                    "lon": round(float(grid_lons[i]), 6),
                    "score": round(float(scores[i]), 2),
                    "rmse": round(float(rmses[i]), 2),
                })

        best_lat = float(grid_lats[best_overall_idx])
        best_lon = float(grid_lons[best_overall_idx])
        best_score = float(scores[best_overall_idx])

        # Prepare modeled vs traced curve comparison
        comparison_points = []
        for i in range(len(traced_curve)):
            comparison_points.append({
                "time": round(float(traced_times[i]), 3),
                "traced_freq": round(float(traced_freqs[i]), 2),
                "modeled_freq": round(float(best_modeled_curve[i]), 2) if best_modeled_curve is not None else 0.0,
            })

        return {
            "flight_id": flight_trajectory.get("flight_id", "UNKNOWN"),
            "callsign": flight_trajectory.get("callsign", "N/A"),
            "aircraft_type": flight_trajectory.get("aircraft_type", "Standard Aircraft"),
            "best_location": {
                "lat": round(best_lat, 6),
                "lon": round(best_lon, 6),
                "score": round(best_score, 2),
                "rmse_hz": round(float(best_overall_rmse), 2),
                "time_offset_sec": round(float(best_offset), 2),
            },
            "heatmap": heatmap_points,
            "curve_comparison": comparison_points,
            "f0_hz": f0,
            "grid_resolution": grid_resolution,
            "total_candidates_evaluated": n_pts,
        }
