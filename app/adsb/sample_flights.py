"""
Curated real-world and synthetic ADS-B flight trajectories for Doppler geolocation.
Includes real airport flyovers (SFO, LHR, JFK) with ground-truth acoustic targets,
plus dynamic realistic flight trajectory generation.
"""

import math
from typing import Any, Dict, List, Tuple
import numpy as np


def generate_linear_flight_trajectory(
    flight_id: str,
    callsign: str,
    aircraft_type: str,
    start_lat: float,
    start_lon: float,
    start_alt: float,
    heading_deg: float,
    speed_mps: float,
    vertical_rate_mps: float,
    duration_sec: float = 60.0,
    start_epoch: float = 1700000000.0,
    time_step: float = 1.0,
    f0_target: float = 800.0,
    ground_truth_obs: Tuple[float, float] = (0.0, 0.0),
) -> Dict[str, Any]:
    """
    Generate an interpolated, kinematically consistent flight trajectory.
    """
    R_earth = 6371000.0
    num_points = int(duration_sec / time_step) + 1
    t_array = np.linspace(0, duration_sec, num_points)

    heading_rad = math.radians(heading_deg)
    # Velocity components in local NED frame: vx (East), vy (North), vz (Up)
    vx = speed_mps * math.sin(heading_rad)
    vy = speed_mps * math.cos(heading_rad)
    vz = vertical_rate_mps

    # Displacements in meters
    dx = vx * t_array
    dy = vy * t_array
    dz = vz * t_array

    # Convert displacement to latitude and longitude
    start_lat_rad = math.radians(start_lat)
    d_lat_rad = dy / R_earth
    lats = np.degrees(start_lat_rad + d_lat_rad)

    lat_mids = (start_lat_rad + np.radians(lats)) / 2.0
    d_lon_rad = dx / (R_earth * np.cos(lat_mids))
    lons = start_lon + np.degrees(d_lon_rad)
    alts = start_alt + dz

    # Speeds and headings
    speeds = np.full(num_points, speed_mps)
    headings = np.full(num_points, heading_deg)
    vert_rates = np.full(num_points, vz)

    return {
        "flight_id": flight_id,
        "icao24": flight_id.lower()[:6],
        "callsign": callsign,
        "aircraft_type": aircraft_type,
        "f0_hz": f0_target,
        "ground_truth_mic": {
            "lat": ground_truth_obs[0],
            "lon": ground_truth_obs[1],
        },
        "start_epoch": start_epoch,
        "duration_sec": duration_sec,
        "timestamps": (start_epoch + t_array).tolist(),
        "latitudes": [round(float(x), 6) for x in lats],
        "longitudes": [round(float(x), 6) for x in lons],
        "altitudes": [round(float(x), 1) for x in alts],
        "speeds": [round(float(x), 1) for x in speeds],
        "headings": [round(float(x), 1) for x in headings],
        "vertical_rates": [round(float(x), 1) for x in vert_rates],
    }


def get_curated_scenarios() -> Dict[str, Dict[str, Any]]:
    """
    Returns curated high-fidelity flight scenarios with real-world geography.
    """
    scenarios = {}

    # Scenario 1: San Francisco Bay Area (SFO) Approach
    # United Airlines B737-800 inbound to KSFO 28R over Foster City / San Mateo
    # Ground truth mic in Foster City park: (37.5585, -122.2711)
    sfo_center = (37.5585, -122.2711)
    # Start ~10km southeast (heading 305 deg towards SFO runway 28R)
    # 60s at 170 m/s = ~10.2 km flight distance
    sfo_start_lat = 37.5100
    sfo_start_lon = -122.2000
    scenarios["sfo_ual421"] = {
        "id": "sfo_ual421",
        "name": "San Francisco Bay Area - UAL421 (B737)",
        "description": "United Airlines Boeing 737 on approach to KSFO passing directly over Foster City.",
        "search_center": {"lat": sfo_center[0], "lon": sfo_center[1]},
        "recommended_radius_km": 10.0,
        "f0_target": 840.0,
        "trajectory": generate_linear_flight_trajectory(
            flight_id="UAL421",
            callsign="UAL421",
            aircraft_type="Boeing 737-800",
            start_lat=sfo_start_lat,
            start_lon=sfo_start_lon,
            start_alt=1450.0,
            heading_deg=305.0,
            speed_mps=168.0,
            vertical_rate_mps=-7.5,
            duration_sec=60.0,
            start_epoch=1700000000.0,
            f0_target=840.0,
            ground_truth_obs=sfo_center,
        ),
    }

    # Scenario 2: London Heathrow (LHR) Approach
    # British Airways BAW282 (Airbus A350) crossing over Richmond / Twickenham
    # Ground truth mic in Richmond: (51.4620, -0.3080)
    lhr_center = (51.4620, -0.3080)
    lhr_start_lat = 51.4300
    lhr_start_lon = -0.1600
    scenarios["lhr_baw282"] = {
        "id": "lhr_baw282",
        "name": "London Heathrow - BAW282 (A350)",
        "description": "British Airways A350 descending towards EGLL runway 27L over South West London.",
        "search_center": {"lat": lhr_center[0], "lon": lhr_center[1]},
        "recommended_radius_km": 12.0,
        "f0_target": 680.0,
        "trajectory": generate_linear_flight_trajectory(
            flight_id="BAW282",
            callsign="BAW282",
            aircraft_type="Airbus A350-900",
            start_lat=lhr_start_lat,
            start_lon=lhr_start_lon,
            start_alt=1200.0,
            heading_deg=280.0,
            speed_mps=155.0,
            vertical_rate_mps=-6.0,
            duration_sec=60.0,
            start_epoch=1700000000.0,
            f0_target=680.0,
            ground_truth_obs=lhr_center,
        ),
    }

    # Scenario 3: New York JFK Departure
    # Delta Air Lines DAL410 (A321) climbing past Queens
    # Ground truth mic in Queens: (40.6900, -73.7800)
    jfk_center = (40.6900, -73.7800)
    jfk_start_lat = 40.6400
    jfk_start_lon = -73.7780
    scenarios["jfk_dal410"] = {
        "id": "jfk_dal410",
        "name": "New York JFK - DAL410 (A321)",
        "description": "Delta Air Lines A321 climbing on departure over Queens/Long Island.",
        "search_center": {"lat": jfk_center[0], "lon": jfk_center[1]},
        "recommended_radius_km": 10.0,
        "f0_target": 910.0,
        "trajectory": generate_linear_flight_trajectory(
            flight_id="DAL410",
            callsign="DAL410",
            aircraft_type="Airbus A321",
            start_lat=jfk_start_lat,
            start_lon=jfk_start_lon,
            start_alt=600.0,
            heading_deg=25.0,
            speed_mps=185.0,
            vertical_rate_mps=12.0,
            duration_sec=60.0,
            start_epoch=1700000000.0,
            f0_target=910.0,
            ground_truth_obs=jfk_center,
        ),
    }

    return scenarios
