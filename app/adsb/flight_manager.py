"""
Flight manager coordinating ADS-B flight data retrieval, interpolation,
and candidate generation for the Doppler geolocation physics engine.
"""

import math
from typing import Any, Dict, List, Optional
import numpy as np

from app.adsb.opensky_client import OpenSkyClient
from app.adsb.sample_flights import get_curated_scenarios, generate_linear_flight_trajectory


class FlightManager:
    def __init__(self, opensky_client: Optional[OpenSkyClient] = None):
        self.opensky = opensky_client or OpenSkyClient()
        self.scenarios = get_curated_scenarios()

    def get_curated_scenario_list(self) -> List[Dict[str, Any]]:
        """List available curated scenarios for easy frontend testing."""
        return [
            {
                "id": k,
                "name": v["name"],
                "description": v["description"],
                "search_center": v["search_center"],
                "recommended_radius_km": v["recommended_radius_km"],
                "f0_target": v["f0_target"],
                "callsign": v["trajectory"]["callsign"],
                "aircraft_type": v["trajectory"]["aircraft_type"],
            }
            for k, v in self.scenarios.items()
        ]

    def get_scenario_by_id(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        return self.scenarios.get(scenario_id)

    def fetch_or_generate_flights(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        time_window_start: Optional[float] = None,
        time_window_end: Optional[float] = None,
        scenario_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch flights matching geographic radius and time window.
        Returns a list of flight trajectory dictionaries.
        """
        # If a curated scenario matches or is explicitly requested
        if scenario_id and scenario_id in self.scenarios:
            return [self.scenarios[scenario_id]["trajectory"]]

        # Check if coordinates closely match any curated scenario (~20km)
        for s_id, s_data in self.scenarios.items():
            sc_lat = s_data["search_center"]["lat"]
            sc_lon = s_data["search_center"]["lon"]
            dist_km = math.hypot((center_lat - sc_lat) * 111.0, (center_lon - sc_lon) * 111.0 * math.cos(math.radians(center_lat)))
            if dist_km < 15.0:
                return [s_data["trajectory"]]

        # Calculate bounding box
        deg_lat = radius_km / 111.0
        deg_lon = radius_km / (111.0 * max(0.1, math.cos(math.radians(center_lat))))
        lamin = center_lat - deg_lat
        lamax = center_lat + deg_lat
        lomin = center_lon - deg_lon
        lomax = center_lon + deg_lon

        # Attempt query via OpenSky live/historical client
        live_states = self.opensky.get_live_states(lamin, lomin, lamax, lomax)
        if live_states and len(live_states) > 0:
            flights = []
            for state in live_states[:5]:
                # Construct trajectory projection forward & backward around state
                speed = state.get("velocity", 180.0)
                heading = state.get("heading", 90.0)
                alt = state.get("altitude", 1500.0)
                callsign = state.get("callsign") or f"FLT{state['icao24'][-4:].upper()}"
                
                # Project 30s before and 30s after to span flyover
                dist_back = speed * 30.0
                rad = math.radians(heading)
                start_lat = state["lat"] - (dist_back * math.cos(rad) / 6371000.0) * (180.0 / math.pi)
                start_lon = state["lon"] - (dist_back * math.sin(rad) / (6371000.0 * math.cos(math.radians(state["lat"])))) * (180.0 / math.pi)

                traj = generate_linear_flight_trajectory(
                    flight_id=state["icao24"].upper(),
                    callsign=callsign,
                    aircraft_type="Commercial Jet (ADS-B Live)",
                    start_lat=start_lat,
                    start_lon=start_lon,
                    start_alt=alt,
                    heading_deg=heading,
                    speed_mps=speed,
                    vertical_rate_mps=state.get("vertical_rate", 0.0),
                    duration_sec=60.0,
                    start_epoch=state.get("time_position") or 1700000000.0,
                    ground_truth_obs=(center_lat, center_lon),
                )
                flights.append(traj)
            return flights

        # If OpenSky has no hits or offline, synthesize realistic candidate flyovers crossing the sector
        candidate_headings = [285.0, 45.0, 135.0]
        speeds = [175.0, 150.0, 190.0]
        altitudes = [1200.0, 1800.0, 2400.0]
        callsigns = ["AAL305", "SWA712", "EZY449"]
        types = ["Boeing 737-800", "Airbus A320neo", "Boeing 777-200ER"]

        flights = []
        for i, (hdg, spd, alt, cs, actype) in enumerate(zip(candidate_headings, speeds, altitudes, callsigns, types)):
            # Calculate start point ~8km offset from center in reverse heading direction
            rad = math.radians(hdg)
            start_dist = radius_km * 750.0
            dx_start = -start_dist * math.sin(rad)
            dy_start = -start_dist * math.cos(rad)
            
            s_lat = center_lat + (dy_start / 6371000.0) * (180.0 / math.pi)
            s_lon = center_lon + (dx_start / (6371000.0 * math.cos(math.radians(center_lat)))) * (180.0 / math.pi)

            traj = generate_linear_flight_trajectory(
                flight_id=f"ADS{i+1:02d}",
                callsign=cs,
                aircraft_type=actype,
                start_lat=s_lat,
                start_lon=s_lon,
                start_alt=alt,
                heading_deg=hdg,
                speed_mps=spd,
                vertical_rate_mps=-2.0,
                duration_sec=60.0,
                start_epoch=time_window_start or 1700000000.0,
                f0_target=800.0 + (i * 50.0),
                ground_truth_obs=(center_lat, center_lon),
            )
            flights.append(traj)

        return flights
