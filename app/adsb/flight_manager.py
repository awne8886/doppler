"""
Flight manager: decides where candidate ADS-B trajectories come from and tags
every trajectory with its provenance.

Honesty rules
-------------
* Curated demo scenarios are returned **only** when a ``scenario_id`` is given
  explicitly, or when ``demo=True``.
* Synthetic flights are generated **only** when ``demo=True``.
* Otherwise only real data is used (uploaded tracks, historical providers or a
  live snapshot).  When none is available the result status is
  ``no_adsb_data`` and the flight list is empty - nothing is invented.

Every trajectory carries ``data_source`` (one of :data:`DATA_SOURCES`) and
``altitude_basis`` (``geometric``, ``barometric`` or ``unknown``).

Live-snapshot limitations (documented, not hidden)
--------------------------------------------------
The live path (OpenSky ``/states/all`` without ``time``, then ADSB.lol) only
sees aircraft *right now*.  Each state is projected linearly for +/-30 s
around its own position timestamp, the nearest ``max_live_aircraft`` aircraft
are kept, and the snapshot is used only when the requested time window (if
any) is within a couple of minutes of now.  Anything older needs a historical
provider or uploaded tracks.
"""

from __future__ import annotations

import copy
import logging
import math
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.adsb.adsblol_client import AdsbLolClient
from app.adsb.opensky_client import OpenSkyClient
from app.adsb.sample_flights import generate_linear_flight_trajectory, get_curated_scenarios

logger = logging.getLogger("doppler.flights")

DATA_SOURCES = ("curated_demo", "synthetic", "opensky_live", "adsblol_live", "historical", "uploaded")
LIVE_PROJECTION_S = 30.0
LIVE_LOOKBACK_S = 120.0
LIVE_LOOKAHEAD_S = 60.0
DEMO_PROXIMITY_KM = 15.0

NO_DATA_MESSAGE = (
    "No real ADS-B trajectories are available for this area and time window. "
    "Upload tracks for the recording time, configure a historical provider, or "
    "explicitly opt in to demo data (scenario_id or demo=true)."
)


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    return math.hypot((lat1 - lat2) * 111.0, (lon1 - lon2) * 111.0 * math.cos(math.radians(lat1)))


def tag_trajectories(flights: Sequence[Dict[str, Any]], data_source: str,
                     altitude_basis: Optional[str] = None) -> List[Dict[str, Any]]:
    """Deep-copy trajectories and stamp provenance fields (existing tags win for altitude basis)."""
    out = []
    for f in flights:
        g = copy.deepcopy(dict(f))
        g["data_source"] = data_source
        if altitude_basis is not None or "altitude_basis" not in g:
            g["altitude_basis"] = altitude_basis or g.get("altitude_basis") or "unknown"
        out.append(g)
    return out


class FlightManager:
    def __init__(self, opensky_client: Optional[OpenSkyClient] = None, adsblol_client: Optional[AdsbLolClient] = None,
                 historical_providers: Optional[Sequence[Any]] = None, max_live_aircraft: int = 8,
                 clock: Callable[[], float] = time.time):
        self.opensky = opensky_client or OpenSkyClient()
        self.adsblol = adsblol_client or AdsbLolClient()
        self.historical_providers = list(historical_providers or [])
        self.max_live_aircraft = max(1, int(max_live_aircraft))
        self.clock = clock
        self.scenarios = get_curated_scenarios()

    # ---------------------------------------------------------------- scenarios
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
                "n_flights": len(v.get("flights", [v["trajectory"]])),
                "audio_duration_sec": v.get("audio_duration_sec"),
                "recording_start_epoch": v.get("recording_start_epoch"),
                "data_source": "curated_demo",
            }
            for k, v in self.scenarios.items()
        ]

    def get_scenario_by_id(self, scenario_id: str) -> Optional[Dict[str, Any]]:
        return self.scenarios.get(scenario_id)

    def scenario_flights(self, scenario_id: str) -> List[Dict[str, Any]]:
        sc = self.scenarios[scenario_id]
        return tag_trajectories(sc.get("flights") or [sc["trajectory"]], "curated_demo", "geometric")

    # ------------------------------------------------------------------ search
    def find_flights(
        self,
        center_lat: float,
        center_lon: float,
        radius_km: float,
        time_window_start: Optional[float] = None,
        time_window_end: Optional[float] = None,
        scenario_id: Optional[str] = None,
        demo: bool = False,
        icao24_hints: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """
        Resolve candidate trajectories and report where they came from.

        Returns ``{"status": "ok"|"no_adsb_data", "data_source", "flights",
        "notes", "limitations"}``.  ``data_source`` is ``None`` when nothing was
        found.
        """
        notes: List[str] = []
        if scenario_id:
            if scenario_id not in self.scenarios:
                raise ValueError(f"Unknown scenario_id '{scenario_id}'.")
            return self._ok("curated_demo", self.scenario_flights(scenario_id),
                            ["Curated demo scenario selected explicitly (simulated trajectories)."], [])

        t0, t1 = _window(time_window_start, time_window_end)

        for provider in self.historical_providers:
            if t0 is None or not provider.available():
                continue
            try:
                res = provider.get_tracks(center_lat, center_lon, radius_km, t0, t1, icao24_hints=icao24_hints)
            except Exception as e:  # pragma: no cover - providers handle their own errors
                logger.warning("Historical provider %s failed: %s", getattr(provider, "name", provider), e)
                continue
            notes.extend(res.get("notes", []))
            if res.get("flights"):
                return self._ok("historical", tag_trajectories(res["flights"], "historical"), notes,
                                res.get("limitations", []))

        if self._live_applicable(t0, t1):
            live = self._live_flights(center_lat, center_lon, radius_km)
            if live["flights"]:
                return self._ok(live["data_source"], live["flights"], notes + live["notes"], [
                    f"Live snapshot: each aircraft is projected linearly +/-{LIVE_PROJECTION_S:.0f} s from a single state vector.",
                    f"Only the nearest {self.max_live_aircraft} aircraft are kept.",
                    "Only valid for recordings made within about two minutes of the query.",
                ])
            notes.extend(live["notes"])
        elif t0 is not None:
            notes.append("Requested time window is not recent; live ADS-B snapshots were not used.")
            if not self.historical_providers:
                notes.append("No historical ADS-B provider is configured (set OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET).")

        if demo:
            for s_id, s_data in self.scenarios.items():
                c = s_data["search_center"]
                if _distance_km(center_lat, center_lon, c["lat"], c["lon"]) < DEMO_PROXIMITY_KM:
                    return self._ok("curated_demo", self.scenario_flights(s_id),
                                    notes + [f"Demo mode: nearby curated scenario '{s_id}' used (simulated trajectories)."], [])
            return self._ok("synthetic", self._synthetic_flights(center_lat, center_lon, radius_km, t0),
                            notes + ["Demo mode: synthetic flights generated around the search centre. "
                                     "They are not real aircraft and cannot locate a real recording."], [])

        return {"status": "no_adsb_data", "data_source": None, "flights": [], "notes": notes + [NO_DATA_MESSAGE],
                "limitations": [], "message": NO_DATA_MESSAGE}

    def fetch_or_generate_flights(self, center_lat: float, center_lon: float, radius_km: float,
                                  time_window_start: Optional[float] = None, time_window_end: Optional[float] = None,
                                  scenario_id: Optional[str] = None, demo: bool = False) -> List[Dict[str, Any]]:
        """Backward-compatible list-only wrapper around :meth:`find_flights` (same opt-in rules)."""
        return self.find_flights(center_lat, center_lon, radius_km, time_window_start, time_window_end,
                                 scenario_id, demo)["flights"]

    @staticmethod
    def _ok(source: str, flights: List[Dict[str, Any]], notes: List[str], limitations: List[str]) -> Dict[str, Any]:
        return {"status": "ok", "data_source": source, "flights": flights, "notes": notes, "limitations": limitations}

    # -------------------------------------------------------------------- live
    def _live_applicable(self, t0: Optional[float], t1: Optional[float]) -> bool:
        if t0 is None:
            return True
        now = self.clock()
        return t1 >= now - LIVE_LOOKBACK_S and t0 <= now + LIVE_LOOKAHEAD_S

    def _live_flights(self, center_lat: float, center_lon: float, radius_km: float) -> Dict[str, Any]:
        deg_lat = radius_km / 111.0
        deg_lon = radius_km / (111.0 * max(0.1, math.cos(math.radians(center_lat))))
        states = self.opensky.get_live_states(center_lat - deg_lat, center_lon - deg_lon,
                                              center_lat + deg_lat, center_lon + deg_lon)
        source = "opensky_live"
        if not states:
            states = self.adsblol.get_live_states(center_lat, center_lon, radius_km)
            source = "adsblol_live"
        usable = [s for s in (states or []) if s.get("time_position") is not None and not s.get("on_ground")]
        skipped = len(states or []) - len(usable)
        usable.sort(key=lambda s: _distance_km(center_lat, center_lon, s["lat"], s["lon"]))
        flights = [self._project_state(s, source) for s in usable[: self.max_live_aircraft]]
        notes = []
        if skipped:
            notes.append(f"{skipped} live state(s) skipped (on ground or without a position timestamp).")
        if len(usable) > self.max_live_aircraft:
            notes.append(f"{len(usable) - self.max_live_aircraft} more distant aircraft ignored (live cap).")
        return {"data_source": source, "flights": flights, "notes": notes}

    @staticmethod
    def _project_state(state: Dict[str, Any], source: str) -> Dict[str, Any]:
        speed, heading, alt = float(state["velocity"]), float(state["heading"]), float(state["altitude"])
        dist_back = speed * LIVE_PROJECTION_S
        rad = math.radians(heading)
        start_lat = state["lat"] - math.degrees(dist_back * math.cos(rad) / 6371000.0)
        start_lon = state["lon"] - math.degrees(dist_back * math.sin(rad) / (6371000.0 * math.cos(math.radians(state["lat"]))))
        vr = float(state.get("vertical_rate") or 0.0)
        icao = str(state.get("icao24") or "unknown")
        traj = generate_linear_flight_trajectory(
            flight_id=icao.upper(),
            callsign=state.get("callsign") or f"FLT{icao[-4:].upper()}",
            aircraft_type=state.get("aircraft_type") or "Unknown (ADS-B live)",
            start_lat=start_lat, start_lon=start_lon, start_alt=alt - vr * LIVE_PROJECTION_S,
            heading_deg=heading, speed_mps=speed, vertical_rate_mps=vr,
            duration_sec=2 * LIVE_PROJECTION_S,
            # The projected start point is LIVE_PROJECTION_S before the reported position.
            start_epoch=float(state["time_position"]) - LIVE_PROJECTION_S,
        )
        traj.pop("f0_hz", None)
        traj.pop("ground_truth_mic", None)
        traj.update({
            "icao24": icao.lower(),
            "data_source": source,
            "altitude_basis": state.get("altitude_basis", "unknown"),
            "projection": "linear_from_single_state",
            "state_time": float(state["time_position"]),
        })
        return traj

    # --------------------------------------------------------------- synthetic
    @staticmethod
    def _synthetic_flights(center_lat: float, center_lon: float, radius_km: float,
                           t0: Optional[float]) -> List[Dict[str, Any]]:
        """Demo-only flights crossing the sector.  Never used unless demo=True."""
        candidate_headings = [285.0, 45.0, 135.0]
        speeds = [175.0, 150.0, 190.0]
        altitudes = [1200.0, 1800.0, 2400.0]
        callsigns = ["SYN305", "SYN712", "SYN449"]
        types = ["Synthetic jet A", "Synthetic jet B", "Synthetic jet C"]
        flights = []
        for i, (hdg, spd, alt, cs, actype) in enumerate(zip(candidate_headings, speeds, altitudes, callsigns, types)):
            rad = math.radians(hdg)
            start_dist = radius_km * 750.0
            dx_start, dy_start = -start_dist * math.sin(rad), -start_dist * math.cos(rad)
            s_lat = center_lat + math.degrees(dy_start / 6371000.0)
            s_lon = center_lon + math.degrees(dx_start / (6371000.0 * math.cos(math.radians(center_lat))))
            traj = generate_linear_flight_trajectory(
                flight_id=f"SYN{i + 1:02d}", callsign=cs, aircraft_type=actype,
                start_lat=s_lat, start_lon=s_lon, start_alt=alt, heading_deg=hdg, speed_mps=spd,
                vertical_rate_mps=-2.0, duration_sec=60.0, start_epoch=t0 if t0 is not None else 1700000000.0,
                f0_target=800.0 + (i * 50.0),
            )
            traj.pop("ground_truth_mic", None)
            traj.update({"data_source": "synthetic", "altitude_basis": "geometric"})
            flights.append(traj)
        return flights


def _window(t0: Optional[float], t1: Optional[float]):
    if t0 is None and t1 is None:
        return None, None
    a = t0 if t0 is not None else t1
    b = t1 if t1 is not None else t0
    return (min(a, b), max(a, b))
