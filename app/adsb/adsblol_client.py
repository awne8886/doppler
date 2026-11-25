"""
ADSB.lol live aircraft client (open, keyless community ADS-B network).

Used as a fallback when OpenSky returns no data.  Values are converted to the
same SI state-vector layout the OpenSky client produces.  Geometric (GNSS)
altitude is preferred over barometric altitude when both are reported, since
Doppler geometry needs a true height; the chosen basis is recorded.  Aircraft
on the ground and states without position, ground speed, track or altitude are
skipped rather than filled with invented defaults.
"""

import logging
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger("doppler.adsblol")

FT_TO_M = 0.3048
KT_TO_MPS = 0.514444
FPM_TO_MPS = 0.00508


class AdsbLolClient:
    BASE_URL = "https://api.adsb.lol/v2"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def get_live_states(self, lat: float, lon: float, radius_km: float) -> Optional[List[Dict[str, Any]]]:
        radius_nm = max(1, min(250, int(round(radius_km / 1.852))))
        url = f"{self.BASE_URL}/point/{lat:.5f}/{lon:.5f}/{radius_nm}"
        try:
            resp = requests.get(url, timeout=self.timeout, headers={"User-Agent": "DopplerAcousticGeolocator/1.1"})
            if resp.status_code != 200:
                logger.warning("ADSB.lol returned status %s", resp.status_code)
                return None
            return self.parse(resp.json())
        except Exception as e:  # network errors are non-fatal
            logger.warning("Failed to query ADSB.lol: %s", e)
            return None

    @staticmethod
    def parse(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        out = []
        now = payload.get("now")
        now_s = float(now) / 1000.0 if now and float(now) > 1e11 else (float(now) if now else None)
        for ac in payload.get("ac") or []:
            if ac.get("lat") is None or ac.get("lon") is None:
                continue
            if ac.get("alt_baro") == "ground" or ac.get("ground") is True:
                continue
            if not isinstance(ac.get("gs"), (int, float)) or not isinstance(ac.get("track"), (int, float)):
                continue
            if isinstance(ac.get("alt_geom"), (int, float)):
                alt_m, basis = float(ac["alt_geom"]) * FT_TO_M, "geometric"
            elif isinstance(ac.get("alt_baro"), (int, float)):
                alt_m, basis = float(ac["alt_baro"]) * FT_TO_M, "barometric"
            else:
                continue
            rate = ac.get("geom_rate", ac.get("baro_rate"))
            seen = float(ac.get("seen_pos", 0.0) or 0.0)
            out.append({
                "icao24": str(ac.get("hex", "")).lower(),
                "callsign": (ac.get("flight") or "").strip(),
                "aircraft_type": ac.get("t"),
                "time_position": (now_s - seen) if now_s else None,
                "lat": float(ac["lat"]),
                "lon": float(ac["lon"]),
                "altitude": alt_m,
                "altitude_basis": basis,
                "velocity": float(ac["gs"]) * KT_TO_MPS,
                "heading": float(ac["track"]),
                "vertical_rate": float(rate) * FPM_TO_MPS if isinstance(rate, (int, float)) else 0.0,
                "vertical_rate_reported": isinstance(rate, (int, float)),
                "on_ground": False,
                "source": "adsb.lol",
            })
        return out
