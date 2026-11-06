"""
OpenSky Network ADS-B API integration client.
Supports querying live state vectors and historical flight tracks,
with robust error handling and authentication support.
"""

import logging
import time
from typing import Any, Dict, List, Optional
import requests

logger = logging.getLogger("doppler.opensky")


class OpenSkyClient:
    BASE_URL = "https://opensky-network.org/api"

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None, timeout: int = 10):
        self.username = username
        self.password = password
        self.timeout = timeout
        self.auth = (username, password) if username and password else None

    def get_live_states(
        self,
        lamin: float,
        lomin: float,
        lamax: float,
        lomax: float,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch live aircraft states within a bounding box.
        
        OpenSky state vector index:
        0: icao24, 1: callsign, 2: origin_country, 3: time_position, 4: last_contact,
        5: longitude, 6: latitude, 7: baro_altitude, 8: on_ground, 9: velocity (m/s),
        10: true_track (deg), 11: vertical_rate (m/s), 12: sensors, 13: geo_altitude,
        14: squawk, 15: spi, 16: position_source
        """
        endpoint = f"{self.BASE_URL}/states/all"
        params = {
            "lamin": lamin,
            "lomin": lomin,
            "lamax": lamax,
            "lomax": lomax,
        }

        try:
            resp = requests.get(
                endpoint,
                params=params,
                auth=self.auth,
                timeout=self.timeout,
                headers={"User-Agent": "DopplerAcousticGeolocator/1.0"},
            )
            if resp.status_code == 200:
                data = resp.json()
                states = data.get("states")
                if not states:
                    return []
                results = []
                for s in states:
                    if s[5] is not None and s[6] is not None:
                        results.append({
                            "icao24": s[0],
                            "callsign": (s[1] or "").strip(),
                            "country": s[2],
                            "time_position": s[3],
                            "lon": float(s[5]),
                            "lat": float(s[6]),
                            "altitude": float(s[7] or s[13] or 1000.0),
                            "velocity": float(s[9] or 150.0),
                            "heading": float(s[10] or 0.0),
                            "vertical_rate": float(s[11] or 0.0),
                            "on_ground": bool(s[8]),
                        })
                return results
            else:
                logger.warning(f"OpenSky API returned status {resp.status_code}: {resp.text[:120]}")
                return None
        except Exception as e:
            logger.warning(f"Failed to query OpenSky API: {e}")
            return None

    def get_track_by_icao(self, icao24: str, epoch_time: int) -> Optional[Dict[str, Any]]:
        """
        Fetch historical flight track for a specific aircraft icao24 at given timestamp.
        """
        endpoint = f"{self.BASE_URL}/tracks/all"
        params = {"icao24": icao24.lower().strip(), "time": epoch_time}

        try:
            resp = requests.get(
                endpoint,
                params=params,
                auth=self.auth,
                timeout=self.timeout,
                headers={"User-Agent": "DopplerAcousticGeolocator/1.0"},
            )
            if resp.status_code == 200:
                return resp.json()
            return None
        except Exception as e:
            logger.warning(f"OpenSky track fetch error for {icao24}: {e}")
            return None
