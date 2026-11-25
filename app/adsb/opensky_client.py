"""
OpenSky Network REST client (live state vectors).

Parsing rules (honesty first):

* Geometric (GNSS) altitude is preferred over barometric altitude because the
  Doppler geometry needs a true height; the chosen basis is recorded in
  ``altitude_basis``.
* Aircraft reported ``on_ground`` are dropped.
* States missing position, ground speed, track, altitude or position time are
  skipped instead of being filled with invented defaults.  A missing vertical
  rate is taken as level flight and flagged with ``vertical_rate_reported``.

Authentication: OpenSky now only accepts OAuth2 client credentials (basic
auth was retired).  See :mod:`app.adsb.providers` for the token handling used
by the historical provider; this live client accepts an optional token
callable with the same semantics.
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger("doppler.opensky")

USER_AGENT = "DopplerAcousticGeolocator/1.2"


class OpenSkyClient:
    BASE_URL = "https://opensky-network.org/api"

    def __init__(self, token_provider: Optional[Callable[[], Optional[str]]] = None, timeout: int = 10):
        self.token_provider = token_provider
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        headers = {"User-Agent": USER_AGENT}
        token = self.token_provider() if self.token_provider else None
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def get_live_states(self, lamin: float, lomin: float, lamax: float, lomax: float) -> Optional[List[Dict[str, Any]]]:
        """Fetch live aircraft states within a bounding box. ``None`` on any failure."""
        params = {"lamin": lamin, "lomin": lomin, "lamax": lamax, "lomax": lomax}
        try:
            resp = requests.get(f"{self.BASE_URL}/states/all", params=params, timeout=self.timeout, headers=self._headers())
            if resp.status_code != 200:
                logger.warning("OpenSky API returned status %s", resp.status_code)
                return None
            return self.parse_states(resp.json())
        except Exception as e:  # network errors are non-fatal
            logger.warning("Failed to query OpenSky API: %s", e)
            return None

    @staticmethod
    def parse_states(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert an OpenSky ``/states/all`` payload into SI state dicts.

        State vector index: 0 icao24, 1 callsign, 2 origin_country, 3 time_position,
        4 last_contact, 5 longitude, 6 latitude, 7 baro_altitude [m], 8 on_ground,
        9 velocity [m/s over ground], 10 true_track [deg], 11 vertical_rate [m/s],
        12 sensors, 13 geo_altitude [m], 14 squawk, 15 spi, 16 position_source.
        """
        results = []
        for s in (payload or {}).get("states") or []:
            if not isinstance(s, (list, tuple)) or len(s) < 14:
                continue
            lon, lat = s[5], s[6]
            if lat is None or lon is None or s[3] is None:
                continue
            if s[8]:  # on_ground
                continue
            if s[9] is None or s[10] is None:
                continue
            if s[13] is not None:
                alt, basis = float(s[13]), "geometric"
            elif s[7] is not None:
                alt, basis = float(s[7]), "barometric"
            else:
                continue
            results.append({
                "icao24": str(s[0]).lower(),
                "callsign": (s[1] or "").strip(),
                "country": s[2],
                "time_position": float(s[3]),
                "lon": float(lon),
                "lat": float(lat),
                "altitude": alt,
                "altitude_basis": basis,
                "velocity": float(s[9]),
                "heading": float(s[10]),
                "vertical_rate": float(s[11]) if s[11] is not None else 0.0,
                "vertical_rate_reported": s[11] is not None,
                "on_ground": False,
                "source": "opensky",
            })
        return results
