"""
Pluggable historical ADS-B providers.

A provider answers "which aircraft were within ``radius_km`` of a point
between ``t0`` and ``t1``, and where exactly were they?".  The interface is
intentionally small::

    class HistoricalProvider:
        name: str
        def available(self) -> bool: ...
        def get_tracks(self, lat, lon, radius_km, t0, t1, icao24_hints=None) -> dict:
            # {"flights": [...trajectory dicts...], "notes": [...], "limitations": [...]}

Trajectories returned by a provider are tagged ``data_source="historical"``
and carry their ``altitude_basis``; they are conditioned (resampled and
smoothed) before matching.

OpenSky Network (REST, https://opensky-network.org/api)
-------------------------------------------------------
* Authentication: OAuth2 client credentials only (HTTP basic auth was
  retired).  Set ``OPENSKY_CLIENT_ID`` and ``OPENSKY_CLIENT_SECRET``; tokens
  from ``auth.opensky-network.org`` last 30 minutes and are refreshed
  automatically.
* ``/states/all?time=T`` returns state vectors for authenticated users up to
  one hour in the past (5 s resolution).  Windows inside the last hour are
  sampled every ``state_step_s`` seconds, giving full state vectors
  (geometric altitude, ground speed, track, vertical rate).
* ``/tracks/all?icao24=X&time=T`` (experimental) reaches back 30 days but
  needs the aircraft's icao24 and only returns sparse waypoints with
  barometric altitude (a waypoint is added on >2.5 deg track or >100 m
  altitude change).  Older windows therefore require ``icao24_hints``;
  waypoints are linearly interpolated (straight segments, as OpenSky intends).
* Responses for windows older than ``cache_min_age_s`` are cached on disk.
* HTTP 429 (credits exhausted) and other errors become notes, never crashes.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import requests

logger = logging.getLogger("doppler.providers")

OPENSKY_API = "https://opensky-network.org/api"
OPENSKY_TOKEN_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
USER_AGENT = "DopplerAcousticGeolocator/1.2"


class DiskCache:
    """Tiny JSON cache keyed by request identity (hashed)."""

    def __init__(self, directory: str):
        self.directory = directory

    def _path(self, key: str) -> str:
        return os.path.join(self.directory, hashlib.sha256(key.encode()).hexdigest() + ".json")

    def get(self, key: str) -> Optional[Any]:
        try:
            with open(self._path(key)) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def set(self, key: str, value: Any) -> None:
        try:
            os.makedirs(self.directory, exist_ok=True)
            tmp = self._path(key) + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(value, fh)
            os.replace(tmp, self._path(key))
        except OSError as e:  # pragma: no cover - disk full etc.
            logger.warning("Cache write failed: %s", e)


class OpenSkyTokenManager:
    """OAuth2 client-credentials token cache for the OpenSky REST API."""

    def __init__(self, client_id: Optional[str], client_secret: Optional[str], session: Optional[Any] = None,
                 token_url: str = OPENSKY_TOKEN_URL, clock: Callable[[], float] = time.time, timeout: float = 10.0):
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = session or requests.Session()
        self.token_url = token_url
        self.clock = clock
        self.timeout = timeout
        self._token: Optional[str] = None
        self._expires = 0.0

    @classmethod
    def from_env(cls, **kw) -> "OpenSkyTokenManager":
        return cls(os.environ.get("OPENSKY_CLIENT_ID"), os.environ.get("OPENSKY_CLIENT_SECRET"), **kw)

    @property
    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    def invalidate(self) -> None:
        self._token, self._expires = None, 0.0

    def token(self) -> Optional[str]:
        if not self.configured:
            return None
        if self._token and self.clock() < self._expires - 60.0:
            return self._token
        try:
            resp = self.session.post(self.token_url, data={
                "grant_type": "client_credentials", "client_id": self.client_id, "client_secret": self.client_secret,
            }, timeout=self.timeout, headers={"User-Agent": USER_AGENT})
        except Exception as e:
            logger.warning("OpenSky token request failed: %s", e)
            return None
        if resp.status_code != 200:
            logger.warning("OpenSky token request returned HTTP %s", resp.status_code)
            return None
        body = resp.json()
        self._token = body.get("access_token")
        self._expires = self.clock() + float(body.get("expires_in", 1800))
        return self._token


class OpenSkyHistoricalProvider:
    name = "opensky_historical"

    def __init__(self, tokens: OpenSkyTokenManager, cache: Optional[DiskCache] = None, session: Optional[Any] = None,
                 clock: Callable[[], float] = time.time, state_step_s: float = 10.0, max_state_requests: int = 60,
                 max_track_requests: int = 10, cache_min_age_s: float = 600.0, timeout: float = 15.0):
        self.tokens = tokens
        self.cache = cache
        self.session = session or requests.Session()
        self.clock = clock
        self.state_step_s = state_step_s
        self.max_state_requests = max_state_requests
        self.max_track_requests = max_track_requests
        self.cache_min_age_s = cache_min_age_s
        self.timeout = timeout

    def available(self) -> bool:
        return self.tokens.configured

    # ----------------------------------------------------------------- HTTP
    def _get(self, path: str, params: Dict[str, Any], t_ref: float, notes: List[str]) -> Optional[Any]:
        key = path + "?" + "&".join(f"{k}={params[k]}" for k in sorted(params))
        cacheable = self.cache is not None and self.clock() - t_ref > self.cache_min_age_s
        if cacheable:
            hit = self.cache.get(key)
            if hit is not None:
                return hit
        for attempt in range(2):
            token = self.tokens.token()
            if not token:
                notes.append("OpenSky authentication failed (check OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET).")
                return None
            try:
                resp = self.session.get(OPENSKY_API + path, params=params, timeout=self.timeout,
                                        headers={"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT})
            except Exception as e:
                notes.append(f"OpenSky request failed: {type(e).__name__}.")
                return None
            if resp.status_code == 401 and attempt == 0:
                self.tokens.invalidate()
                continue
            if resp.status_code == 429:
                retry = resp.headers.get("X-Rate-Limit-Retry-After-Seconds")
                notes.append(f"OpenSky API credits exhausted (HTTP 429){f'; retry after {retry} s' if retry else ''}.")
                return None
            if resp.status_code == 404:
                return {}
            if resp.status_code != 200:
                notes.append(f"OpenSky returned HTTP {resp.status_code} for {path}.")
                return None
            body = resp.json()
            if cacheable:
                self.cache.set(key, body)
            return body
        return None

    # ---------------------------------------------------------------- query
    def get_tracks(self, lat: float, lon: float, radius_km: float, t0: float, t1: float,
                   icao24_hints: Optional[Sequence[str]] = None) -> Dict[str, Any]:
        notes: List[str] = []
        now = self.clock()
        bbox = _bbox(lat, lon, radius_km)
        if t0 >= now - 3600.0 + 30.0:
            flights = self._from_states(bbox, t0, t1, notes)
            lim = [f"OpenSky state vectors sampled every {self.state_step_s:.0f} s (5 s resolution) and resampled."]
        elif t0 >= now - 30 * 86400.0:
            if not icao24_hints:
                notes.append("OpenSky's REST API cannot list aircraft in an area more than one hour in the past. "
                             "Provide icao24 hex codes of candidate aircraft (icao24_hints) or upload tracks.")
                return {"flights": [], "notes": notes, "limitations": []}
            flights = self._from_tracks(bbox, t0, t1, icao24_hints, notes)
            lim = ["OpenSky /tracks waypoints are sparse (straight segments between >2.5 deg / >100 m changes) and "
                   "barometric; the endpoint is marked experimental by OpenSky."]
        else:
            notes.append("OpenSky REST data older than 30 days is not available; upload tracks instead.")
            return {"flights": [], "notes": notes, "limitations": []}
        return {"flights": flights, "notes": notes, "limitations": lim if flights else []}

    def _from_states(self, bbox, t0, t1, notes) -> List[Dict[str, Any]]:
        # OpenSky floors `time` to 5 s for authenticated users; align samples to the step.
        start = math.floor((t0 - 30.0) / self.state_step_s) * self.state_step_s
        times = np.arange(start, t1 + 30.0 + 1e-9, self.state_step_s)
        if times.size > self.max_state_requests:
            notes.append(f"Time window truncated to {self.max_state_requests} OpenSky requests.")
            times = times[: self.max_state_requests]
        per_ac: Dict[str, Dict[str, Any]] = {}
        for T in times:
            body = self._get("/states/all", {"time": int(T), "lamin": round(bbox[0], 4), "lomin": round(bbox[1], 4),
                                             "lamax": round(bbox[2], 4), "lomax": round(bbox[3], 4)}, float(T), notes)
            if body is None:
                break
            for s in body.get("states") or []:
                if not isinstance(s, list) or len(s) < 14 or s[3] is None or s[5] is None or s[6] is None or s[8]:
                    continue
                ac = per_ac.setdefault(str(s[0]).lower(), {"callsign": (s[1] or "").strip(), "rows": {}})
                ac["rows"][float(s[3])] = s
        flights = []
        for icao, ac in per_ac.items():
            rows = [ac["rows"][k] for k in sorted(ac["rows"])]
            geo_ok = all(r[13] is not None for r in rows)
            alt_idx, basis = (13, "geometric") if geo_ok else (7, "barometric")
            rows = [r for r in rows if r[alt_idx] is not None]
            if len(rows) < 3:
                continue
            full = all(r[9] is not None and r[10] is not None for r in rows)
            flights.append({
                "flight_id": icao.upper(), "icao24": icao, "callsign": ac["callsign"] or icao.upper(),
                "aircraft_type": "Unknown (OpenSky)",
                "timestamps": [float(r[3]) for r in rows],
                "latitudes": [float(r[6]) for r in rows], "longitudes": [float(r[5]) for r in rows],
                "altitudes": [float(r[alt_idx]) for r in rows],
                "speeds": [float(r[9]) for r in rows] if full else [],
                "headings": [float(r[10]) for r in rows] if full else [],
                "vertical_rates": [float(r[11]) for r in rows] if all(r[11] is not None for r in rows) else [],
                "data_source": "historical", "altitude_basis": basis, "provider": self.name,
            })
        return flights

    def _from_tracks(self, bbox, t0, t1, hints, notes) -> List[Dict[str, Any]]:
        flights = []
        for icao in list(hints)[: self.max_track_requests]:
            icao = str(icao).strip().lower()
            body = self._get("/tracks/all", {"icao24": icao, "time": int(0.5 * (t0 + t1))}, t1, notes)
            if not body:
                notes.append(f"No OpenSky track for {icao} at the requested time.")
                continue
            path = [w for w in body.get("path") or [] if w[1] is not None and w[2] is not None and w[3] is not None
                    and not w[5]]
            if len(path) < 2:
                continue
            tw = np.array([float(w[0]) for w in path])
            lo, hi = max(t0 - 120.0, tw[0]), min(t1 + 120.0, tw[-1])
            if hi - lo < 5.0:
                notes.append(f"OpenSky track for {icao} does not cover the requested window.")
                continue
            tg = np.arange(lo, hi + 1e-9, 1.0)
            lat = np.interp(tg, tw, [float(w[1]) for w in path])
            lon = np.interp(tg, tw, [float(w[2]) for w in path])
            alt = np.interp(tg, tw, [float(w[3]) for w in path])
            if not np.any((lat >= bbox[0]) & (lat <= bbox[2]) & (lon >= bbox[1]) & (lon <= bbox[3])):
                notes.append(f"Aircraft {icao} was not inside the search area during the window.")
                continue
            flights.append({
                "flight_id": icao.upper(), "icao24": icao,
                "callsign": (body.get("callsign") or body.get("calllsign") or icao.upper()).strip(),
                "aircraft_type": "Unknown (OpenSky)",
                "timestamps": tg.tolist(), "latitudes": lat.tolist(), "longitudes": lon.tolist(),
                "altitudes": alt.tolist(), "speeds": [], "headings": [], "vertical_rates": [],
                "data_source": "historical", "altitude_basis": "barometric", "provider": self.name,
                "waypoint_interpolated": True,
            })
        return flights


def _bbox(lat: float, lon: float, radius_km: float):
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(0.1, math.cos(math.radians(lat))))
    return (lat - dlat, lon - dlon, lat + dlat, lon + dlon)


def build_providers(cache_dir: str) -> List[Any]:
    """Providers enabled by the environment (currently OpenSky when credentials are set)."""
    tokens = OpenSkyTokenManager.from_env()
    if not tokens.configured:
        return []
    return [OpenSkyHistoricalProvider(tokens, DiskCache(os.path.join(cache_dir, "opensky")))]
