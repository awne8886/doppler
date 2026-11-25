"""
Optional per-cell observer ground elevation.

By default the observer sits at ``observer_alt_m`` everywhere (flat ground),
which needs no data and works offline.  Two optional sources refine that:

* a **user-provided** regular lat/lon elevation grid
  (``{"lat_min", "lat_max", "lon_min", "lon_max", "values": [[...], ...]}``,
  ``values[i][j]`` at latitude row ``i`` from ``lat_min`` and longitude column
  ``j`` from ``lon_min``);
* an opt-in **DEM lookup** against an OpenTopoData-compatible API
  (``DOPPLER_DEM_URL``, default the public SRTM 90 m dataset), sampled on a
  coarse grid over the search area and cached on disk.  Any failure falls
  back to flat ground with a note.

Datum: DEMs give height above mean sea level (geoid).  ADS-B geometric
altitude is height above the WGS-84 ellipsoid, so pass the local geoid
undulation as ``datum_offset_m`` when matching against geometric tracks
(barometric/QNH altitudes are already MSL-like).
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import requests

from app.physics.doppler_engine import DopplerEngine

logger = logging.getLogger("doppler.elevation")
DEFAULT_DEM_URL = "https://api.opentopodata.org/v1/srtm90m"


class LatLonElevation:
    """Bilinear elevation on a regular lat/lon grid, exposed as a local (x, y) -> metres callable."""

    def __init__(self, grid: Dict[str, Any], center_lat: float, center_lon: float, datum_offset_m: float = 0.0):
        self.values = np.asarray(grid["values"], dtype=np.float64)
        if self.values.ndim != 2 or min(self.values.shape) < 2 or not np.all(np.isfinite(self.values)):
            raise ValueError("Elevation grid 'values' must be a finite 2-D array of at least 2x2.")
        self.lat_min, self.lat_max = float(grid["lat_min"]), float(grid["lat_max"])
        self.lon_min, self.lon_max = float(grid["lon_min"]), float(grid["lon_max"])
        if not (self.lat_max > self.lat_min and self.lon_max > self.lon_min):
            raise ValueError("Elevation grid bounds are inverted or empty.")
        self.center = (center_lat, center_lon)
        self.offset = float(datum_offset_m)
        self.engine = DopplerEngine()

    def at_latlon(self, lat: float, lon: float) -> float:
        ny, nx = self.values.shape
        fy = min(max((lat - self.lat_min) / (self.lat_max - self.lat_min) * (ny - 1), 0.0), ny - 1.0)
        fx = min(max((lon - self.lon_min) / (self.lon_max - self.lon_min) * (nx - 1), 0.0), nx - 1.0)
        iy, ix = min(int(fy), ny - 2), min(int(fx), nx - 2)
        ty, tx = fy - iy, fx - ix
        v = self.values
        return float((1 - ty) * ((1 - tx) * v[iy, ix] + tx * v[iy, ix + 1])
                     + ty * ((1 - tx) * v[iy + 1, ix] + tx * v[iy + 1, ix + 1])) + self.offset

    def __call__(self, x: float, y: float) -> float:
        lat, lon, _ = self.engine.cartesian_to_geodetic(x, y, 0.0, self.center[0], self.center[1], 0.0)
        return self.at_latlon(float(lat), float(lon))


class OpenTopoDataProvider:
    """Batched elevation lookups (<=100 points per request) with an optional disk cache."""

    def __init__(self, url: str = DEFAULT_DEM_URL, cache: Optional[Any] = None, session: Optional[Any] = None,
                 timeout: float = 10.0):
        self.url = url
        self.cache = cache
        self.session = session or requests.Session()
        self.timeout = timeout

    def elevations(self, points: Sequence[Tuple[float, float]]) -> Optional[List[float]]:
        out: List[Optional[float]] = []
        for i in range(0, len(points), 100):
            chunk = points[i:i + 100]
            locs = "|".join(f"{la:.5f},{lo:.5f}" for la, lo in chunk)
            key = f"{self.url}?{locs}"
            hit = self.cache.get(key) if self.cache else None
            if hit is None:
                try:
                    resp = self.session.get(self.url, params={"locations": locs}, timeout=self.timeout,
                                            headers={"User-Agent": "DopplerAcousticGeolocator/1.2"})
                except Exception as e:
                    logger.warning("DEM lookup failed: %s", e)
                    return None
                if resp.status_code != 200:
                    logger.warning("DEM lookup returned HTTP %s", resp.status_code)
                    return None
                hit = [r.get("elevation") for r in (resp.json() or {}).get("results", [])]
                if len(hit) != len(chunk) or any(v is None for v in hit):
                    return None
                if self.cache:
                    self.cache.set(key, hit)
            out.extend(float(v) for v in hit)
        return out  # type: ignore[return-value]


def dem_grid(provider: OpenTopoDataProvider, center_lat: float, center_lon: float, radius_km: float,
             n: int = 9) -> Optional[Dict[str, Any]]:
    """Sample an n x n lat/lon grid covering the search square; None if the lookup fails."""
    dlat = radius_km / 111.0 * 1.05
    dlon = radius_km / (111.0 * max(0.1, math.cos(math.radians(center_lat)))) * 1.05
    lats = np.linspace(center_lat - dlat, center_lat + dlat, n)
    lons = np.linspace(center_lon - dlon, center_lon + dlon, n)
    pts = [(float(la), float(lo)) for la in lats for lo in lons]
    vals = provider.elevations(pts)
    if vals is None:
        return None
    return {"lat_min": float(lats[0]), "lat_max": float(lats[-1]), "lon_min": float(lons[0]),
            "lon_max": float(lons[-1]), "values": np.asarray(vals).reshape(n, n).tolist()}


def build_dem_provider(cache_dir: str) -> Optional[OpenTopoDataProvider]:
    from app.adsb.providers import DiskCache

    url = os.environ.get("DOPPLER_DEM_URL", DEFAULT_DEM_URL)
    if not url:
        return None
    return OpenTopoDataProvider(url, DiskCache(os.path.join(cache_dir, "dem")))


ElevationFn = Callable[[float, float], float]
