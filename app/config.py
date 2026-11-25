"""
Runtime configuration read from environment variables.

All settings have safe defaults so the app runs with no configuration.

=========================  =======================================================
Variable                   Meaning
=========================  =======================================================
DOPPLER_DATA_DIR           Writable directory for uploads and caches
                           (default: ``<repo>/var``; never inside ``app/static``).
DOPPLER_MAX_UPLOAD_MB      Maximum accepted upload size in MiB (default 50).
DOPPLER_UPLOAD_TTL_S       Uploaded audio is deleted after this many seconds
                           (default 21600 = 6 h).
DOPPLER_CORS_ORIGINS       Comma-separated allowed CORS origins. ``*`` (default)
                           allows any origin *without* credentials; an empty
                           value disables CORS entirely (same-origin only).
DOPPLER_CARTO_KEY          Optional CARTO basemap key served to the frontend via
                           ``/js/config.js`` (restrict it by HTTP referrer).
DOPPLER_MAX_LIVE_AIRCRAFT  Live-snapshot aircraft kept, nearest first (default 8).
DOPPLER_WORKERS            Matcher worker processes (default min(4, CPUs); 1 = serial).
DOPPLER_MAX_JOBS           Concurrent async geolocation jobs (default 2).
DOPPLER_JOB_TTL_S          Finished jobs are forgotten after this many seconds (default 3600).
=========================  =======================================================
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class Settings:
    data_dir: str = os.path.join(REPO_ROOT, "var")
    max_upload_bytes: int = 50 * 1024 * 1024
    upload_ttl_s: float = 6 * 3600.0
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    carto_key: str = ""
    max_live_aircraft: int = 8
    workers: int = 1
    max_jobs: int = 2
    job_ttl_s: float = 3600.0

    @property
    def upload_dir(self) -> str:
        return os.path.join(self.data_dir, "uploads")

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.data_dir, "cache")

    @classmethod
    def from_env(cls) -> Settings:
        origins_raw = os.environ.get("DOPPLER_CORS_ORIGINS", "*")
        origins = [o.strip() for o in origins_raw.split(",") if o.strip()]
        return cls(
            data_dir=os.path.abspath(os.environ.get("DOPPLER_DATA_DIR") or os.path.join(REPO_ROOT, "var")),
            max_upload_bytes=int(_env_float("DOPPLER_MAX_UPLOAD_MB", 50.0) * 1024 * 1024),
            upload_ttl_s=_env_float("DOPPLER_UPLOAD_TTL_S", 6 * 3600.0),
            cors_origins=origins,
            carto_key=os.environ.get("DOPPLER_CARTO_KEY", ""),
            max_live_aircraft=int(_env_float("DOPPLER_MAX_LIVE_AIRCRAFT", 8)),
            workers=max(1, int(_env_float("DOPPLER_WORKERS", max(1, min(4, os.cpu_count() or 1))))),
            max_jobs=max(1, int(_env_float("DOPPLER_MAX_JOBS", 2))),
            job_ttl_s=_env_float("DOPPLER_JOB_TTL_S", 3600.0),
        )
