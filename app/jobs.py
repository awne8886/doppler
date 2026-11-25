"""
In-memory asynchronous job runner for long geolocation requests.

``POST /api/jobs/geolocate`` queues a job and returns its id; clients poll
``GET /api/jobs/{id}`` or stream ``GET /api/jobs/{id}/events`` (Server-Sent
Events) for progress, and read the same result schema as ``/api/geolocate``
from the finished job.  Jobs run in a small thread pool (the matcher itself
may use worker processes), can be cancelled with ``DELETE /api/jobs/{id}``,
and are forgotten ``ttl_s`` seconds after they finish.  State is per server
process; this is not a durable queue.
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, Optional

STAGE_WEIGHTS = {"queued": (0.0, 0.0), "flights": (0.0, 0.05), "grid": (0.05, 0.6), "refine": (0.6, 0.75),
                 "sensitivity": (0.75, 1.0)}


class JobCancelled(Exception):
    pass


class JobStore:
    def __init__(self, max_workers: int = 2, ttl_s: float = 3600.0, clock: Callable[[], float] = time.time):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="doppler-job")
        self.ttl_s = ttl_s
        self.clock = clock

    def _cleanup(self) -> None:
        now = self.clock()
        for jid in [j for j, v in self._jobs.items()
                    if v["status"] in ("done", "error", "cancelled") and now - v["updated"] > self.ttl_s]:
            self._jobs.pop(jid, None)

    def submit(self, fn: Callable[[Callable[[str, float], None]], Any]) -> str:
        jid = uuid.uuid4().hex
        now = self.clock()
        with self._lock:
            self._cleanup()
            self._jobs[jid] = {"job_id": jid, "status": "queued", "stage": "queued", "progress": 0.0,
                               "created": now, "updated": now, "result": None, "error": None, "cancel": False}

        def progress(stage: str, frac: float) -> None:
            with self._lock:
                job = self._jobs.get(jid)
                if job is None or job["cancel"]:
                    raise JobCancelled()
                lo, hi = STAGE_WEIGHTS.get(stage, (job["progress"], job["progress"]))
                job.update(stage=stage, progress=round(max(job["progress"], lo + (hi - lo) * max(0.0, min(1.0, frac))), 4),
                           updated=self.clock())

        def run() -> None:
            with self._lock:
                job = self._jobs.get(jid)
                if job is None or job["cancel"]:
                    return
                job.update(status="running", stage="flights", updated=self.clock())
            try:
                result = fn(progress)
                status, error = "done", None
            except JobCancelled:
                result, status, error = None, "cancelled", None
            except Exception as e:  # HTTPException and friends: keep status code + detail
                result, status = None, "error"
                error = {"status_code": getattr(e, "status_code", 500),
                         "detail": getattr(e, "detail", None) or "Internal error while geolocating."}
            with self._lock:
                job = self._jobs.get(jid)
                if job is not None:
                    job.update(status=status, result=result, error=error, updated=self.clock(),
                               progress=1.0 if status == "done" else job["progress"],
                               stage="done" if status == "done" else job["stage"])

        self._executor.submit(run)
        return jid

    def get(self, jid: str, include_result: bool = True) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._cleanup()
            job = self._jobs.get(jid)
            if job is None:
                return None
            out = {k: v for k, v in job.items() if k != "cancel" and (include_result or k != "result")}
            return out

    def cancel(self, jid: str) -> bool:
        with self._lock:
            job = self._jobs.get(jid)
            if job is None:
                return False
            if job["status"] in ("queued", "running"):
                job["cancel"] = True
                if job["status"] == "queued":
                    job.update(status="cancelled", updated=self.clock())
            return True
