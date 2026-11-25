"""
Process pool for the matcher.

Grid chunks and the independent Nelder-Mead refinements (alternatives and
sensitivity scenarios) are distributed over worker processes.  The work done
is identical to the serial path, so results do not depend on the number of
workers.

A ``forkserver`` context is used where available: forking a multi-threaded
web server process directly is unsafe, while the fork server is a clean,
single-threaded process that preloads the matcher so workers start quickly.
Any failure to create or use the pool falls back to serial execution.
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import multiprocessing as mp
import os
import pickle
import threading
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, List, Optional, Sequence

logger = logging.getLogger("doppler.parallel")

_POOL: Optional[ProcessPoolExecutor] = None
_POOL_WORKERS = 0
_LOCK = threading.Lock()
_WORKER_CACHE: dict = {}


def default_workers() -> int:
    env = os.environ.get("DOPPLER_WORKERS")
    if env is not None:
        try:
            return max(1, int(env))
        except ValueError:
            return 1
    return max(1, min(4, os.cpu_count() or 1))


def get_pool(workers: int) -> Optional[ProcessPoolExecutor]:
    global _POOL, _POOL_WORKERS
    if workers <= 1:
        return None
    with _LOCK:
        if _POOL is not None and _POOL_WORKERS == workers:
            return _POOL
        shutdown_pool()
        try:
            methods = mp.get_all_start_methods()
            ctx = mp.get_context("forkserver" if "forkserver" in methods else "spawn")
            if ctx.get_start_method() == "forkserver":
                ctx.set_forkserver_preload(["app.physics.matching"])
            _POOL = ProcessPoolExecutor(max_workers=workers, mp_context=ctx)
            _POOL_WORKERS = workers
        except Exception as e:  # pragma: no cover - platform specific
            logger.warning("Process pool unavailable (%s); running serially.", e)
            _POOL, _POOL_WORKERS = None, 0
        return _POOL


def shutdown_pool(wait: bool = False) -> None:
    global _POOL, _POOL_WORKERS
    if _POOL is not None:
        _POOL.shutdown(wait=wait, cancel_futures=True)
    _POOL, _POOL_WORKERS = None, 0


atexit.register(shutdown_pool, True)


def _state(token: str, blob: Optional[bytes]):
    """Unpickle a matcher once per worker per distinct state (keyed by a content hash)."""
    if token not in _WORKER_CACHE:
        if len(_WORKER_CACHE) > 4:
            _WORKER_CACHE.clear()
        _WORKER_CACHE[token] = pickle.loads(blob)
    return _WORKER_CACHE[token]


def _call(token: str, blob: bytes, method: str, args: tuple, kwargs: dict):
    matcher = _state(token, blob)
    return getattr(matcher, method)(*args, **kwargs)


def map_matcher(matcher: Any, method: str, calls: Sequence[tuple], workers: int,
                progress: Optional[Callable[[float], None]] = None) -> List[Any]:
    """
    ``[getattr(matcher, method)(*args, **kwargs) for args, kwargs in calls]``,
    in order, using the process pool when ``workers > 1``.
    """
    pool = get_pool(workers) if len(calls) > 1 else None
    if pool is not None:
        try:
            blob = pickle.dumps(matcher, protocol=pickle.HIGHEST_PROTOCOL)
            # Content-addressed: any change to the matcher state (e.g. a finished grid) is a new key.
            token = hashlib.blake2b(blob, digest_size=16).hexdigest()
            futures = [pool.submit(_call, token, blob, method, args, kwargs) for args, kwargs in calls]
            out = []
            for i, f in enumerate(futures):
                out.append(f.result())
                if progress:
                    progress((i + 1) / len(futures))
            return out
        except Exception as e:  # pragma: no cover - fall back on any pool failure
            logger.warning("Parallel execution failed (%s); retrying serially.", str(e).strip().splitlines()[0])
            shutdown_pool()
    out = []
    for i, (args, kwargs) in enumerate(calls):
        out.append(getattr(matcher, method)(*args, **kwargs))
        if progress:
            progress((i + 1) / len(calls))
    return out
