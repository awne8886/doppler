"""
Shared test fixtures.

Tests must never touch the network: every HTTP request made through
``requests`` raises, so code paths that would query live ADS-B services see a
failure (and must handle it) instead of silently depending on the internet.
Uploads and caches go to a per-session temporary data directory.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

_DATA_DIR = tempfile.mkdtemp(prefix="doppler-test-")
os.environ.setdefault("DOPPLER_DATA_DIR", _DATA_DIR)
os.environ.setdefault("DOPPLER_WORKERS", "1")  # tests run the matcher serially unless they opt in

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class NetworkDisabled(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    import requests

    def blocked(*args, **kwargs):
        raise NetworkDisabled("Network access is disabled in tests; use recorded fixtures or mocks.")

    monkeypatch.setattr(requests.sessions.Session, "request", blocked)
    monkeypatch.setattr(requests, "get", blocked)
    monkeypatch.setattr(requests, "post", blocked)
    yield
