# src/net.py
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests

BASE = "https://api.collegefootballdata.com"

# Secrets/knobs (can be set in GitHub Actions env)
_API = (os.environ.get("CFBD_API_KEY") or "").strip()
_MIN_SLEEP = float(os.environ.get("CFBD_MIN_SLEEP_MS", "300")) / 1000.0  # min gap between calls
_MAX_RETRIES = int(os.environ.get("CFBD_MAX_RETRIES", "8"))
_TIMEOUT = float(os.environ.get("CFBD_TIMEOUT", "30"))

_session = requests.Session()
_session.headers.update({"Accept": "application/json"})
if _API:
    _session.headers.update({"Authorization": f"Bearer {_API}"})


_last_call_monotonic = 0.0


def _throttle() -> None:
    """Ensure a minimum gap between calls to avoid 429s."""
    global _last_call_monotonic
    now = time.monotonic()
    wait = _MIN_SLEEP - (now - _last_call_monotonic)
    if wait > 0:
        time.sleep(wait)
    _last_call_monotonic = time.monotonic()


def cfbd_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None):
    """
    GET wrapper with throttle + retry/backoff (handles 429/5xx).
    Returns parsed JSON (or raises on final failure).
    """
    url = urljoin(BASE, path)
    params = params or {}
    timeout = timeout or _TIMEOUT

    backoff_base = 1.5  # seconds, exponential

    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        _throttle()
        try:
            r = _session.get(url, params=params, timeout=timeout)
        except Exception as e:
            last_exc = e
            time.sleep(min(backoff_base * (2 ** attempt), 60.0))
            continue

        # Handle 429 rate limit explicitly
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            sleep_s = float(retry_after) if retry_after else backoff_base * (2 ** attempt)
            time.sleep(min(max(sleep_s, _MIN_SLEEP), 60.0))
            continue

        # Retry on transient 5xx
        if 500 <= r.status_code < 600:
            time.sleep(min(backoff_base * (2 ** attempt), 60.0))
            continue

        # Success or other non-retry error
        r.raise_for_status()
        if not r.content:
            return None
        ct = (r.headers.get("Content-Type") or "").lower()
        return r.json() if "json" in ct or r.text.startswith("{") else r.text

    # If we fall through, raise the last error or a generic one
    if last_exc:
        raise last_exc
    raise requests.HTTPError(f"CFBD GET failed after {_MAX_RETRIES} attempts: {url} {params}")
