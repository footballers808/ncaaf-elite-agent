# src/net.py
from __future__ import annotations

import os
import random
import time
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import requests

BASE = "https://api.collegefootballdata.com"

# Config via env
_API = (os.environ.get("CFBD_API_KEY") or "").strip()
_MIN_SLEEP = float(os.environ.get("CFBD_MIN_SLEEP_MS", "300")) / 1000.0
_MAX_RETRIES = int(os.environ.get("CFBD_MAX_RETRIES", "8"))
_TIMEOUT = float(os.environ.get("CFBD_TIMEOUT", "30"))
_BACKOFF_BASE = float(os.environ.get("CFBD_BACKOFF_BASE_S", "2.0"))

_session = requests.Session()
_session.headers.update({"Accept": "application/json"})
if _API:
    _session.headers.update({"Authorization": f"Bearer {_API}"})


_last_call_monotonic = 0.0


def _throttle() -> None:
    """Respect a minimum gap between calls to avoid 429s."""
    global _last_call_monotonic
    now = time.monotonic()
    wait = _MIN_SLEEP - (now - _last_call_monotonic)
    if wait > 0:
        time.sleep(wait)
    _last_call_monotonic = time.monotonic()


def cfbd_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: Optional[float] = None):
    """
    GET with throttle + retry/backoff (429 & 5xx).
    Returns parsed JSON or raises on final failure.
    """
    url = urljoin(BASE, path)
    params = params or {}
    timeout = timeout or _TIMEOUT

    last_exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        _throttle()
        try:
            r = _session.get(url, params=params, timeout=timeout)
        except Exception as e:
            last_exc = e
            sleep_s = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), 90.0)
            time.sleep(sleep_s)
            continue

        # 429: Too Many Requests
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_s = float(retry_after)
                except Exception:
                    sleep_s = _BACKOFF_BASE * (2 ** attempt)
            else:
                sleep_s = _BACKOFF_BASE * (2 ** attempt)
            sleep_s = min(max(sleep_s + random.uniform(0, 0.5), _MIN_SLEEP), 90.0)
            time.sleep(sleep_s)
            continue

        # Retry on transient 5xx
        if 500 <= r.status_code < 600:
            sleep_s = min(_BACKOFF_BASE * (2 ** attempt) + random.uniform(0, 0.5), 90.0)
            time.sleep(sleep_s)
            continue

        # Success or other non-retry error
        r.raise_for_status()
        if not r.content:
            return None
        ct = (r.headers.get("Content-Type") or "").lower()
        return r.json() if "json" in ct or (r.text and r.text.strip().startswith("{")) else r.text

    if last_exc:
        raise last_exc
    raise requests.HTTPError(f"CFBD GET failed after {_MAX_RETRIES} attempts: {url} {params}")
