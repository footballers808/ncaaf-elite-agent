# src/cfbd_client.py
from __future__ import annotations

import time
import math
import random
from typing import Any, Dict, Optional, List

import requests

from .common import get_env, cache_requests


CFBD = "https://api.collegefootballdata.com"


def _headers() -> Dict[str, str]:
    key = get_env("CFBD_API_KEY", "")
    if not key:
        # Allow running in limited/non-key mode (some endpoints still work)
        return {}
    return {"Authorization": f"Bearer {key}"}


def _throttle_ms() -> int:
    """Small inter-request delay to avoid 429s during loops."""
    try:
        return max(0, int(get_env("CFBD_THROTTLE_MS", "250")))
    except Exception:
        return 250


def _max_retries() -> int:
    try:
        return max(1, int(get_env("CFBD_MAX_RETRIES", "8")))
    except Exception:
        return 8


def _base_backoff_seconds() -> float:
    try:
        return max(0.1, float(get_env("CFBD_BACKOFF_BASE_SEC", "0.6")))
    except Exception:
        return 0.6


def _sleep(seconds: float) -> None:
    # Cap to something reasonable in CI
    time.sleep(min(seconds, 15.0))


def _respect_retry_after(resp: requests.Response) -> Optional[float]:
    """Return seconds to wait if server instructed us via Retry-After."""
    ra = resp.headers.get("Retry-After")
    if not ra:
        return None
    try:
        return float(ra)
    except Exception:
        return None


def _get(path: str, params: Dict[str, Any]) -> Any:
    """
    Robust GET with:
      - requests-cache (set up via common.cache_requests())
      - exponential backoff + jitter
      - respect Retry-After on 429/5xx
      - gentle throttle between attempts
    """
    cache_requests()  # enable persistent cache across the job
    url = f"{CFBD}{path}"

    # Normalize params: requests struggles with None sometimes
    clean_params: Dict[str, Any] = {k: v for k, v in (params or {}).items() if v is not None}

    session = requests.Session()
    tries = _max_retries()
    base = _base_backoff_seconds()
    for attempt in range(tries):
        if attempt > 0:
            # gentle throttle between subsequent attempts
            _sleep(_throttle_ms() / 1000.0)

        try:
            r = session.get(url, params=clean_params, headers=_headers(), timeout=30)
        except requests.RequestException as e:
            # Backoff on network problems, then retry
            wait = base * (2 ** attempt) + random.random() * 0.3
            _sleep(wait)
            if attempt == tries - 1:
                raise
            continue

        # Cached hits return status_code from original; treat like normal
        status = r.status_code

        if status == 200:
            return r.json()

        # Rate limited: respect Retry-After if provided, else exponential backoff
        if status == 429:
            ra = _respect_retry_after(r)
            if ra is None:
                ra = base * (2 ** attempt) + random.random() * 0.5
            _sleep(float(ra))
            if attempt == tries - 1:
                r.raise_for_status()
            continue

        # Transient server errors -> retry with backoff
        if 500 <= status < 600:
            ra = _respect_retry_after(r)
            if ra is None:
                ra = base * (2 ** attempt) + random.random() * 0.5
            _sleep(float(ra))
            if attempt == tries - 1:
                r.raise_for_status()
            continue

        # Client errors other than 429: fail fast (bad params, etc.)
        r.raise_for_status()

    # If we somehow exit loop without return/raise, raise generic
    raise RuntimeError(f"CFBD request failed after {tries} attempts: {url}")


# -------- Public endpoints (best-effort) --------

def games(year: int, season_type: str = "regular", week: Optional[int] = None) -> Any:
    p: Dict[str, Any] = {"year": year, "seasonType": season_type}
    if week is not None:
        p["week"] = week
    return _get("/games", p)


def lines(year: int, week: Optional[int] = None, season_type: str = "regular") -> Any:
    p: Dict[str, Any] = {"year": year, "seasonType": season_type}
    if week is not None:
        p["week"] = week
    return _get("/lines", p)


def injuries(year: int, week: Optional[int] = None, season_type: str = "regular") -> Any:
    # Not always populated, so callers should be tolerant
    p: Dict[str, Any] = {"year": year, "seasonType": season_type}
    if week is not None:
        p["week"] = week
    return _get("/injuries", p)


def game_stats(year: int, season_type: str = "regular", week: Optional[int] = None) -> Any:
    p: Dict[str, Any] = {"year": year, "seasonType": season_type}
    if week is not None:
        p["week"] = week
    return _get("/game/statistics", p)


def venues() -> Any:
    return _get("/venues", {})
