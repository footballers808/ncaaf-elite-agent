# src/cfbd_client.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional

import requests

from .common import get_env, cache_requests

CFBD_BASE = "https://api.collegefootballdata.com"

# ---- Tunables come from env (with safe defaults) ----
_MAX_RETRIES: int = int(get_env("CFBD_MAX_RETRIES", "8"))          # total attempts
_BACKOFF_BASE: float = float(get_env("CFBD_BACKOFF_BASE_SEC", "0.6"))  # seconds
_THROTTLE_MS: int = int(get_env("CFBD_THROTTLE_MS", "250"))        # min gap between calls
_TIMEOUT: int = 30

# single-process simple throttle
_last_call = [0.0]


def _headers() -> Dict[str, str]:
    key = get_env("CFBD_API_KEY", "")
    h = {
        "User-Agent": "ncaaf-elite-agent/1.0 (+github-actions)",
        "Accept": "application/json",
    }
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def _throttle() -> None:
    """Ensure at least THROTTLE_MS between consecutive requests."""
    if _THROTTLE_MS <= 0:
        return
    now = time.time()
    min_gap = _THROTTLE_MS / 1000.0
    dt = now - _last_call[0]
    if dt < min_gap:
        time.sleep(min_gap - dt)
    _last_call[0] = time.time()


def _sleep_backoff(attempt: int, status: int) -> None:
    # exp backoff with small linear jitter, cap to 10s
    delay = min(_BACKOFF_BASE * (2 ** attempt) + 0.05 * attempt, 10.0)
    # Be a bit more patient on explicit 429
    if status == 429:
        delay = max(delay, _BACKOFF_BASE * 2)
    time.sleep(delay)


def _get(path: str, params: Dict[str, Any]) -> Any:
    """
    Robust GET with throttle + retries for 429/5xx.
    Raises for non-retriable errors.
    """
    cache_requests()  # install requests_cache once per process
    url = CFBD_BASE + path
    headers = _headers()

    exc: Optional[Exception] = None
    for attempt in range(_MAX_RETRIES):
        try:
            _throttle()
            r = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT)

            if r.status_code in (429, 500, 502, 503, 504):
                _sleep_backoff(attempt, r.status_code)
                continue  # retry

            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            exc = e
            # network hiccup: backoff and retry
            _sleep_backoff(attempt, getattr(e.response, "status_code", 0))
            continue

    # last attempt failed
    if exc:
        raise exc
    raise RuntimeError("CFBD request failed without exception detail")


# ------------- Public endpoints (best-effort helpers) -----------------

def games(year: int, season_type: str = "regular", week: Optional[int] = None) -> Any:
    p: Dict[str, Any] = {"year": year, "seasonType": season_type}
    if week is not None:
        p["week"] = week
    return _get("/games", p)


def lines(year: int, season_type: str = "regular", week: Optional[int] = None) -> Any:
    p: Dict[str, Any] = {"year": year, "seasonType": season_type}
    if week is not None:
        p["week"] = week
    return _get("/lines", p)


def game_stats(year: int, season_type: str = "regular", week: Optional[int] = None) -> Any:
    p: Dict[str, Any] = {"year": year, "seasonType": season_type}
    if week is not None:
        p["week"] = week
    return _get("/game/statistics", p)


def injuries(year: int, week: Optional[int] = None) -> Any:
    # CFBD injuries aren’t complete for every year; treat as optional
    p: Dict[str, Any] = {"year": year}
    if week is not None:
        p["week"] = week
    try:
        return _get("/injuries", p)
    except Exception:
        return []


def venues() -> Any:
    return _get("/venues", {})
