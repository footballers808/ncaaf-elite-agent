# src/cfbd_client.py
from __future__ import annotations

import time
from typing import Any, Dict, Optional, Tuple

import requests

from .common import get_env, cache_requests

CFBD_BASE = "https://api.collegefootballdata.com"

# ---- Tunables from env (with safe defaults) ----
_MAX_RETRIES: int = int(get_env("CFBD_MAX_RETRIES", "8"))                 # total attempts
_BACKOFF_BASE: float = float(get_env("CFBD_BACKOFF_BASE_SEC", "0.6"))     # seconds
_THROTTLE_MS: int = int(get_env("CFBD_THROTTLE_MS", "250"))               # min gap between calls (ms)
_TIMEOUT: int = 30                                                         # per-request timeout seconds

# process-local simple throttle timer
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
    # Exponential backoff with slight linear jitter, capped
    delay = min(_BACKOFF_BASE * (2 ** attempt) + 0.05 * attempt, 12.0)
    # Be a bit more patient on 429
    if status == 429:
        delay = max(delay, _BACKOFF_BASE * 2)
    time.sleep(delay)


def _diagnostics() -> str:
    """Return a short, safe diagnostics string about auth presence."""
    key_present = bool(get_env("CFBD_API_KEY", ""))
    return f"auth_present={key_present}, retries={_MAX_RETRIES}, throttle_ms={_THROTTLE_MS}, backoff_base={_BACKOFF_BASE}"


def _get(path: str, params: Dict[str, Any]) -> Any:
    """
    Robust GET with throttle + retries for 429/5xx.
    After exhausting retries, raise with precise status/body diagnostics.
    """
    cache_requests()  # install requests_cache once per process
    url = CFBD_BASE + path
    headers = _headers()

    last_status: Optional[int] = None
    last_text: Optional[str] = None
    last_exc: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES):
        try:
            _throttle()
            resp = requests.get(url, params=params, headers=headers, timeout=_TIMEOUT)

            # capture for diagnostics
            last_status = resp.status_code
            # only keep a small preview of body to avoid huge logs
            last_text = (resp.text or "")[:300].replace("\n", "\\n")

            # Retry on typical transient statuses
            if resp.status_code in (429, 500, 502, 503, 504):
                _sleep_backoff(attempt, resp.status_code)
                continue

            # Non-retriable: raise if error, else return JSON
            resp.raise_for_status()
            return resp.json()

        except requests.RequestException as e:
            last_exc = e
            # Network/HTTP error -> backoff and retry
            status = getattr(getattr(e, "response", None), "status_code", 0) or (last_status or 0)
            _sleep_backoff(attempt, status)
            continue

    # Out of retries — construct an informative error
    diag = _diagnostics()
    if last_exc:
        # Attach last response info if we have it
        msg = f"CFBD request failed after retries: {url} params={params} status={last_status} body_preview={last_text} ({diag})"
        raise requests.HTTPError(msg) from last_exc

    # We never raised/returned (likely kept getting 429/5xx)
    msg = f"CFBD request exhausted retries: {url} params={params} status={last_status} body_preview={last_text} ({diag})"
    raise requests.HTTPError(msg)


# ------------- Public endpoints -----------------

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
    # Not always populated by CFBD; treat as optional best-effort
    p: Dict[str, Any] = {"year": year}
    if week is not None:
        p["week"] = week
    try:
        return _get("/injuries", p)
    except Exception:
        return []


def venues() -> Any:
    return _get("/venues", {})
