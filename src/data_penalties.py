# SPDX-License-Identifier: MIT
"""
Builds rolling penalty features by team/season and writes:
    artifacts/features/penalties.parquet

CLI:
    python -m src.data_penalties --years 3 --windows 3,5,10 [--min-games 1] [--end-year 2024]

Notes
-----
- Designed to be *robust first*: if API calls fail or rate limits are hit,
  we still emit a well-formed empty parquet so the rest of the pipeline
  continues (model can fall back or ignore these features).
- Uses src.net.cfbd_get() if available (preferred) which respects CFBD_*
  throttle/retry envs. Otherwise falls back to a simple local requester.
- The produced schema is intentionally simple and stable:
      ['season', 'team', 'games', 'roll_penalties_w{W}']
  Downstream code can join on ['season', 'team'] and treat missing values
  as 0/NaN or drop as needed.

Future upgrade
--------------
Replace the placeholder rollup logic with real penalty counts retrieved from
plays/drives endpoints, then compute rolling sums by window. The I/O contract
(parquet columns) can remain the same.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Optional import of src.net
# -----------------------------
_CFBD_VIA_NET = False
def _maybe_import_net():
    global _CFBD_VIA_NET, cfbd_get
    try:
        # Prefer your repo helper that already handles throttling + retries
        from src.net import cfbd_get  # type: ignore
        _CFBD_VIA_NET = True
        return cfbd_get
    except Exception:
        _CFBD_VIA_NET = False
        return None


cfbd_get = _maybe_import_net()


# -----------------------------
# Fallback requester if src.net is not present
# -----------------------------
def _fallback_cfbd_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Minimal, throttle-aware fetcher for CFBD when src.net.cfbd_get is unavailable.
    Reads:
      CFBD_API_KEY, CFBD_MIN_SLEEP_MS, CFBD_MAX_RETRIES, CFBD_BACKOFF_BASE_S
    """
    import requests
    base = "https://api.collegefootballdata.com"
    api_key = os.getenv("CFBD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CFBD_API_KEY is not set; cannot fetch data.")

    min_sleep_ms = max(int(os.getenv("CFBD_MIN_SLEEP_MS", "1000")), 500)
    max_retries = max(int(os.getenv("CFBD_MAX_RETRIES", "8")), 1)
    backoff_base = float(os.getenv("CFBD_BACKOFF_BASE_S", "1.6"))

    url = f"{base}{path}"
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    params = params or {}

    last_exc: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            # backoff on 4xx/5xx
            last_exc = RuntimeError(
                f"HTTP {resp.status_code} for {url} params={params} body={resp.text[:200]}"
            )
        except Exception as e:
            last_exc = e

        sleep_s = (min_sleep_ms / 1000.0) + (backoff_base ** attempt) * 0.2
        time.sleep(sleep_s)

    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to GET {url} after {max_retries} attempts.")


def _safe_cfbd_get(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    if cfbd_get is not None:
        return cfbd_get(path, params or {})
    return _fallback_cfbd_get(path, params)


# -----------------------------
# CLI parsing
# -----------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build rolling penalty features.")
    p.add_argument(
        "--years",
        type=int,
        default=1,
        help="Number of seasons to include (counting backwards).",
    )
    p.add_argument(
        "--windows",
        type=str,
        default="3,5,10",
        help="Comma-separated rolling windows used to name the output columns.",
    )
    p.add_argument(
        "--min-games",
        type=int,
        default=1,
        dest="min_games",
        help="Min games per team-season to keep in the table.",
    )
    p.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Optional last season to include. If omitted, uses current UTC year.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="artifacts/features/penalties.parquet",
        help="Output parquet path.",
    )
    return p.parse_args()


# -----------------------------
# Helpers
# -----------------------------
@dataclass
class SeasonTeams:
    season: int
    teams: List[str]


def _seasons_to_build(years: int, end_year: Optional[int]) -> List[int]:
    if end_year is None:
        end_year = datetime.now(timezone.utc).year
    # Build descending: [end_year, end_year-1, ...]
    out = [end_year - i for i in range(max(years, 1))]
    return sorted(out)  # ascending or descending is fine for features


def _fetch_teams_for_season(season: int) -> List[str]:
    """
    Very light discovery of teams that played in a given season by reading /games.
    We avoid heavy endpoints. If rate-limited, bubble up exception.
    """
    # Regular season is enough to enumerate most teams
    js = _safe_cfbd_get("/games", {"year": season, "seasonType": "regular"})
    teams: set[str] = set()
    for g in js or []:
        # Some responses use keys home/away team names:
        ht = g.get("home_team") or g.get("homeTeam")
        at = g.get("away_team") or g.get("awayTeam")
        if ht: teams.add(str(ht))
        if at: teams.add(str(at))
    return sorted(teams)


def _build_placeholder_penalty_frame(
    season: int,
    teams: List[str],
    windows: List[int],
) -> pd.DataFrame:
    """
    Placeholder feature rollups. Since we are not pulling the full play/drive
    data here (expensive and rate-limited), we emit zeros for the rolling
    penalty counts. This keeps the pipeline unblocked and schema-stable.

    Produced columns:
        ['season', 'team', 'games'] + [f'roll_penalties_w{W}' for W in windows]
    """
    if not teams:
        return pd.DataFrame(
            columns=["season", "team", "games"] + [f"roll_penalties_w{w}" for w in windows]
        )

    rows = []
    for t in teams:
        row = {"season": season, "team": t, "games": 0}
        for w in windows:
            row[f"roll_penalties_w{w}"] = 0
        rows.append(row)
    return pd.DataFrame(rows)


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)


# -----------------------------
# Main build
# -----------------------------
def main() -> int:
    args = _parse_args()

    years = max(args.years, 1)
    windows = [int(w) for w in str(args.windows).split(",") if str(w).strip().isdigit()]
    if not windows:
        windows = [3, 5, 10]

    seasons = _seasons_to_build(years, args.end_year)

    all_frames: List[pd.DataFrame] = []
    errors: List[Tuple[int, str]] = []

    for season in seasons:
        try:
            teams = _fetch_teams_for_season(season)
        except Exception as e:
            errors.append((season, f"team discovery failed: {e}"))
            # produce an empty placeholder for the season to keep schema stable
            teams = []

        try:
            df = _build_placeholder_penalty_frame(season, teams, windows)
            # Basic row pruning by min games (here games==0 because placeholders)
            # In future, once you compute real rolling counts & games, this filter
            # will remove very low-sample team-seasons.
            if args.min_games > 1 and not df.empty:
                df = df.loc[df["games"] >= int(args.min_games)].copy()
            all_frames.append(df)
        except Exception as e:
            errors.append((season, f"build failed: {e}"))

    # Concatenate & de-duplicate
    try:
        if all_frames:
            out_df = pd.concat(all_frames, ignore_index=True, sort=False)
            out_df = out_df.drop_duplicates(subset=["season", "team"], keep="last")
        else:
            out_df = pd.DataFrame(
                columns=["season", "team", "games"] + [f"roll_penalties_w{w}" for w in windows]
            )
    except Exception as e:
        print(f"[feat_penalties] ERROR concatenating frames: {e}", file=sys.stderr)
        # fall back to an empty but well-formed table
        out_df = pd.DataFrame(
            columns=["season", "team", "games"] + [f"roll_penalties_w{w}" for w in windows]
        )

    # Emit parquet
    _ensure_parent_dir(args.out)
    try:
        out_df.to_parquet(args.out, index=False)
    except Exception as e:
        # As a last resort, write CSV to avoid hard failure
        csv_fallback = os.path.splitext(args.out)[0] + ".csv"
        print(
            f"[feat_penalties] WARNING: to_parquet failed ({e}); writing CSV fallback {csv_fallback}",
            file=sys.stderr,
        )
        out_df.to_csv(csv_fallback, index=False)

    # Log any issues
    if errors:
        for y, msg in errors:
            print(f"[feat_penalties] Season {y}: {msg}", file=sys.stderr)
        print(
            "[feat_penalties] Built with warnings. Consider increasing throttling or "
            "implementing the real penalties rollup when ready.",
            file=sys.stderr,
        )
    else:
        print(f"[feat_penalties] OK. Wrote {len(out_df):,} rows to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
