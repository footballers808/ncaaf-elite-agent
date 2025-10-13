# src/labeler.py
from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
import requests


CFBD = "https://api.collegefootballdata.com"


def _pick(cols: Iterable[str], choices: Iterable[str]) -> Optional[str]:
    """
    Return first matching column (case-insensitive) from `choices`.
    Works with snake_case and camelCase because we compare lowercased names.
    """
    lower_map = {c.lower(): c for c in cols}
    for c in choices:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _headers() -> dict:
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing CFBD_API_KEY secret.")
    return {"Authorization": f"Bearer {api_key}"}


def fetch_completed_games(year: int, season_type: str = "regular") -> pd.DataFrame:
    """
    Pull games for a season; keep only rows that have final scores and
    normalize to a standard schema:
        game_id, home, away, home_points, away_points
    """
    url = f"{CFBD}/games"
    params = {"year": year, "seasonType": season_type}
    r = requests.get(url, params=params, headers=_headers(), timeout=60)
    r.raise_for_status()
    raw = pd.DataFrame(r.json())

    if raw.empty:
        print(f"⚠️ CFBD returned 0 rows for year={year} seasonType={season_type}")
        return raw

    cols = list(raw.columns)

    # Accept both snake_case and camelCase
    id_col   = _pick(cols, ["id", "game_id", "gameId"])
    home_col = _pick(cols, ["home_team", "home", "homeName", "homeTeam"])
    away_col = _pick(cols, ["away_team", "away", "awayName", "awayTeam"])
    hp_col   = _pick(cols, ["home_points", "home_score", "HomePoints", "homePoints"])
    ap_col   = _pick(cols, ["away_points", "away_score", "AwayPoints", "awayPoints"])

    missing = []
    if not id_col:   missing.append("id/game_id")
    if not home_col: missing.append("home_team/home/homeTeam")
    if not away_col: missing.append("away_team/away/awayTeam")
    if not hp_col:   missing.append("home_points/home_score")
    if not ap_col:   missing.append("away_points/away_score")

    if missing:
        # Help debug by printing available columns
        raise ValueError(
            f"CFBD /games missing expected columns: {missing}\n"
            f"Available columns: {cols}"
        )

    out = pd.DataFrame(
        {
            "game_id":     raw[id_col],
            "home":        raw[home_col],
            "away":        raw[away_col],
            "home_points": raw[hp_col],
            "away_points": raw[ap_col],
        }
    )

    # Final only
    out = out[out["home_points"].notna() & out["away_points"].notna()].copy()

    # Normalize dtypes
    out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce")
    out = out.dropna(subset=["game_id"]).copy()
    out["game_id"] = out["game_id"].astype("int64")

    out = out.drop_duplicates(subset=["game_id"]).reset_index(drop=True)
    return out


def _save_labels(df: pd.DataFrame) -> str:
    os.makedirs("store", exist_ok=True)
    path = os.path.join("store", "labels.parquet")
    df.to_parquet(path, index=False)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=datetime.utcnow().year)
    parser.add_argument(
        "--season-type",
        type=str,
        default="regular",
        help="regular|postseason|both (CFBD seasonType)",
    )
    args = parser.parse_args()

    df = fetch_completed_games(args.year, args.season_type)
    if df.empty:
        print("⚠️ No completed games found; nothing to write.")
        return

    path = _save_labels(df)
    print(f"✅ Wrote {path} with {len(df)} rows and columns {list(df.columns)}")


if __name__ == "__main__":
    main()
