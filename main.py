"""
main.py — NCAAF Elite Agent Predictor (robust)
----------------------------------------------
Fetches schedule from CFBD, normalizes columns, generates model predictions,
and writes a standardized predictions.csv for downstream steps.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd
import requests

# ===============================
# CONFIG
# ===============================
CFBD_API_KEY = os.environ.get("CFBD_API_KEY")
if not CFBD_API_KEY:
    raise EnvironmentError("❌ Missing CFBD_API_KEY in environment secrets.")

CFBD = "https://api.collegefootballdata.com"
HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# Helpers
# ===============================
def pick_col(cols: Iterable[str], choices: Iterable[str]) -> Optional[str]:
    """Return the first column present from choices (case-insensitive)."""
    lower = {c.lower(): c for c in cols}
    for name in choices:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def normalize_games_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with standardized columns: game_id, home, away, start_date?."""
    if df.empty:
        return df.copy()

    cols = list(df.columns)  # keep original case

    game_id_col = pick_col(cols, ["id", "game_id", "gameid"])
    home_col    = pick_col(cols, ["home_team", "home", "hometeam", "home_name", "homeTeam"])
    away_col    = pick_col(cols, ["away_team", "away", "awayteam", "away_name", "awayTeam"])

    # Date can vary a lot; try several possibilities
    date_col = pick_col(
        cols,
        [
            "start_date",
            "start_time",
            "start_time_tbd",
            "game_date",
            "kickoff",
            "start",
        ],
    )

    out = df.copy()

    # Copy/rename when present
    if game_id_col:
        out["game_id"] = out[game_id_col]
    if home_col:
        out["home"] = out[home_col]
    if away_col:
        out["away"] = out[away_col]
    if date_col:
        out["start_date"] = pd.to_datetime(out[date_col], errors="coerce")

    # Keep only what we care about if available
    keep = [c for c in ["game_id", "home", "away", "start_date", "season", "week"] if c in out.columns]
    if not keep:
        return pd.DataFrame()

    out = out[keep].copy()

    # Drop rows missing team names
    if {"home", "away"}.issubset(out.columns):
        out = out[out["home"].notna() & out["away"].notna()]

    return out.reset_index(drop=True)


# ===============================
# DATA FETCH
# ===============================
def fetch_games_any_schema(year: int) -> pd.DataFrame:
    """Fetch games for the given year and normalize columns safely."""
    url = f"{CFBD}/games"
    params = {"year": year, "seasonType": "regular"}
    print(f"📡 Fetching games for {year} ...")

    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    r.raise_for_status()
    raw = pd.DataFrame(r.json())

    if raw.empty:
        print("⚠️ CFBD returned 0 games.")
        return raw

    print(f"✅ Retrieved {len(raw)} raw rows from CFBD.")
    games = normalize_games_cols(raw)

    print(f"✅ Normalized to {len(games)} rows with columns: {list(games.columns)}")
    return games


# ===============================
# MODEL / PREDICTIONS
# ===============================
def generate_predictions(games: pd.DataFrame) -> pd.DataFrame:
    """Generate placeholder predictions (replace with your model)."""
    if games.empty:
        print("⚠️ No games to predict.")
        return pd.DataFrame()

    if not {"home", "away"}.issubset(games.columns):
        raise ValueError("❌ Normalized games missing 'home'/'away' columns.")

    df = games.copy()

    # Example placeholder model — replace with your real model
    df["model_spread"] = (df["home"].astype(str).apply(hash) % 21) - 10  # ~[-10, +10]
    df["model_total"] = 45 + (df["away"].astype(str).apply(hash) % 21)   # ~[45..65]
    df["confidence"] = 0.5 + (df["model_spread"].abs() / 25.0)

    print(f"✅ Generated predictions for {len(df)} games.")
    return df


# ===============================
# OUTPUT WRITER
# ===============================
def write_predictions(df: pd.DataFrame) -> str:
    """
    Standardize and save predictions.csv
    Required columns: home, away, model_spread, model_total
    """
    if df.empty:
        raise ValueError("❌ No predictions to write — dataframe is empty.")

    # Ensure required columns exist
    required = {"home", "away", "model_spread", "model_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing required columns for output: {missing}")

    out_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} ({len(df)} rows)")
    return out_path


# ===============================
# MAIN
# ===============================
def main():
    year = datetime.utcnow().year
    games = fetch_games_any_schema(year)
    preds = generate_predictions(games)
    write_predictions(preds)


if __name__ == "__main__":
    main()
