# src/labeler.py
from __future__ import annotations
import os, time
from typing import Any, Dict, List, Optional
import pandas as pd
import requests

CFBD = "https://api.collegefootballdata.com"
API_KEY = os.getenv("CFBD_API_KEY", "")
HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

STORE_DIR = "store"
os.makedirs(STORE_DIR, exist_ok=True)

STORE_DIR = "store"
os.makedirs(STORE_DIR, exist_ok=True)  # make sure it exists

def main(year: Optional[int] = None, week: Optional[int] = None):
    if year is None:
        year = int(time.strftime("%Y"))
    games = fetch_completed_games(year, week)
    labels = build_labels(games)
    if labels.empty:
        print("⚠️ No completed games with scores found — skipping label save.")
        # still write an empty placeholder to avoid FileNotFoundError later
        pd.DataFrame().to_parquet(os.path.join(STORE_DIR, "labels.parquet"))
        return
    path = os.path.join(STORE_DIR, "labels.parquet")
    labels.to_parquet(path, index=False)
    print(f"✅ Wrote labels: {path} (rows={len(labels)})")

def _get(path: str, params: Optional[Dict[str, Any]] = None):
    url = f"{CFBD}{path}"
    r = requests.get(url, params=params or {}, headers=HEADERS, timeout=45)
    r.raise_for_status()
    return r.json()

def fetch_completed_games(year: int, week: Optional[int] = None) -> pd.DataFrame:
    params = {"year": year}
    if week is not None:
        params["week"] = week
    games = _get("/games", params)
    rows: List[Dict[str, Any]] = []
    for g in games:
        if (g.get("home_points") is None) or (g.get("away_points") is None):
            continue
        rows.append({
            "game_id": g.get("id") or g.get("gameId"),
            "season": g.get("season"),
            "week": g.get("week"),
            "start": g.get("start_date") or g.get("startDate"),
            "home": g.get("home_team") or g.get("homeTeam"),
            "away": g.get("away_team") or g.get("awayTeam"),
            "home_points": g.get("home_points"),
            "away_points": g.get("away_points"),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_parquet(os.path.join(STORE_DIR, "games.parquet"), index=False)
    return df

def build_labels(df_games: pd.DataFrame) -> pd.DataFrame:
    if df_games.empty:
        return df_games
    df = df_games.copy()
    df["actual_spread"] = df["home_points"] - df["away_points"]
    df["actual_total"] = df["home_points"] + df["away_points"]
    df["home_win"] = (df["actual_spread"] > 0).astype(int)
    return df[[
        "game_id","season","week","start","home","away",
        "home_points","away_points","actual_spread","actual_total","home_win"
    ]]

def main(year: Optional[int] = None, week: Optional[int] = None):
    if year is None:
        year = int(time.strftime("%Y"))
    games = fetch_completed_games(year, week)
    labels = build_labels(games)
    if labels.empty:
        print("No completed games with scores found.")
        return
    path = os.path.join(STORE_DIR, "labels.parquet")
    labels.to_parquet(path, index=False)
    print(f"Wrote labels: {path} (rows={len(labels)})")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--year", type=int, default=None)
    p.add_argument("--week", type=int, default=None)
    args = p.parse_args()
    main(args.year, args.week)
