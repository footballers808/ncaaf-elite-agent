api_key = os.environ.get("CFBD_API_KEY")
headers = {"Authorization": f"Bearer {api_key}"}
r = requests.get(f"{CFBD}/games", params=params, headers=headers, timeout=60)# src/labeler.py
from __future__ import annotations
import os, time, requests, pandas as pd
from typing import Optional

STORE_DIR = "store"
os.makedirs(STORE_DIR, exist_ok=True)

CFBD = "https://api.collegefootballdata.com"

def fetch_completed_games(year: int, week: Optional[int] = None):
    """Fetch games with final scores; requires CFBD_API_KEY in env."""
    api_key = os.environ.get("CFBD_API_KEY")
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        # Fail fast with a clear message (but no secret leak)
        raise RuntimeError("CFBD_API_KEY is not set in environment.")

    params = {"year": year}
    if week is not None:
        params["week"] = week

    r = requests.get(f"{CFBD}/games", params=params, headers=headers, timeout=60)
    r.raise_for_status()

    games = [
        g for g in r.json()
        if g.get("home_points") is not None and g.get("away_points") is not None
    ]
    return games

def build_labels(games: list[dict]) -> pd.DataFrame:
    if not games:
        return pd.DataFrame()
    df = pd.DataFrame(games).rename(columns={"id": "game_id", "home_team":"home", "away_team":"away"})
    df["actual_spread"] = df["home_points"] - df["away_points"]
    df["actual_total"]  = df["home_points"] + df["away_points"]
    return df[["game_id","home","away","actual_spread","actual_total"]]

def main(year: Optional[int] = None, week: Optional[int] = None):
    if year is None:
        year = int(time.strftime("%Y"))

    games = fetch_completed_games(year, week)
    labels = build_labels(games)

    path = os.path.join(STORE_DIR, "labels.parquet")
    if labels.empty:
        print("⚠️ No completed games found; writing placeholder labels.parquet")
        pd.DataFrame(columns=["game_id","home","away","actual_spread","actual_total"]).to_parquet(path, index=False)
    else:
        labels.to_parquet(path, index=False)
        print(f"✅ Wrote labels: {path} (rows={len(labels)})")

if __name__ == "__main__":
    main()

