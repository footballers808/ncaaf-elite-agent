"""
main.py — NCAAF Elite Agent Predictor
-------------------------------------
Fetches current or upcoming games, generates model predictions,
and writes standardized predictions.csv for downstream use.
"""

import os
import pandas as pd
import requests
from datetime import datetime

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
# DATA FETCH
# ===============================
def fetch_upcoming_games(year: int) -> pd.DataFrame:
    """Fetch games for the given year — handles any CFBD schema safely."""
    url = f"{CFBD}/games"
    params = {"year": year, "seasonType": "regular"}
    print(f"📡 Fetching games for {year} ...")

    resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    games = pd.DataFrame(resp.json())

    if games.empty:
        print("⚠️ No games returned from CFBD API.")
        return games

    # Normalize possible schema differences
    games.columns = [c.lower() for c in games.columns]

    # Some seasons use 'start_date', others use 'start_time_tbd' or similar
    for alt in ["start_date", "start_time", "start_time_tbd", "game_date"]:
        if alt in games.columns:
            games["start_date"] = pd.to_datetime(games[alt], errors="coerce")
            break
    else:
        games["start_date"] = pd.NaT

    # Keep safe subset of columns (if they exist)
    keep = [c for c in ["id", "home_team", "away_team", "start_date", "season", "week"] if c in games.columns]
    games = games[keep].copy()

    # Filter out entries with missing teams
    if "home_team" in games.columns and "away_team" in games.columns:
        games = games[games["home_team"].notna() & games["away_team"].notna()]

    print(f"✅ Retrieved {len(games)} valid games from CFBD.")
    return games


# ===============================
# MODEL / PREDICTIONS
# ===============================
def generate_predictions(games: pd.DataFrame) -> pd.DataFrame:
    """Generate mock predictions (replace with your model later)."""
    if games.empty:
        print("⚠️ No games to predict.")
        return pd.DataFrame()

    df = games.copy()
    df["model_spread"] = (df["home_team"].apply(hash) % 20) - 10
    df["model_total"] = 45 + (df["away_team"].apply(hash) % 20)
    df["confidence"] = 0.5 + (abs(df["model_spread"]) / 25)
    print(f"✅ Generated predictions for {len(df)} games.")
    return df


# ===============================
# OUTPUT WRITER
# ===============================
def write_predictions(df: pd.DataFrame):
    """Standardize columns and write predictions.csv."""
    if df.empty:
        raise ValueError("❌ No predictions to write — dataframe is empty.")

    df = df.rename(columns={"id": "game_id", "home_team": "home", "away_team": "away"})

    required = {"home", "away", "model_spread", "model_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"❌ Missing required columns for output: {missing}")

    output_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Wrote {output_path} ({len(df)} rows)")
    return output_path


# ===============================
# MAIN
# ===============================
def main():
    year = datetime.now().year
    games = fetch_upcoming_games(year)
    preds = generate_predictions(games)
    write_predictions(preds)


if __name__ == "__main__":
    main()
