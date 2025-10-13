"""
main.py — NCAAF Elite Agent Predictor
-------------------------------------
Runs your college football model, fetches upcoming games, produces predictions,
and writes a standardized predictions.csv for downstream workflows.
"""

import os
import pandas as pd
import requests
from datetime import datetime

# ===============================
# CONFIG
# ===============================
CFBD_API_KEY = os.environ.get("CFBD_API_KEY")
CFBD = "https://api.collegefootballdata.com"

HEADERS = {"Authorization": f"Bearer {CFBD_API_KEY}"} if CFBD_API_KEY else {}

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===============================
# DATA FETCH
# ===============================
def fetch_upcoming_games(year: int):
    """Fetch games that have not yet started (future schedule)."""
    url = f"{CFBD}/games"
    params = {"year": year, "seasonType": "regular"}
    print(f"Fetching games for {year} ...")
    resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    games = pd.DataFrame(resp.json())

    # Keep only future games with teams populated
    if not games.empty:
        games = games[games["start_date"].notna()]
        games = games[["id", "home_team", "away_team", "start_date", "season", "week"]]
        games["start_date"] = pd.to_datetime(games["start_date"], errors="coerce")

    print(f"✅ Retrieved {len(games)} games from CFBD.")
    return games


# ===============================
# MODEL PLACEHOLDER / LOGIC
# ===============================
def generate_predictions(games: pd.DataFrame):
    """
    Generate dummy predictions.
    Replace this logic with your actual model code.
    """
    if games.empty:
        print("⚠️ No games to predict.")
        return pd.DataFrame()

    preds = games.copy()

    # Example dummy logic for now
    preds["model_spread"] = (preds["home_team"].apply(hash) % 20) - 10
    preds["model_total"] = 50 + (preds["away_team"].apply(hash) % 20)
    preds["confidence"] = 0.5 + (abs(preds["model_spread"]) / 20)

    print("✅ Generated predictions for all games.")
    return preds


# ===============================
# OUTPUT WRITER
# ===============================
def write_predictions(df: pd.DataFrame):
    """
    Standardize and save predictions.csv
    Must include columns: home, away, model_spread, model_total
    """
    if df.empty:
        raise ValueError("No predictions to write — dataframe is empty.")

    # Rename columns to match required schema
    df = df.rename(
        columns={
            "home_team": "home",
            "away_team": "away",
            "id": "game_id",
        }
    )

    # Validate required columns
    required = {"home", "away", "model_spread", "model_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for output: {missing}")

    # Write file
    output_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    df.to_csv(output_path, index=False)
    print(f"✅ Wrote {output_path} ({len(df)} rows)")
    return output_path


# ===============================
# MAIN RUNNER
# ===============================
def main():
    year = datetime.now().year
    games = fetch_upcoming_games(year)
    preds = generate_predictions(games)
    write_predictions(preds)


if __name__ == "__main__":
    main()
