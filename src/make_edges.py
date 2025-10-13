# src/make_edges.py
from __future__ import annotations
import os, json, sys
from datetime import datetime
import pandas as pd

CAL_PATH = os.path.join("store", "calibration.json")

REQUIRED = ["game_id", "home", "away", "model_spread", "model_total"]

# Flexible column rename map if your predictions.csv uses slightly different names
RENAME_MAP = {
    # ids/teams
    "id": "game_id",
    "gameId": "game_id",
    "home_team": "home",
    "away_team": "away",
    # model outputs
    "pred_spread": "model_spread",
    "spread_pred": "model_spread",
    "pred_total": "model_total",
    "total_pred": "model_total",
    # market lines
    "line_spread": "market_spread",
    "spread_line": "market_spread",
    "line_total": "market_total",
    "total_line": "market_total",
}

def _apply_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Optionally adjust model_spread/model_total using store/calibration.json."""
    if not os.path.exists(CAL_PATH):
        return df
    try:
        with open(CAL_PATH, "r", encoding="utf-8") as f:
            cal = json.load(f)
        a_s = float(cal.get("spread", {}).get("a", 0.0))
        b_s = float(cal.get("spread", {}).get("b", 1.0))
        a_t = float(cal.get("total",  {}).get("a", 0.0))
        b_t = float(cal.get("total",  {}).get("b", 1.0))
        if "model_spread" in df:
            df["model_spread"] = a_s + b_s * df["model_spread"].astype(float)
        if "model_total" in df:
            df["model_total"] = a_t + b_t * df["model_total"].astype(float)
    except Exception:
        pass
    return df

def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

def main() -> None:
    # Locate your current predictions file
    candidates = ["predictions.csv", os.path.join("output", "predictions.csv")]
    src_path = next((p for p in candidates if os.path.exists(p)), None)
    if src_path is None:
        print("❌ Could not find predictions.csv (looked in ./ and ./output/).", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(src_path)

    # Standardize column names
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    # Ensure required columns exist
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"❌ predictions.csv missing required columns: {missing}", file=sys.stderr)
        print("   Required columns (min):", REQUIRED, file=sys.stderr)
        sys.exit(1)

    # Optional market columns
    if "market_spread" not in df.columns:
        df["market_spread"] = pd.NA
    if "market_total" not in df.columns:
        df["market_total"]  = pd.NA

    # Coerce numbers
    _coerce_numeric(df, ["model_spread", "model_total", "market_spread", "market_total"])

    # Apply calibration if available
    df = _apply_calibration(df)

    # Compute edges if market lines are present
    df["edge_spread"] = df["model_spread"] - df["market_spread"]
    df["edge_total"]  = df["model_total"]  - df["market_total"]

    # Select/Order final columns
    out_cols = [
        "game_id", "home", "away",
        "model_spread", "model_total",
        "market_spread", "market_total",
        "edge_spread", "edge_total",
    ]
    # Include any extra columns you had
    final = df[[c for c in out_cols if c in df.columns]].copy()

    # Save to output/edges_YYYYMMDD.csv
    os.makedirs("output", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d")
    out_path = f"output/edges_{ts}.csv"
    final.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} (rows={len(final)})")

if __name__ == "__main__":
    main()
