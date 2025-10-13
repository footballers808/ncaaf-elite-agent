# src/make_edges.py
from __future__ import annotations
import os, sys, glob, json
from datetime import datetime
import pandas as pd

CAL_PATH = os.path.join("store", "calibration.json")

# minimum columns the learn loop needs downstream
REQUIRED = ["game_id", "home", "away", "model_spread", "model_total"]

# flexible renames for common variations
RENAME_MAP = {
    "id": "game_id",
    "gameId": "game_id",
    "game_id": "game_id",
    "home_team": "home",
    "Home": "home",
    "away_team": "away",
    "Away": "away",
    "pred_spread": "model_spread",
    "spread_pred": "model_spread",
    "modelSpread": "model_spread",
    "pred_total": "model_total",
    "total_pred": "model_total",
    "modelTotal": "model_total",
    "line_spread": "market_spread",
    "spread_line": "market_spread",
    "consensus_spread": "market_spread",
    "line_total": "market_total",
    "total_line": "market_total",
    "consensus_total": "market_total",
}

def _find_source_csv() -> str | None:
    # 1) Prefer a root-level predictions.csv
    if os.path.exists("predictions.csv"):
        return "predictions.csv"
    # 2) Then output/predictions.csv
    if os.path.exists(os.path.join("output", "predictions.csv")):
        return os.path.join("output", "predictions.csv")
    # 3) Otherwise, search for any CSV in output/ that likely contains model columns
    candidates = sorted(glob.glob("output/*.csv"))
    for path in reversed(candidates):
        try:
            head = pd.read_csv(path, nrows=50)
            cols = set([RENAME_MAP.get(c, c) for c in head.columns])
            if {"model_spread", "model_total"}.intersection(cols) and {"home","away","game_id"}.intersection(cols):
                return path
        except Exception:
            continue
    return None

def _rename_and_require(df: pd.DataFrame) -> pd.DataFrame:
    # normalize names
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    # fill missing team/ids if alternate columns exist
    if "home" not in df and "home_team" in df: df = df.rename(columns={"home_team":"home"})
    if "away" not in df and "away_team" in df: df = df.rename(columns={"away_team":"away"})
    # make sure required exist
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns after normalization: {missing}")
    return df

def _apply_calibration(df: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(CAL_PATH):
        return df
    try:
        cal = json.load(open(CAL_PATH, "r", encoding="utf-8"))
        a_s = float(cal.get("spread", {}).get("a", 0.0))
        b_s = float(cal.get("spread", {}).get("b", 1.0))
        a_t = float(cal.get("total", {}).get("a", 0.0))
        b_t = float(cal.get("total", {}).get("b", 1.0))
        df["model_spread"] = a_s + b_s * pd.to_numeric(df["model_spread"], errors="coerce")
        df["model_total"]  = a_t + b_t * pd.to_numeric(df["model_total"], errors="coerce")
    except Exception:
        pass
    return df

def main():
    src_path = _find_source_csv()
    if src_path is None:
        print("❌ No source predictions CSV found. Expected predictions.csv or a CSV in output/ with model_spread/model_total.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(src_path)
    df = _rename_and_require(df)

    # Ensure numeric
    for c in ["model_spread","model_total","market_spread","market_total"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Market columns optional
    if "market_spread" not in df.columns: df["market_spread"] = pd.NA
    if "market_total"  not in df.columns: df["market_total"]  = pd.NA

    # Apply calibration if available
    df = _apply_calibration(df)

    # Edges (if market present these will be numeric; otherwise NA)
    df["edge_spread"] = df["model_spread"] - df["market_spread"]
    df["edge_total"]  = df["model_total"]  - df["market_total"]

    # Final order
    out_cols = ["game_id","home","away","model_spread","model_total",
                "market_spread","market_total","edge_spread","edge_total"]
    out = df[[c for c in out_cols if c in df.columns]].copy()

    os.makedirs("output", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d")
    out_path = f"output/edges_{ts}.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} (rows={len(out)}) from {src_path}")

if __name__ == "__main__":
    main()
