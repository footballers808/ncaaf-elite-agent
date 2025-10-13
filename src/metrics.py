# src/metrics.py
from __future__ import annotations

import glob
import json
import os
from datetime import datetime
from typing import Iterable, Optional

import numpy as np
import pandas as pd


STORE_DIR = "store"
OUTPUT_DIR = "output"


def _pick(cols: Iterable[str], choices: Iterable[str]) -> Optional[str]:
    lower = {c.lower(): c for c in cols}
    for c in choices:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _latest_edges_path() -> str:
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "edges_*.csv")))
    if not files:
        raise FileNotFoundError("No edges CSV found in output/edges_*.csv. "
                                "Make sure Predict workflow uploaded predictions_bundle.")
    return files[-1]


def _load_edges() -> pd.DataFrame:
    p = _latest_edges_path()
    df = pd.read_csv(p)

    # Normalize column names
    cols = list(df.columns)
    id_col = _pick(cols, ["game_id", "id", "gameId"])
    home_col = _pick(cols, ["home", "home_team"])
    away_col = _pick(cols, ["away", "away_team"])
    sp_col = _pick(cols, ["model_spread", "pred_spread", "spread_pred", "modelSpread"])
    tot_col = _pick(cols, ["model_total", "pred_total", "total_pred", "modelTotal"])

    missing = []
    if not id_col: missing.append("game_id (or id)")
    if not home_col: missing.append("home/home_team")
    if not away_col: missing.append("away/away_team")
    if not sp_col: missing.append("model_spread/pred_spread")
    if not tot_col: missing.append("model_total/pred_total")
    if missing:
        raise ValueError(f"Edges file {p} missing columns: {missing}")

    out = pd.DataFrame({
        "game_id": df[id_col],
        "home": df[home_col],
        "away": df[away_col],
        "model_spread": df[sp_col],
        "model_total": df[tot_col],
    })

    # Coerce game_id to int safely
    out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce")
    out = out.dropna(subset=["game_id"]).copy()
    out["game_id"] = out["game_id"].astype("int64")

    # Dedupe in case predictor produced duplicates
    out = out.drop_duplicates(subset=["game_id"]).reset_index(drop=True)

    print(f"Loaded edges: {len(out)} rows from {p}")
    return out


def _load_labels() -> pd.DataFrame:
    # try store/labels.parquet first, fall back to any store/*.parquet
    candidates = []
    lp = os.path.join(STORE_DIR, "labels.parquet")
    if os.path.exists(lp):
        candidates.append(lp)
    candidates += sorted(glob.glob(os.path.join(STORE_DIR, "*.parquet")))

    if not candidates:
        raise FileNotFoundError("No labels parquet found in store/. "
                                "Run the 'Label completed games' step first.")

    p = candidates[0]
    df = pd.read_parquet(p)

    cols = list(df.columns)
    id_col = _pick(cols, ["game_id", "id", "gameId"])
    home_col = _pick(cols, ["home", "home_team"])
    away_col = _pick(cols, ["away", "away_team"])
    hp_col = _pick(cols, ["home_points", "home_score"])
    ap_col = _pick(cols, ["away_points", "away_score"])

    missing = []
    if not id_col: missing.append("game_id (or id)")
    if not home_col: missing.append("home/home_team")
    if not away_col: missing.append("away/away_team")
    if not hp_col: missing.append("home_points/home_score")
    if not ap_col: missing.append("away_points/away_score")
    if missing:
        raise ValueError(f"Labels parquet {p} missing columns: {missing}")

    out = pd.DataFrame({
        "game_id": df[id_col],
        "home": df[home_col],
        "away": df[away_col],
        "home_points": df[hp_col],
        "away_points": df[ap_col],
    })

    # Completed games only (must have points)
    out = out[out["home_points"].notna() & out["away_points"].notna()].copy()

    # Normalize types
    out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce")
    out = out.dropna(subset=["game_id"]).copy()
    out["game_id"] = out["game_id"].astype("int64")

    # Compute actuals
    out["actual_spread"] = out["home_points"] - out["away_points"]
    out["actual_total"] = out["home_points"] + out["away_points"]

    out = out.drop_duplicates(subset=["game_id"]).reset_index(drop=True)

    print(f"Loaded labels: {len(out)} completed games from {p}")
    return out


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"mae": mae, "rmse": rmse, "n": int(len(y_true))}


def evaluate():
    os.makedirs(STORE_DIR, exist_ok=True)

    edges = _load_edges()
    labels = _load_labels()

    # Merge by integer game_id
    merged = pd.merge(labels, edges, on="game_id", how="inner", suffixes=("_label", "_model"))

    print(f"Merged rows on game_id: {len(merged)}")
    if len(merged) == 0:
        # Write diagnostics & a non-failing metrics file so the loop keeps running
        diag = {
            "message": "No overlap between labels and predictions on game_id",
            "labels_n": int(len(labels)),
            "edges_n": int(len(edges)),
            "labels_id_sample": labels["game_id"].astype(int).head(10).tolist() if len(labels) else [],
            "edges_id_sample": edges["game_id"].astype(int).head(10).tolist() if len(edges) else [],
        }
        stamp = datetime.utcnow().strftime("%Y%m%d")
        latest_path = os.path.join(STORE_DIR, "metrics_latest.json")
        dated_path = os.path.join(STORE_DIR, f"metrics_{stamp}.json")
        for p in (latest_path, dated_path):
            with open(p, "w") as f:
                json.dump({"overlap": 0, "diagnostics": diag}, f, indent=2)
        print("⚠️ No overlap; wrote diagnostics to store/metrics_latest.json")
        return

    # Compute metrics
    m_spread = _metrics(
        y_true=merged["actual_spread"].to_numpy(float),
        y_pred=merged["model_spread"].to_numpy(float),
    )
    m_total = _metrics(
        y_true=merged["actual_total"].to_numpy(float),
        y_pred=merged["model_total"].to_numpy(float),
    )

    results = {
        "overlap": int(len(merged)),
        "spread": m_spread,
        "total": m_total,
        "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds"),
    }

    stamp = datetime.utcnow().strftime("%Y%m%d")
    latest_path = os.path.join(STORE_DIR, "metrics_latest.json")
    dated_path = os.path.join(STORE_DIR, f"metrics_{stamp}.json")
    for p in (latest_path, dated_path):
        with open(p, "w") as f:
            json.dump(results, f, indent=2)

    print(f"✅ Wrote {latest_path} and {dated_path}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    evaluate()
