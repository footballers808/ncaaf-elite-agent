# src/metrics.py
from __future__ import annotations
import os, glob, json, time
from typing import Dict, Any
import pandas as pd

STORE = "store"
OUT_METRICS = os.path.join(STORE, "metrics.jsonl")

def _latest_edges(path_glob: str = "output/edges_*.csv") -> str | None:
    files = sorted(glob.glob(path_glob))
    return files[-1] if files else None

def evaluate(labels_path: str = os.path.join(STORE, "labels.parquet"),
             preds_path: str | None = None) -> Dict[str, Any]:
    if not os.path.exists(labels_path):
        raise FileNotFoundError(labels_path)
    labels = pd.read_parquet(labels_path)

    if preds_path is None:
        preds_path = _latest_edges()
    if preds_path is None or not os.path.exists(preds_path):
        raise FileNotFoundError("No predictions CSV found in output/edges_*.csv")

    preds = pd.read_csv(preds_path)

    needed = {"game_id", "model_spread", "model_total"}
    missing = needed - set(preds.columns)
    if missing:
        raise ValueError(f"Predictions missing columns: {missing}")

    df = labels.merge(preds, on="game_id", how="inner")
    if df.empty:
        raise ValueError("No overlap between labels and predictions on game_id")

    df["spread_error"] = df["actual_spread"] - df["model_spread"]
    df["total_error"]  = df["actual_total"] - df["model_total"]

mae_spread  = float(df["spread_error"].abs().mean())
rmse_spread = float((df["spread_error"]**2).mean() ** 0.5)
mae_total   = float(df["total_error"].abs().mean())
rmse_total  = float((df["total_error"]**2).mean() ** 0.5)

