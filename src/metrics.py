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

    # Merge predictions with labels on game_id
    df = labels.merge(preds, on="game_id", how="inner")
    if df.empty:
        raise ValueError("No overlap between labels and predictions on game_id")

    # Errors
    df["spread_error"] = df["actual_spread"] - df["model_spread"]
    df["total_error"]  = df["actual_total"]  - df["model_total"]

    mae_spread  = float(df["spread_error"].abs().mean())
    rmse_spread = float((df["spread_error"]**2).mean() ** 0.5)
    mae_total   = float(df["total_error"].abs().mean())
    rmse_total  = float((df["total_error"]**2).mean() ** 0.5)

    # Edge hit-rates by bucket (if these columns exist)
    bucket_stats = []
    if {"edge_spread","edge_total","market_spread","market_total"}.issubset(df.columns):
        df2 = df.copy()
        df2["cover"] = (df2["actual_spread"] > df2["market_spread"]).astype(int)
        df2["over"]  = (df2["actual_total"]  > df2["market_total"]).astype(int)
        df2["edge_spread_abs"] = df2["edge_spread"].abs()
        df2["edge_total_abs"]  = df2["edge_total"].abs()

        buckets = [(0,1),(1,2),(2,3),(3,4),(4,10)]
        for lo, hi in buckets:
            seg = df2[(df2["edge_spread_abs"]>=lo) & (df2["edge_spread_abs"]<hi)]
            if len(seg):
                hit = float(((seg["edge_spread"]>0) == (seg["cover"]==1)).mean())
                bucket_stats.append({"type":"spread","bucket":f"{lo}-{hi}","n":int(len(seg)),"hit":hit})
            segt = df2[(df2["edge_total_abs"]>=lo) & (df2["edge_total_abs"]<hi)]
            if len(segt):
                hit = float(((segt["edge_total"]>0) == (segt["over"]==1)).mean())
                bucket_stats.append({"type":"total","bucket":f"{lo}-{hi}","n":int(len(segt)),"hit":hit})

    report: Dict[str, Any] = {
        "ts": int(time.time()),
        "preds_path": preds_path,
        "n_games": int(len(df)),
        "mae_spread": mae_spread,
        "rmse_spread": rmse_spread,
        "mae_total": mae_total,
        "rmse_total": rmse_total,
        "buckets": bucket_stats,
    }

    os.makedirs(STORE, exist_ok=True)
    with open(OUT_METRICS, "a", encoding="utf-8") as f:
        f.write(json.dumps(report) + "\n")
    with open(os.path.join(STORE, "metrics_latest.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report

if __name__ == "__main__":
    out = evaluate()
    print(json.dumps(out, indent=2))

