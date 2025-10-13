# src/make_edges.py
from __future__ import annotations

import glob
import os
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd


def _pick(cols: Iterable[str], choices: Iterable[str]) -> Optional[str]:
    """Return first matching column (case-insensitive) from `choices` present in `cols`."""
    lower = {c.lower(): c for c in cols}
    for c in choices:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def _load_predictions() -> pd.DataFrame:
    """
    Load predictions.csv from repo root or ./output/.
    """
    candidates = []
    if os.path.exists("predictions.csv"):
        candidates.append("predictions.csv")
    candidates += sorted(glob.glob("output/predictions.csv"))
    if not candidates:
        raise FileNotFoundError("Could not find predictions.csv (looked in ./ and ./output/).")

    path = candidates[0]
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"{path} is empty.")
    return df


def _normalize_predictions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize to columns:
      game_id, home, away, model_spread, model_total
    """
    cols = list(df.columns)

    id_col = _pick(cols, ["game_id", "id", "gameId"])
    home_col = _pick(cols, ["home", "home_team", "Home", "homeName"])
    away_col = _pick(cols, ["away", "away_team", "Away", "awayName"])
    spread_col = _pick(cols, ["model_spread", "pred_spread", "spread_pred", "modelSpread"])
    total_col = _pick(cols, ["model_total", "pred_total", "total_pred", "modelTotal"])

    missing = []
    if not id_col:
        missing.append("game_id (or id)")
    if not home_col:
        missing.append("home/home_team")
    if not away_col:
        missing.append("away/away_team")
    if not spread_col:
        missing.append("model_spread (or pred_spread)")
    if not total_col:
        missing.append("model_total (or pred_total)")

    if missing:
        raise ValueError(f"Predictions file missing required columns: {missing}")

    out = pd.DataFrame(
        {
            "game_id": df[id_col],
            "home": df[home_col],
            "away": df[away_col],
            "model_spread": df[spread_col],
            "model_total": df[total_col],
        }
    )

    # Ensure consistent dtypes
    # CFBD ids are ints; cast safely, dropping rows without ids
    out = out[out["game_id"].notna()].copy()
    out["game_id"] = out["game_id"].astype("int64", errors="ignore")
    # if astype doesn't work (pandas < 2.0), fallback:
    try:
        out["game_id"] = out["game_id"].astype(int)
    except Exception:
        pass

    # Drop duplicates on (game_id)
    out = out.drop_duplicates(subset=["game_id"]).reset_index(drop=True)
    return out


def main():
    df = _load_predictions()
    edges = _normalize_predictions(df)

    if edges.empty:
        raise ValueError("No usable rows after normalization.")

    os.makedirs("output", exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d")
    out_path = os.path.join("output", f"edges_{stamp}.csv")
    edges.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} (rows={len(edges)}) with columns {list(edges.columns)}")


if __name__ == "__main__":
    main()
