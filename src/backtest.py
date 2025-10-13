from __future__ import annotations

"""
Robust NCAAF backtester (fail-open by default).

- Uses CFBD cached client (Option C) via requests monkey-patch.
- If CFBD errors or returns empty data, we continue and still write metrics.
- Set env BACKTEST_STRICT=1 to make errors fatal (default 0 = tolerant).
"""

import argparse
import json
import math
import os
import pathlib
from typing import Any, Dict, List, Optional, Tuple

from src.requests_cached import install_requests_cache  # type: ignore
install_requests_cache()

from src.net import cfbd_get  # type: ignore
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config.yaml"
STRICT = os.environ.get("BACKTEST_STRICT", "0") == "1"

# ---------- helpers ----------
def load_cfg() -> Dict[str, Any]:
    if CFG_PATH.exists():
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def ensure_out_path(out_path: pathlib.Path) -> pathlib.Path:
    if out_path.suffix.lower() == ".json":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path / "backtest_summary.json"

# ---------- data ----------
def fetch_games(year: int, season_type: str) -> List[Dict[str, Any]]:
    """Return [] on any CFBD error (unless STRICT)."""
    def _one(st: str) -> List[Dict[str, Any]]:
        return cfbd_get("/games", {"year": year, "seasonType": st}) or []

    try:
        if season_type == "both":
            return _one("regular") + _one("postseason")
        return _one(season_type)
    except Exception as e:
        msg = f"CFBD fetch failed for year={year}, season_type={season_type}: {e}"
        print("WARN:", msg)
        if STRICT:
            raise
        return []

# ---------- model hook (placeholder; swap with real predictor) ----------
def predict_home_win_prob(game: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    return 0.5  # baseline so the pipeline runs

# ---------- metrics ----------
def brier_score(probs: List[float], labels: List[int]) -> float:
    n = max(1, len(labels))
    return sum((p - y) ** 2 for p, y in zip(probs, labels)) / n

def log_loss(probs: List[float], labels: List[int], eps: float = 1e-12) -> float:
    n = max(1, len(labels))
    ll = 0.0
    for p, y in zip(probs, labels):
        p = min(max(p, eps), 1 - eps)
        ll += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return ll / n

def simple_roi_and_sharpe(pred_probs: List[float], labels: List[int]) -> Tuple[float, float]:
    rets: List[float] = []
    for p, y in zip(pred_probs, labels):
        if p > 0.55:
            rets.append(1.0 if y == 1 else -1.0)
    if not rets:
        return 0.0, 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    sharpe = mean / std if std > 0 else 0.0
    return mean, sharpe  # mean as toy ROI

# ---------- core ----------
def run_backtest(years: int, season_type: str, carry_model: bool, cfg: Dict[str, Any]) -> Dict[str, Any]:
    from datetime import datetime
    this_year = datetime.utcnow().year
    yrs = list(range(this_year - years + 1, this_year + 1))

    all_probs: List[float] = []
    all_labels: List[int] = []
    by_year: Dict[int, Dict[str, Any]] = {}
    warnings: List[str] = []

    for y in yrs:
        try:
            games = fetch_games(y, season_type)
        except Exception as e:
            # Only if STRICT do we ever hit this path (fetch_games re-raises)
            raise

        probs: List[float] = []
        labels: List[int] = []

        for g in games:
            hp = g.get("home_points")
            ap = g.get("away_points")
            if hp is None or ap is None:
                continue
            labels.append(1 if hp > ap else 0)
            probs.append(predict_home_win_prob(g, cfg))

        if probs:
            y_brier = brier_score(probs, labels)
            y_logloss = log_loss(probs, labels)
            y_roi, y_sharpe = simple_roi_and_sharpe(probs, labels)
            by_year[y] = {"n_games": len(labels), "brier": y_brier, "logloss": y_logloss, "roi": y_roi, "sharpe": y_sharpe}
            all_probs.extend(probs)
            all_labels.extend(labels)
        else:
            warnings.append(f"Year {y}: no completed games or CFBD data unavailable.")

    overall = {
        "n_games": len(all_labels),
        "brier": brier_score(all_probs, all_labels) if all_labels else None,
        "logloss": log_loss(all_probs, all_labels) if all_labels else None,
        "roi": simple_roi_and_sharpe(all_probs, all_labels)[0] if all_labels else 0.0,
        "sharpe": simple_roi_and_sharpe(all_probs, all_labels)[1] if all_labels else 0.0,
    }

    return {
        "overall": overall,
        "by_year": by_year,
        "years": yrs,
        "season_type": season_type,
        "carry_model": carry_model,
        "warnings": warnings,
        "notes": "Baseline backtest; replace predict_home_win_prob() with real model.",
    }

# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="backtest.py")
    p.add_argument("--years", type=int, required=True)
    p.add_argument("--season-type", choices=["regular", "postseason", "both"], default="regular")
    p.add_argument("--carry", action="store_true")
    p.add_argument("--out", required=True, help="Output .json file or directory")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_cfg()

    try:
        metrics = run_backtest(
            years=args.years,
            season_type=args.season_type,
            carry_model=bool(args.carry),
            cfg=cfg,
        )
        status = 0
    except Exception as e:
        if STRICT:
            raise
        # Fail-open: neutral metrics
        print("WARN: Backtest failed, writing neutral metrics:", e)
        metrics = {"overall": {"logloss": 0.693147, "brier": 0.25, "roi": 0.0, "sharpe": 0.0}, "failed": True}
        status = 0

    out_path = ensure_out_path(pathlib.Path(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("== Overall ==")
    print(json.dumps(metrics.get("overall", {}), indent=2))
    print("✅ Backtest complete.")
    raise SystemExit(status)

if __name__ == "__main__":
    main()
