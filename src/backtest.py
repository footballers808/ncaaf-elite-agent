from __future__ import annotations

"""
Lightweight NCAAF backtester.

- Pulls games from CFBD using the cached request layer (Option C)
- Computes simple classification metrics on a naive baseline (p=0.5) so the pipeline runs
- Emits JSON metrics to --out (file or directory)
- Designed to be upgraded: plug your real model into `predict_home_win_prob(...)`
"""

import argparse
import json
import math
import pathlib
from typing import Any, Dict, List, Optional, Tuple

# ---- CFBD cache+retry for all requests.get calls ----
from src.requests_cached import install_requests_cache  # type: ignore
install_requests_cache()

# Optional: direct helper if you want to call it explicitly here
from src.net import cfbd_get  # type: ignore

# ---- Config loader ----
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config.yaml"


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def load_cfg() -> Dict[str, Any]:
    if CFG_PATH.exists():
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def ensure_out_path(out_path: pathlib.Path) -> pathlib.Path:
    """
    If --out is a directory (or has no suffix), we create it and write metrics to 'backtest_summary.json'.
    If --out ends with .json, we write directly to that file.
    """
    if out_path.suffix.lower() == ".json":
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return out_path
    out_path.mkdir(parents=True, exist_ok=True)
    return out_path / "backtest_summary.json"


# --------------------------------------------------------------------------------------
# Data access
# --------------------------------------------------------------------------------------
def fetch_games(year: int, season_type: str = "regular") -> List[Dict[str, Any]]:
    """
    Fetch games from CFBD. season_type in {'regular','postseason','both'}
    If 'both', we merge regular+postseason.
    """
    def _one(st: str) -> List[Dict[str, Any]]:
        return cfbd_get("/games", {"year": year, "seasonType": st}) or []

    if season_type == "both":
        games = _one("regular") + _one("postseason")
    else:
        games = _one(season_type)
    # filter to FBS vs FBS if the payload contains subdivisions; otherwise leave as is
    return games


# --------------------------------------------------------------------------------------
# Model hook (replace later with your real pipeline)
# --------------------------------------------------------------------------------------
def predict_home_win_prob(game: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """
    Placeholder model: returns 0.5 for all games.
    Swap this with calls into your real predict code (e.g., importing your model and features).
    """
    # Example of where you'd call your actual model:
    #   features = featurize(game, cfg)
    #   p_home = model.predict_proba(features)
    return 0.5


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------
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
    """
    Placeholder finance metrics so downstream code has fields to read.
    Strategy: bet the home team when p > 0.55 at flat +100 odds (toy).
    """
    rets: List[float] = []
    for p, y in zip(pred_probs, labels):
        if p > 0.55:
            # Win +1, lose -1 (toy)
            rets.append(1.0 if y == 1 else -1.0)
    if not rets:
        return 0.0, 0.0
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / max(1, len(rets) - 1)
    std = math.sqrt(var) if var > 0 else 0.0
    sharpe = mean / std if std > 0 else 0.0
    roi = mean  # toy definition per bet
    return roi, sharpe


# --------------------------------------------------------------------------------------
# Backtest core
# --------------------------------------------------------------------------------------
def run_backtest(years: int, season_type: str, carry_model: bool, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Very simple rolling backtest:
    - Loops last `years` seasons (inclusive of current year if CFBD has data)
    - Predicts each game with `predict_home_win_prob`
    - Computes classification metrics
    - 'carry_model' is accepted for interface parity; noop in this baseline
    """
    from datetime import datetime
    this_year = datetime.utcnow().year
    yrs = list(range(this_year - years + 1, this_year + 1))

    all_probs: List[float] = []
    all_labels: List[int] = []
    by_year: Dict[int, Dict[str, Any]] = {}

    for y in yrs:
        games = fetch_games(y, season_type)
        probs: List[float] = []
        labels: List[int] = []

        for g in games:
            hp = g.get("home_points")
            ap = g.get("away_points")
            if hp is None or ap is None:
                continue  # skip unplayed/canceled games

            label_home_win = 1 if (hp > ap) else 0
            p_home = predict_home_win_prob(g, cfg)
            probs.append(p_home)
            labels.append(label_home_win)

        if probs:
            y_brier = brier_score(probs, labels)
            y_logloss = log_loss(probs, labels)
            y_roi, y_sharpe = simple_roi_and_sharpe(probs, labels)
            by_year[y] = {
                "n_games": len(labels),
                "brier": y_brier,
                "logloss": y_logloss,
                "roi": y_roi,
                "sharpe": y_sharpe,
            }
            all_probs.extend(probs)
            all_labels.extend(labels)

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
        "notes": "Baseline backtest. Replace predict_home_win_prob() with your real model for meaningful metrics.",
    }


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="backtest.py")
    p.add_argument("--years", type=int, required=True, help="How many seasons to backtest (e.g., 5)")
    p.add_argument("--season-type", choices=["regular", "postseason", "both"], default="regular")
    p.add_argument("--carry", action="store_true", help="(noop here) Carry model from year to year")
    p.add_argument("--out", required=True, help="Output .json file or directory")
    return p.parse_args()

def main():
    args = parse_args()
    cfg = load_cfg()

    metrics = run_backtest(
        years=args.years,
        season_type=args.season_type,
        carry_model=bool(args.carry),
        cfg=cfg,
    )

    out_path = ensure_out_path(pathlib.Path(args.out))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print("== Overall ==")
    print(json.dumps(metrics.get("overall", {}), indent=2))
    print("✅ Backtest complete.")

if __name__ == "__main__":
    main()
