# src/predict.py
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import yaml
from src.net import cfbd_get
from src.score_predict import predict_scores

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
CFG_PATH = ROOT / "config.yaml"

@dataclass
class GameRow:
    start_date: str
    season: int
    week: int
    home_team: str
    away_team: str
    neutral_site: Optional[bool]
    venue: Optional[str]
    p_home: float
    pred_spread: float
    pred_home_score: int
    pred_away_score: int
    market_spread: Optional[float]
    market_total: Optional[float]
    books: int

# ---------------------------- helpers ----------------------------

def load_cfg() -> Dict[str, Any]:
    if CFG_PATH.exists():
        with open(CFG_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}

def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z").astimezone(timezone.utc)
        except Exception:
            return None

def _season_now_utc() -> int:
    now = datetime.now(timezone.utc)
    # CFBD season is calendar year; during bowls in Jan it's still previous season for some endpoints,
    # but for predictions we typically use current year.
    return now.year

def _resolve_week_auto(year: int) -> int:
    """
    Try to pick the next upcoming regular-season week using CFBD calendar.
    Fallback: last known week if all games are in the past.
    """
    cal = cfbd_get("/calendar", {"year": year}) or []
    if not cal:
        return 1
    now = datetime.now(timezone.utc)
    upcoming: List[Tuple[int, datetime]] = []
    latest_week = 1
    for row in cal:
        wk = int(row.get("week") or row.get("week_number") or latest_week)
        latest_week = max(latest_week, wk)
        start = _parse_dt(row.get("firstGameStart")) or _parse_dt(row.get("first_game_start"))
        if start and start >= now:
            upcoming.append((wk, start))
    if upcoming:
        upcoming.sort(key=lambda t: t[1])
        return upcoming[0][0]
    return latest_week

def _fetch_games(year: int, week: int) -> List[Dict[str, Any]]:
    g = cfbd_get("/games", {"year": year, "week": week, "seasonType": "regular"}) or []
    # ensure consistent keys
    return g

def _consensus_lines(year: int, week: int) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Build consensus market spread/total per (home, away).
    CFBD /lines returns per-book quotes; we average across books.
    """
    lines = cfbd_get("/lines", {"year": year, "week": week}) or []
    by_match: Dict[Tuple[str, str], List[Tuple[Optional[float], Optional[float]]]] = {}
    for row in lines:
        home = row.get("homeTeam") or row.get("home_team") or row.get("home")
        away = row.get("awayTeam") or row.get("away_team") or row.get("away")
        if not home or not away:
            continue
        totals = []
        spreads = []
        # Row may contain 'lines' list (per book) or direct 'spread'/'total' fields
        if "lines" in row and isinstance(row["lines"], list) and row["lines"]:
            for l in row["lines"]:
                total = l.get("overUnder") or l.get("total")
                spread = l.get("spread")
                totals.append(float(total)) if _is_num(total) else None
                spreads.append(float(spread)) if _is_num(spread) else None
        else:
            total = row.get("overUnder") or row.get("total")
            spread = row.get("spread")
            if _is_num(total): totals.append(float(total))
            if _is_num(spread): spreads.append(float(spread))

        key = (str(home), str(away))
        by_match.setdefault(key, [])
        if totals or spreads:
            # Aggregate once per book; store pair (spread, total) even if one is None
            mean_total = sum(totals) / len(totals) if totals else None
            mean_spread = sum(spreads) / len(spreads) if spreads else None
            by_match[key].append((mean_spread, mean_total))

    # Reduce to consensus
    consensus: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, pairs in by_match.items():
        if not pairs:
            continue
        sp = [p for p, _ in pairs if p is not None]
        to = [t for _, t in pairs if t is not None]
        consensus[key] = {
            "market_spread": (sum(sp) / len(sp)) if sp else None,
            "market_total": (sum(to) / len(to)) if to else None,
            "books": len(pairs),
        }
    return consensus

def _is_num(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False

# -------------------------- model hook --------------------------

def _predict_home_win_prob(game: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """
    Replace this with your real model. For now, return 0.5 to keep the pipeline flowing.
    If you have a trained model, load it here (e.g., from artifacts/model.bin) and
    compute the probability using your features.
    """
    return 0.5

# ----------------------------- main -----------------------------

def build_predictions(year: int, week: int, cfg: Dict[str, Any]) -> List[GameRow]:
    games = _fetch_games(year, week)
    market = _consensus_lines(year, week)

    out: List[GameRow] = []
    for g in games:
        if g.get("season_type") and str(g.get("season_type")).lower() != "regular":
            # Just in case; we specifically fetched regular but keep guard
            continue

        start_raw = g.get("start_date") or g.get("startTime") or g.get("start_time") or ""
        start = _parse_dt(start_raw)
        start_iso = start.isoformat().replace("+00:00", "Z") if start else (start_raw or "")
        home = str(g.get("home_team") or g.get("homeTeam") or "")
        away = str(g.get("away_team") or g.get("awayTeam") or "")
        if not home or not away:
            continue

        key = (home, away)
        m = market.get(key, {})
        market_spread = m.get("market_spread")
        market_total = m.get("market_total")
        books = int(m.get("books", 0))

        p_home = _predict_home_win_prob(g, cfg)
        phs, pas, spread_pred = predict_scores(
            p_home=p_home,
            market_total=market_total,
            sigma=cfg.get("scores", {}).get("sigma_margin", 16.0),
            fallback_total=cfg.get("scores", {}).get("avg_total", 55.0),
        )

        row = GameRow(
            start_date=start_iso,
            season=int(g.get("season") or year),
            week=int(g.get("week") or week),
            home_team=home,
            away_team=away,
            neutral_site=bool(g.get("neutral_site")) if g.get("neutral_site") is not None else None,
            venue=g.get("venue") or g.get("venue_id") or None,
            p_home=float(p_home),
            pred_spread=float(spread_pred),
            pred_home_score=int(phs),
            pred_away_score=int(pas),
            market_spread=float(market_spread) if market_spread is not None else None,
            market_total=float(market_total) if market_total is not None else None,
            books=books,
        )
        out.append(row)

    # sort by start time if possible
    out.sort(key=lambda r: r.start_date or "")
    return out

def write_predictions_csv(rows: List[GameRow], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "start_date", "season", "week",
        "home_team", "away_team", "neutral_site", "venue",
        "p_home", "pred_spread", "pred_home_score", "pred_away_score",
        "market_spread", "market_total", "books",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({
                "start_date": r.start_date,
                "season": r.season,
                "week": r.week,
                "home_team": r.home_team,
                "away_team": r.away_team,
                "neutral_site": r.neutral_site,
                "venue": r.venue,
                "p_home": f"{r.p_home:.6f}",
                "pred_spread": f"{r.pred_spread:.2f}",
                "pred_home_score": r.pred_home_score,
                "pred_away_score": r.pred_away_score,
                "market_spread": f"{r.market_spread:.2f}" if r.market_spread is not None else "",
                "market_total": f"{r.market_total:.1f}" if r.market_total is not None else "",
                "books": r.books,
            })

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--week", default="auto", help="ISO week (yyyy-ww) or integer (e.g., '7') or 'auto'")
    ap.add_argument("--out", required=True, help="Output CSV path, e.g., artifacts/predictions.csv")
    args = ap.parse_args()

    cfg = load_cfg()
    out_path = pathlib.Path(args.out)

    # Resolve year & week
    # Accept formats: "auto", "7", "2025-07"
    if args.week == "auto":
        year = _season_now_utc()
        week = _resolve_week_auto(year)
    elif "-" in str(args.week):
        parts = str(args.week).split("-", 1)
        year = int(parts[0])
        week = int(parts[1])
    else:
        year = _season_now_utc()
        week = int(args.week)

    rows = build_predictions(year, week, cfg)
    write_predictions_csv(rows, out_path)

    # Print a tiny JSON summary for logs
    print(json.dumps({"year": year, "week": week, "games": len(rows)}, indent=2))
    print(f"✅ Wrote predictions → {out_path}")

if __name__ == "__main__":
    main()
