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
from src.model import predict_home_win_prob  # model-driven if model exists

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
    return datetime.now(timezone.utc).year

def _resolve_week_auto(year: int) -> int:
    cal = cfbd_get("/calendar", {"year": year}) or []
    if not cal:
        return 1
    now = datetime.now(timezone.utc)
    upcoming: List[Tuple[int, datetime]] = []
    latest_week = 1
    for row in cal:
        wk = int(row.get("week") or row.get("week_number") or latest_week)
        latest_week = max(latest_week, wk)
        start = _parse_dt(row.get("firstGameStart") or row.get("first_game_start"))
        if start and start >= now:
            upcoming.append((wk, start))
    if upcoming:
        upcoming.sort(key=lambda t: t[1])
        return upcoming[0][0]
    return latest_week

def _fetch_games(year: int, week: int) -> List[Dict[str, Any]]:
    return cfbd_get("/games", {"year": year, "week": week, "seasonType": "regular"}) or []

def _is_num(x: Any) -> bool:
    try:
        float(x); return True
    except Exception:
        return False

def _consensus_lines(year: int, week: int) -> Dict[Tuple[str, str], Dict[str, Any]]:
    lines = cfbd_get("/lines", {"year": year, "week": week}) or []
    by_match: Dict[Tuple[str, str], List[Tuple[Optional[float], Optional[float]]]] = {}
    for row in lines:
        home = row.get("homeTeam") or row.get("home_team") or row.get("home")
        away = row.get("awayTeam") or row.get("away_team") or row.get("away")
        if not home or not away:
            continue
        totals, spreads = [], []
        if "lines" in row and isinstance(row["lines"], list) and row["lines"]:
            for l in row["lines"]:
                t = l.get("overUnder") or l.get("total")
                s = l.get("spread")
                if _is_num(t): totals.append(float(t))
                if _is_num(s): spreads.append(float(s))
        else:
            t = row.get("overUnder") or row.get("total")
            s = row.get("spread")
            if _is_num(t): totals.append(float(t))
            if _is_num(s): spreads.append(float(s))
        key = (str(home), str(away))
        by_match.setdefault(key, [])
        if totals or spreads:
            by_match[key].append((
                (sum(spreads)/len(spreads)) if spreads else None,
                (sum(totals)/len(totals)) if totals else None,
            ))
    consensus: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for key, pairs in by_match.items():
        if not pairs: continue
        sp = [p for p, _ in pairs if p is not None]
        to = [t for _, t in pairs if t is not None]
        consensus[key] = {
            "market_spread": (sum(sp)/len(sp)) if sp else None,
            "market_total": (sum(to)/len(to)) if to else None,
            "books": len(pairs),
        }
    return consensus

def build_predictions(year: int, week: int, cfg: Dict[str, Any]) -> List[GameRow]:
    games = _fetch_games(year, week)
    market = _consensus_lines(year, week)
    out: List[GameRow] = []
    for g in games:
        if g.get("season_type") and str(g.get("season_type")).lower() != "regular":
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

        p_home = predict_home_win_prob(g, cfg)
        phs, pas, spread_pred = predict_scores(
            p_home=p_home,
            market_total=market_total,
            sigma=cfg.get("scores", {}).get("sigma_margin", 16.0),
            fallback_total=cfg.get("scores", {}).get("avg_total", 55.0),
        )

        out.append(GameRow(
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
        ))
    out.sort(key=lambda r: r.start_date or "")
    return out

def write_predictions_csv(rows: List[GameRow], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "start_date","season","week",
        "home_team","away_team","neutral_site","venue",
        "p_home","pred_spread","pred_home_score","pred_away_score",
        "market_spread","market_total","books",
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
    ap.add_argument("--week", default="auto", help="ISO week (yyyy-ww) or integer or 'auto'")
    ap.add_argument("--out", required=True, help="path to predictions.csv")
    args = ap.parse_args()

    cfg = load_cfg()
    out_path = pathlib.Path(args.out)

    if args.week == "auto":
        year = _season_now_utc()
        week = _resolve_week_auto(year)
    elif "-" in str(args.week):
        parts = str(args.week).split("-", 1)
        year, week = int(parts[0]), int(parts[1])
    else:
        year = _season_now_utc()
        week = int(args.week)

    rows = build_predictions(year, week, cfg)
    write_predictions_csv(rows, out_path)
    print(json.dumps({"year": year, "week": week, "games": len(rows)}, indent=2))
    print(f"✅ Wrote predictions → {out_path}")

if __name__ == "__main__":
    main()
