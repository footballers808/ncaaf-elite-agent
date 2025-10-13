from __future__ import annotations
import json, pathlib, re
from typing import Dict, Any, List, Tuple, Optional

import pandas as pd

from src.net import cfbd_get  # cached, retrying HTTP client

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts" / "features"
CACHE = ROOT / ".cache" / "penalties"
ART_DIR.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

# ------------------------------- helpers ---------------------------------

def _to_dt(s: Any) -> pd.Timestamp:
    try:
        return pd.to_datetime(s, utc=True)
    except Exception:
        return pd.NaT

def _normalize_team(name: str) -> str:
    # Light normalization to improve joins (you can extend later)
    return (name or "").strip()

def _attempt_games_with_penalties(year: int) -> Optional[pd.DataFrame]:
    """
    Try to fetch penalties per team per game from a 'games' style endpoint
    where home/away penalties might be present (if CFBD adds/changes fields).
    Returns None if fields aren't available.
    """
    g = cfbd_get("/games", {"year": year, "seasonType": "regular"}) or []
    if not g:
        return None
    df = pd.DataFrame(g)
    cols = {"home_team", "away_team", "home_penalties", "away_penalties", "start_date"}
    if cols.issubset(set(df.columns)):
        out = df[["start_date", "home_team", "away_team", "home_penalties", "away_penalties"]].copy()
        out["date"] = out["start_date"].map(_to_dt)
        return out
    return None

def _attempt_games_teams_boxscores(year: int) -> Optional[pd.DataFrame]:
    """
    Fallback: team-game boxscores endpoint (common on CFBD).
    Look for a 'stat' like 'penalties' on each team, per game.
    """
    try:
        rows = cfbd_get("/games/teams", {"year": year, "seasonType": "regular"}) or []
    except Exception:
        rows = []
    if not rows:
        return None

    # This structure commonly contains 'teams' list with 'school' and 'statistics'
    recs: List[Dict[str, Any]] = []
    for r in rows:
        date = _to_dt(r.get("start_date") or r.get("start_time_tbd") or r.get("startTime"))
        # home/away team dicts may vary field names; handle both sides
        for side in r.get("teams", []):
            school = side.get("school") or side.get("team")
            stats = side.get("statistics") or []
            pens = None
            for st in stats:
                key = (st.get("category") or st.get("stat") or "").lower()
                if "penalt" in key:  # matches 'penalties'
                    try:
                        pens = int(re.findall(r"-?\d+", str(st.get("value") or st.get("stat") or ""))[0])
                    except Exception:
                        pens = None
                    break
            if school is None:
                continue
            recs.append({"date": date, "team": _normalize_team(school), "pens": pens})
    if not recs:
        return None
    return pd.DataFrame(recs)

def _attempt_plays_count(year: int) -> Optional[pd.DataFrame]:
    """
    Slowest fallback: count penalties from plays.
    Looks for a boolean/flag or 'Penalty' in text/type. This is generous but robust.
    """
    # Pull plays by year (you can consider paging by week if rate limits bite)
    try:
        plays = cfbd_get("/plays", {"year": year, "seasonType": "regular"}) or []
    except Exception:
        plays = []
    if not plays:
        return None

    df = pd.DataFrame(plays)
    # expected columns often include offense/defense or home/away team name; keep it generic.
    # We'll map penalties using a few heuristics.
    text_cols = [c for c in df.columns if "text" in c.lower() or "desc" in c.lower()]
    has_flag_cols = [c for c in df.columns if "penalt" in c.lower()]
    team_col = None
    for guess in ["offense", "offense_team", "offense_school", "team", "home", "possession"]:
        if guess in df.columns:
            team_col = guess
            break
    date_col = None
    for guess in ["start_date", "game_start", "play_date", "date"]:
        if guess in df.columns:
            date_col = guess
            break

    if team_col is None or date_col is None:
        return None

    def _penalty_row(row) -> int:
        # direct flag
        for c in has_flag_cols:
            v = row.get(c)
            if isinstance(v, (int, bool)) and bool(v):
                return 1
            if isinstance(v, str) and v.strip().lower() in {"true", "t", "yes", "y", "1"}:
                return 1
        # look in text
        for c in text_cols:
            v = str(row.get(c, "")).lower()
            if "penalty" in v:
                return 1
        # also check a 'play_type' semantic
        pt = str(row.get("play_type", "")).lower()
        if "penalt" in pt:
            return 1
        return 0

    df["is_penalty"] = df.apply(_penalty_row, axis=1)
    df["date"] = df[date_col].map(_to_dt)
    df["team"] = df[team_col].map(_normalize_team)
    agg = df.groupby(["team", "date"], as_index=False)["is_penalty"].sum()
    agg.rename(columns={"is_penalty": "pens"}, inplace=True)
    return agg

def _team_game_penalties(year: int) -> pd.DataFrame:
    """
    Unified accessor that returns long table:
        columns: team, date, pens (penalties that team committed in that game)
    """
    # Try fastest -> slowest
    df = _attempt_games_with_penalties(year)
    if df is not None:
        home = df[["date", "home_team", "home_penalties"]].rename(
            columns={"home_team": "team", "home_penalties": "pens"}
        )
        away = df[["date", "away_team", "away_penalties"]].rename(
            columns={"away_team": "team", "away_penalties": "pens"}
        )
        tall = pd.concat([home, away], ignore_index=True)
        tall["team"] = tall["team"].map(_normalize_team)
        return tall.dropna(subset=["team", "date"])

    df = _attempt_games_teams_boxscores(year)
    if df is not None:
        return df.dropna(subset=["team", "date"])

    df = _attempt_plays_count(year)
    if df is not None:
        return df.dropna(subset=["team", "date"])

    # If all fails, return empty
    return pd.DataFrame(columns=["team", "date", "pens"])

# -------------------------- public: rollups -------------------------------

def build_penalty_rollups(
    years: List[int],
    windows: Tuple[int, ...] = (3, 5, 10),
    min_games: int = 2,
    save_parquet: bool = True,
) -> pd.DataFrame:
    """
    Returns a long DataFrame with rolling penalty features *by team* up to each game date.
    Columns:
        team, date, pens_pg_w{W} for W in windows
    """
    frames = []
    for y in years:
        tall = _team_game_penalties(y)
        if tall.empty:
            continue
        tall = tall.sort_values(["team", "date"])
        # penalties per game rolling mean
        for w in windows:
            tall[f"pens_pg_w{w}"] = (
                tall.groupby("team", group_keys=False)["pens"]
                .rolling(w, min_periods=min_games)
                .mean()
                .reset_index(level=0, drop=True)
            )
        frames.append(tall[["team", "date"] + [f"pens_pg_w{w}" for w in windows]])

    if not frames:
        feats = pd.DataFrame(columns=["team", "date"] + [f"pens_pg_w{w}" for w in windows])
    else:
        feats = pd.concat(frames, ignore_index=True).dropna(subset=["team"])

    if save_parquet:
        out = ART_DIR / "penalties.parquet"
        feats.to_parquet(out, index=False)
        print(f"Wrote penalty rollups → {out} (rows={len(feats)})")

    return feats

# --------------------------- CLI convenience -----------------------------

def _years_from_latest(n_years: int) -> List[int]:
    from datetime import datetime
    cur = datetime.utcnow().year
    return list(range(cur - n_years + 1, cur + 1))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=5, help="How many seasons back to build")
    ap.add_argument("--windows", type=str, default="3,5,10", help="CSV of rolling windows")
    ap.add_argument("--min-games", type=int, default=2, help="Min games before rolling mean")
    args = ap.parse_args()

    wins = tuple(int(x) for x in str(args.windows).split(",") if x.strip())
    years = _years_from_latest(args.years)

    build_penalty_rollups(years, windows=wins, min_games=args.min_games, save_parquet=True)
