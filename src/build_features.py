from __future__ import annotations
import pandas as pd, numpy as np
from typing import List, Dict
from . import cfbd_api as api
from .common import save_parquet

# Helper to flatten lines into per-game row
def _lines_to_df(rows: List[Dict]) -> pd.DataFrame:
    out = []
    for g in rows:
        gid = g.get("id")
        home = g.get("homeTeam")
        away = g.get("awayTeam")
        # pick a consensus-ish provider if available; else last
        provs = g.get("lines") or []
        pick = None
        # try to prefer "consensus" or "Vegas" style names
        for cand in ["CONSENSUS","consensus","Vegas","Caesars","DraftKings","FanDuel","Pinnacle"]:
            pick = next((l for l in provs if (l.get("provider") or "").lower()==cand.lower()), None) or pick
        if not pick and provs:
            pick = provs[-1]
        spread = total = None
        if pick:
            spread = pick.get("spread")
            total  = pick.get("overUnder")
        out.append({"game_id": gid, "market_spread": spread, "market_total": total, "home": home, "away": away})
    return pd.DataFrame(out)

def _games_to_df(rows: List[Dict]) -> pd.DataFrame:
    # Includes basic weather if present
    out = []
    for r in rows:
        out.append({
            "game_id": r.get("id"),
            "season": r.get("season"),
            "week": r.get("week"),
            "season_type": r.get("season_type") or r.get("seasonType"),
            "home": r.get("home_team") or r.get("homeTeam"),
            "away": r.get("away_team") or r.get("awayTeam"),
            "home_points": r.get("home_points"),
            "away_points": r.get("away_points"),
            "venue": r.get("venue"),
            "weather": r.get("weather"),
            "temp": r.get("temperature"),
            "wind": r.get("windSpeed"),
            "precip": r.get("precipitation")
        })
    return pd.DataFrame(out)

def _team_stats_to_df(rows: List[Dict]) -> pd.DataFrame:
    # Convert team-statistics endpoint to simple team-game features
    # We’ll pull points, yards per play and plays where available
    out = []
    for r in rows:
        gid = r.get("gameId") or r.get("game_id")
        week = r.get("week")
        season = r.get("season")
        for side in r.get("teams", []):
            team = side.get("team")
            stats = {s.get("category"): s.get("stat") for s in side.get("statistics", [])}
            pts = float(stats.get("points", np.nan))
            plays = float(stats.get("plays", np.nan))
            ypp = float(stats.get("yardsPerPlay", np.nan))
            out.append({"game_id": gid, "team": team, "season": season, "week": week,
                        "pts": pts, "plays": plays, "ypp": ypp})
    return pd.DataFrame(out)

def _recent_form(team_games: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    # rolling means over last n games (exclude current)
    team_games = team_games.sort_values(["team", "season", "week"])
    for col in ["pts", "plays", "ypp"]:
        team_games[f"rf_{col}"] = (team_games.groupby("team")[col]
                                   .apply(lambda s: s.shift(1).rolling(n, min_periods=1).mean())
                                   .values)
    return team_games

def _pivot_pair(games_df: pd.DataFrame, team_df: pd.DataFrame) -> pd.DataFrame:
    # Join team-level recent form to game rows as (home_*, away_*)
    home_df = team_df.rename(columns={c: f"home_{c}" for c in team_df.columns if c not in ["game_id","team","season","week"]})
    home_df = home_df.rename(columns={"team":"home"})
    away_df = team_df.rename(columns={c: f"away_{c}" for c in team_df.columns if c not in ["game_id","team","season","week"]})
    away_df = away_df.rename(columns={"team":"away"})
    out = games_df.merge(home_df, on=["game_id","home","season","week"], how="left") \
                  .merge(away_df, on=["game_id","away","season","week"], how="left")
    return out

def _injury_signal(rows: List[Dict]) -> pd.DataFrame:
    # best-effort: count "Out" / "Doubtful" per team per week
    out = []
    for r in rows:
        team = r.get("team")
        week = r.get("week")
        season = r.get("season")
        players = r.get("injuries") or []
        cnt = 0
        for p in players:
            status = (p.get("status") or "").lower()
            if any(k in status for k in ["out","doubt","question"]):
                cnt += 1
        out.append({"team": team, "season": season, "week": week, "injury_count": cnt})
    return pd.DataFrame(out)

def build_features(years_back: int = 3, season_type: str = "regular") -> pd.DataFrame:
    """
    Build a single features parquet: artifacts/features/features.parquet
    """
    frames = []
    import datetime as dt
    this_year = dt.date.today().year
    years = list(range(this_year - years_back + 1, this_year + 1))
    for y in years:
        g = _games_to_df(api.games(y, season_type))
        if g.empty: 
            continue
        l = _lines_to_df(api.lines(y, season_type=season_type))
        ts = _team_stats_to_df(api.team_game_stats(y, season_type=season_type))
        if ts.empty:
            # create neutral placeholders so model still trains
            # derive team rows from games
            tmph = g[["game_id","season","week","home"]].rename(columns={"home":"team"})
            tmpa = g[["game_id","season","week","away"]].rename(columns={"away":"team"})
            ts = pd.concat([tmph,tmpa]).assign(pts=np.nan, plays=np.nan, ypp=np.nan)

        ts = _recent_form(ts, n=3)
        inj = _injury_signal(api.injuries(y)) if hasattr(api, "injuries") else pd.DataFrame()
        if inj.empty:
            inj = pd.DataFrame(columns=["team","season","week","injury_count"])

        # pair features
        pair = _pivot_pair(g, ts)
        # market join
        pair = pair.merge(l[["game_id","market_spread","market_total"]], on="game_id", how="left")

        # attach injury counts to both sides
        pair = pair.merge(inj.rename(columns={"team":"home"})[["home","season","week","injury_count"]]
                          .rename(columns={"injury_count":"home_injury_count"}),
                          on=["home","season","week"], how="left")
        pair = pair.merge(inj.rename(columns={"team":"away"})[["away","season","week","injury_count"]]
                          .rename(columns={"injury_count":"away_injury_count"}),
                          on=["away","season","week"], how="left")

        # pace proxy: recent plays per game
        pair["home_pace"] = pair["home_rf_plays"]
        pair["away_pace"] = pair["away_rf_plays"]

        # weather cleanups
        for c in ["temp","wind","precip"]:
            if c in pair:
                pair[c] = pd.to_numeric(pair[c], errors="coerce")

        frames.append(pair)

    feats = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    # basic hygiene
    for col in feats.columns:
        if feats[col].dtype == "O":
            feats[col] = feats[col]
    # save
    save_parquet(feats, "artifacts/features/features.parquet")
    return feats

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=3)
    p.add_argument("--season-type", default="regular")
    args = p.parse_args()
    build_features(args.years, args.season_type)
