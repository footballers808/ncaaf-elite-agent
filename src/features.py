import pandas as pd
import numpy as np
import datetime as dt
from typing import Dict, Any, List, Optional
from .common import ART, safe_write_parquet
from . import cfbd_client as cfbd
from .weather import summarize as wx_summary

def _iso_to_year_week(iso: str) -> (int, int):
    y, w = iso.split("-")
    return int(y), int(w)

def _roll_mean(s: pd.Series, w: int):
    return s.rolling(w, min_periods=1).mean()

def _extract_game_rows(year: int, season_type: str) -> pd.DataFrame:
    games = cfbd.games(year, season_type)
    rows = []
    for g in games:
        if g.get("completed") is False: 
            continue
        # Home/away rows for supervised learning
        for side in ("home","away"):
            opp = "away" if side=="home" else "home"
            rows.append({
                "game_id": g["id"],
                "season": g["season"],
                "week": g.get("week"),
                "kickoff": g.get("start_date"),
                "team": g[f"{side}_team"],
                "opponent": g[f"{opp}_team"],
                "neutral_site": g.get("neutral_site"),
                "home_team": g["home_team"],
                "away_team": g["away_team"],
                "points_for": g.get(f"{side}_points"),
                "points_against": g.get(f"{opp}_points"),
                "spread_close": g.get("spread"),    # not always present
                "total_close": g.get("over_under"), # not always present
                "venue_id": g.get("venue_id"),
            })
    return pd.DataFrame(rows)

def _pace_from_stats(year: int, season_type: str) -> pd.DataFrame:
    stats = cfbd.game_stats(year, season_type)
    # Use plays and time-of-possession to derive pace proxies
    rows = []
    for g in stats:
        gid = g["id"]
        for team in g.get("teams", []):
            stat_map = {s["category"]: s["stat"] for s in team.get("stats", [])}
            plays = float(stat_map.get("plays", 0))
            top   = stat_map.get("possessionTime", "00:00")
            mins = sum(x * int(t) for x,t in zip([60,1], top.split(":"))) / 60 if ":" in top else 0.0
            pace = plays / max(1.0, mins) if mins else np.nan
            rows.append({"game_id": gid, "team": team["team"], "pace": pace})
    return pd.DataFrame(rows)

def _injury_counts(year: int, week: Optional[int], window_days: int) -> pd.DataFrame:
    inj = cfbd.injuries(year, week)
    # Count injuries per team over recent window (API already weekly; treat as per-week count)
    rows = []
    for it in inj:
        team = it.get("team")
        cnt  = len(it.get("injuries", []))
        rows.append({"team": team, "injuries_recent": cnt})
    return pd.DataFrame(rows).groupby("team", as_index=False)["injuries_recent"].sum()

def _merge_market(year: int, season_type: str, week: Optional[int]) -> pd.DataFrame:
    ln = cfbd.lines(year, season_type, week)
    rows = []
    for l in ln:
        gid = l.get("id")
        # Pick consensus/close if available; otherwise last line posted
        if not l.get("lines"): 
            continue
        best = l["lines"][-1]
        rows.append({
            "game_id": gid,
            "market_spread": best.get("spread"),
            "market_total": best.get("overUnder")
        })
    return pd.DataFrame(rows).drop_duplicates("game_id")

def _add_weather(df: pd.DataFrame, hours_before: int) -> pd.DataFrame:
    # Try attach venue lat/lon then call Open-Meteo
    venues = {v["id"]: v for v in cfbd.venues()}
    def wx(row):
        import dateutil.parser as dp
        v = venues.get(row.get("venue_id") or -1)
        if not v: return pd.Series({"wx_temp": np.nan, "wx_wind": np.nan, "wx_precip": np.nan})
        lat, lon = v.get("latitude"), v.get("longitude")
        if lat is None or lon is None or not row.get("kickoff"): 
            return pd.Series({"wx_temp": np.nan, "wx_wind": np.nan, "wx_precip": np.nan})
        kickoff = dp.isoparse(row["kickoff"]).astimezone(dt.timezone.utc)
        s = wx_summary(lat, lon, kickoff, hours_before)
        if not s: return pd.Series({"wx_temp": np.nan, "wx_wind": np.nan, "wx_precip": np.nan})
        return pd.Series({"wx_temp": s["temp"], "wx_wind": s["wind"], "wx_precip": s["precip"]})
    wx_cols = df.apply(wx, axis=1)
    return pd.concat([df, wx_cols], axis=1)

def build_features(years: List[int], season_type: str, cfg: Dict[str, Any]):
    frames = []
    for y in years:
        base = _extract_game_rows(y, season_type)
        if base.empty: 
            continue
        pace = _pace_from_stats(y, season_type)
        mkt  = _merge_market(y, season_type, week=None)
        inj  = _injury_counts(y, week=None, window_days=cfg["injury_window_days"])
        df = base.merge(pace, on=["game_id","team"], how="left") \
                 .merge(mkt, on="game_id", how="left") \
                 .merge(inj, on="team", how="left")
        # Rolling recent form per team by season
        df = df.sort_values(["team","season","week"])
        df["pf_mean"] = df.groupby(["team","season"])["points_for"].apply(lambda s: _roll_mean(s.shift(1), cfg["form_games"]))
        df["pa_mean"] = df.groupby(["team","season"])["points_against"].apply(lambda s: _roll_mean(s.shift(1), cfg["form_games"]))
        df["pace_mean"] = df.groupby(["team","season"])["pace"].apply(lambda s: _roll_mean(s.shift(1), cfg["pace_games"]))
        # Weather
        if cfg.get("weather_enabled", True):
            df = _add_weather(df, hours_before=cfg.get("weather_hours_before", 3))
        frames.append(df)
    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path = ART / "features" / "penalties.parquet"  # keep your expected filename
    safe_write_parquet(full, path)
    return path
