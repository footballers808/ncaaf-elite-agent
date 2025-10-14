# src/feat_penalties.py
from __future__ import annotations

import pathlib
import subprocess
import sys
from typing import Any, Dict, Iterable, Tuple

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FEAT_DIR = ART / "features"
FEAT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET = FEAT_DIR / "penalties.parquet"


def _ensure_built(years: int = 5, windows: Iterable[int] = (3, 5, 10)) -> None:
    """
    Ensure penalties parquet exists. If missing, build it once by invoking data builder.
    """
    if PARQUET.exists():
        return

    win_str = ",".join(str(w) for w in windows)
    cmd = [sys.executable, "-m", "src.data_penalties", "--years", str(years), "--windows", win_str]
    print(f"[feat_penalties] penalties.parquet missing → building with: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if not PARQUET.exists():
        raise FileNotFoundError(
            f"Failed to create {PARQUET}. Please run: python -m src.data_penalties --years {years} --windows {win_str}"
        )


def load_penalty_features(years: int = 5, windows: Iterable[int] = (3, 5, 10)) -> pd.DataFrame:
    """
    Loads (and if missing, builds) penalty features parquet.
    Columns are expected to include:
      - season, week, start_dt
      - home_team, away_team
      - rolling metrics per team (e.g., home_pen_3, home_pen_5, ...; away_pen_3, ...)
    """
    _ensure_built(years=years, windows=windows)
    df = pd.read_parquet(PARQUET)
    # Normalize team name columns to str
    for c in ("home_team", "away_team"):
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df


def _first_non_null(d: Dict[str, Any], keys: Iterable[str], default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


def _game_identity(game: Dict[str, Any]) -> Tuple[str, str, pd.Timestamp]:
    """
    Extract (home, away, start_dt) tuple from a CFBD game row or our internal format.
    """
    home = str(_first_non_null(game, ("home_team", "homeTeam", "home"), ""))
    away = str(_first_non_null(game, ("away_team", "awayTeam", "away"), ""))
    # start_date can be ISO with or without Z
    sraw = _first_non_null(game, ("start_date", "startTime", "start_time"), None)
    start_dt = None
    if sraw:
        try:
            start_dt = pd.to_datetime(sraw, utc=True)
        except Exception:
            start_dt = None
    # Fallback: if season/week only, still return a consistent key; features_for_game uses nearest <= date
    if start_dt is None:
        # Try season/week into a rough date ordering (week helps ordering within a season)
        season = int(_first_non_null(game, ("season",), 0) or 0)
        week = int(_first_non_null(game, ("week",), 0) or 0)
        # Use a synthetic timestamp monotonically increasing within the season
        start_dt = pd.Timestamp(f"{season}-01-01", tz="UTC") + pd.to_timedelta(week, unit="W")
    return home, away, start_dt


def features_for_game(game: Dict[str, Any], pen_df: pd.DataFrame, windows: Iterable[int] = (3, 5, 10)) -> Dict[str, float]:
    """
    Look up rolling penalty features for a given game row.

    Expected columns in pen_df:
      - 'home_team', 'away_team', 'start_dt'
      - for each window w in windows:
          home_pen_{w}, away_pen_{w}, home_pen_yds_{w}, away_pen_yds_{w}
    This function returns a flat dict of features for the game’s (home, away) at the latest
    row in pen_df with start_dt <= game.start_dt.
    """
    home, away, start_dt = _game_identity(game)
    if not home or not away:
        return {}

    cand = pen_df[(pen_df["home_team"] == home) & (pen_df["away_team"] == away)]
    if "start_dt" in cand.columns:
        cand = cand[cand["start_dt"] <= start_dt].sort_values("start_dt")
    if cand.empty:
        # Try reversed home/away if dataset stored differently
        cand2 = pen_df[(pen_df["home_team"] == away) & (pen_df["away_team"] == home)]
        if "start_dt" in cand2.columns:
            cand2 = cand2[cand2["start_dt"] <= start_dt].sort_values("start_dt")
        cand = cand2

    if cand.empty:
        return {}

    row = cand.iloc[-1].to_dict()
    feats: Dict[str, float] = {}
    for w in windows:
        for name in (f"home_pen_{w}", f"away_pen_{w}",
                     f"home_pen_yds_{w}", f"away_pen_yds_{w}",
                     f"home_pen_rate_{w}", f"away_pen_rate_{w}"):
            if name in row:
                try:
                    feats[name] = float(row[name])
                except Exception:
                    pass
    return feats

