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


def _build(years: int, windows: Iterable[int]) -> None:
    win_str = ",".join(str(w) for w in windows)
    cmd = [sys.executable, "-m", "src.data_penalties", "--years", str(years), "--windows", win_str]
    print(f"[feat_penalties] building penalties with: {' '.join(cmd)}")
    # capture output so we can show the real error on failure
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if proc.returncode != 0:
        print("[feat_penalties] ---- builder stdout ----")
        print(proc.stdout)
        print("[feat_penalties] ---- builder stderr ----")
        print(proc.stderr)
        raise RuntimeError(
            f"penalty builder failed (rc={proc.returncode}) for years={years}, windows={win_str}"
        )


def _ensure_built(years: int = 5, windows: Iterable[int] = (3, 5, 10)) -> None:
    """
    Ensure penalties parquet exists. If missing, build it; on failure, retry with fewer years (3).
    """
    if PARQUET.exists():
        return

    try:
        _build(years=years, windows=windows)
    except Exception as e:
        print(f"[feat_penalties] initial build failed: {e} — retrying with years=3 ...")
        _build(years=3, windows=windows)

    if not PARQUET.exists():
        raise FileNotFoundError(
            f"Failed to create {PARQUET}. Try running manually: "
            f"python -m src.data_penalties --years {years} --windows {','.join(map(str, windows))}"
        )


def load_penalty_features(years: int = 5, windows: Iterable[int] = (3, 5, 10)) -> pd.DataFrame:
    _ensure_built(years=years, windows=windows)
    df = pd.read_parquet(PARQUET)
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
    home = str(_first_non_null(game, ("home_team", "homeTeam", "home"), ""))
    away = str(_first_non_null(game, ("away_team", "awayTeam", "away"), ""))
    sraw = _first_non_null(game, ("start_date", "startTime", "start_time"), None)
    start_dt = None
    if sraw:
        try:
            start_dt = pd.to_datetime(sraw, utc=True)
        except Exception:
            start_dt = None
    if start_dt is None:
        season = int(_first_non_null(game, ("season",), 0) or 0)
        week = int(_first_non_null(game, ("week",), 0) or 0)
        start_dt = pd.Timestamp(f"{season}-01-01", tz="UTC") + pd.to_timedelta(week, unit="W")
    return home, away, start_dt


def features_for_game(game: Dict[str, Any], pen_df: pd.DataFrame, windows: Iterable[int] = (3, 5, 10)) -> Dict[str, float]:
    home, away, start_dt = _game_identity(game)
    if not home or not away:
        return {}

    cand = pen_df[(pen_df["home_team"] == home) & (pen_df["away_team"] == away)]
    if "start_dt" in cand.columns:
        cand = cand[cand["start_dt"] <= start_dt].sort_values("start_dt")
    if cand.empty:
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

