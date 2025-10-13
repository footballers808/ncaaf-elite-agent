from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Iterable, Optional, Dict, Any, List

import pandas as pd
import requests

CFBD = "https://api.collegefootballdata.com"


# ----------------------- helpers (unchanged) -----------------------
def _pick(cols: Iterable[str], choices: Iterable[str]) -> Optional[str]:
    """
    Return first matching column (case-insensitive) from `choices`.
    Works with snake_case and camelCase because we compare lowercased names.
    """
    lower_map = {c.lower(): c for c in cols}
    for c in choices:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def _headers() -> dict:
    api_key = os.environ.get("CFBD_API_KEY")
    if not api_key:
        raise EnvironmentError("Missing CFBD_API_KEY secret.")
    return {"Authorization": f"Bearer {api_key}"}


# ----------------------- public fetch (your existing logic) -----------------------
def fetch_completed_games(year: int, season_type: str = "regular") -> pd.DataFrame:
    """
    Pull games for a season; keep only rows that have final scores and
    normalize to a standard schema:
        game_id, home, away, home_points, away_points
    """
    url = f"{CFBD}/games"
    params = {"year": year, "seasonType": season_type}
    r = requests.get(url, params=params, headers=_headers(), timeout=60)
    r.raise_for_status()
    raw = pd.DataFrame(r.json())

    if raw.empty:
        print(f"⚠️ CFBD returned 0 rows for year={year} seasonType={season_type}")
        return raw

    cols = list(raw.columns)

    # Accept both snake_case and camelCase
    id_col = _pick(cols, ["id", "game_id", "gameId"])
    home_col = _pick(cols, ["home_team", "home", "homeName", "homeTeam"])
    away_col = _pick(cols, ["away_team", "away", "awayName", "awayTeam"])
    hp_col = _pick(cols, ["home_points", "home_score", "HomePoints", "homePoints"])
    ap_col = _pick(cols, ["away_points", "away_score", "AwayPoints", "awayPoints"])

    missing = []
    if not id_col:
        missing.append("id/game_id")
    if not home_col:
        missing.append("home_team/home/homeTeam")
    if not away_col:
        missing.append("away_team/away/awayTeam")
    if not hp_col:
        missing.append("home_points/home_score")
    if not ap_col:
        missing.append("away_points/away_score")

    if missing:
        raise ValueError(
            f"CFBD /games missing expected columns: {missing}\n"
            f"Available columns: {cols}"
        )

    out = pd.DataFrame(
        {
            "game_id": raw[id_col],
            "home": raw[home_col],
            "away": raw[away_col],
            "home_points": raw[hp_col],
            "away_points": raw[ap_col],
        }
    )

    # Final only
    out = out[out["home_points"].notna() & out["away_points"].notna()].copy()

    # Normalize dtypes
    out["game_id"] = pd.to_numeric(out["game_id"], errors="coerce")
    out = out.dropna(subset=["game_id"]).copy()
    out["game_id"] = out["game_id"].astype("int64")

    out = out.drop_duplicates(subset=["game_id"]).reset_index(drop=True)
    return out


def _save_labels(df: pd.DataFrame) -> str:
    os.makedirs("store", exist_ok=True)
    path = os.path.join("store", "labels.parquet")
    df.to_parquet(path, index=False)
    return path


# ----------------------- NEW: pipeline adapter -----------------------
def label_latest_results(cfg: Dict[str, Any], predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Adapter used by the runner. It fetches completed games for the configured season
    and (if possible) merges with predictions on gameId to build supervised rows.
    If predictions don’t include gameId, we still return label rows (no y_true).
    """
    try:
        year = int(cfg.get("season_year") or datetime.now(timezone.utc).year)
    except Exception:
        year = datetime.now(timezone.utc).year

    df = fetch_completed_games(year, "regular")
    if df.empty:
        print("ℹ️ No completed games available for labeling.")
        return []

    # Map by game_id for quick join to predictions (which may store key as 'gameId')
    label_by_id = {int(gid): row for gid, row in df.set_index("game_id").iterrows()}
    out: List[Dict[str, Any]] = []

    # Try to label the predictions if ids match; otherwise emit label-only rows.
    matched = 0
    for p in predictions or []:
        gid = p.get("gameId") or p.get("game_id")
        if gid is None:
            continue
        try:
            gid = int(gid)
        except Exception:
            continue
        lab = label_by_id.get(gid)
        if lab is None:
            continue
        matched += 1
        home_pts = float(lab["home_points"])
        away_pts = float(lab["away_points"])
        row = dict(p)  # copy prediction fields
        row.update(
            {
                "y_true": 1 if (home_pts - away_pts) > 0 else 0,  # simple win label (example)
                "home_points": int(home_pts),
                "away_points": int(away_pts),
                # optional regression truth if your model outputs these preds:
                "spread_true": away_pts - home_pts,  # home negative if home won big
                "total_true": home_pts + away_pts,
            }
        )
        out.append(row)

    if matched == 0:
        # Fallback: return label rows only (useful for metrics later), runner will handle empty y_true gracefully
        for _, r in df.iterrows():
            out.append(
                {
                    "game_id": int(r["game_id"]),
                    "home": r["home"],
                    "away": r["away"],
                    "home_points": int(r["home_points"]),
                    "away_points": int(r["away_points"]),
                    # no y_true because we don’t know your exact target when no prediction row exists
                }
            )
    print(f"✅ Built {len(out)} labeled rows (matched {matched} predictions).")
    return out


# ----------------------- CLI (kept for manual use) -----------------------
def _cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=datetime.utcnow().year)
    parser.add_argument(
        "--season-type",
        type=str,
        default="regular",
        help="regular|postseason|both (CFBD seasonType)",
    )
    args = parser.parse_args()

    df = fetch_completed_games(args.year, args.season_type)
    if df.empty:
        print("⚠️ No completed games found; nothing to write.")
        return

    path = _save_labels(df)
    print(f"✅ Wrote {path} with {len(df)} rows and columns {list(df.columns)}")


if __name__ == "__main__":
    _cli()
