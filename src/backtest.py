from __future__ import annotations
import os
import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import numpy as np
import requests
import yaml

# Reuse your model + labeler helpers
from .model import (
    predict_game,
    train_model, load_model, save_model,
    _season_year_from_cfg as _year_infer,
)
from .model import _headers as _cfbd_headers
from .labeler import _pick as _pick_col

CFBD = "https://api.collegefootballdata.com"


# -------------------------- Fetch historical season --------------------------

def fetch_season_games(year: int, season_type: str = "regular") -> pd.DataFrame:
    """All games for a season with finals; sorted chronologically."""
    url = f"{CFBD}/games"
    params = {"year": int(year), "seasonType": season_type}
    r = requests.get(url, params=params, headers=_cfbd_headers(), timeout=90)
    r.raise_for_status()
    raw = pd.DataFrame(r.json())
    if raw.empty:
        return pd.DataFrame(columns=["id","home_team","away_team","start_date","neutral_site","home_points","away_points"])

    cols = list(raw.columns)
    gid   = _pick_col(cols, ["id","game_id","gameId"])
    home  = _pick_col(cols, ["home_team","home","homeTeam"])
    away  = _pick_col(cols, ["away_team","away","awayTeam"])
    start = _pick_col(cols, ["start_date","startDate","start_time_tbd","startTimeTBD"])
    neu   = _pick_col(cols, ["neutral_site","neutralSite"])
    hp    = _pick_col(cols, ["home_points","home_score","homePoints","HomePoints"])
    ap    = _pick_col(cols, ["away_points","away_score","awayPoints","AwayPoints"])

    df = pd.DataFrame({
        "id": raw[gid],
        "home_team": raw[home],
        "away_team": raw[away],
        "start_date": raw[start] if start in raw else None,
        "neutral_site": raw[neu] if neu in raw else False,
        "home_points": raw[hp],
        "away_points": raw[ap],
    }).copy()

    df = df[df["home_points"].notna() & df["away_points"].notna()].copy()
    df["id"] = pd.to_numeric(df["id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["id"]).copy()
    df["id"] = df["id"].astype("int64")

    if "start_date" in df and df["start_date"].notna().any():
        df["start_dt"] = pd.to_datetime(df["start_date"], errors="coerce", utc=True)
        df = df.sort_values("start_dt", na_position="last").drop(columns=["start_dt"])
    return df.reset_index(drop=True)


def fetch_season_lines(year: int, season_type: str = "regular") -> pd.DataFrame:
    """
    Fetch provider lines and collapse to last line per (game, provider).
    Uses tolerant column picking (camelCase/snake_case).
    """
    url = f"{CFBD}/lines"
    params = {"year": int(year), "seasonType": season_type}
    r = requests.get(url, params=params, headers=_cfbd_headers(), timeout=90)
    r.raise_for_status()
    raw = pd.DataFrame(r.json())
    if raw.empty or "lines" not in raw.columns:
        return pd.DataFrame(columns=["game_id","provider","home_spread","total","last_updated"])

    cols = list(raw.columns)
    gid  = _pick_col(cols, ["id","game_id","gameId"])
    home = _pick_col(cols, ["home_team","home","homeTeam"])
    away = _pick_col(cols, ["away_team","away","awayTeam"])

    # Build a minimal base frame with tolerant names
    base = pd.DataFrame({
        "id": raw[gid],
        "home_team": raw[home],
        "away_team": raw[away],
        "lines": raw["lines"],
    })

    # Explode nested lines rows
    base = base.explode("lines", ignore_index=True)
    if base.empty:
        return pd.DataFrame(columns=["game_id","provider","home_spread","total","last_updated"])

    ln = pd.json_normalize(base["lines"]).add_prefix("ln.")
    df = pd.concat([base.drop(columns=["lines"]), ln], axis=1)

    cols2 = list(df.columns)
    prov = _pick_col(cols2, ["ln.provider","ln.provider_name","ln.providerName"])
    spr  = _pick_col(cols2, ["ln.home_spread","ln.homeSpread","ln.spread","ln.formattedSpread","ln.spreadOpen"])
    tot  = _pick_col(cols2, ["ln.over_under","ln.overUnder","ln.total","ln.totalOpen"])
    upd  = _pick_col(cols2, ["ln.last_updated","ln.lastUpdated","ln.updated"])

    def _to_num(x):
        if x is None:
            return None
        try:
            s = str(x).strip().lower()
            if s in ("pk","pick","pickem","pick'em"):
                return 0.0
            return float(x)
        except Exception:
            return None

    out = pd.DataFrame({
        "game_id": pd.to_numeric(df["id"], errors="coerce"),
        "provider": df[prov],
        "home_spread": df[spr].map(_to_num) if spr in df else None,
        "total": df[tot].map(_to_num) if tot in df else None,
        "last_updated": pd.to_datetime(df[upd], errors="coerce", utc=True) if upd in df else None,
    }).dropna(subset=["game_id","provider"]).copy()

    out["game_id"] = out["game_id"].astype("int64")
    out = out.sort_values(["game_id","provider","last_updated"], na_position="last").drop_duplicates(
        subset=["game_id","provider"], keep="last"
    )
    return out.reset_index(drop=True)


def consensus_lines(lines: pd.DataFrame, preferred: Optional[List[str]] = None) -> pd.DataFrame:
    if lines.empty:
        return pd.DataFrame(columns=["game_id","mkt_spread","mkt_total"])
    df = lines.copy()
    if preferred:
        df["is_pref"] = df["provider"].isin(preferred)
    rows = []
    for gid, grp in df.groupby("game_id"):
        use = grp[grp["is_pref"]] if (preferred and grp["is_pref"].any()) else grp
        sp = use["home_spread"].dropna()
        tt = use["total"].dropna()
        rows.append({
            "game_id": int(gid),
            "mkt_spread": float(sp.mean()) if not sp.empty else None,
            "mkt_total": float(tt.mean()) if not tt.empty else None,
        })
    return pd.DataFrame(rows)


# -------------------------- Backtest Engine --------------------------

@dataclass
class BTConfig:
    years: List[int]
    season_type: str
    use_prev_year_end_as_seed: bool  # if True, powers carry across seasons; else reset each season


def load_cfg(path: str = "config.yaml") -> Dict[str, Any]:
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _init_team_power() -> pd.DataFrame:
    # fresh neutral table (same as model’s initializer)
    url = f"{CFBD}/teams/fbs"
    r = requests.get(url, headers=_cfbd_headers(), timeout=60)
    r.raise_for_status()
    raw = pd.DataFrame(r.json())
    if raw.empty:
        return pd.DataFrame(columns=["team","power","off_ppg","def_ppg","pace_ppg"])
    df = pd.DataFrame({"team": raw["school"]})
    df["power"] = 0.0; df["off_ppg"]=28.0; df["def_ppg"]=27.0; df["pace_ppg"]=70.0
    return df


def simulate_year(year: int, cfg: Dict[str, Any], carry_model: Optional[pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, float], pd.DataFrame]:
    """
    Chronological, no-lookahead sim:
      predict BEFORE learning each game, then learn from final.
    """
    cfg = dict(cfg)
    cfg["season_year"] = int(year)

    # start-of-season powers
    if isinstance(carry_model, pd.DataFrame) and not carry_model.empty:
        team_tbl = carry_model.copy()
    else:
        team_tbl = _init_team_power()

    games = fetch_season_games(year, cfg.get("backtest_season_type","regular"))
    lines = fetch_season_lines(year, cfg.get("backtest_season_type","regular"))
    mkt = consensus_lines(lines, preferred=cfg.get("odds",{}).get("preferred_providers"))
    mkt_by_gid = {int(r["game_id"]): (r["mkt_spread"], r["mkt_total"]) for _, r in mkt.iterrows()}

    rows: List[Dict[str, Any]] = []

    for _, g in games.iterrows():
        game_dict = {
            "id": int(g["id"]),
            "homeTeam": g["home_team"],
            "awayTeam": g["away_team"],
            "start_local": g.get("start_date"),
            "neutralSite": bool(g.get("neutral_site", False)),
        }
        try:
            pred = predict_game(game_dict, team_tbl, cfg)
        except Exception as e:
            print(f"predict error on game {g['id']}: {e}")
            pred = None

        m_spread, m_total = mkt_by_gid.get(int(g["id"]), (None, None))
        row = {
            "game_id": int(g["id"]),
            "home": g["home_team"],
            "away": g["away_team"],
            "home_points": int(g["home_points"]),
            "away_points": int(g["away_points"]),
            "mkt_spread": m_spread,
            "mkt_total": m_total,
        }
        if pred:
            row.update({
                "pred_spread": float(pred["spread"]),
                "pred_total": float(pred["total"]),
                "pred_home_pts": int(pred["pred_home_pts"]),
                "pred_away_pts": int(pred["pred_away_pts"]),
            })
            if m_spread is not None:
                row["edge_spread"] = round(float(pred["spread"]) - float(m_spread), 2)
            if m_total is not None:
                row["edge_total"] = round(float(pred["total"]) - float(m_total), 2)

        rows.append(row)

        # learn from final
        labeled = [{
            "home": g["home_team"],
            "away": g["away_team"],
            "home_points": int(g["home_points"]),
            "away_points": int(g["away_points"]),
        }]
        team_tbl = train_model(cfg, team_tbl, labeled)

    df = pd.DataFrame(rows)

    # metrics
    met: Dict[str, float] = {}
    if not df.empty:
        df["actual_margin"] = df["home_points"] - df["away_points"]
        if "pred_spread" in df:
            met["MAE_spread"] = float((df["pred_spread"] - df["actual_margin"]).abs().mean())
        if "pred_total" in df:
            df["actual_total"] = df["home_points"] + df["away_points"]
            met["MAE_total"] = float((df["pred_total"] - df["actual_total"]).abs().mean())

        if "pred_spread" in df and df["mkt_spread"].notna().any():
            preds = np.sign(df["pred_spread"] - df["mkt_spread"].fillna(0))
            outcomes = np.sign(df["actual_margin"] - df["mkt_spread"].fillna(0))
            valid = df["mkt_spread"].notna()
            met["ATS_agreement_rate"] = float((preds[valid] == outcomes[valid]).mean())

        tiers = (cfg.get("edge_tiers") or {})
        th_spread = (tiers.get("spread") or [1.5, 2.5, 4.0])[0]

        roi_u = 0.0; bets = 0
        if "edge_spread" in df and df["mkt_spread"].notna().any():
            take = df["edge_spread"].abs() >= th_spread
            for _, r in df[take & df["mkt_spread"].notna()].iterrows():
                model_pick_home = (r["pred_spread"] < r["mkt_spread"])
                covered = (r["actual_margin"] < r["mkt_spread"]) if model_pick_home else (r["actual_margin"] > r["mkt_spread"])
                roi_u += 1.0 if covered else -1.1
                bets += 1
        met["ROI_units_spread"] = float(roi_u)
        met["Bets_spread"] = int(bets)

    return df, met, team_tbl


def backtest(cfg: Dict[str, Any], years: List[int], season_type: str, carry_across_seasons: bool) -> Dict[str, Any]:
    results: List[pd.DataFrame] = []
    model_at_end: Optional[pd.DataFrame] = None
    metrics_per_year: Dict[int, Dict[str, float]] = {}

    for y in years:
        carry_model = model_at_end if carry_across_seasons else None
        df, met, model_at_end = simulate_year(y, {**cfg, "backtest_season_type": season_type}, carry_model)
        results.append(df.assign(season=y))
        metrics_per_year[y] = met
        print(f"[{y}] {met}")

    all_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    overall: Dict[str, float] = {}
    if not all_df.empty:
        if "pred_spread" in all_df:
            overall["MAE_spread"] = float((all_df["pred_spread"] - (all_df["home_points"] - all_df["away_points"])).abs().mean())
        if "pred_total" in all_df:
            overall["MAE_total"] = float((all_df["pred_total"] - (all_df["home_points"] + all_df["away_points"])).abs().mean())
        overall["Bets_spread"] = int(sum(m.get("Bets_spread",0) for m in metrics_per_year.values()))
        overall["ROI_units_spread"] = float(sum(m.get("ROI_units_spread",0.0) for m in metrics_per_year.values()))

    return {
        "by_year": metrics_per_year,
        "overall": overall,
        "predictions": all_df,
        "final_model": model_at_end,
    }


# -------------------------- CLI --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=str, required=True, help="e.g. 2021,2022,2023")
    ap.add_argument("--season-type", type=str, default="regular", choices=["regular","postseason","both"])
    ap.add_argument("--carry", action="store_true", help="carry powers across seasons (use previous season end as seed)")
    ap.add_argument("--out", type=str, default="backtest_out")
    args = ap.parse_args()

    cfg = load_cfg()
    years = [int(x) for x in args.years.split(",") if x.strip()]

    res = backtest(cfg, years, args.season_type, args.carry)

    os.makedirs(args.out, exist_ok=True)
    pd.DataFrame(res["predictions"]).to_csv(os.path.join(args.out, "predictions_all.csv"), index=False)
    pd.DataFrame([{"year": y, **res["by_year"][y]} for y in res["by_year"]]).to_csv(os.path.join(args.out, "metrics_by_year.csv"), index=False)
    pd.DataFrame([res["overall"]]).to_csv(os.path.join(args.out, "metrics_overall.csv"), index=False)
    if isinstance(res["final_model"], pd.DataFrame) and not res["final_model"].empty:
        res["final_model"].to_csv(os.path.join(args.out, "team_power_final.csv"), index=False)

    print("== Overall ==")
    print(res["overall"])
    print("✅ Backtest complete.")

if __name__ == "__main__":
    main()

