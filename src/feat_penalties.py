from __future__ import annotations
from typing import Dict, Any, Iterable, Tuple
import pandas as pd
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART_DIR = ROOT / "artifacts" / "features"

def load_penalty_features() -> pd.DataFrame:
    p = ART_DIR / "penalties.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} not found. Run: python -m src.data_penalties --years 5 --windows 3,5,10"
        )
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["team"] = df["team"].astype(str)
    return df

def features_for_game(game: Dict[str, Any], pen_df: pd.DataFrame, windows=(3,5,10)) -> Dict[str, float]:
    """
    Leak-safe lookup: use the **latest row strictly before the game date** for each team.
    Returns dict of home/away features and diffs.
    """
    date = pd.to_datetime(game.get("start_date") or game.get("startTime") or game.get("date"), utc=True)
    h = str(game.get("home_team") or game.get("home"))
    a = str(game.get("away_team") or game.get("away"))
    if not h or not a or pd.isna(date):
        return {}

    hrow = pen_df[(pen_df.team == h) & (pen_df.date < date)].tail(1)
    arow = pen_df[(pen_df.team == a) & (pen_df.date < date)].tail(1)

    out: Dict[str, float] = {}
    for w in windows:
        col = f"pens_pg_w{w}"
        hv = float(hrow[col].iloc[0]) if (not hrow.empty and col in hrow) else float("nan")
        av = float(arow[col].iloc[0]) if (not arow.empty and col in arow) else float("nan")
        out[f"h_pens_pg_w{w}"] = hv
        out[f"a_pens_pg_w{w}"] = av
        out[f"diff_pens_pg_w{w}"] = hv - av if (hv == hv and av == av) else float("nan")  # NaN-safe
    return out
