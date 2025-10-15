import pandas as pd
from . import cfbd_client as cfbd

def attach_market_edges(preds: pd.DataFrame, year: int, season_type: str, week: int):
    ln = cfbd.lines(year, season_type, week)
    rows = []
    for l in ln:
        if not l.get("lines"): 
            continue
        best = l["lines"][-1]
        rows.append({
            "game_id": l["id"],
            "market_spread": best.get("spread"),
            "market_total": best.get("overUnder")
        })
    mkt = pd.DataFrame(rows).drop_duplicates("game_id")
    return preds.merge(mkt, on="game_id", how="left")
