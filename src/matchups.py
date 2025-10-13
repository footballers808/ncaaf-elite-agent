import numpy as np
from typing import Tuple

def _safe(v, d=0.0):
    try: 
        x = float(v)
        if np.isfinite(x): return x
    except: 
        pass
    return d

def derive_matchup_row(team_row):
    # Features from your team power table (off_ppg/def_ppg/pace_ppg/power)
    return {
        "off": _safe(team_row.off_ppg, 24.0),
        "def": _safe(team_row.def_ppg, 24.0),
        "pace": _safe(team_row.pace_ppg, 70.0),
        "pow": _safe(team_row.power, 0.0),
    }

def matchup_adjustments(game, team_tbl, cfg) -> Tuple[float,float,str]:
    """
    Small, data-driven nudges:
      - If home offense >> away defense → spread & total edge for home
      - Pace alignment pushes total slightly
    """
    home = game["homeTeam"]; away = game["awayTeam"]
    th = team_tbl[team_tbl.team==home]
    ta = team_tbl[team_tbl.team==away]
    if th.empty or ta.empty: 
        return 0.0, 0.0, ""

    H = derive_matchup_row(th.iloc[0])
    A = derive_matchup_row(ta.iloc[0])

    # Off-vs-Def deltas (bigger = better for the offense)
    edge_home = (H["off"] - A["def"])
    edge_away = (A["off"] - H["def"])

    # Spread: home edge minus away edge
    spread_adj = (edge_home - edge_away) * cfg["matchup_spread_scale"]
    spread_adj = float(max(-cfg["max_spread_adj_component"], min(spread_adj, cfg["max_spread_adj_component"])))

    # Total: both offenses strong + fast pace → slight bump; defensive dominance → trim
    pace_factor = (H["pace"] + A["pace"]) / 140.0
    total_adj = (edge_home + edge_away) * cfg["matchup_total_scale"] * float(np.clip(pace_factor, 0.85, 1.15))
    total_adj = float(max(-cfg["max_total_adj_component"], min(total_adj, cfg["max_total_adj_component"])))

    note = f"Edge(H-OvsD {edge_home:.1f}, A-OvsD {edge_away:.1f})"
    return spread_adj, total_adj, note
