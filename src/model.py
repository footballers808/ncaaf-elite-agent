import numpy as np

def prob_home_covers(line, mu, sigma=13.0):
    from math import erf
    z = (mu - line)/max(1e-6, sigma)
    return 0.5*(1+erf(z/np.sqrt(2)))

def predict_game(game, team_tbl, cfg):
    away, home = game["awayTeam"], game["homeTeam"]
    th = team_tbl[team_tbl.team==home]
    ta = team_tbl[team_tbl.team==away]
    if th.empty or ta.empty: return None

    ph, pa = th.iloc[0].power, ta.iloc[0].power
    off_h, off_a = th.iloc[0].off_ppg, ta.iloc[0].off_ppg
    def_h, def_a = th.iloc[0].def_ppg, ta.iloc[0].def_ppg
    pace_h, pace_a = th.iloc[0].pace_ppg, ta.iloc[0].pace_ppg

    spread = cfg["power_scale"]*(ph - pa) + (0.5 if game.get("neutralSite") else cfg["hfa_points"])
    pace_factor = float(np.clip((pace_h + pace_a)/140.0, 0.85, 1.15))
    total = float(np.clip(off_h + off_a + 0.25*(def_h + def_a),
                          cfg["min_total_floor"], cfg["max_total_cap"])) * pace_factor
    total = float(np.clip(total, cfg["min_total_floor"], cfg["max_total_cap"]))

    home_pts = (total + spread)/2.0
    away_pts = (total - spread)/2.0
    p_cover = prob_home_covers(spread, spread, sigma=cfg["sigma_points"])

    return dict(
        gameId=game.get("id"),
        start_local=game.get("start_local"),
        home=home, away=away,
        spread=round(spread,2), total=round(total,1),
        pred_home_pts=int(round(home_pts)), pred_away_pts=int(round(away_pts)),
        p_home_cover=round(p_cover,3)
    )
