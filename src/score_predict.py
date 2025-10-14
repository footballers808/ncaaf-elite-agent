# src/score_predict.py
from __future__ import annotations
from typing import Optional, Tuple
from statistics import NormalDist

# Tunables (you can move these into config.yaml later)
ND = NormalDist()           # standard normal
SIGMA_MARGIN = 16.0         # typical CFB scoring-margin stdev
AVG_TOTAL = 55.0            # fallback total if market total missing

def spread_from_win_prob(p_home: float, sigma: float = SIGMA_MARGIN) -> float:
    """
    Convert win probability -> implied point spread using a normal-margin model:
      P(home win) = Phi(mu / sigma)  =>  mu = sigma * Phi^{-1}(p)
    """
    p = min(max(float(p_home), 1e-6), 1 - 1e-6)
    return sigma * ND.inv_cdf(p)

def scores_from_spread_total(spread: float, total: float) -> Tuple[float, float]:
    """
    Given spread (home - away) and total points:
        H - A = spread
        H + A = total
      => H = (total + spread) / 2
         A = (total - spread) / 2
    """
    h = 0.5 * (total + spread)
    a = 0.5 * (total - spread)
    return h, a

def predict_scores(
    p_home: float,
    market_total: Optional[float] = None,
    sigma: float = SIGMA_MARGIN,
    fallback_total: float = AVG_TOTAL,
) -> Tuple[int, int, float]:
    """
    Returns (pred_home_score:int, pred_away_score:int, spread_pred:float).
    - Spread is derived from win prob.
    - Total comes from market_total, else a fallback constant.
    """
    spread = spread_from_win_prob(p_home, sigma=sigma)
    total = market_total if (market_total is not None and market_total > 0) else fallback_total
    h, a = scores_from_spread_total(spread, total)
    return max(0, round(h)), max(0, round(a)), spread
