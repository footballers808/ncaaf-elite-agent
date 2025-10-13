# src/exact_score.py
from __future__ import annotations

import os
import glob
from typing import List

import numpy as np
import pandas as pd


N_SIMS = 40000  # number of Monte Carlo draws per game (tune as you like)

# Reasonable default scoring variance; we keep it simple and fast.
SIG_HOME = 14.0
SIG_AWAY = 14.0
RHO = 0.35  # modest positive correlation between team scores


def _latest_edges() -> str:
    files = sorted(glob.glob(os.path.join("output", "edges_*.csv")))
    if not files:
        raise FileNotFoundError("No edges CSV found in output/edges_*.csv")
    return files[-1]


def _means_from_spread_total(model_spread: float, model_total: float) -> tuple[float, float]:
    """
    Convert spread & total into implied mean points for home/away.

      model_spread = home_mean - away_mean
      model_total  = home_mean + away_mean

      => home_mean = (total + spread)/2
         away_mean = (total - spread)/2
    """
    mu_home = 0.5 * (model_total + model_spread)
    mu_away = 0.5 * (model_total - model_spread)
    return mu_home, mu_away


def _simulate_scores(mu_home: float, mu_away: float, n: int = N_SIMS) -> pd.DataFrame:
    """
    Draw integer scoring outcomes from a bivariate normal approximation.
    We clamp to 0..90 and round to integers.
    """
    cov = np.array(
        [
            [SIG_HOME ** 2, RHO * SIG_HOME * SIG_AWAY],
            [RHO * SIG_HOME * SIG_AWAY, SIG_AWAY ** 2],
        ]
    )
    sims = np.random.multivariate_normal([mu_home, mu_away], cov, size=n).astype(np.float64)

    # Round to nearest whole number and clip into a reasonable range
    sims = np.rint(sims)
    sims = np.clip(sims, 0, 90)

    df = pd.DataFrame(sims, columns=["home_score", "away_score"])
    return df


def _key_number_nudge(x: float) -> float:
    """
    Nudge scores that land near football key numbers so the distribution
    doesn't look too Gaussian. Lightweight way to shape the tail.
    """
    keys = [3, 7, 10, 13, 14, 17, 20, 21, 24, 27, 28, 31, 34, 35, 38, 41, 44, 47, 48, 51, 55]
    for k in keys:
        if abs(x - k) <= 1:
            return float(k)
    return float(x)


def _prob_table(home: str, away: str, mu_h: float, mu_a: float, n: int = N_SIMS) -> pd.DataFrame:
    sims = _simulate_scores(mu_h, mu_a, n=n)
    # optional nudge to key numbers
    sims["home_score"] = sims["home_score"].apply(_key_number_nudge)
    sims["away_score"] = sims["away_score"].apply(_key_number_nudge)

    # probability table (top 20 outcomes)
    probs = sims.value_counts(normalize=True).reset_index(name="prob")
    probs.insert(0, "home_team", home)
    probs.insert(1, "away_team", away)
    probs = probs.rename(columns={"home_score": "home_score", "away_score": "away_score"})
    probs = probs.sort_values("prob", ascending=False).head(20)

    # order columns
    return probs[["home_team", "away_team", "home_score", "away_score", "prob"]]


def run():
    edges_path = _latest_edges()
    edges = pd.read_csv(edges_path)

    required = {"home", "away", "model_spread", "model_total"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"{edges_path} missing columns: {missing}")

    rows: List[pd.DataFrame] = []
    for _, r in edges.iterrows():
        home = str(r["home"])
        away = str(r["away"])
        spread = float(r["model_spread"])
        total = float(r["model_total"])
        mu_h, mu_a = _means_from_spread_total(spread, total)
        table = _prob_table(home, away, mu_h, mu_a, n=N_SIMS)
        rows.append(table)

    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["home_team", "away_team", "home_score", "away_score", "prob"]
    )

    os.makedirs("output", exist_ok=True)
    out_path = os.path.join("output", "exact_scores.csv")
    out.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} (rows={len(out)})")


if __name__ == "__main__":
    run()
