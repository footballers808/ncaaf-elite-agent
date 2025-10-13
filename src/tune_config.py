from __future__ import annotations
import itertools
import json
import os
from typing import Any, Dict, List

import yaml

from .backtest import backtest, load_cfg


def _save_yaml(path: str, data: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def run_sweep(base_cfg: Dict[str, Any], years: List[int]) -> List[Dict[str, Any]]:
    """
    Simple grid search over a handful of knobs.
    You can expand/alter these ranges based on compute budget.
    """
    # Define the grid
    grid = {
        # Learn loop
        ("learn","k_factor"):         [0.18, 0.22, 0.25, 0.30],
        ("learn","margin_cap"):       [18, 24, 28],
        ("learn","weekly_decay"):     [0.00, 0.01],
        ("learn","hfa_points_for_learn"): [1.5, 2.0, 2.5],

        # Model & scoring split
        ("hfa_points",):              [1.5, 2.0, 2.5],
        ("power_scale",):             [0.9, 1.0, 1.1],
        ("score_split_gain",):        [0.02, 0.03, 0.05],

        # Star tiers (first threshold only – controls bet count)
        ("edge_tiers","spread","t0"): [1.0, 1.5, 2.0],
        ("edge_tiers","total","t0"):  [1.5, 2.0, 2.5],
    }

    # explode to list of parameter combinations
    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    results: List[Dict[str, Any]] = []
    for combo in itertools.product(*values):
        cfg = _deep_copy(base_cfg)

        # apply combo
        for k, v in zip(keys, combo):
            if k == ("edge_tiers","spread","t0"):
                tiers = cfg.get("edge_tiers", {}).get("spread", [1.5,2.5,4.0])
                tiers = [float(v), tiers[1], tiers[2]]
                cfg.setdefault("edge_tiers", {}).update({"spread": tiers})
            elif k == ("edge_tiers","total","t0"):
                tiers = cfg.get("edge_tiers", {}).get("total", [2.0,3.0,4.5])
                tiers = [float(v), tiers[1], tiers[2]]
                cfg.setdefault("edge_tiers", {}).update({"total": tiers})
            else:
                _set(cfg, list(k), float(v) if isinstance(v, (int,float)) else v)

        # always ensure learning enabled during tuning
        cfg.setdefault("learn", {}).setdefault("enabled", True)

        # run backtest over chosen years (no carry across, to reduce bias)
        res = backtest(cfg, years, season_type="regular", carry_across_seasons=False)
        overall = res["overall"]
        score = _score(overall)  # objective: maximize ROI and minimize MAE

        results.append({
            "params": cfg,
            "overall": overall,
            "score": score,
        })
        print(f"done combo: score={score:.4f}, overall={overall}")

    # sort best to worst (higher score is better)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _score(overall: Dict[str, Any]) -> float:
    """
    Composite score:
      + ROI units (primary)
      - MAE_spread penalty
      (adjust weights as you like)
    """
    roi = float(overall.get("ROI_units_spread", 0.0))
    mae = float(overall.get("MAE_spread", 8.0))
    # Scale penalties to comparable range
    return roi - 0.1 * mae


def _deep_copy(x):
    return yaml.safe_load(yaml.safe_dump(x))


def _set(d: Dict[str, Any], path: List[str], value: Any):
    cur = d
    for p in path[:-1]:
        cur = cur.setdefault(p, {})
    cur[path[-1]] = value


if __name__ == "__main__":
    # Example usage: runs a small sweep on 2022-2023
    base = load_cfg()
    years = [2022, 2023]
    results = run_sweep(base, years)

    os.makedirs("tuning_out", exist_ok=True)
    # save top 10
    for i, item in enumerate(results[:10], 1):
        with open(f"tuning_out/top_{i:02d}.json", "w", encoding="utf-8") as f:
            json.dump(item, f, indent=2)

    # write the single best config suggestion (as YAML for easy diff)
    _save_yaml("tuning_out/best_config.yaml", results[0]["params"])
    print("✅ tuning complete. See tuning_out/ for artifacts.")
