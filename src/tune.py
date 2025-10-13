import os, json, yaml, pathlib, shutil, time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
import optuna

# ---- Paths ----
ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config.yaml"
OUT_DIR = ROOT / "tuning-results"
STUDY_DB = OUT_DIR / "study.db"      # enables resume
BEST_CFG = OUT_DIR / "best_config.yaml"
TRIALS_CSV = OUT_DIR / "trials.csv"

# ---- Your project entrypoints (import your modules) ----
# Expect these modules to exist already in your repo
# and honor the config dict passed in.
from src import data as data_mod
from src import model as model_mod
from src import eval as eval_mod

def load_yaml(p: pathlib.Path) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(obj: Dict[str, Any], p: pathlib.Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def set_nested(cfg: Dict[str, Any], path, value):
    d = cfg
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value

def get_nested(cfg: Dict[str, Any], path):
    d = cfg
    for k in path:
        d = d[k]
    return d

# ---- Define the search space here (adjust to your stack) ----
SEARCH_SPACE = {
    ("train","learning_rate"): ("loguniform", 1e-4, 5e-2),
    ("train","weight_decay"):  ("loguniform", 1e-7, 1e-2),
    ("train","max_depth"):     ("int", 3, 10),           # tree models
    ("train","n_estimators"):  ("int", 100, 1200),
    ("data","lookback_days"):  ("int", 21, 120),
    ("data","gap_days"):       ("int", 0, 3),
    ("eval","cv_folds"):       ("int", 3, 6),
    ("eval","calibration"):    ("choice", ["none","isotonic","temperature"]),
    # Edges / thresholds (affects play selection quality)
    ("edge","threshold_spread"): ("uniform", 1.0, 3.5),
    ("edge","threshold_total"):  ("uniform", 1.5, 4.0),
}

def suggest_param(trial: optuna.Trial, spec):
    kind, *args = spec
    if kind == "uniform":      return trial.suggest_float("p", *args)
    if kind == "loguniform":   return trial.suggest_float("p", *args, log=True)
    if kind == "int":          return trial.suggest_int("p", *args)
    if kind == "choice":       return trial.suggest_categorical("p", args[0])
    raise ValueError(f"unknown search space kind {kind}")

def oof_score_for_cfg(cfg: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    """
    Train/evaluate with time-series CV using your project modules.
    Return a scalar to minimize (e.g., LogLoss or -Sharpe) and any metrics to log.
    """
    # Load data per cfg
    Xy = data_mod.load_for_training(cfg)  # implement in src/data.py
    # Do rolling/temporal CV; implement in src/eval.py
    metrics = eval_mod.cv_eval(model_mod.fit_and_predict, Xy, cfg)  
    # metrics expected to include at least: {"logloss": ..., "brier": ..., "roi": ..., "sharpe": ...}
    # Primary objective: minimize logloss (more stable than ROI during tuning)
    objective = metrics.get("logloss", 1.0)
    return objective, metrics

def build_cfg_from_trial(base_cfg: Dict[str, Any], trial: optuna.Trial) -> Dict[str, Any]:
    cfg = yaml.safe_load(yaml.dump(base_cfg))  # deep copy
    for path, spec in SEARCH_SPACE.items():
        trial._tell_param_name = ".".join(path)  # for CSV clarity
        val = suggest_param(trial, spec)
        set_nested(cfg, path, val)
    # Ensure temporal split
    cfg.setdefault("eval", {}).setdefault("split", "temporal")
    return cfg

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = load_yaml(CFG_PATH)

    storage = f"sqlite:///{STUDY_DB}"
    study = optuna.create_study(
        study_name="ncaaf-elite-agent",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )

    trial_rows = []
    def objective(trial: optuna.Trial):
        cfg = build_cfg_from_trial(base_cfg, trial)
        t0 = time.time()
        score, metrics = oof_score_for_cfg(cfg)
        dt = time.time() - t0

        # Log additional metrics in the trial
        for k, v in metrics.items():
            trial.set_user_attr(k, float(v) if isinstance(v, (int, float, np.floating)) else v)

        # Keep a local CSV of trials
        row = {"number": trial.number, "value": float(score), "seconds": dt}
        row.update({ ".".join(k): get_nested(cfg, k) for k in SEARCH_SPACE.keys() })
        row.update(metrics)
        trial_rows.append(row)
        pd.DataFrame(trial_rows).to_csv(TRIALS_CSV, index=False)
        return score

    # Iterations come from env or default
    iters = int(os.environ.get("TUNE_ITERS", "60"))
    study.optimize(objective, n_trials=iters, gc_after_trial=True)

    # Build the best config
    best_trial = study.best_trial
    best_cfg = build_cfg_from_trial(base_cfg, best_trial)

    # Pick calibration that performed best (if your metrics include ece/calibration)
    # (Already in SEARCH_SPACE, so best_cfg contains it.)

    save_yaml(best_cfg, BEST_CFG)

    # Also export the study summary JSON
    summary = {
        "best_value": study.best_value,
        "best_trial": best_trial.number,
        "n_trials": len(study.trials),
        "metrics": {k: best_trial.user_attrs.get(k) for k in best_trial.user_attrs},
    }
    json.dump(summary, open(OUT_DIR / "summary.json", "w"), indent=2)

    print(f"✅ best_config.yaml written to {BEST_CFG}")
    print(f"Best objective (lower is better): {study.best_value}")

if __name__ == "__main__":
    main()
