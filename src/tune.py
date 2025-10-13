# src/tune.py
import os, json, yaml, pathlib, shutil, time, subprocess, sys
from typing import Dict, Any, Tuple
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config.yaml"
OUT_DIR = ROOT / "tuning-results"
BEST_CFG = OUT_DIR / "best_config.yaml"
TRIALS_CSV = OUT_DIR / "trials.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
BACKTEST_JSON = ARTIFACTS_DIR / "backtest.json"

SEARCH_SPACE = {
    ("train","learning_rate"): ("loguniform", 1e-4, 5e-2),
    ("train","weight_decay"):  ("loguniform", 1e-7, 1e-2),
    ("train","max_depth"):     ("int", 3, 10),            # if tree-based, safe to ignore otherwise
    ("train","n_estimators"):  ("int", 100, 1200),
    ("data","lookback_days"):  ("int", 21, 120),
    ("data","gap_days"):       ("int", 0, 3),
    ("eval","cv_folds"):       ("int", 3, 6),
    ("eval","calibration"):    ("choice", ["none","isotonic","temperature"]),
    ("edge","threshold_spread"): ("uniform", 1.0, 3.5),
    ("edge","threshold_total"):  ("uniform", 1.5, 4.0),
}

def load_yaml(p): 
    with open(p, "r", encoding="utf-8") as f: 
        return yaml.safe_load(f)

def save_yaml(obj, p):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)

def set_nested(cfg: Dict[str, Any], path, value):
    d = cfg
    for k in path[:-1]:
        d = d.setdefault(k, {})
    d[path[-1]] = value

def suggest(trial, spec):
    kind, *args = spec
    if kind == "uniform":    return trial.suggest_float("p_"+str(trial.number)+"_"+str(len(trial.params)), *args)
    if kind == "loguniform": return trial.suggest_float("p_"+str(trial.number)+"_"+str(len(trial.params)), *args, log=True)
    if kind == "int":        return trial.suggest_int("p_"+str(trial.number)+"_"+str(len(trial.params)), *args)
    if kind == "choice":     return trial.suggest_categorical("p_"+str(trial.number)+"_"+str(len(trial.params)), args[0])
    raise ValueError(kind)

def build_cfg_from_trial(base_cfg: Dict[str, Any], trial) -> Dict[str, Any]:
    cfg = yaml.safe_load(yaml.dump(base_cfg))  # deep copy
    for path, spec in SEARCH_SPACE.items():
        val = suggest(trial, spec)
        set_nested(cfg, path, val)
    cfg.setdefault("eval", {}).setdefault("split", "temporal")
    return cfg

def run_backtest() -> Dict[str, Any]:
    # Clean prior artifact so we don't read stale data
    BACKTEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    if BACKTEST_JSON.exists():
        BACKTEST_JSON.unlink()
    # Call your existing CLI
    cmd = [sys.executable, "-m", "src.backtest", "--years", "5", "--cv", "time", "--out", str(BACKTEST_JSON)]
    r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if r.returncode != 0:
        print(r.stdout)
        raise RuntimeError("backtest command failed")
    if not BACKTEST_JSON.exists():
        # Fallback: still raise — tuner needs a metric
        print(r.stdout)
        raise FileNotFoundError(f"{BACKTEST_JSON} not produced by backtest")
    with open(BACKTEST_JSON, "r", encoding="utf-8") as f:
        return json.load(f)

def objective_from_metrics(metrics: Dict[str, Any]) -> float:
    # Prefer logloss if present; otherwise minimize (1 / (Sharpe+eps)) as a stable fallback
    if "logloss" in metrics and isinstance(metrics["logloss"], (int, float)):
        return float(metrics["logloss"])
    sharpe = float(metrics.get("sharpe", 0.0))
    return 1.0 / (abs(sharpe) + 1e-6)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = load_yaml(CFG_PATH)

    # Keep a pristine copy and restore after each trial
    base_copy = ROOT / "config.base.yaml"
    shutil.copyfile(CFG_PATH, base_copy)

    study = optuna.create_study(
        study_name="ncaaf-elite-agent-cli",
        storage=f"sqlite:///{OUT_DIR/'study.db'}",
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=8),
    )

    trial_rows = []
    iters = int(os.environ.get("TUNE_ITERS", "60"))

    def objective(trial):
        # Build temp config for the trial
        cfg = build_cfg_from_trial(base_cfg, trial)
        temp_cfg = ROOT / "config.temp.yaml"
        save_yaml(cfg, temp_cfg)

        # Swap in the temp config
        shutil.copyfile(temp_cfg, CFG_PATH)

        t0 = time.time()
        try:
            metrics = run_backtest()
        finally:
            # Always restore base file so repo stays clean
            shutil.copyfile(base_copy, CFG_PATH)

        dt = time.time() - t0
        value = objective_from_metrics(metrics)

        # Record trial info (lightweight CSV)
        flat_cfg = {}
        for path in SEARCH_SPACE.keys():
            d = cfg
            for k in path:
                d = d[k]
            flat_cfg[".".join(path)] = d
        flat_metrics = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in metrics.items()}
        row = {"trial": trial.number, "value": float(value), "seconds": dt, **flat_cfg, **flat_metrics}
        trial_rows.append(row)
        try:
            import pandas as pd
            pd.DataFrame(trial_rows).to_csv(TRIALS_CSV, index=False)
        except Exception:
            # Pandas optional; ignore if not installed
            with open(TRIALS_CSV, "w", encoding="utf-8") as f:
                f.write(",".join(row.keys()) + "\n")
                for r in trial_rows:
                    f.write(",".join(str(r[k]) for k in row.keys()) + "\n")

        # Log to Optuna user attrs
        for k, v in flat_metrics.items():
            if isinstance(v, (int, float)):
                trial.set_user_attr(k, float(v))
        return value

    study.optimize(objective, n_trials=iters, gc_after_trial=True)

    # Reconstruct best config by re-applying best trial params to base
    best = study.best_trial
    best_cfg = yaml.safe_load(yaml.dump(base_cfg))
    idx = 0
    for path, spec in SEARCH_SPACE.items():
        # Replay the suggestion order: use recorded params in the same sequence
        key = list(best.params.keys())[idx]
        set_nested(best_cfg, path, best.params[key])
        idx += 1
    save_yaml(best_cfg, BEST_CFG)

    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_value": study.best_value,
            "best_trial": best.number,
            "n_trials": len(study.trials),
            "metrics": {k: best.user_attrs.get(k) for k in best.user_attrs},
        }, f, indent=2)

    print(f"✅ best_config.yaml written to {BEST_CFG}")

if __name__ == "__main__":
    main()
