# src/tune.py
import os, json, yaml, pathlib, shutil, time, subprocess, sys, random, textwrap
from typing import Dict, Any, Optional
import optuna

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG_PATH = ROOT / "config.yaml"
OUT_DIR = ROOT / "tuning-results"
BEST_CFG = OUT_DIR / "best_config.yaml"
TRIALS_CSV = OUT_DIR / "trials.csv"
ARTIFACTS_DIR = ROOT / "artifacts"
BACKTEST_OUT_DIR = ARTIFACTS_DIR / "backtest_out"

# If set to "1", we will return neutral metrics on repeated backtest failure
FAIL_OPEN = os.environ.get("TUNE_FAIL_OPEN", "0") == "1"

SEARCH_SPACE = {
    ("train","learning_rate"): ("loguniform", 1e-4, 5e-2),
    ("train","weight_decay"):  ("loguniform", 1e-7, 1e-2),
    ("train","max_depth"):     ("int", 3, 10),
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

def _s(trial, spec):
    kind, *args = spec
    key = f"p_{trial.number}_{len(trial.params)}"
    if kind == "uniform":    return trial.suggest_float(key, *args)
    if kind == "loguniform": return trial.suggest_float(key, *args, log=True)
    if kind == "int":        return trial.suggest_int(key, *args)
    if kind == "choice":     return trial.suggest_categorical(key, args[0])
    raise ValueError(kind)

def build_cfg_from_trial(base_cfg: Dict[str, Any], trial) -> Dict[str, Any]:
    cfg = yaml.safe_load(yaml.dump(base_cfg))
    for path, spec in SEARCH_SPACE.items():
        set_nested(cfg, path, _s(trial, spec))
    cfg.setdefault("eval", {}).setdefault("split", "temporal")
    return cfg

def _tail(s: str, n: int = 200) -> str:
    lines = [ln for ln in s.splitlines() if ln.strip() != ""]
    return "\n".join(lines[-n:])

def _find_new_json(snap_before: set[pathlib.Path]) -> Optional[pathlib.Path]:
    candidates = [p for p in ARTIFACTS_DIR.rglob("*.json") if p not in snap_before]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in candidates:
        if "backtest" in p.name.lower():
            return p
    return candidates[0]

def run_backtest() -> Dict[str, Any]:
    # Clean output dir
    if BACKTEST_OUT_DIR.exists():
        if BACKTEST_OUT_DIR.is_dir():
            shutil.rmtree(BACKTEST_OUT_DIR)
        else:
            BACKTEST_OUT_DIR.unlink()
    BACKTEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    before = set(ARTIFACTS_DIR.rglob("*.json"))
    cmd = [sys.executable, "-m", "src.backtest", "--years", "5", "--out", str(BACKTEST_OUT_DIR)]

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print("\n--- backtest stdout (attempt", attempt, ") ---")
        print(_tail(proc.stdout, 400))
        print("--- end backtest stdout ---\n")

        if proc.returncode == 0:
            p = _find_new_json(before) or max(BACKTEST_OUT_DIR.rglob("*.json"), default=None, key=lambda x: x.stat().st_mtime)
            if p is None:
                raise FileNotFoundError("Backtest succeeded but no JSON found in artifacts/")
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)

        # Non-zero: back off and retry
        if attempt < max_attempts:
            sleep_s = min(30, 2 ** attempt) + random.uniform(0, 0.5)
            print(f"Backtest failed (exit {proc.returncode}); retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)

    # Reached here = all attempts failed
    msg = "backtest command failed after retries"
    if FAIL_OPEN:
        print("WARNING:", msg, "— returning neutral metrics to continue tuning.")
        return {
            "overall": {"logloss": 0.693147, "brier": 0.25, "roi": 0.0, "sharpe": 0.0},
            "failed": True,
            "note": msg,
        }
    raise RuntimeError(msg)

def objective_from_metrics(metrics: Dict[str, Any]) -> float:
    overall = metrics.get("overall", {})
    if "logloss" in overall and isinstance(overall["logloss"], (int, float)):
        return float(overall["logloss"])
    # Fallback to inverse Sharpe if logloss missing
    sharpe = float(overall.get("sharpe", 0.0))
    return 1.0 / (abs(sharpe) + 1e-6)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    base_cfg = load_yaml(CFG_PATH)
    base_copy = ROOT / "config.base.yaml"
    shutil.copyfile(CFG_PATH, base_copy)

    study = optuna.create_study(
        study_name="ncaaf-elite-agent-cli",
        storage=f"sqlite:///{OUT_DIR/'study.db'}",
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=6),
    )

    # Trial CSV
    header_written = False
    iters = int(os.environ.get("TUNE_ITERS", "8"))

    def write_row(row: Dict[str, Any]):
        nonlocal header_written
        TRIALS_CSV.parent.mkdir(parents=True, exist_ok=True)
        if not header_written or not pathlib.Path(TRIALS_CSV).exists():
            with open(TRIALS_CSV, "w", encoding="utf-8") as f:
                f.write(",".join(row.keys()) + "\n")
            header_written = True
        with open(TRIALS_CSV, "a", encoding="utf-8") as f:
            f.write(",".join(str(row[k]) for k in row.keys()) + "\n")

    def objective(trial):
        # Build temp config
        cfg = build_cfg_from_trial(base_cfg, trial)
        temp_cfg = ROOT / "config.temp.yaml"
        save_yaml(cfg, temp_cfg)
        shutil.copyfile(temp_cfg, CFG_PATH)

        t0 = time.time()
        try:
            metrics = run_backtest()
        finally:
            shutil.copyfile(base_copy, CFG_PATH)
        dt = time.time() - t0

        overall = metrics.get("overall", {})
        value = objective_from_metrics(metrics)

        # Flatten config used
        flat = {"trial": trial.number, "value": float(value), "seconds": round(dt, 3)}
        for path in SEARCH_SPACE:
            d = cfg
            for k in path:
                d = d[k]
            flat[".".join(path)] = d
        for k, v in overall.items():
            if isinstance(v, (int, float)):
                flat[k] = float(v)
                trial.set_user_attr(k, float(v))
        write_row(flat)

        # Gentle pause between trials
        time.sleep(1.5)
        return value

    study.optimize(objective, n_trials=iters, gc_after_trial=True)

    best = study.best_trial
    best_cfg = yaml.safe_load(yaml.dump(base_cfg))
    for i, path in enumerate(SEARCH_SPACE.keys()):
        key = list(best.params.keys())[i]
        set_nested(best_cfg, path, best.params[key])
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
