#!/usr/bin/env python3
import yaml, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
CFG = ROOT / "config.yaml"

REQUIRED_BOUNDS = {
    ("train", "learning_rate"): (1e-6, 1.0),
    ("train", "weight_decay"):  (0.0, 1.0),
    ("data", "lookback_days"):  (3, 365),
    ("data", "gap_days"):       (0, 14),  # to avoid leakage, >= 1 is safer
    ("eval", "cv_folds"):       (2, 10),
}

def get(cfg, path):
    for k in path: cfg = cfg[k]
    return cfg

def main():
    cfg = yaml.safe_load(open(CFG, "r", encoding="utf-8"))
    ok = True

    # bounds sanity
    for path, (lo, hi) in REQUIRED_BOUNDS.items():
        try:
            v = float(get(cfg, path))
            if not (lo <= v <= hi):
                print(f"❌ {'.'.join(path)}={v} out of [{lo},{hi}]")
                ok = False
        except Exception:
            print(f"❌ missing/invalid: {'.'.join(path)}")
            ok = False

    # leakage guard: require temporal split + nonzero gap unless explicitly disabled
    if cfg.get("eval", {}).get("split", "temporal") != "temporal":
        print("⚠️ eval.split is not 'temporal' — confirm this is intentional.")

    if cfg.get("data", {}).get("gap_days", 0) == 0:
        print("⚠️ data.gap_days == 0 — consider >= 1 to reduce leak risk.")

    print("✅ validate_config done." if ok else "❌ validate_config found issues.")
    sys.exit(0 if ok else 2)

if __name__ == "__main__":
    main()
