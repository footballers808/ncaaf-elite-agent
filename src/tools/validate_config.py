#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight config validator used by CI before jobs run.

- Loads YAML safely
- Ensures required sections/keys exist (with sensible defaults allowed)
- Validates types/ranges for knobs used in the workflow
- Exits non-zero on error so the CI step fails fast

Usage:
  python -m src.tools.validate_config --path config.yaml
  # or
  python src/tools/validate_config.py --path config.yaml
"""
import argparse
import sys
from typing import Any, Dict, Iterable

try:
    import yaml
except Exception:
    print("[ERR] pyyaml is required to validate config.", file=sys.stderr)
    sys.exit(2)


REQUIRED_TOP = ["features", "model"]
# Optional sections that we still sanity-check if present
OPTIONAL_SECTIONS = ["market", "edges", "tuning"]

def _expect(d: Dict[str, Any], key: str, types: Iterable[type], where: str):
    if key not in d:
        raise ValueError(f"Missing key '{key}' in {where}")
    if not isinstance(d[key], tuple(types)):
        ts = ", ".join(t.__name__ for t in types)
        raise TypeError(f"Key '{key}' in {where} must be type(s): {ts}")

def _check_features(cfg: Dict[str, Any]):
    feats = cfg["features"]
    _expect(feats, "windows", (list,), "features")
    if not feats["windows"]:
        raise ValueError("features.windows must be a non-empty list")
    if not all(isinstance(w, int) and w > 0 for w in feats["windows"]):
        raise TypeError("features.windows must be positive integers (weeks)")
    # allow more keys but forbid obviously wrong types if present
    if "use_penalties" in feats and not isinstance(feats["use_penalties"], bool):
        raise TypeError("features.use_penalties must be bool if present")

def _check_model(cfg: Dict[str, Any]):
    model = cfg["model"]
    _expect(model, "type", (str,), "model")
    if model["type"] not in ("logistic", "xgboost", "lightgbm", "gbm"):
        # keep permissive; just warn
        print(f"[WARN] Unrecognized model.type='{model['type']}', continuing…", file=sys.stderr)
    # generic hyperparams if present
    if "params" in model and not isinstance(model["params"], dict):
        raise TypeError("model.params must be a mapping if present")

def _check_edges(cfg: Dict[str, Any]):
    edges = cfg.get("edges")
    if not edges:
        return
    for k in ("edge_threshold_spread", "edge_threshold_total"):
        if k in edges and not isinstance(edges[k], (int, float)):
            raise TypeError(f"edges.{k} must be numeric")
        if k in edges and edges[k] < 0:
            raise ValueError(f"edges.{k} must be >= 0")

def _check_tuning(cfg: Dict[str, Any]):
    t = cfg.get("tuning")
    if not t:
        return
    if "trials" in t:
        if not isinstance(t["trials"], int) or t["trials"] <= 0:
            raise ValueError("tuning.trials must be a positive integer")
    if "pr_apply" in t and not isinstance(t["pr_apply"], bool):
        raise TypeError("tuning.pr_apply must be boolean if present")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="config.yaml")
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # required presence
    for sec in REQUIRED_TOP:
        if sec not in cfg:
            raise ValueError(f"Missing required top-level section: {sec}")

    _check_features(cfg)
    _check_model(cfg)
    _check_edges(cfg)
    _check_tuning(cfg)

    print(f"[OK] Config validated: {args.path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERR] {e}", file=sys.stderr)
        sys.exit(1)
