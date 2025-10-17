# src/tools/validate_config.py
from __future__ import annotations

import pathlib
import sys
from typing import Any, Dict

import yaml


# === defaults the pipeline expects ===
DEFAULTS: Dict[str, Any] = {
    "season_type": "regular",
    "form_games": 4,
    "pace_games": 4,
    "injury_window_days": 28,
    "injury_spread_weight_per_point": 0.5,
    "injury_total_weight_per_point": 0.3,
    "max_spread_adj_component": 3.0,
    "max_total_adj_component": 3.0,
    "weather_enabled": True,
    "weather_hours_before": 3,
}

DEFAULT_MODEL: Dict[str, Any] = {
    "type": "random_forest",
    "random_state": 42,
    "n_estimators": 400,
    "max_depth": 12,
    "min_samples_leaf": 2,
}


def _load_cfg(path: pathlib.Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise SystemExit("config.yaml must be a YAML mapping/object")
    return data


def _validate_and_fill(cfg: Dict[str, Any]) -> Dict[str, Any]:
    changed = False

    # model block required
    if "model" not in cfg or not isinstance(cfg.get("model"), dict):
        cfg["model"] = dict(DEFAULT_MODEL)
        changed = True

    # accept only known types; fall back if unknown
    model_type = str(cfg["model"].get("type", "")).lower()
    if model_type not in {"random_forest", "linear", "xgboost", "lightgbm"}:
        cfg["model"]["type"] = DEFAULT_MODEL["type"]
        changed = True

    # fill top-level defaults
    for k, v in DEFAULTS.items():
        if k not in cfg:
            cfg[k] = v
            changed = True

    return cfg, changed


def main() -> None:
    cfg_path = pathlib.Path("config.yaml")
    cfg = _load_cfg(cfg_path)
    cfg, changed = _validate_and_fill(cfg)

    if changed:
        # persist self-healed config so downstream steps agree on values
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        print("info: filled missing defaults into config.yaml")

    # minimal, friendly summary
    m = cfg["model"]
    print(
        "config ok -> "
        f"season_type={cfg['season_type']}, "
        f"form_games={cfg['form_games']}, pace_games={cfg['pace_games']}, "
        f"weather_enabled={cfg['weather_enabled']}, "
        f"model={m.get('type')} (n_estimators={m.get('n_estimators', '-')}, "
        f"max_depth={m.get('max_depth', '-')})"
    )


if __name__ == "__main__":
    main()
