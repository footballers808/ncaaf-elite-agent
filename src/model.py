# src/model.py
from __future__ import annotations
import pathlib
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd

from src.feat_penalties import load_penalty_features, features_for_game

ROOT = pathlib.Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
MODEL_PATHS = [ART / "model.joblib", ART / "model.bin", ART / "model.pkl"]

# Module-level caches
_PEN_DF: Optional[pd.DataFrame] = None
_MODEL: Optional[Any] = None
_MODEL_META: Dict[str, Any] = {}

def _load_pen_df() -> pd.DataFrame:
    global _PEN_DF
    if _PEN_DF is None:
        _PEN_DF = load_penalty_features()
    return _PEN_DF

def _find_model_path() -> Optional[pathlib.Path]:
    for p in MODEL_PATHS:
        if p.exists():
            return p
    return None

def load_model() -> Optional[Any]:
    """Load a trained sklearn model bundle if present."""
    global _MODEL, _MODEL_META
    if _MODEL is not None:
        return _MODEL
    mp = _find_model_path()
    if not mp:
        return None
    obj = joblib.load(mp)
    if isinstance(obj, dict) and "estimator" in obj:
        _MODEL = obj["estimator"]
        _MODEL_META = obj.get("meta", {})
    else:
        _MODEL = obj
        _MODEL_META = {}
    return _MODEL

def model_meta() -> Dict[str, Any]:
    return dict(_MODEL_META)

def game_features_vector(game: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, float]:
    """Assemble features for ONE game (currently: rolling penalties)."""
    wins = tuple(cfg.get("features", {}).get("penalties", {}).get("windows", [3, 5, 10]))
    pen_df = _load_pen_df()
    fx = features_for_game(game, pen_df, windows=wins)
    return fx

def predict_home_win_prob(game: Dict[str, Any], cfg: Dict[str, Any]) -> float:
    """
    Predict P(home win) using trained model if available.
    Falls back to 0.5 if no model or features are missing.
    """
    est = load_model()
    x = game_features_vector(game, cfg)
    if not est:
        return 0.5

    cols = _MODEL_META.get("feature_order") or sorted(x.keys())
    if not cols:
        return 0.5

    v = np.array([[float(x.get(c, np.nan)) for c in cols]], dtype=float)

    means: Optional[Dict[str, float]] = _MODEL_META.get("feature_means")
    if means:
        for i, c in enumerate(cols):
            if np.isnan(v[0, i]):
                v[0, i] = float(means.get(c, 0.0))
    else:
        v = np.nan_to_num(v, nan=0.0)

    scaler_ref = _MODEL_META.get("scaler")
    if scaler_ref:
        try:
            scaler = joblib.load(ROOT / scaler_ref) if isinstance(scaler_ref, str) else None
            if scaler is not None:
                v = scaler.transform(v)
        except Exception:
            pass

    try:
        proba = est.predict_proba(v)[0, 1]
        return float(proba)
    except Exception:
        return 0.5
